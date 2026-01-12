import argparse
import pandas as pd
import numpy as np
import datetime
import os
import yaml
from tqdm import tqdm

from envs.task_scheduling_env import TaskSchedulingEnv
from simulation.cluster_manager import DatacenterClusterManager
from rewards.predefined.composite_reward import CompositeReward

# Import Controllers
from controllers.heuristics import RandomController, LowestCarbonController, LowestPriceController
from controllers.llm_controller import LLMController

def load_yaml(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def get_controller(name, cluster, **kwargs):
    if name == 'random': return RandomController(cluster)
    if name == 'carbon': return LowestCarbonController(cluster)
    if name == 'price': return LowestPriceController(cluster)
    if name == 'llm': return LLMController(cluster, **kwargs)
    raise ValueError(f"Unknown agent: {name}")

def run_evaluation(agent_name, use_rag=False, episodes=1, **model_kwargs):
    # --- SETUP ---
    sim_cfg = load_yaml("configs/env/sim_config.yaml")["simulation"]
    dc_cfg = load_yaml("configs/env/datacenters.yaml")["datacenters"]
    reward_cfg = load_yaml("configs/env/reward_config.yaml")["reward"]

    # Evaluation usually runs on unseen data or a specific test month
    # Let's shift the date slightly to ensure we aren't testing on training data
    start = pd.Timestamp(datetime.datetime(sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"] + 10,
                                           sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=2) # Short eval for testing

    cluster = DatacenterClusterManager(
        config_list=dc_cfg,
        simulation_year=sim_cfg["year"],
        init_day=int(sim_cfg["month"] * 30.5) + 10,
        init_hour=sim_cfg["init_hour"],
        strategy="manual_rl",
        tasks_file_path=sim_cfg["workload_path"],
        cloud_provider=sim_cfg["cloud_provider"]
    )
    
    env = TaskSchedulingEnv(cluster, start, end, CompositeReward(reward_cfg["components"]))
    
    # Init Controller
    agent = get_controller(agent_name, cluster, use_rag=use_rag, **model_kwargs)
    print(f"Starting Evaluation for Agent: {agent.name}")

    episode_metrics = []

    # --- LOOP ---
    for ep in range(episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        
        # Accumulators
        total_carbon = 0
        total_water = 0
        total_cost = 0
        total_energy = 0
        total_sla_violated = 0
        total_reward = 0
        
        # For Averages
        pue_values = []
        latency_values = []
        steps = 0
        
        pbar = tqdm(desc=f"Episode {ep+1}")
        
        while not done:
            action, meta = agent.get_action(env, obs)
            obs, reward, done, truncated, info = env.step(action)
            
            # Aggregate Totals
            total_carbon += info['total_carbon_kg']
            total_water += info['total_water_liters']
            total_cost += info['total_cost_usd']
            total_energy += info['total_energy_kwh']
            total_sla_violated += info['sla_violated']
            total_reward += reward
            
            # Collect for Averages
            pue_values.append(info['cluster_pue'])
            if info['avg_latency_min'] > 0:
                latency_values.append(info['avg_latency_min'])

            steps += 1
            pbar.update(1)
        
        pbar.close()

        # Compute Episode Summary
        episode_metrics.append({
            "Agent": agent.name,
            "Episode": ep,
            "Carbon_kg": total_carbon,
            "Water_L": total_water,
            "Cost_$": total_cost,
            "Energy_kWh": total_energy,
            "SLA_Violations": total_sla_violated,
            "Avg_PUE": np.mean(pue_values) if pue_values else 0,
            "Avg_Latency_min": np.mean(latency_values) if latency_values else 0,
            "Total_Reward": total_reward
        })
        pbar.close()

    # --- SAVE RESULTS ---
    df = pd.DataFrame(episode_metrics)
    
    print("\n--- RESULTS SUMMARY ---")
    print(df.mean(numeric_only=True))
    
    os.makedirs("results/evaluation", exist_ok=True)
    filename = f"results/evaluation/eval_{agent.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved detailed results to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default='random', choices=['random', 'carbon', 'price', 'llm'])
    parser.add_argument("--use_rag", action="store_true", help="Enable RAG for LLM agent")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--use_history", action="store_true", help="Enable short-term history window")
    
    parser.add_argument("--model_path", type=str, default="mock", 
                        help="Path to local HF model folder (e.g. ./models/Llama-3-8B). Use 'mock' for debugging.")

    args = parser.parse_args()

    # Pass the model_path to the controller
    run_evaluation(args.agent, 
                   use_rag=args.use_rag, 
                   episodes=args.episodes,
                   # We pass these as kwargs to LLMController via get_controller
                   model_path=args.model_path,
                   use_history=args.use_history 
                   )