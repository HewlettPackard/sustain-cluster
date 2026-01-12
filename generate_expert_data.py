# generate_expert_data.py

import pandas as pd
import json
import os
import yaml
import datetime
import numpy as np
import random
from tqdm import tqdm

from envs.task_scheduling_env import TaskSchedulingEnv
from simulation.cluster_manager import DatacenterClusterManager
from rewards.predefined.composite_reward import CompositeReward
from utils.llm_serialization import EnvSerializer

# --- CONFIGURATION ---
NUM_SIMULATIONS = 10      # How many episodes to generate
DAYS_PER_SIM = 30         # Duration of each episode
OUTPUT_FILE = "data/expert_trajectories/heuristic_dataset_v2.json"

# --- HELPER FUNCTIONS ---
def load_yaml(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def policy_lowest_carbon(cluster_manager):
    """Returns index of DC with lowest Carbon Intensity"""
    dcs = list(cluster_manager.datacenters.values())
    ci_values = [dc.ci_manager.get_current_ci(norm=False) for dc in dcs]
    best_idx = np.argmin(ci_values)
    return int(best_idx), "Lowest Carbon Intensity"

def policy_lowest_price(cluster_manager):
    """Returns index of DC with lowest Electricity Price"""
    dcs = list(cluster_manager.datacenters.values())
    price_values = [dc.price_manager.get_current_price() for dc in dcs]
    best_idx = np.argmin(price_values)
    return int(best_idx), "Lowest Energy Price"

def get_random_start_time(year):
    """Generates a random valid start date and corresponding day-of-year."""
    # Pick a random month (1-11) to allow room for a 30-day run
    month = random.randint(1, 11) 
    
    # Pick a random day (1-28) to stay safe across all months
    day = random.randint(1, 28)
    
    # Pick a random hour
    hour = random.randint(0, 23)
    
    start_date = datetime.datetime(year, month, day, hour, 0, tzinfo=datetime.timezone.utc)
    
    # Calculate Day of Year (0-364) for SustainCluster logic
    day_of_year = start_date.timetuple().tm_yday - 1
    
    return start_date, day_of_year, hour

def main():
    # 1. LOAD BASE CONFIGS
    sim_cfg = load_yaml("configs/env/sim_config.yaml")["simulation"]
    dc_cfg = load_yaml("configs/env/datacenters.yaml")["datacenters"]
    reward_cfg = load_yaml("configs/env/reward_config.yaml")["reward"]
    
    output_data = []
    
    print(f"🚀 Starting Data Generation: {NUM_SIMULATIONS} Sims x {DAYS_PER_SIM} Days")

    # Global Progress Bar
    total_steps_expected = NUM_SIMULATIONS * DAYS_PER_SIM * 24 * 4 # 15 min steps
    pbar = tqdm(total=total_steps_expected, desc="Generating Trajectories")

    # 2. SIMULATION LOOP
    for sim_idx in range(NUM_SIMULATIONS):
        
        # A. Randomize Start Time
        start_dt, init_day_of_year, init_hour = get_random_start_time(sim_cfg["year"])
        
        # FIX: Convert to Pandas Timestamp for the Environment
        start_time = pd.Timestamp(start_dt)
        end_time = start_time + pd.Timedelta(days=DAYS_PER_SIM)
        
        # Update Description
        pbar.set_description(f"Sim {sim_idx+1}/{NUM_SIMULATIONS} | Start: {start_time.strftime('%Y-%m-%d')}")

        # B. Initialize Cluster with Random Start
        cluster = DatacenterClusterManager(
            config_list=dc_cfg,
            simulation_year=sim_cfg["year"],
            init_day=init_day_of_year,
            init_hour=init_hour,
            strategy="manual_rl",
            tasks_file_path=sim_cfg["workload_path"],
            cloud_provider=sim_cfg["cloud_provider"]
        )
        
        reward_fn = CompositeReward(components=reward_cfg["components"])
        
        # C. Initialize Environment
        env = TaskSchedulingEnv(cluster, start_time, end_time, reward_fn)
        serializer = EnvSerializer(cluster)
        
        obs, _ = env.reset(seed=42 + sim_idx) # Vary seed slightly just in case
        
        step_count = 0
        
        # D. Run Episode
        while env.current_time < env.end_time:
            # 1. Serialize State
            state_text = serializer.serialize_state(env, env.current_tasks)
            
            # 2. Choose Action (Teacher Policy)
            # 50/50 Split between Carbon and Price focus
            if np.random.rand() < 0.5:
                action, strategy_name = policy_lowest_carbon(cluster)
            else:
                action, strategy_name = policy_lowest_price(cluster)
            
            # 3. Store Data (Only if tasks exist and state_text valid)
            if state_text and len(env.current_tasks) > 0:
                data_point = {
                    "sim_id": sim_idx,
                    "global_step": len(output_data),
                    "prompt": state_text,
                    "chosen_action": action,
                    "strategy": strategy_name,
                    "timestamp": str(env.current_time)
                }
                output_data.append(data_point)

            # 4. Step Environment
            obs, reward, done, truncated, info = env.step(action)
            
            step_count += 1
            pbar.update(1)
            
            if done: break
            
        # Clean up to free memory before next sim
        del env
        del cluster

    pbar.close()

    # 3. SAVE TO JSON
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\n✅ Generation Complete.")
    print(f"   Total Samples: {len(output_data)}")
    print(f"   Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()