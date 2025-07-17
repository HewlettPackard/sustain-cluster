# collect_reward_stats.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from rewards.predefined.composite_reward import CompositeReward
from rewards.registry_utils import get_reward_function
# Important: Make sure this script can import from your project structure
from train_worker_only_sb3 import make_env  # Assuming your main script is train_worker_multi.py
from utils.config_loader import load_yaml

# --- Configuration ---
SIM_CFG = "configs/env/sim_config_ma.yaml"
DC_CFG = "configs/env/datacenters_ma.yaml"
# Use the reward config you want to analyze
REWARD_CFG = "configs/env/reward_config_worker_ci_only.yaml"
NUM_STEPS_TO_RUN = 20000  # A good number to get a decent sample

def build_reward(rwd_cfg: dict):
    if "components" in rwd_cfg:
        return CompositeReward(
            components=rwd_cfg["components"],
            normalize=rwd_cfg.get("normalize", False),
            freeze_stats_after_steps=rwd_cfg.get("freeze_stats_after_steps"),
        )
    if "name" in rwd_cfg:
        return get_reward_function(rwd_cfg["name"], **rwd_cfg.get("args", {}))
    raise ValueError("Reward YAML must contain either 'components' or 'name'.")

def main():
    print("Creating a single environment to collect reward statistics...")
    
    sim_cfg    = load_yaml(SIM_CFG)["simulation"]
    dc_cfg     = load_yaml(DC_CFG)["datacenters"]
    reward_cfg = load_yaml(REWARD_CFG)["reward"]


    # Create a single, non-parallel environment
    env = make_env(SIM_CFG, DC_CFG, REWARD_CFG)
    
    # We need access to the reward function to calculate individual components
    reward_fn = build_reward(reward_cfg)

    # Lists to store the raw component values at each step
    throughputs = []
    carbon_penalties = []
    sla_penalties = []
    aging_penalties = []

    print(f"Running for {NUM_STEPS_TO_RUN} steps with a random agent...")
    obs, _ = env.reset()

    for _ in tqdm(range(NUM_STEPS_TO_RUN)):
        action = env.action_space.sample()  # Take a random action
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Re-calculate reward components using the info dict ---
        # This mirrors the logic inside your WorkerThroughputReward function
        cluster_info = info.get("raw_results", {})
        
        # Throughput
        num_tasks_completed = sum(dc["__common__"].get("finished_tasks_count", 0) for dc in cluster_info.get("datacenter_infos", {}).values())
        throughputs.append(num_tasks_completed) # Use raw value, not multiplied by weight
        
        # Carbon
        total_kg_co2 = sum(dc["__common__"].get("carbon_emissions_kg", 0.0) for dc in cluster_info.get("datacenter_infos", {}).values())
        carbon_penalties.append(-total_kg_co2) # Raw penalty
        
        # SLA
        num_sla_violations = sum(dc["__common__"].get("__sla__", {}).get("violated", 0) for dc in cluster_info.get("datacenter_infos", {}).values())
        sla_penalties.append(-num_sla_violations) # Raw penalty
        
        # Aging
        task_info = info.get("task_info", {})
        total_queue_age_minutes = sum(dc_info.get("age_mean", 0.0) * dc_info.get("queue_length", 0) for dc_info in task_info.values())
        aging_penalties.append(-total_queue_age_minutes) 

        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()

    # --- Calculate and Print Statistics ---
    print("\n--- Reward Component Statistics ---")
    stats = {}
    
    components = {
        "throughput": throughputs,
        "carbon": carbon_penalties,
        "sla": sla_penalties,
        "aging": aging_penalties,
    }

    for name, data in components.items():
        if data:
            stats[name] = {
                "mean": np.mean(data),
                "std": np.std(data),
                "min": np.min(data),
                "max": np.max(data),
            }
            print(f"\nComponent: {name}")
            print(f"  Mean: {stats[name]['mean']:.6f}")
            print(f"  Std Dev: {stats[name]['std']:.6f}")
            print(f"  Min: {stats[name]['min']:.6f}")
            print(f"  Max: {stats[name]['max']:.6f}")
        else:
            print(f"\nComponent: {name} - No data collected.")
            
    print("\n--- YAML Configuration Snippet ---")
    print("Copy these 'mean' and 'std' values into your reward config file:")
    print("\nnormalization_stats:")
    for name, stat_values in stats.items():
        print(f"  {name}:")
        print(f"    mean: {stat_values['mean']:.6f}")
        print(f"    std: {stat_values['std']:.6f}")


if __name__ == "__main__":
    main()