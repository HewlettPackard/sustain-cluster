# test_random_agent.py
#%%
import os
import sys
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# --- Add project root to Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Local Imports ---
from envs.sustaincluster_ma_env import SustainClusterMAEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rewards.registry_utils import get_reward_function
from rewards.predefined.composite_reward import CompositeReward
from utils.config_loader import load_yaml
from utils.config_logger import setup_logger

#%%

"""
Initializes the simplified SustainCluster environment and runs a full episode
with a random agent to test for stability and basic functionality.
"""
print("--- Initializing Test for Random Agent ---")

# --- 1. Load Configurations ---
# We use the same configuration files as the training script
try:
    sim_cfg_full = load_yaml("configs/env/sim_config_ma.yaml")
    dc_cfg_full = load_yaml("configs/env/datacenters_ma.yaml")
    reward_cfg_full = load_yaml("configs/env/reward_config_manager_ci_only.yaml")
    
    sim_cfg = sim_cfg_full["simulation"]
    dc_cfg = dc_cfg_full["datacenters"]
    reward_cfg = reward_cfg_full["reward"]
    
    # Ensure your dc_cfg has been simplified to 3 datacenters
    if len(dc_cfg) != 3:
        print(f"WARNING: Found {len(dc_cfg)} datacenters. This test is designed for the simplified 3-DC setup.")

except FileNotFoundError as e:
    print(f"ERROR: Configuration file not found. Make sure you are running from the project root.")
    print(e)

# --- 2. Create the Environment in Simple Mode ---
print("Creating the environment in 'simple_obs_mode'...")

# We don't need a real logger for this test
run_name = f"test_random_agent_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
# Set up a dummy logger that writes to a file
dummy_logger = setup_logger(f"logs/{run_name}", enable_logger=True)

# Create the reward function instance
if "name" in reward_cfg:
    reward_fn = get_reward_function(reward_cfg["name"], **reward_cfg.get("args", {}))
else:
    raise ValueError("Reward config must have a 'name' for this test.")

for cfg in dc_cfg:
    cfg.setdefault("simulation_year", sim_cfg["year"])

    
# Create the cluster manager backend
cluster = DatacenterClusterManagerMA(
    config_list=dc_cfg,
    simulation_year=sim_cfg["year"],
    tasks_file_path=sim_cfg["workload_path"],
    cloud_provider=sim_cfg["cloud_provider"],
    max_total_options=sim_cfg["max_total_options"],
    logger=dummy_logger,
)
EVALUATION_DURATION_DAYS = 7

# Create the environment instance, explicitly setting simple_obs_mode=True
env = SustainClusterMAEnv(
    cluster_manager_ma=cluster,
    start_time=pd.Timestamp(f"{sim_cfg['year']}-{sim_cfg['month']}-{sim_cfg['init_day']}"),
    end_time=pd.Timestamp(f"{sim_cfg['year']}-{sim_cfg['month']}-{sim_cfg['init_day'] + EVALUATION_DURATION_DAYS}"), # Run for 7 days
    reward_fn=reward_fn,
    logger=dummy_logger,
    simple_obs_mode=True,
    use_meta_tasks=True
)

print("Environment created successfully.")
print(f"Observation Space: {env.observation_space('manager_1')}")
print(f"Action Space: {env.action_space('manager_1')}")

# --- 3. Run the Test Episode ---
seed = 42
print(f"\n--- Starting Episode with Seed {seed} ---")
all_step_infos = []

try:
    obs_dict, info = env.reset(seed=seed)
    done = False
    truncated = False
    total_reward = 0.0
    step_count = 0

    # Use tqdm for a progress bar
    pbar = tqdm(desc="Running Random Actions", total=EVALUATION_DURATION_DAYS * 24 * 4)
    
    while not (done or truncated):
        # --- Assemble the action dictionary with random actions ---
        actions_dict = {}
        for agent_id in env.manager_agent_ids:
            # Sample a random action from the manager's action space
            actions_dict[agent_id] = env.action_space(agent_id).sample()
        
        # Add the fixed worker policy
        actions_dict.update({agent_id: 1 for agent_id in env.worker_agent_ids})

        # --- Step the environment ---
        next_obs_dict, rewards_dict, dones_dict, trunc_dict, infos_dict = env.step(actions_dict)
        all_step_infos.append(infos_dict)

        # --- Update state and track metrics ---
        obs_dict = next_obs_dict
        done = dones_dict["__all__"]
        truncated = trunc_dict["__all__"]
        
        # Calculate the average global reward for this step
        step_reward = np.mean(list(rewards_dict.values()))
        total_reward += step_reward
        step_count += 1
        
        pbar.update(1)
        pbar.set_postfix({"step": step_count, "last_reward": f"{step_reward:.3f}", "total_reward": f"{total_reward:.2f}"})

    pbar.close()
    print("\n--- Episode Finished ---")
    print(f"Termination Reason: {'Done' if done else 'Truncated'}")
    print(f"Total Steps: {step_count}")
    print(f"Total Cumulative Reward: {total_reward:.2f}")
    print(f"Average Reward per Step: {total_reward / step_count:.4f}")
    
    print("\n✅ Test Passed: Environment ran a full episode without crashing.")

except Exception as e:
    print(f"\n❌ Test Failed: An error occurred during the simulation.")
    import traceback
    traceback.print_exc()
    
finally:
    env.close()

#%%
# --- 4. Metrics Aggregation and Analysis (Copied from your eval script) ---
print("\n--- Aggregating Results for Random Agent ---")

flat_records = []
for t, step_info in enumerate(all_step_infos):
    raw_results = step_info.get("__global__", {}).get("raw_results", {})
    dc_infos = raw_results.get("datacenter_infos", {})
    
    for dc_id, dc_info_step in dc_infos.items():
        common = dc_info_step.get("__common__", {})
        sla_info = common.get("__sla__", {"met": 0, "violated": 0})
        
        record = {
            "timestep": t,
            "datacenter_id": dc_id,
            "energy_kwh": common.get("energy_consumption_kwh", 0.0),
            "carbon_kg": common.get("carbon_emissions_kg", 0.0),
            "cpu_util_pct": common.get("cpu_util_percent", 0.0),
            "gpu_util_pct": common.get("gpu_util_percent", 0.0),
            "running_tasks": common.get("running_tasks", 0),
            "sla_met": sla_info.get("met", 0),
            "sla_violated": sla_info.get("violated", 0),
        }
        flat_records.append(record)

if not flat_records:
    print("No data collected, skipping analysis and plotting.")
#%%
df_results = pd.DataFrame(flat_records)

summary = df_results.groupby("datacenter_id").agg(
    total_energy_kwh=("energy_kwh", "sum"),
    total_carbon_kg=("carbon_kg", "sum"),
    avg_cpu_util_pct=("cpu_util_pct", "mean"),
    avg_gpu_util_pct=("gpu_util_pct", "mean"),
    total_sla_met=("sla_met", "sum"),
    total_sla_violated=("sla_violated", "sum"),
).reset_index()

print("\n--- Random Agent Performance Summary per Datacenter ---")
print(summary.round(2))
#%%
# --- 5. Plotting Section ---
print("\n--- Generating Plots ---")
sns.set_theme(style="whitegrid")

dc_name_to_location_label = {dc_cfg['dc_id']: dc_cfg['location'] for dc_cfg in dc_cfg_full['datacenters']}
df_results['datacenter_label'] = df_results['datacenter_id'].map(dc_name_to_location_label)

# Plot Running Tasks per Datacenter
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_results, x="timestep", y="running_tasks", hue="datacenter_label")
plt.title("Running Tasks per Datacenter (Random Agent)")
plt.xlabel("Timestep (15 min intervals)")
plt.ylabel("Number of Running 'Meta-Tasks'")
plt.tight_layout()
plt.show()

# Plot CPU Utilization
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_results, x="timestep", y="cpu_util_pct", hue="datacenter_label")
plt.title("CPU Utilization (%) (Random Agent)")
plt.xlabel("Timestep (15 min intervals)")
plt.ylabel("CPU Utilization (%)")
plt.tight_layout()
plt.show()

# Plot Carbon Emissions
plt.figure(figsize=(12, 6))
# Plotting the cumulative sum can be insightful
df_results['cumulative_carbon_kg'] = df_results.groupby('datacenter_id')['carbon_kg'].cumsum()
sns.lineplot(data=df_results, x="timestep", y="cumulative_carbon_kg", hue="datacenter_label")
plt.title("Cumulative Carbon Emissions (kg CO₂) (Random Agent)")
plt.xlabel("Timestep (15 min intervals)")
plt.ylabel("Cumulative kg CO₂")
plt.tight_layout()
plt.show()
#%%