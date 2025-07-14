# %%
import sys
import os
import copy
import datetime
import logging
import random

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

sns.set_theme(style="whitegrid")

# --- Local Imports ---
from envs.sustaincluster_ma_env import SustainClusterMAEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
# ### --- SIMPLIFICATION CHANGE: Import the new simple MLP actor --- ###
from rl_components.agent_net_simple import ManagerActorMLP 
from utils.config_loader import load_yaml
from rewards.registry_utils import get_reward_function
from rewards.predefined.composite_reward import CompositeReward
from utils.vec_normalize import VecNormalize # <-- Add this import
from utils.controllers import BaseController, RandomController, LowestCarbonController, LowestTemperatureController, PPOController

# --- Configuration ---
CONFIG_DIR = "configs/env"
BASE_SIM_CONFIG_PATH = os.path.join(CONFIG_DIR, "sim_config_ma.yaml")
BASE_DC_CONFIG_PATH = os.path.join(CONFIG_DIR, "datacenters_ma.yaml")
BASE_REWARD_CONFIG_PATH = os.path.join(CONFIG_DIR, "reward_config_manager_ci_only.yaml")
# ### --- SIMPLIFICATION CHANGE: Point to the PPO config --- ###
BASE_ALGO_CONFIG_PATH = os.path.join(CONFIG_DIR, "ppo_algorithm_config.yaml")


# --- IMPORTANT: Set this to your trained Manager agent's checkpoint ---
# Example path, you MUST update this to your actual checkpoint file
DEFAULT_RL_CHECKPOINT_PATH = "checkpoints/PPO_MGR_ONLY_128hidden_20250708_205052/best_model.pth"

# Evaluation settings
EVALUATION_DURATION_DAYS = 7
SEED = 125  # We will evaluate on a single, fixed seed for reproducibility

# --- Logger and Output Setup ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/eval_manager_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"evaluation_manager_{timestamp}.log")
results_csv_path = os.path.join(log_dir, f"results_manager_{timestamp}.csv")
logger = logging.getLogger("eval_manager_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(file_handler)
logger.info(f"Evaluation log will be saved to: {log_path}")
print(f"Evaluation log will be saved to: {log_path}")


# --- Environment Creation Function ---
def make_eval_env(base_sim_cfg, base_dc_cfg, base_reward_cfg, duration_days, seed, simple_obs_mode=True):
    """Creates the SustainCluster-MA evaluation environment."""
    sim_cfg = copy.deepcopy(base_sim_cfg)
    dc_cfg = copy.deepcopy(base_dc_cfg)
    reward_cfg = copy.deepcopy(base_reward_cfg)

    sim_cfg["simulation"]["duration_days"] = duration_days

    # This function was updated in the training script to handle single vs. composite rewards
    if "components" in reward_cfg["reward"]:
        reward_fn = CompositeReward(components=reward_cfg["reward"]["components"], normalize=False)
    elif "name" in reward_cfg["reward"]:
        reward_name = reward_cfg["reward"]["name"]
        reward_args = reward_cfg["reward"].get("args", {})
        reward_fn = get_reward_function(reward_name, **reward_args)
    else:
        raise ValueError("Invalid reward configuration.")
    
    for cfg in dc_cfg["datacenters"]:
        cfg.setdefault("simulation_year", sim_cfg["simulation"]["year"])
        

    cluster = DatacenterClusterManagerMA(
        config_list=dc_cfg["datacenters"],
        simulation_year=sim_cfg["simulation"]["year"],
        tasks_file_path=sim_cfg["simulation"]["workload_path"],
        cloud_provider=sim_cfg["simulation"]["cloud_provider"],
        max_total_options=sim_cfg["simulation"]["max_total_options"],
        duration_days=sim_cfg["simulation"]["duration_days"],
        
        logger=logger,
    )

    start = pd.Timestamp(datetime.datetime(sim_cfg["simulation"]["year"], sim_cfg["simulation"]["month"], sim_cfg["simulation"]["init_day"],
                                           sim_cfg["simulation"]["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=duration_days)
    

    env = SustainClusterMAEnv(
        cluster_manager_ma=cluster, start_time=start, end_time=end,
        reward_fn=reward_fn, logger=logger, simple_obs_mode=simple_obs_mode
    )
    return env


# %%
# --- Main Evaluation Script ---
logger.info("--- Starting PPO Agent Evaluation Run ---")
logger.info(f"RL Checkpoint: {DEFAULT_RL_CHECKPOINT_PATH}")
logger.info(f"Seed: {SEED}, Duration: {EVALUATION_DURATION_DAYS} days")

# Load all base configurations
base_sim_cfg_dict = load_yaml(BASE_SIM_CONFIG_PATH)
base_dc_cfg_dict = load_yaml(BASE_DC_CONFIG_PATH)
base_reward_cfg_dict = load_yaml(BASE_REWARD_CONFIG_PATH)
algo_cfg_dict = load_yaml(BASE_ALGO_CONFIG_PATH)

# --- Load the Trained Manager Actor ---
if not os.path.exists(DEFAULT_RL_CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found at {DEFAULT_RL_CHECKPOINT_PATH}")

device = torch.device("cpu")
# NOTE: The checkpoint saving function might need to be adjusted.
# Assuming it saves a dictionary with 'actor_state_dict'.
checkpoint = torch.load(DEFAULT_RL_CHECKPOINT_PATH, map_location=device, weights_only=False)

# ### --- SIMPLIFICATION CHANGE: Recreate the simple MLP actor --- ###
# Dynamically get network dimensions from the simplified environment
base_eval_env = make_eval_env(base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict, 1, SEED, simple_obs_mode=True)
temp_env = VecNormalize(venv=base_eval_env)
first_mgr_id = f"manager_{temp_env.venv._dc_ids[0]}"
obs_dim = temp_env.observation_space(first_mgr_id).shape[0]
action_dim = temp_env.action_space(first_mgr_id).n
del temp_env

# Re-create the MLP actor network
actor = ManagerActorMLP(obs_dim, action_dim, hidden_dim=128).to(device)
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.eval()
logger.info("Successfully loaded PPO Manager Actor policy.")

# --- Create the final evaluation environment in simple mode ---
base_eval_env = make_eval_env(base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict,
                    EVALUATION_DURATION_DAYS, SEED, simple_obs_mode=True)
env = VecNormalize(venv=base_eval_env)

# Load the saved running mean/std from the checkpoint
if 'obs_rms' in checkpoint:
    env.obs_rms = checkpoint['obs_rms']
    logger.info("Successfully loaded observation normalization stats.")
else:
    print("WARNING: No observation normalization stats found in checkpoint. Using default stats.")
    
env.eval() 

# %%
# --- Simulation Loop ---
obs_dict, _ = env.reset(seed=SEED)
num_steps = EVALUATION_DURATION_DAYS * 24 * 4
all_step_infos = []

for step in tqdm(range(num_steps), desc=f"Simulating PPO Agent (Seed {SEED})"):

    # ### --- SIMPLIFICATION CHANGE: Updated simulation loop --- ###

    # 1. Prepare observations for the Manager actor from the current `obs_dict`.
    #    The observation is now a flattened array for each agent.
    obs_list = [obs_dict[f"manager_{dc_id}"] for dc_id in env._dc_ids]
    
    # Create a single batch tensor for the model
    obs_tensor = torch.from_numpy(np.stack(obs_list)).float().to(device)

    # 2. Get a deterministic action `a_t` from the actor.
    with torch.no_grad():
        logits = actor(obs_tensor)
        actions = torch.argmax(logits, dim=1)

    # 3. Assemble the full action dictionary for the single, unified `step` method.
    actions_dict = {f"manager_{dc_id}": actions[i].item() for i, dc_id in enumerate(env._dc_ids)}
    actions_dict.update({f"worker_{dc_id}": 1 for dc_id in env._dc_ids})

    # 4. Call the single `env.step()` method.
    next_obs, rew_dict, dones_dict, trunc_dict, info_dict = env.step(actions_dict)
    
    # 5. Store info and update the state for the next loop iteration.
    all_step_infos.append(info_dict)
    obs_dict = next_obs
    
    if dones_dict["__all__"] or trunc_dict["__all__"]:
        logger.info(f"Simulation ended early at step {step+1}.")
        break

# %%
# --- Metrics Aggregation and Analysis ---
logger.info("Simulation finished. Aggregating results...")

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
            "energy_cost_usd": common.get("energy_cost_USD", 0.0),
            "energy_kwh": common.get("energy_consumption_kwh", 0.0),
            "carbon_kg": common.get("carbon_emissions_kg", 0.0),
            "price_usd_per_kwh": common.get("price_USD_kwh", 0.0),
            "ci_g_per_kwh": common.get("ci", 0.0),
            "external_temp_c": common.get("weather", 0.0),
            "cpu_util_pct": common.get("cpu_util_percent", 0.0),
            "gpu_util_pct": common.get("gpu_util_percent", 0.0),
            "mem_util_pct": common.get("mem_util_percent", 0.0),
            "running_tasks": common.get("running_tasks", 0),
            "sla_met": sla_info.get("met", 0),
            "sla_violated": sla_info.get("violated", 0),
        }
        flat_records.append(record)

df_results = pd.DataFrame(flat_records)
df_results.to_csv(results_csv_path, index=False)
logger.info(f"Detailed step-by-step results saved to: {results_csv_path}")


# --- Create Summary Table ---
summary = df_results.groupby("datacenter_id").agg(
    total_energy_cost_usd=("energy_cost_usd", "sum"),
    total_energy_kwh=("energy_kwh", "sum"),
    total_carbon_kg=("carbon_kg", "sum"),
    avg_price_usd_per_kwh=("price_usd_per_kwh", "mean"),
    avg_ci_g_per_kwh=("ci_g_per_kwh", "mean"),
    avg_cpu_util_pct=("cpu_util_pct", "mean"),
    avg_gpu_util_pct=("gpu_util_pct", "mean"),
    avg_mem_util_pct=("mem_util_pct", "mean"),
    total_sla_met=("sla_met", "sum"),
    total_sla_violated=("sla_violated", "sum"),
).reset_index()

# Calculate overall SLA Violation Rate
summary["sla_violation_rate_pct"] = (
    summary["total_sla_violated"] / (summary["total_sla_met"] + summary["total_sla_violated"] + 1e-6)
) * 100

summary = summary.round(2)
print("\n--- Evaluation Summary per Datacenter ---")
print(summary)
logger.info("\n--- Evaluation Summary per Datacenter ---\n" + summary.to_string())


# --- Create Overall Totals ---
overall_totals = summary.sum(numeric_only=True)
# Recompute averages and rates for the overall totals
total_tasks = overall_totals["total_sla_met"] + overall_totals["total_sla_violated"]
overall_totals["sla_violation_rate_pct"] = (overall_totals["total_sla_violated"] / (total_tasks + 1e-6)) * 100
overall_totals["avg_cpu_util_pct"] = df_results["cpu_util_pct"].mean()
overall_totals["avg_gpu_util_pct"] = df_results["gpu_util_pct"].mean()

print("\n--- Overall System Totals & Averages ---")
print(overall_totals.round(2))
logger.info("\n--- Overall System Totals & Averages ---\n" + overall_totals.round(2).to_string())


# %%
# --- Plotting Section ---
# This section uses the generated df_results DataFrame

# Get location labels for plotting
dc_name_to_location_label = {
    dc_cfg['dc_id']: dc_cfg['location'] for dc_cfg in base_dc_cfg_dict['datacenters']
}
df_results['datacenter_label'] = df_results['datacenter_id'].map(dc_name_to_location_label)


# Plot Carbon Intensity
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_results, x="timestep", y="ci_g_per_kwh", hue="datacenter_label", palette="colorblind")
plt.title("Carbon Intensity (gCO₂/kWh) over Time")
plt.xlabel("Timestep (15 min intervals)")
plt.ylabel("gCO₂/kWh")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title="Datacenter Location")
plt.tight_layout()
plt.show()

# Plot Running Tasks per Datacenter
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_results, x="timestep", y="running_tasks", hue="datacenter_label", palette="colorblind")
plt.title("Running Tasks per Datacenter over Time")
plt.xlabel("Timestep (15 min intervals)")
plt.ylabel("Number of Running Tasks")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title="Datacenter Location")
plt.tight_layout()
plt.show()

# Plot the external temperature
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_results, x="timestep", y="external_temp_c", hue="datacenter_label", palette="colorblind")
plt.title("External Temperature (°C) over Time")
plt.xlabel("Timestep (15 min intervals)")
plt.ylabel("Temperature (°C)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(title="Datacenter Location")
plt.tight_layout()
plt.show()

# %%
# --- 1x3 Plot: CPU, GPU, and Memory Utilization per Datacenter ---

fig, axes = plt.subplots(3, 1, figsize=(18, 9), sharex=True)

# CPU Utilization
sns.lineplot(
    data=df_results,
    x="timestep",
    y="cpu_util_pct",
    hue="datacenter_label",
    palette="colorblind",
    ax=axes[0]
)
axes[0].set_title("CPU Utilization (%)")
axes[0].set_xlabel("Timestep (15 min intervals)")
axes[0].set_ylabel("CPU Utilization (%)")
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
axes[0].legend(title="Datacenter", loc="upper right")

# GPU Utilization
sns.lineplot(
    data=df_results,
    x="timestep",
    y="gpu_util_pct",
    hue="datacenter_label",
    palette="colorblind",
    ax=axes[1],
    legend=False
)
axes[1].set_title("GPU Utilization (%)")
axes[1].set_xlabel("Timestep (15 min intervals)")
axes[1].set_ylabel("GPU Utilization (%)")
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

# Memory Utilization (if available)
if "mem_util_pct" in df_results.columns:
    sns.lineplot(
        data=df_results,
        x="timestep",
        y="mem_util_pct",
        hue="datacenter_label",
        palette="colorblind",
        ax=axes[2],
        legend=False
    )
    axes[2].set_ylabel("Memory Utilization (%)")
else:
    axes[2].text(0.5, 0.5, "No memory_util_pct data", ha='center', va='center', fontsize=12)
    axes[2].set_ylabel("Memory Utilization (%)")
axes[2].set_title("Memory Utilization (%)")
axes[2].set_xlabel("Timestep (15 min intervals)")
axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

# %%
