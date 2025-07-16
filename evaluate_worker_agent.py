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
from stable_baselines3 import PPO

from envs.sustaincluster_worker_multi_env import SustainClusterWorkerMultiEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from utils.config_loader import load_yaml
from rewards.registry_utils import get_reward_function
from rewards.predefined.composite_reward import CompositeReward

# --- Configuration ---
CONFIG_DIR = "configs/env"
BASE_SIM_CONFIG_PATH = os.path.join(CONFIG_DIR, "sim_config_ma.yaml")
BASE_DC_CONFIG_PATH = os.path.join(CONFIG_DIR, "datacenters_ma.yaml")
BASE_REWARD_CONFIG_PATH = os.path.join(CONFIG_DIR, "reward_config_worker_ci_only.yaml")
# ### --- SIMPLIFICATION CHANGE: Point to the PPO config --- ###
BASE_ALGO_CONFIG_PATH = os.path.join(CONFIG_DIR, "ppo_algorithm_config.yaml")


# --- IMPORTANT: Set this to your trained Manager agent's checkpoint ---
# Example path, you MUST update this to your actual checkpoint file
DEFAULT_RL_CHECKPOINT_PATH = "checkpoints/train_worker_multi_SB3_PPO_worker_20250715_154951/PPO_model.zip"

# Evaluation settings
EVALUATION_DURATION_DAYS = 7
SEED = 123  # We will evaluate on a single, fixed seed for reproducibility

# --- Logger and Output Setup ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/eval_worker_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"evaluation_worker_{timestamp}.log")
results_csv_path = os.path.join(log_dir, f"results_worker_{timestamp}.csv")
logger = logging.getLogger("eval_worker_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(file_handler)
logger.info(f"Evaluation log will be saved to: {log_path}")
print(f"Evaluation log will be saved to: {log_path}")


# --- Environment Creation Function ---
def make_eval_env(base_sim_cfg, base_dc_cfg, base_reward_cfg, duration_days, seed, simple_obs_mode=True):
    """Creates the SustainCluster Worker only evaluation environment."""
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
    

    env = SustainClusterWorkerMultiEnv(
        cluster_manager=cluster, start_time=start, end_time=end,
        reward_fn=reward_fn, logger=logger)
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

# --- Load the Trained Worker Actor ---
if not os.path.exists(DEFAULT_RL_CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found at {DEFAULT_RL_CHECKPOINT_PATH}")

device = torch.device("cpu")
model = PPO.load(DEFAULT_RL_CHECKPOINT_PATH, device=device)

# --- Create the evaluation environment ---
eval_env = make_eval_env(base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict, EVALUATION_DURATION_DAYS, SEED)

# %%
# --- Simulation Loop ---
obs_dict, _ = eval_env.reset(seed=SEED)
num_steps = EVALUATION_DURATION_DAYS * 24 * 4
all_infos = []

for step in tqdm(range(num_steps), desc=f"Evaluating PPO Worker (Seed {SEED})"):
    
    actions, _states = model.predict(obs_dict, deterministic=True)
    print(f"Actions at step {step}: {actions}")
    obs, reward, terminated, truncated, info = eval_env.step(actions)
    all_infos.append(info)

    if terminated or truncated:
        logger.info(f"Simulation ended early at step {step+1}.")
        break
    

# %%
# --- Metrics Aggregation and Analysis ---
logger.info("Simulation finished. Aggregating results...")

flat_records = []
# print(all_infos[:5])
for t, step_info in enumerate(all_infos):
    dc_infos = step_info.get("raw_results", {}).get("datacenter_infos", {})
    for dc_id, dc in dc_infos.items():
        common = dc["__common__"]
        sla = common.get("__sla__", {"met": 0, "violated": 0})
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
            "sla_met": sla.get("met", 0),
            "sla_violated": sla.get("violated", 0),
        }
        flat_records.append(record)
        # print(record)

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

# Plot Commited Tasks each time step per Datacenter
plt.figure(figsize=(10, 5))
sns.lineplot(data=df_results, x="timestep", y="committed_this_step", hue="datacenter_label", palette="colorblind")
plt.title("Committed Tasks per Datacenter (per step)")
plt.xlabel("Timestep (15 min intervals)")
plt.ylabel("Committed Tasks")
plt.legend(title="Datacenter Location")
plt.grid(True, linestyle='--', linewidth=0.5)
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
