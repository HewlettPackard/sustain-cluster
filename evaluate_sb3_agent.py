# evaluate_sb3_agent.py

# %%
import sys
import os
import copy
import datetime
import logging
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# --- Stable Baselines 3 and Local Imports ---
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.sustaincluster_ma_env import SustainClusterMAEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from utils.config_loader import load_yaml
from rewards.registry_utils import get_reward_function
from utils.sb3_compat_wrapper import SB3MultiAgentWrapper

# --- Configuration ---
# ### --- IMPORTANT: Update these paths to your SB3 training run --- ###
RUN_NAME = "SB3_PPO_sb3_20250711_221934/best_model.zip" # The name of the folder for your SB3 run
MODEL_SAVE_DIR = f"checkpoints/{RUN_NAME}"
DEFAULT_RL_CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.zip")
VEC_NORMALIZE_PATH = os.path.join(MODEL_SAVE_DIR, "vec_normalize.pkl")

# Use the same base configs as your training script
CONFIG_DIR = "configs/env"
BASE_SIM_CONFIG_PATH = os.path.join(CONFIG_DIR, "sim_config_ma.yaml")
BASE_DC_CONFIG_PATH = os.path.join(CONFIG_DIR, "datacenters_ma.yaml")
BASE_REWARD_CONFIG_PATH = os.path.join(CONFIG_DIR, "reward_config_manager_ci_only.yaml")

# Evaluation settings
EVALUATION_DURATION_DAYS = 7
SEED = 123

# --- Logger and Output Setup ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logs/eval_manager_SB3_{timestamp}"
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


# --- Environment Creation Function (can be shared with the training script) ---
def make_sb3_eval_env(args, logger=None):
    """A thunk for creating the evaluation environment."""
    from train_ppo_manager_only import make_simple_env # Assuming this is in another file
    def _init():
        base_env = make_simple_env(
            sim_cfg_path=args.sim_config,
            dc_cfg_path=args.dc_config,
            reward_cfg_path=args.reward_config,
            logger=logger,
        )
        env = SB3MultiAgentWrapper(base_env)
        env.reset(seed=SEED)
        return env
    return _init

def process_infos_to_dataframe(all_step_infos: list) -> pd.DataFrame:
    """Processes the list of info dicts from a simulation run into a clean DataFrame."""
    
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
    
    return df_results

# %%
# --- Main Evaluation Script ---
logger.info("--- Starting SB3 PPO Agent Evaluation Run ---")
# ... (load configs) ...

# --- Load the Trained SB3 Model and Normalization Stats ---
if not os.path.exists(DEFAULT_RL_CHECKPOINT_PATH):
    raise FileNotFoundError(f"SB3 model not found at {DEFAULT_RL_CHECKPOINT_PATH}")
# if not os.path.exists(VEC_NORMALIZE_PATH):
    # raise FileNotFoundError(f"VecNormalize stats not found at {VEC_NORMALIZE_PATH}")

# 1. Create the evaluation environment
# We use DummyVecEnv because we only need one instance for evaluation
# We pass dummy args object for the make_env function
class Args:
    sim_config = BASE_SIM_CONFIG_PATH
    dc_config = BASE_DC_CONFIG_PATH
    reward_config = BASE_REWARD_CONFIG_PATH
args = Args()
eval_env = DummyVecEnv([make_sb3_eval_env(args, logger=logger)])

# 2. Load the normalization stats and wrap the environment
# IMPORTANT: This must be done *before* loading the model
# eval_env = VecNormalize.load(VEC_NORMALIZE_PATH, eval_env)
# Set to evaluation mode
eval_env.training = False
eval_env.norm_reward = False

# 3. Load the PPO model
# The model already knows its policy and network architecture
model = PPO.load(DEFAULT_RL_CHECKPOINT_PATH, env=eval_env)
logger.info("Successfully loaded SB3 PPO model and normalization stats.")

# %%
# --- Simulation Loop ---
obs = eval_env.reset()
num_steps = EVALUATION_DURATION_DAYS * 24 * 4
all_step_infos = []
done = False

for step in tqdm(range(num_steps), desc=f"Simulating SB3 PPO Agent (Seed {SEED})"):
    if done:
        logger.info(f"Simulation ended early at step {step}.")
        break

    # Get deterministic actions from the loaded model
    action, _states = model.predict(obs, deterministic=True)
    
    # Step the vectorized environment
    obs, reward, done, info = eval_env.step(action)
    
    # The info from a VecEnv is a list of dicts
    all_step_infos.append(info[0])
    
    # Since we have only one env, done is a boolean array of size 1
    done = done[0]

#%%
# --- Metrics Aggregation and Analysis ---
# ... (This entire section of your script for processing infos, creating summaries,
#      and plotting is PERFECTLY FINE and does not need to be changed.) ...
base_dc_cfg_dict = load_yaml(BASE_DC_CONFIG_PATH)

df_results = process_infos_to_dataframe(all_step_infos)

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
