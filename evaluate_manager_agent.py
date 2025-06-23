# %%
import sys
import os
import copy
import datetime
import logging
import random

# Add the project root directory to the Python path
# This allows imports from rl_components, envs, etc.
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

# --- Local Imports for SustainCluster-MA ---
from envs.sustaincluster_ma_env import SustainClusterMAEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rl_components.agent_net_ma import ManagerActor, ManagerCritic # We only need Manager networks
from utils.checkpoint_manager_ma import load_checkpoint_data
from rewards.predefined.composite_reward import CompositeReward
from utils.config_loader import load_yaml
from rewards.registry_utils import get_reward_function


# --- Configuration ---
CONFIG_DIR = "configs/env"
# Base configs for the MA environment
BASE_SIM_CONFIG_PATH = os.path.join(CONFIG_DIR, "sim_config_ma.yaml")
BASE_DC_CONFIG_PATH = os.path.join(CONFIG_DIR, "datacenters_ma.yaml")
# Use the reward config you trained the manager with
BASE_REWARD_CONFIG_PATH = os.path.join(CONFIG_DIR, "reward_config_manager_ci_only.yaml")
BASE_ALGO_CONFIG_PATH = os.path.join(CONFIG_DIR, "algorithm_config_ma.yaml")


# --- IMPORTANT: Set this to your trained Manager agent's checkpoint ---
# Example path, you MUST update this to your actual checkpoint file
DEFAULT_RL_CHECKPOINT_PATH = "checkpoints/train_MGR_ONLY_manager_only_16heads_20250623_143955/best_eval_ckpt.pth"

# Evaluation settings
EVALUATION_DURATION_DAYS = 7
SEED = 123  # We will evaluate on a single, fixed seed for reproducibility

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
def make_eval_env(base_sim_cfg, base_dc_cfg, base_reward_cfg, duration_days, seed):
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
        logger=logger,
    )

    start = pd.Timestamp(datetime.datetime(sim_cfg["simulation"]["year"], sim_cfg["simulation"]["month"], sim_cfg["simulation"]["init_day"],
                                           sim_cfg["simulation"]["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=duration_days)
    

    env = SustainClusterMAEnv(
        cluster_manager_ma=cluster, start_time=start, end_time=end,
        reward_fn=reward_fn, logger=logger
    )
    return env


# %%
# --- Main Evaluation Script ---
logger.info("--- Starting Evaluation Run ---")
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

device = torch.device("cpu") # Run evaluation on CPU
checkpoint_data, _ = load_checkpoint_data(path=DEFAULT_RL_CHECKPOINT_PATH, device=device)
if checkpoint_data is None:
    raise ValueError("Failed to load checkpoint data.")

# Dynamically get network dimensions from the environment
temp_env = make_eval_env(base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict, 1, SEED)
D_GLOBAL = 4 # From env design
D_META_MANAGER = 8 # From utils/marl_utils.py
D_OPT = temp_env.cluster_manager_ma.D_OPTION_FEAT
MAX_OPT = temp_env.cluster_manager_ma.max_total_options
del temp_env

# Get network architecture from algo config
hidden_dim = algo_cfg_dict["algorithm"].get("hidden_dim", 256)
attention_cfg = algo_cfg_dict["algorithm"].get("attention", {})
embed_dim = attention_cfg.get("embed_dim", 64)
num_heads = attention_cfg.get("num_heads", 4)
num_layers = attention_cfg.get("num_attention_layers", 2)

# Re-create the actor network with the same architecture it was trained with
actor = ManagerActor(
    D_META_MANAGER, D_GLOBAL, D_OPT, MAX_OPT,
).to(device)

actor.load_state_dict(checkpoint_data["model_state_dict"]["mgr_actor"])
actor.eval()
logger.info("Successfully loaded Manager Actor policy.")


# --- Create the evaluation environment ---
env = make_eval_env(base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict,
                    EVALUATION_DURATION_DAYS, SEED)

# %%
# --- Simulation Loop ---
obs_dict, _ = env.reset(seed=SEED)
num_steps = EVALUATION_DURATION_DAYS * 24 * 4
all_step_infos = []

for step in tqdm(range(num_steps), desc=f"Simulating RL Agent (Seed {SEED})"):

    # ### --- THE FIX IS HERE --- ###
    # --- Start of Corrected Evaluation Logic ---

    # 1. Prepare observations for the Manager actor (same as before)
    meta_m, opt_m, mask_m, glob_m = [], [], [], []
    for dc_id in env._dc_ids:
        o_mgr = obs_dict[f"manager_{dc_id}"]
        meta_m.append(o_mgr["obs_manager_meta_task_i"])
        opt_m.append(o_mgr["obs_all_options_set_padded"])
        mask_m.append(o_mgr["all_options_padding_mask"])
        glob_m.append(o_mgr["global_context"])

    # Create batch tensor for the model
    meta_m_t = torch.from_numpy(np.asarray(meta_m)).float().to(device)
    opt_m_t = torch.from_numpy(np.asarray(opt_m)).float().to(device)
    mask_m_t = torch.from_numpy(np.asarray(mask_m)).bool().to(device)
    glob_m_t = torch.from_numpy(np.asarray(glob_m)).float().to(device)

    # 2. Get actions from the actor (greedy evaluation)
    with torch.no_grad():
        logits_m = actor(meta_m_t, glob_m_t, opt_m_t, mask_m_t)
        act_m = torch.argmax(logits_m, dim=-1)

    # 3. Mimic the training loop's three-step sequence
    
    # Step 3a: Call manager_step
    mgr_acts = {dc: act_m[i].item() for i, dc in enumerate(env._dc_ids)}
    _ = env.manager_step(mgr_acts) # We don't need the obs_after_mgr here

    # Step 3b: Call worker_step with the fixed policy
    wrk_acts = {dc: 1 for dc in env._dc_ids} # Fixed worker policy: Execute Now
    _ = env.worker_step(wrk_acts)

    # Step 3c: Call env_step to run physics and get final results
    next_obs, rew_dict, dones_dict, trunc_dict, info_dict = env.env_step()
    
    # --- End of Corrected Evaluation Logic ---
    # ### --- END OF FIX --- ###

    # 4. Store info and update state for the next loop iteration
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
