import numpy as np
import pandas as pd
import yaml
import datetime
from envs.task_scheduling_env import TaskSchedulingEnv
from simulation.cluster_manager import DatacenterClusterManager
from rewards.predefined.composite_reward import CompositeReward  # Adjust import if you use a different reward

# Helper to load yaml
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def verify_environment():
    print("==================================================")
    print("      STARTING SUSTAINCLUSTER VERIFICATION        ")
    print("==================================================")

    # 1. LOAD CONFIGURATIONS
    # Adjust these paths if your config files are named differently
    try:
        sim_cfg = load_yaml("configs/env/sim_config.yaml")["simulation"]
        dc_cfg = load_yaml("configs/env/datacenters.yaml")["datacenters"]
        reward_cfg = load_yaml("configs/env/reward_config.yaml")["reward"]
        print("[✓] Configurations loaded successfully.")
    except FileNotFoundError as e:
        print(f"[X] Configuration file not found: {e}")
        return

    # 2. SETUP CLUSTER MANAGER
    # Setting up dates based on config
    start = pd.Timestamp(datetime.datetime(sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"],
                                           sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=1) # Run just for 1 day for verification

    print(f"[*] Initializing Cluster Manager (Provider: {sim_cfg['cloud_provider']})...")
    cluster = DatacenterClusterManager(
        config_list=dc_cfg,
        simulation_year=sim_cfg["year"],
        init_day=int(sim_cfg["month"] * 30.5),
        init_hour=sim_cfg["init_hour"],
        strategy="manual_rl", # Crucial: We want manual control
        tasks_file_path=sim_cfg["workload_path"],
        shuffle_datacenter_order=False, # Keep false for debugging determinism
        cloud_provider=sim_cfg["cloud_provider"],
        logger=None
    )

    # 3. SETUP REWARD
    reward_fn = CompositeReward(
        components=reward_cfg["components"],
        normalize=False
    )

    # 4. INITIALIZE SIMPLIFIED ENV
    print("[*] Initializing Simplified TaskSchedulingEnv...")
    env = TaskSchedulingEnv(
        cluster_manager=cluster,
        start_time=start,
        end_time=end,
        reward_fn=reward_fn,
        writer=None
    )

    # 5. VERIFICATION CHECKS
    print("\n--- CHECKING SPACES ---")
    
    # Check Action Space
    num_dcs = len(dc_cfg)
    print(f"Number of Data Centers: {num_dcs}")
    print(f"Env Action Space: {env.action_space}")
    
    assert env.action_space.n == num_dcs, \
        f"Mismatch! Action space size ({env.action_space.n}) != Num DCs ({num_dcs}). Did you remove the defer action?"
    print("[✓] Action Space is correct (Discrete, equal to Num DCs).")

    # Check Observation Space
    obs, _ = env.reset(seed=42)
    print(f"Observation Shape: {obs.shape}")
    print(f"Observation Sample (First 10 dims): {obs[:10]}")
    
    assert len(obs.shape) == 1, "Observation should be a 1D array (Aggregated), not a list of tasks."
    print("[✓] Observation Space is correct (1D Aggregated Vector).")

    # 6. RANDOM AGENT LOOP
    print("\n--- RUNNING RANDOM AGENT LOOP (5 Steps) ---")
    
    total_reward = 0
    steps = 5
    
    for i in range(steps):
        # Sample random action (0 to N-1)
        action = env.action_space.sample()
        
        # Take step
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Extract meaningful info for print
        batch_size = info.get('batch_size', 'N/A')
        selected_dc = info.get('selected_dc', 'N/A')
        energy = info.get('total_energy_kwh', 0)
        
        print(f"Step {i+1}: Action={action} (DC_{selected_dc}) | Batch Size={batch_size} tasks | Reward={reward:.4f} | Energy={energy:.2f}kWh")
        
        # Verify Observation update
        if not np.array_equal(obs, next_obs):
            pass # Good, observation changed
        else:
            print("    [!] Warning: Observation didn't change (might happen if no tasks/inputs changed).")
            
        obs = next_obs
        total_reward += reward

        if done:
            print("Episode finished early.")
            break

    print("\n==================================================")
    print(f"      VERIFICATION COMPLETE. Total Reward: {total_reward:.4f}")
    print("==================================================")

if __name__ == "__main__":
    verify_environment()