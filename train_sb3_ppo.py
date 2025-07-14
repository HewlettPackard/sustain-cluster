# train_sb3_ppo.py

import os
import datetime
import argparse
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# --- Local Imports ---
# Assuming these files exist from our previous discussions
from envs.sustaincluster_ma_env import SustainClusterMAEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rewards.registry_utils import get_reward_function
from utils.config_loader import load_yaml
from utils.sb3_compat_wrapper import SB3MultiAgentWrapper # The crucial wrapper
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from utils.custom_callbacks import SaveVecNormalizeCallback

# --- The Thunk for Creating Environments in Parallel ---
def make_env(rank: int, seed: int, args, log_dir: str): # Pass log_dir
    """
    A thunk (a function that returns a function) to be used by the vector env.
    This ensures that each parallel environment is created with a unique seed
    and the correct configuration.
    """
    def _init():
        # This function will be called by the VecEnv in a separate process
        from train_ppo_manager_only import make_simple_env # Import inside the thunk
        
        # Create the base environment using your existing function
        base_env = make_simple_env(
            sim_cfg_path=args.sim_config,
            dc_cfg_path=args.dc_config,
            reward_cfg_path=args.reward_config
        )
        
        # === START OF NEW LOGIC ===
        # 1. Wrap the base environment with the SB3 compatibility wrapper first.
        env = SB3MultiAgentWrapper(base_env)
        
        # 2. Then, wrap it with the Monitor wrapper.
        # This will watch for the info["episode"] dictionary upon termination.
        # We create a unique log file for each parallel environment.
        monitor_log_path = os.path.join(log_dir, f'monitor_{rank}')
        os.makedirs(monitor_log_path, exist_ok=True)
        env = Monitor(env, filename=monitor_log_path)
        # === END OF NEW LOGIC ===
        return env
    return _init

def main(args):
    # --- 1. Configuration and Setup ---
    run_name = f"SB3_PPO_{args.tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"runs/{run_name}"
    model_save_dir = f"checkpoints/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    algo_cfg = load_yaml(args.algo_config)["ppo_algorithm"]
    num_envs = 1

    # --- 2. Create the Vectorized Training Environment ---
    print(f"Creating {num_envs} parallel environments for training...")
    # Create a list of thunks, one for each parallel environment
    train_env_thunks = [make_env(i, args.seed, args, log_dir) for i in range(num_envs)]
    
    # Use SubprocVecEnv for true parallel performance on multi-core machines
    # Use DummyVecEnv if you need to debug or are on a single-core machine
    train_env = DummyVecEnv(train_env_thunks)
    
    # CRITICAL: Wrap the vectorized environment with SB3's VecNormalize.
    # This automatically handles observation normalization and reward scaling.
    # train_env = VecNormalize(train_vec_env, norm_obs=True, norm_reward=True, gamma=algo_cfg["gamma"])
    print("Training environments created.")

    # --- 3. Create the Vectorized Evaluation Environment ---
    print("Creating evaluation environment...")
    eval_env_thunks = [make_env(99, args.seed, args, log_dir)]
    eval_env = DummyVecEnv(eval_env_thunks) # Eval env doesn't need to be multi-process
    
    # IMPORTANT: Wrap the eval env in VecNormalize, but set `training=False` and `norm_reward=False`.
    # This ensures it uses the same observation stats as the training env but doesn't update them.
    # eval_env = VecNormalize(eval_vec_env, training=False, norm_obs=True, norm_reward=False, gamma=algo_cfg["gamma"])

    # --- 4. Define the PPO Model ---
    # The policy_kwargs can be used to specify network architecture.
    # by default, MlpPolicy uses [64, 64] with Tanh activations.
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[128, 128])) # Example: Wider value network

    # n_steps (int) – The number of steps to run for each environment per update 
    # (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel) 
    # NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization) 
    
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=96, # 2048 // num_envs -> 976 timmesteps
        batch_size=96,
        n_epochs=algo_cfg["update_epochs"],
        gamma=algo_cfg["gamma"], # 0.95
        gae_lambda=algo_cfg["gae_lambda"],
        clip_range=algo_cfg["clip_coef"],
        ent_coef=algo_cfg["ent_coef"],
        vf_coef=algo_cfg["vf_coef"],
        learning_rate=algo_cfg["learning_rate"],
        tensorboard_log=log_dir,
        device="cpu",
    )

    # --- 5. Set up Callbacks for Evaluation and Saving ---
    # eval_callback = SaveVecNormalizeCallback(
    #     eval_env,
    #     best_model_save_path=os.path.join(model_save_dir, "best_model.zip"), # SB3 adds .zip automatically
    #     log_path=os.path.join(model_save_dir, "eval_logs"),
    #     eval_freq=max(algo_cfg.get("eval_frequency", 5000) // num_envs, 1),
    #     deterministic=True,
    #     render=False
    # )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_dir, "best_model.zip"), # SB3 adds .zip automatically
        log_path=os.path.join(model_save_dir, "eval_logs"),
        n_eval_episodes=5, # Number of episodes to evaluate
        eval_freq=1, # Frequency of evaluation
    )
    
    # --- 6. Train the Model ---
    print("--- Starting Stable Baselines 3 PPO Training ---")
    model.learn(
        total_timesteps=algo_cfg["total_timesteps"],
        callback=eval_callback
    )
    print("--- Training Complete ---")

    # --- 7. Save the Final Model and Normalization Stats ---
    final_model_path = os.path.join(model_save_dir, "final_model")
    model.save(final_model_path)
    # train_env.save(os.path.join(model_save_dir, "final_model_vec_normalize.pkl"))
    
    # The normalization stats are saved separately
    norm_stats_path = os.path.join(model_save_dir, "vec_normalize.pkl")
    train_env.save(norm_stats_path)
    
    print(f"Final model saved to: {final_model_path}.zip")
    print(f"Normalization stats saved to: {norm_stats_path}")

    train_env.close()

if __name__ == "__main__":
    # We need to import the make_simple_env function from your other script
    # This is a bit of a hacky way to do it, but works for a single file setup.
    # A better long-term solution is to move make_simple_env to a shared utils file.
    from train_ppo_manager_only import make_simple_env
    
    # We also need a dummy parse_args function if we're not running from the command line
    # For this script, we'll assume the real one is used.
    from train_ppo_manager_only import parse_args
    
    # This is required for SubprocVecEnv to work correctly on some platforms
    torch.multiprocessing.set_start_method("fork")
    
    args = parse_args() # Get config paths from command line
    main(args)