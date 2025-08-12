# train_sb3_ppo_worker_llm_enhanced.py - Enhanced Worker Training with LLM Advice

import os
import datetime
import argparse
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import asyncio
import numpy as np

# --- Local Imports ---
from envs.sustaincluster_ma_env_llm import LLMIntegratedWorkerEnv  # Import from main file
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rewards.registry_utils import get_reward_function
from utils.config_loader import load_yaml
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor

def make_env(rank: int, seed: int, args, log_dir: str, enable_llm: bool = True):
    """Enhanced environment factory with LLM integration."""
    def _init():
        from envs.sustaincluster_worker_multi_env import SustainClusterWorkerMultiEnv
        from simulation.cluster_manager_ma import DatacenterClusterManagerMA
        from rewards.registry_utils import get_reward_function
        from rewards.predefined.composite_reward import CompositeReward
        import pandas as pd
        import datetime
        
        # Load configurations
        sim_cfg_full = load_yaml(args.sim_config)
        sim_cfg = sim_cfg_full["simulation"]
        dc_cfg = load_yaml(args.dc_config)["datacenters"]
        reward_cfg = load_yaml(args.reward_config)["reward"]

        for cfg in dc_cfg:
            cfg.setdefault("simulation_year", sim_cfg["year"])
        
        start = pd.Timestamp(datetime.datetime(sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"],
                                             sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc))
        end = start + datetime.timedelta(days=sim_cfg["duration_days"])

        cluster = DatacenterClusterManagerMA(
            config_list=dc_cfg,
            simulation_year=sim_cfg["year"],
            tasks_file_path=sim_cfg["workload_path"],
            cloud_provider=sim_cfg["cloud_provider"],
            max_total_options=sim_cfg["max_total_options"],
            duration_days=sim_cfg["duration_days"],
        )

        # Build reward function
        if "components" in reward_cfg:
            reward_fn = CompositeReward(components=reward_cfg["components"], 
                                      normalize=reward_cfg['normalize'], 
                                      freeze_stats_after_steps=reward_cfg['freeze_stats_after_steps'])
        elif "name" in reward_cfg:
            reward_name = reward_cfg["name"]
            reward_args = reward_cfg.get("args", {})
            reward_fn = get_reward_function(reward_name, **reward_args)
        else:
            raise ValueError("Invalid reward configuration.")

        # Create base worker environment
        base_env = SustainClusterWorkerMultiEnv(
            cluster_manager=cluster,
            start_time=start,
            end_time=end,
            reward_fn=reward_fn
        )
        
        # Wrap with LLM integration
        env = LLMIntegratedWorkerEnv(
            base_worker_env=base_env,
            enable_llm_advice=enable_llm,
            llm_service_url=args.llm_service_url,
            llm_timeout=args.llm_timeout,
            llm_max_concurrent=args.llm_max_concurrent,
            llm_history_window=args.llm_history_window
        )
        
        # Monitor wrapper
        monitor_log_path = os.path.join(log_dir, f'monitor_{rank}')
        os.makedirs(monitor_log_path, exist_ok=True)
        env = Monitor(env, filename=monitor_log_path)
        
        return env
    return _init

def parse_args():
    parser = argparse.ArgumentParser(description="LLM-Enhanced Worker Training with SB3")
    parser.add_argument("--sim-config", type=str, default="configs/env/sim_config_ma.yaml")
    parser.add_argument("--reward-config", type=str, default="configs/env/reward_config_worker_ci_only.yaml")
    parser.add_argument("--dc-config", type=str, default="configs/env/datacenters_ma.yaml")
    parser.add_argument("--algo-config", type=str, default="configs/env/ppo_algorithm_config.yaml")
    parser.add_argument("--tag", type=str, default="llm_enhanced", help="Run tag")
    parser.add_argument("--seed", type=int, default=42)
    
    # LLM Integration Arguments
    parser.add_argument("--enable-llm", type=bool, default=True, help="Enable LLM advice integration")
    parser.add_argument("--llm-service-url", type=str, default="http://10.93.232.106:8000", help="LLM service URL")
    parser.add_argument("--llm-timeout", type=float, default=2.0, help="LLM request timeout")
    parser.add_argument("--llm-max-concurrent", type=int, default=16, help="Max concurrent LLM requests")
    parser.add_argument("--llm-history-window", type=int, default=10, help="LLM context history window")
    
    return parser.parse_args()

def main(args):
    """Enhanced main function with LLM integration."""
    run_name = f"SB3_PPO_Worker_LLM_{args.tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"runs/{run_name}"
    model_save_dir = f"checkpoints/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    algo_cfg = load_yaml(args.algo_config)["ppo_algorithm"]
    num_envs = 1

    # Log LLM configuration
    if args.enable_llm:
        print(f"LLM Integration Enabled:")
        print(f"  Service URL: {args.llm_service_url}")
        print(f"  Timeout: {args.llm_timeout}s")
        print(f"  Max Concurrent: {args.llm_max_concurrent}")
        print(f"  History Window: {args.llm_history_window}")
    else:
        print("LLM Integration Disabled - Using standard observations")

    # Create vectorized training environment
    print(f"Creating {num_envs} parallel environments for training...")
    train_env_thunks = [make_env(i, args.seed, args, log_dir, args.enable_llm) for i in range(num_envs)]
    train_env = DummyVecEnv(train_env_thunks)
    print("Training environments created.")

    # Create vectorized evaluation environment
    print("Creating evaluation environment...")
    eval_env_thunks = [make_env(99, args.seed, args, log_dir, args.enable_llm)]
    eval_env = DummyVecEnv(eval_env_thunks)

    # Define PPO model with enhanced observation space
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[128, 128]))

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=96,
        batch_size=96,
        n_epochs=algo_cfg["update_epochs"],
        gamma=algo_cfg["gamma"],
        gae_lambda=algo_cfg["gae_lambda"],
        clip_range=algo_cfg["clip_coef"],
        ent_coef=algo_cfg["ent_coef"],
        vf_coef=algo_cfg["vf_coef"],
        learning_rate=algo_cfg["learning_rate"],
        tensorboard_log=log_dir,
        device="cpu",
    )

    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_dir, "best_model.zip"),
        log_path=os.path.join(model_save_dir, "eval_logs"),
        n_eval_episodes=5,
        eval_freq=max(algo_cfg.get("eval_frequency", 5000) // num_envs, 1),
        deterministic=True,
        render=False
    )
    
    # Train the model
    print("--- Starting Enhanced SB3 PPO Worker Training ---")
    model.learn(
        total_timesteps=algo_cfg["total_timesteps"],
        callback=eval_callback
    )
    print("--- Training Complete ---")

    # Save final model
    final_model_path = os.path.join(model_save_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}.zip")

    train_env.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)