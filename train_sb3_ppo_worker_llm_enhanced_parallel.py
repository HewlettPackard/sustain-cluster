# train_sb3_ppo_worker_llm_enhanced_parallel.py - UPDATED TO USE FIXED ENVIRONMENT FILES
# Now uses the fixed sustaincluster_ma_env_llm.py instead of custom implementations

import os
import datetime
import argparse
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import multiprocessing
from utils.config_loader import load_yaml
import time

def make_env(rank: int, seed: int, args, log_dir: str, enable_llm: bool = True):
    """UPDATED: Environment factory using the fixed LLM environment files"""
    def _init():
        from envs.sustaincluster_worker_multi_env import SustainClusterWorkerMultiEnv
        from envs.sustaincluster_ma_env_llm import LLMIntegratedWorkerEnv  # FIXED: Use the updated file
        from simulation.cluster_manager_ma import DatacenterClusterManagerMA
        from rewards.registry_utils import get_reward_function
        from rewards.predefined.composite_reward import CompositeReward
        from utils.config_loader import load_yaml
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
        
        # UPDATED: Use the fixed LLMIntegratedWorkerEnv from sustaincluster_ma_env_llm.py
        env = LLMIntegratedWorkerEnv(
            base_worker_env=base_env,
            enable_llm_advice= True,#enable_llm and not getattr(args, 'disable_llm_for_subprocess', False),
            llm_service_url=args.llm_service_url,
            llm_timeout=args.llm_timeout,
            llm_max_concurrent=max(4, args.llm_max_concurrent // args.num_envs),  # Distribute concurrent requests
            llm_history_window=args.llm_history_window
        )
        
        # Monitor wrapper with unique seed
        monitor_log_path = os.path.join(log_dir, f'monitor_{rank}')
        os.makedirs(monitor_log_path, exist_ok=True)
        env = Monitor(env, filename=monitor_log_path)
        
        # Set seed
        env.reset(seed=seed + rank)
        return env
    return _init

def parse_args():
    parser = argparse.ArgumentParser(description="UPDATED: Parallelized LLM-Enhanced Worker Training")
    parser.add_argument("--sim-config", type=str, default="configs/env/sim_config_ma.yaml")
    parser.add_argument("--reward-config", type=str, default="configs/env/reward_config_worker_ci_only.yaml")
    parser.add_argument("--dc-config", type=str, default="configs/env/datacenters_ma.yaml")
    parser.add_argument("--algo-config", type=str, default="configs/env/ppo_algorithm_config.yaml")
    parser.add_argument("--tag", type=str, default="llm_enhanced_fixed", help="Run tag")
    parser.add_argument("--seed", type=int, default=42)
    
    # Parallelization Arguments
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--use-subprocess", type=bool, default=False, help="Use SubprocVecEnv (disabled by default for stability)")
    parser.add_argument("--disable-llm-for-subprocess", type=bool, default=True, help="Disable LLM when using subprocess")
    
    # LLM Integration Arguments
    parser.add_argument("--enable-llm", type=bool, default=True, help="Enable LLM advice integration")
    parser.add_argument("--llm-service-url", type=str, default="http://10.93.232.106:8000", help="LLM service URL")
    parser.add_argument("--llm-timeout", type=float, default=5.0, help="LLM request timeout")
    parser.add_argument("--llm-max-concurrent", type=int, default=16, help="Max concurrent LLM requests")
    parser.add_argument("--llm-history-window", type=int, default=5, help="LLM context history window")
    
    return parser.parse_args()

def main():
    """UPDATED: Main function using fixed environment files"""
    args = parse_args()
    
    run_name = f"SB3_PPO_Worker_LLM_FIXED_{args.tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"runs/{run_name}"
    model_save_dir = f"checkpoints/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    algo_cfg = load_yaml(args.algo_config)["ppo_algorithm"]

    print(f"UPDATED Worker Training Configuration (using fixed environment files):")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Use subprocess: {args.use_subprocess}")
    print(f"  Disable LLM for subprocess: {args.disable_llm_for_subprocess}")
    
    if args.enable_llm and not (args.use_subprocess and args.disable_llm_for_subprocess):
        print(f"LLM Integration Enabled (using FIXED client):")
        print(f"  Service URL: {args.llm_service_url}")
        print(f"  Timeout: {args.llm_timeout}s")
        print(f"  Max Concurrent: {args.llm_max_concurrent}")

    # Create environments
    print(f"Creating {args.num_envs} parallel environments (using FIXED LLMIntegratedWorkerEnv)...")
    train_env_thunks = [make_env(i, args.seed, args, log_dir, args.enable_llm) for i in range(args.num_envs)]
    
    # Use DummyVecEnv by default for better stability with LLM integration
    if args.use_subprocess and args.num_envs > 1:
        # print("⚠️ Using SubprocVecEnv - LLM advice will be disabled for stability")
        train_env = SubprocVecEnv(train_env_thunks, start_method='spawn')
    else:
        print("Using DummyVecEnv for better LLM integration stability")
        train_env = DummyVecEnv(train_env_thunks)

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env_thunks = [make_env(99, args.seed, args, log_dir, args.enable_llm)]
    eval_env = DummyVecEnv(eval_env_thunks)

    # PPO model with conservative settings for stability
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[128, 128]))

    # Conservative batch sizing for stability
    n_steps = 96 if args.num_envs == 1 else max(48, 96 // args.num_envs)
    batch_size = n_steps * args.num_envs

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=n_steps,
        batch_size=batch_size,
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

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_dir, "best_model.zip"),
        log_path=os.path.join(model_save_dir, "eval_logs"),
        n_eval_episodes=3,  # Reduced for faster evaluation
        eval_freq=max(2000, 5000 // args.num_envs),
        deterministic=True,
        render=False
    )
    
    # Train the model
    print("--- Starting UPDATED Worker Training (with FIXED LLM integration) ---")
    print(f"Total timesteps: {algo_cfg['total_timesteps']}")
    print(f"Steps per update: {n_steps}")
    print(f"Batch size: {batch_size}")
    print("Using fixed sustaincluster_ma_env_llm.py and sync_llm_client_fixed.py")
    
    try:
        model.learn(
            total_timesteps=algo_cfg["total_timesteps"],
            callback=eval_callback,
            progress_bar=True
        )
        print("--- Training Complete ---")
    except Exception as e:
        print(f"❌ Training error: {e}")
        raise
    finally:
        # Cleanup
        train_env.close()
        eval_env.close()

    # Save final model
    final_model_path = os.path.join(model_save_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}.zip")

if __name__ == "__main__":
    main()