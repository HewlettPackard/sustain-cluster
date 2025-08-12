# =========================================================================
# SPEED-OPTIMIZED WORKER TRAINING SCRIPT  
# worker_training_speed_optimized.py
# =========================================================================



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
from stable_baselines3.common.vec_env import VecNormalize
D_LLM_ADVICE = 10

def make_env(rank: int, seed: int, args, log_dir: str, enable_llm: bool = True):
    """SPEED-OPTIMIZED: Environment factory with optimized LLM settings and verification."""
    def _init():
        from envs.sustaincluster_worker_multi_env import SustainClusterWorkerMultiEnv
        from envs.sustaincluster_ma_env_llm_speed_optimized import SpeedOptimizedLLMIntegratedWorkerEnv
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
        
        # SPEED-OPTIMIZED: Wrap with speed-optimized LLM integration
        env = SpeedOptimizedLLMIntegratedWorkerEnv(
            base_worker_env=base_env,
            enable_llm_advice=enable_llm,
            llm_service_url=args.llm_service_url,
            llm_timeout=args.llm_timeout,
            llm_max_concurrent=args.llm_max_concurrent,
            llm_history_window=args.llm_history_window,
            num_envs=args.num_envs
        )
        
        # ✅ FIX: Verify environment observation space
        sample_obs, _ = env.reset(seed=seed + rank)
        print(f"Environment {rank} observation shape: {sample_obs.shape} (LLM: {enable_llm})")
        
        # Monitor wrapper with unique seed
        monitor_log_path = os.path.join(log_dir, f'monitor_{rank}')
        os.makedirs(monitor_log_path, exist_ok=True)
        env = Monitor(env, filename=monitor_log_path)
        
        return env
    return _init

def parse_args():
    parser = argparse.ArgumentParser(description="SPEED-OPTIMIZED: Worker Training with SB3")
    parser.add_argument("--sim-config", type=str, default="configs/env/sim_config_ma.yaml")
    parser.add_argument("--reward-config", type=str, default="configs/env/reward_config_worker_ci_only.yaml")
    parser.add_argument("--dc-config", type=str, default="configs/env/datacenters_ma.yaml")
    parser.add_argument("--algo-config", type=str, default="configs/env/ppo_algorithm_config.yaml")
    parser.add_argument("--tag", type=str, default="speed_optimized", help="Run tag")
    parser.add_argument("--seed", type=int, default=42)
    
    # Parallelization Arguments
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--use-subprocess", type=bool, default=True, help="Use SubprocVecEnv")
    
    # SPEED-OPTIMIZED: LLM Integration Arguments
    parser.add_argument("--enable-llm", type=bool, default=True, help="Enable LLM advice")
    parser.add_argument("--llm-service-url", type=str, default="http://10.93.232.106:8000")
    parser.add_argument("--llm-timeout", type=float, default=2.0, help="REDUCED: Fast timeout")
    parser.add_argument("--llm-max-concurrent", type=int, default=64, help="INCREASED: High concurrency")
    parser.add_argument("--llm-history-window", type=int, default=8, help="REDUCED: Optimized history")
    
    return parser.parse_args()

def main():
    """SPEED-OPTIMIZED: Main function with performance focus"""
    args = parse_args()
    
    import torch
    if torch.cuda.is_available():
        print("🔧 CUDA available but forcing CPU usage for MLP policy (recommended)")
    
    # Determine optimal number of environments
    if args.num_envs == -1:
        args.num_envs = min(multiprocessing.cpu_count(), 8)
    
    run_name = f"SB3_PPO_Worker_SPEED_{args.tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"runs/{run_name}"
    model_save_dir = f"checkpoints/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    algo_cfg = load_yaml(args.algo_config)["ppo_algorithm"]

    # Log configuration
    print(f"⚡ SPEED-OPTIMIZED Worker Training Configuration:")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Vectorization: {'SubprocVecEnv' if args.use_subprocess else 'DummyVecEnv'}")
    print(f"  CPU cores: {multiprocessing.cpu_count()}")
    
    if args.enable_llm:
        print(f"SPEED-OPTIMIZED LLM Integration:")
        print(f"  Service URL: {args.llm_service_url}")
        print(f"  Timeout: {args.llm_timeout}s (fast)")
        print(f"  Max Concurrent: {args.llm_max_concurrent} (high)")
        print(f"  History Window: {args.llm_history_window} (optimized)")

    # Create vectorized training environment
    print(f"Creating {args.num_envs} SPEED-OPTIMIZED parallel environments...")
    train_env_thunks = [make_env(i, args.seed, args, log_dir, args.enable_llm) for i in range(args.num_envs)]
    
    if args.use_subprocess and args.num_envs > 1:
        print("Using SubprocVecEnv for SPEED-OPTIMIZED parallel execution")
        base_train_env = SubprocVecEnv(train_env_thunks, start_method='spawn')
    else:
        print("Using DummyVecEnv for sequential execution")
        base_train_env = DummyVecEnv(train_env_thunks)
    
    # ✅ FIX: Add VecNormalize wrapper for training environment
    train_env = VecNormalize(
        base_train_env,
        norm_obs=True,
        norm_reward=True,
        gamma=algo_cfg.get("gamma", 0.99)
    )
    
    print("SPEED-OPTIMIZED training environments created with normalization.")

    # Create evaluation environment
    print("Creating SPEED-OPTIMIZED evaluation environment...")
    eval_env_thunks = [make_env(99, args.seed, args, log_dir, args.enable_llm)]
    base_eval_env = DummyVecEnv(eval_env_thunks)
    
    # ✅ FIX: Add VecNormalize wrapper for evaluation environment
    eval_env = VecNormalize(
        base_eval_env,
        training=False,  # Don't update normalization stats
        norm_obs=True,
        norm_reward=False  # Don't normalize rewards for evaluation logging
    )

    # ✅ FIX: Verify observation dimensions
    sample_obs = train_env.reset()
    obs_shape = sample_obs.shape
    expected_base_dim = 16  # D_TASK_FEAT (7) + D_DC_FEAT (5) + D_GLOBAL (4)
    expected_total_dim = expected_base_dim + (D_LLM_ADVICE if args.enable_llm else 0)
    
    print(f"\n📊 Worker Observation Space Verification:")
    print(f"  Observation shape: {obs_shape}")
    print(f"  Expected base dim per DC: {expected_base_dim}")
    print(f"  LLM advice dim per DC: {D_LLM_ADVICE if args.enable_llm else 0}")
    print(f"  Expected total dim per DC: {expected_total_dim}")
    print(f"  Actual dim per DC: {obs_shape[1] if len(obs_shape) > 1 else 'N/A'}")
    
    # if len(obs_shape) > 1 and obs_shape[1] != expected_total_dim:
    #     print(f"⚠️  WARNING: Worker observation dimension mismatch!")
    #     print(f"  This may cause training instability.")
    # else:
    #     print(f"✅ Worker observation dimensions verified correctly!")
        
    if len(obs_shape) == 3:  # (n_envs, n_dcs, obs_dim)
        actual_obs_dim = obs_shape[2]  # This is the correct dimension per DC
        actual_n_dcs = obs_shape[1] 
        print(f"  Actual dim per DC: {actual_obs_dim}")
        print(f"  Actual n_dcs: {actual_n_dcs}")
        
        if actual_obs_dim != expected_total_dim:
            print(f"⚠️  WARNING: Worker observation dimension mismatch!")
            print(f"  Expected: {expected_total_dim}, Got: {actual_obs_dim}")
        else:
            print(f"✅ Worker observation dimensions verified correctly!")
    elif len(obs_shape) == 2:  # (n_dcs, obs_dim) - single environment
        actual_obs_dim = obs_shape[1]
        print(f"  Actual dim per DC: {actual_obs_dim}")
        
        if actual_obs_dim != expected_total_dim:
            print(f"⚠️  WARNING: Worker observation dimension mismatch!")
        else:
            print(f"✅ Worker observation dimensions verified correctly!")
    else:
        print(f"❌ Unexpected observation shape: {obs_shape}")

    # SPEED-OPTIMIZED: PPO model with optimized hyperparameters
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[256, 256]))

    # SPEED-OPTIMIZED: Larger batches and fewer steps for faster training
    adjusted_n_steps = max(128 // args.num_envs, 32)
    rollout_buffer_size = adjusted_n_steps * args.num_envs
    adjusted_batch_size = rollout_buffer_size
    # adjusted_batch_size = adjusted_n_steps * args.num_envs * 2
    
    print(f"📊 PPO Configuration:")
    print(f"  n_steps: {adjusted_n_steps}")
    print(f"  n_envs: {args.num_envs}")
    print(f"  rollout_buffer_size: {rollout_buffer_size}")
    print(f"  batch_size: {adjusted_batch_size}")

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=adjusted_n_steps,
        batch_size=adjusted_batch_size,
        n_epochs=max(algo_cfg["update_epochs"] - 2, 3),
        gamma=algo_cfg["gamma"],
        gae_lambda=algo_cfg["gae_lambda"],
        clip_range=algo_cfg["clip_coef"],
        ent_coef=algo_cfg["ent_coef"],
        vf_coef=algo_cfg["vf_coef"],
        learning_rate=algo_cfg["learning_rate"] * 1.5,
        tensorboard_log=log_dir,
        device="cpu",
    )

    # ✅ FIX: Enhanced evaluation callback that shares normalization stats
    class EnhancedEvalCallback(EvalCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.training_env_ref = None
        
        def set_training_env(self, training_env):
            self.training_env_ref = training_env
        
        def _on_step(self) -> bool:
            if (self.training_env_ref is not None and 
                hasattr(self.training_env_ref, 'obs_rms') and 
                hasattr(self.eval_env, 'obs_rms')):
                self.eval_env.obs_rms = self.training_env_ref.obs_rms
                if hasattr(self.training_env_ref, 'ret_rms') and hasattr(self.eval_env, 'ret_rms'):
                    self.eval_env.ret_rms = self.training_env_ref.ret_rms
            return super()._on_step()
    
    eval_callback = EnhancedEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_save_dir, "best_model.zip"),
        log_path=os.path.join(model_save_dir, "eval_logs"),
        n_eval_episodes=3,
        eval_freq=max(algo_cfg.get("eval_frequency", 5000) // args.num_envs, 1000),
        deterministic=True,
        render=False
    )
    eval_callback.set_training_env(train_env)
    
    # SPEED-OPTIMIZED: Training with reduced total timesteps for faster completion
    print("--- Starting SPEED-OPTIMIZED SB3 PPO Worker Training ---")
    training_timesteps = int(algo_cfg["total_timesteps"] * 0.8)
    print(f"Total timesteps: {training_timesteps} (optimized for speed)")
    print(f"Effective timesteps per environment: {training_timesteps // args.num_envs}")
    
    model.learn(
        total_timesteps=training_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    print("--- SPEED-OPTIMIZED Training Complete ---")

    # ✅ FIX: Save final model with normalization stats
    final_model_path = os.path.join(model_save_dir, "final_model")
    model.save(final_model_path)
    
    # Save normalization statistics separately
    normalization_path = os.path.join(model_save_dir, "vec_normalize_final.pkl")
    train_env.save(normalization_path)
    
    print(f"Final SPEED-OPTIMIZED model saved to: {final_model_path}.zip")
    print(f"Normalization stats saved to: {normalization_path}")

    # Cleanup
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()