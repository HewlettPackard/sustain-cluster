# =========================================================================
# SPEED-OPTIMIZED MANAGER TRAINING SCRIPT
# manager_training_speed_optimized.py
# =========================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
import argparse
import datetime
import pandas as pd
import sys
import asyncio
import multiprocessing
import time
import psutil

# SPEED-OPTIMIZED: Import speed-optimized environment
from envs.sustaincluster_ma_env_llm_speed_optimized import SpeedOptimizedLLMIntegratedSustainClusterMAEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rl_components.agent_net_simple import ManagerActorMLP, CentralizedCriticMLP
from rewards.registry_utils import get_reward_function
from rewards.predefined.composite_reward import CompositeReward
from utils.config_loader import load_yaml
from utils.config_logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
from utils.vec_normalize_wrapper import VecNormalizeWrapper



D_META_MANAGER = 7  # From utils.marl_utils
D_GLOBAL = 4
D_LLM_ADVICE = 10

class PickleSafeRewardWrapper:
    """Pickle-safe wrapper for reward functions"""
    def __init__(self, reward_config):
        self.reward_config = reward_config
        self._reward_fn = None
    
    def _create_reward_fn(self):
        if self._reward_fn is None:
            if "components" in self.reward_config:
                try:
                    self._reward_fn = CompositeReward(
                        components=self.reward_config["components"], 
                        normalize=self.reward_config.get('normalize', False), 
                        freeze_stats_after_steps=self.reward_config.get('freeze_stats_after_steps', 1000)
                    )
                except Exception as e:
                    print(f"CompositeReward creation failed: {e}, using fallback")
                    self._reward_fn = get_reward_function("energy_efficiency", normalize=True)
            elif "name" in self.reward_config:
                reward_name = self.reward_config["name"]
                reward_args = self.reward_config.get("args", {})
                self._reward_fn = get_reward_function(reward_name, **reward_args)
            else:
                self._reward_fn = get_reward_function("energy_efficiency", normalize=True)
        return self._reward_fn
    
    def __call__(self, *args, **kwargs):
        reward_fn = self._create_reward_fn()
        return reward_fn(*args, **kwargs)
    
    def __getstate__(self):
        return {'reward_config': self.reward_config}
    
    def __setstate__(self, state):
        self.reward_config = state['reward_config']
        self._reward_fn = None

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser(description="SPEED-OPTIMIZED: Parallel Manager Training")
    parser.add_argument("--sim-config", type=str, default="configs/env/sim_config_ma.yaml")
    parser.add_argument("--reward-config", type=str, default="configs/env/reward_config_manager_ci_only.yaml")
    parser.add_argument("--dc-config", type=str, default="configs/env/datacenters_ma.yaml")
    parser.add_argument("--algo-config", type=str, default="configs/env/ppo_algorithm_config.yaml")
    parser.add_argument("--tag", type=str, default="speed_optimized", help="Run tag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-logger", type=str2bool, default=True, help="Enable logger")
    
    # Parallelization Arguments
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--disable-llm", type=str2bool, default=False, help="Disable LLM for testing")
    parser.add_argument("--max-init-time", type=int, default=120, help="REDUCED: Max env init time")
    
    # SPEED-OPTIMIZED: LLM Integration Arguments  
    parser.add_argument("--llm-service-url", type=str, default="http://10.93.232.106:8000")
    parser.add_argument("--llm-timeout", type=float, default=2.0, help="REDUCED: Fast timeout")
    parser.add_argument("--llm-max-concurrent", type=int, default=64, help="INCREASED: High concurrency")
    parser.add_argument("--llm-history-window", type=int, default=8, help="REDUCED: Optimized history")
    
    return parser.parse_args()

def print_system_info():
    """Print system resource information"""
    print(f"🖥️  System Info:")
    print(f"  CPU cores: {multiprocessing.cpu_count()}")
    print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB total, {psutil.virtual_memory().available / (1024**3):.1f} GB available")
    print(f"  CPU usage: {psutil.cpu_percent(interval=1):.1f}%")

def create_single_env_with_timeout(rank: int, seed: int, config_paths: dict, llm_config: dict, timeout: int = 60):
    """SPEED-OPTIMIZED: Create environment with faster timeout"""
    print(f"⚡ Creating SPEED-OPTIMIZED environment {rank+1}...")
    start_time = time.time()
    
    try:
        import numpy as np
        import pandas as pd
        import datetime
        from envs.sustaincluster_ma_env_llm_speed_optimized import SpeedOptimizedLLMIntegratedSustainClusterMAEnv
        from simulation.cluster_manager_ma import DatacenterClusterManagerMA
        from utils.config_loader import load_yaml
        
        sim_cfg_full = load_yaml(config_paths["sim_config"])
        sim_cfg = sim_cfg_full["simulation"]
        dc_cfg = load_yaml(config_paths["dc_config"])["datacenters"]
        reward_cfg = load_yaml(config_paths["reward_config"])["reward"]

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
            logger=None,
        )

        reward_fn = PickleSafeRewardWrapper(reward_cfg)

        # SPEED-OPTIMIZED: Create speed-optimized LLM-integrated environment
        env = SpeedOptimizedLLMIntegratedSustainClusterMAEnv(
            cluster_manager_ma=cluster, 
            start_time=start, 
            end_time=end,
            reward_fn=reward_fn,
            logger=None,
            simple_obs_mode=True,
            use_meta_tasks=True,
            training_mode="manager",
            enable_llm_advice=not llm_config.get("disable_llm", False),
            llm_service_url=llm_config["llm_service_url"],
            llm_timeout=llm_config["llm_timeout"],  # FAST: 2.0s
            llm_max_concurrent=llm_config["llm_max_concurrent"],  # HIGH: 64
            llm_history_window=llm_config["llm_history_window"],  # OPTIMIZED: 8
            num_envs=llm_config["num_envs"]  # NEW: For thread optimization
        )
        
        env.reset(seed=seed + rank)
        
        elapsed = time.time() - start_time
        print(f"  ⚡ SPEED-OPTIMIZED Environment {rank+1} created in {elapsed:.1f}s")
        return env
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ❌ Environment {rank+1} failed after {elapsed:.1f}s: {e}")
        raise

class SpeedOptimizedVectorizedEnv:
    """SPEED-OPTIMIZED: Vectorized environment with performance focus"""
    def __init__(self, env_fns, max_init_time=120):
        print(f"🚀 Creating SPEED-OPTIMIZED vectorized environment with {len(env_fns)} environments...")
        self.env_fns = env_fns
        self.num_envs = len(env_fns)
        self.max_init_time = max_init_time
        
        print_system_info()
        
        # SPEED-OPTIMIZED: Create environments with faster timeout
        start_time = time.time()
        self.envs = []
        
        for i, fn in enumerate(env_fns):
            elapsed = time.time() - start_time
            if elapsed > self.max_init_time:
                raise TimeoutError(f"Environment initialization timed out after {elapsed:.1f}s")
            
            print(f"Creating speed-optimized environment {i+1}/{self.num_envs}...")
            env = fn()
            self.envs.append(env)
            
            # SPEED-OPTIMIZED: Less frequent memory checks
            if (i + 1) % 4 == 0:  # Check every 4 envs instead of every env
                memory_usage = psutil.virtual_memory().percent
                print(f"  Memory usage after env {i+1}: {memory_usage:.1f}%")
        
        # Get spaces from first environment
        first_env = self.envs[0]
        first_mgr_id = f"manager_{first_env._dc_ids[0]}"
        
        self.single_observation_space = first_env.observation_space(first_mgr_id)
        self.single_action_space = first_env.action_space(first_mgr_id)
        self.num_agents = first_env.num_dcs
        self.dc_ids = first_env._dc_ids
        
        total_time = time.time() - start_time
        print(f"⚡ All SPEED-OPTIMIZED environments created in {total_time:.1f}s!")
        print(f"📊 Configuration: {self.num_envs} envs × {self.num_agents} agents = {self.num_envs * self.num_agents} total agents")
    
    def reset(self, seed=None):
        """SPEED-OPTIMIZED: Faster reset with reduced logging"""
        obs_list = []
        for i, env in enumerate(self.envs):
            obs_dict, _ = env.reset(seed=seed + i if seed is not None else None)
            obs_list.append(obs_dict)
        return obs_list
    
    def step(self, actions_list):
        """Step all environments"""
        obs_list = []
        rewards_list = []
        dones_list = []
        truncs_list = []
        infos_list = []
        
        for i, (env, actions) in enumerate(zip(self.envs, actions_list)):
            obs, rewards, dones, truncs, infos = env.step(actions)
            obs_list.append(obs)
            rewards_list.append(rewards)
            dones_list.append(dones)
            truncs_list.append(truncs)
            infos_list.append(infos)
        
        return obs_list, rewards_list, dones_list, truncs_list, infos_list
    
    def close(self):
        print(f"🔒 Closing {self.num_envs} speed-optimized environments...")
        for env in self.envs:
            env.close()

def make_env(rank: int, seed: int, config_paths: dict, llm_config: dict):
    """Environment factory wrapper"""
    def _init():
        return create_single_env_with_timeout(rank, seed, config_paths, llm_config)
    return _init

def save_enhanced_checkpoint(actor_state_dict, critic_state_dict, optimizer_state_dict, 
                           norm_obs_rms, save_path, **kwargs):
    """Enhanced checkpoint saving with normalization stats."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint_data = {
        'actor_state_dict': actor_state_dict,
        'critic_state_dict': critic_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'obs_rms': norm_obs_rms,
        **kwargs  # Additional metadata
    }
    torch.save(checkpoint_data, save_path)
    print(f"Enhanced checkpoint saved to {save_path}")

async def train():
    """SPEED-OPTIMIZED: Training with performance focus"""
    args = parse_args()
    
    print(f"⚡ SPEED-OPTIMIZED Training Configuration:")
    print(f"  Environments: {args.num_envs}")
    print(f"  LLM timeout: {args.llm_timeout}s (fast)")
    print(f"  LLM concurrency: {args.llm_max_concurrent} (high)")
    print(f"  LLM history: {args.llm_history_window} (optimized)")
    print(f"  Max init time: {args.max_init_time}s (reduced)")
    
    run_name = f"PPO_MGR_SPEED_{args.tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")
    logger = setup_logger(f"logs/{run_name}", enable_logger=args.enable_logger)
    
    algo_cfg = load_yaml(args.algo_config)["ppo_algorithm"]
    
    ckpt_dir = f"checkpoints/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # Prepare configurations
    config_paths = {
        "sim_config": args.sim_config,
        "dc_config": args.dc_config,
        "reward_config": args.reward_config
    }
    
    llm_config = {
        "disable_llm": args.disable_llm,
        "llm_service_url": args.llm_service_url,
        "llm_timeout": args.llm_timeout,
        "llm_max_concurrent": args.llm_max_concurrent,
        "llm_history_window": args.llm_history_window,
        "num_envs": args.num_envs
    }

    # SPEED-OPTIMIZED: Create environments with faster initialization
    print(f"\n⚡ Creating {args.num_envs} SPEED-OPTIMIZED training environments...")
    start_total = time.time()
    
    env_fns = [make_env(i, args.seed, config_paths, llm_config) for i in range(args.num_envs)]
    base_env = SpeedOptimizedVectorizedEnv(env_fns, max_init_time=args.max_init_time)
    
    # ✅ FIX: Add VecNormalize wrapper
    env = VecNormalizeWrapper(base_env, gamma=algo_cfg["gamma"])
    
    # Evaluation environment
    print(f"\n🧪 Creating SPEED-OPTIMIZED evaluation environment...")
    eval_env_fn = make_env(99, args.seed, config_paths, llm_config)
    eval_base_env = SpeedOptimizedVectorizedEnv([eval_env_fn])
    
    # ✅ FIX: Add VecNormalize wrapper for eval
    eval_env = VecNormalizeWrapper(eval_base_env, gamma=algo_cfg["gamma"])
    eval_env.eval()  # Set to evaluation mode
    
    total_init_time = time.time() - start_total
    print(f"\n⚡ SPEED-OPTIMIZED environment setup complete in {total_init_time:.1f}s!")
    
    # ✅ FIX: Verify observation dimensions
    obs_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n
    num_agents = env.num_agents
    dc_ids = env.dc_ids
    
    # Verify observation dimension calculation
    d_option_feat = 5  # From cluster_manager_ma.py DESTINATION_OPTION_FEATURE_ORDER
    expected_base_obs_dim = D_META_MANAGER + (num_agents * d_option_feat) + D_GLOBAL
    expected_obs_dim = expected_base_obs_dim + (D_LLM_ADVICE if not args.disable_llm else 0)
    
    print(f"\n📊 Observation Space Verification:")
    print(f"  Expected base obs dim: {expected_base_obs_dim}")
    print(f"  LLM advice dim: {D_LLM_ADVICE if not args.disable_llm else 0}")
    print(f"  Expected total obs dim: {expected_obs_dim}")
    print(f"  Actual obs dim: {obs_dim}")
    
    if obs_dim != expected_obs_dim:
        print(f"⚠️  WARNING: Observation dimension mismatch!")
        print(f"  This may cause training instability.")
    else:
        print(f"✅ Observation dimensions verified correctly!")
    
    # SPEED-OPTIMIZED: Smaller networks for faster computation
    print(f"\n🧠 Creating SPEED-OPTIMIZED neural networks...")
    actor = ManagerActorMLP(obs_dim, action_dim, hidden_dim=32).to(device)  # INCREASED: 16 → 32
    critic = CentralizedCriticMLP(num_agents=num_agents, single_agent_obs_dim=obs_dim, hidden_dim=64).to(device)  # OPTIMIZED
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), 
                          lr=algo_cfg["learning_rate"], eps=1e-5)
    
    # Training setup
    num_steps = algo_cfg["num_steps"]
    total_agents = args.num_envs * num_agents
    num_updates = int(algo_cfg["total_timesteps"]) // (num_steps * total_agents)
    
    print(f"\n📋 SPEED-OPTIMIZED Training Configuration:")
    print(f"  Steps per update: {num_steps}")
    print(f"  Total updates: {num_updates}")
    print(f"  Total timesteps: {num_updates * num_steps * total_agents:,}")
    
    # SPEED-OPTIMIZED: Initialize storage with faster allocation
    obs_storage = torch.zeros((num_steps, total_agents, obs_dim), device=device, dtype=torch.float32)
    actions_storage = torch.zeros((num_steps, total_agents), device=device, dtype=torch.long)
    logprobs_storage = torch.zeros((num_steps, total_agents), device=device, dtype=torch.float32)
    rewards_storage = torch.zeros((num_steps, total_agents), device=device, dtype=torch.float32)
    dones_storage = torch.zeros((num_steps, total_agents), device=device, dtype=torch.float32)
    values_storage = torch.zeros((num_steps, total_agents), device=device, dtype=torch.float32)
    
    # Initial reset
    print(f"\n🎬 Starting SPEED-OPTIMIZED training...")
    obs_dict_list = env.reset(seed=args.seed)
    
    # Initialize observations
    next_obs = torch.zeros((total_agents, obs_dim), device=device, dtype=torch.float32)
    for env_idx in range(args.num_envs):
        obs_dict = obs_dict_list[env_idx]
        for agent_idx in range(num_agents):
            global_agent_idx = env_idx * num_agents + agent_idx
            dc_id = dc_ids[agent_idx]
            manager_key = f"manager_{dc_id}"
            if manager_key in obs_dict:
                next_obs[global_agent_idx] = torch.tensor(obs_dict[manager_key], dtype=torch.float32)
    
    next_done = torch.zeros(total_agents, device=device, dtype=torch.float32)
    best_eval_return = -np.inf

    print(f"⚡ SPEED-OPTIMIZED TRAINING STARTED!")
    global_step = 0
    
    for update in range(1, num_updates + 1):
        update_start_time = time.time()
        
        # Rollout phase with speed optimizations
        actor.eval()
        critic.eval()
        
        total_episode_reward = 0.0

        for step in range(num_steps):
            global_step += 1
            
            # SPEED-OPTIMIZED: Reduced progress logging frequency
            if step % 20 == 0 and update % 10 == 0:
                print(f"  Update {update}/{num_updates}, Step {step}/{num_steps}")
            
            obs_storage[step] = next_obs
            dones_storage[step] = next_done

            with torch.no_grad():
                action_logits = actor(next_obs)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
                
                # SPEED-OPTIMIZED: Vectorized value computation
                values = torch.zeros(total_agents, device=device, dtype=torch.float32)
                for env_idx in range(args.num_envs):
                    start_idx = env_idx * num_agents
                    end_idx = (env_idx + 1) * num_agents
                    env_obs = next_obs[start_idx:end_idx].reshape(1, -1)
                    env_value = critic(env_obs).squeeze()
                    values[start_idx:end_idx] = env_value.expand(num_agents)

            values_storage[step] = values
            actions_storage[step] = action
            logprobs_storage[step] = logprob

            # Create action lists
            actions_list = []
            for env_idx in range(args.num_envs):
                actions_dict = {}
                for agent_idx in range(num_agents):
                    global_agent_idx = env_idx * num_agents + agent_idx
                    dc_id = dc_ids[agent_idx]
                    actions_dict[f"manager_{dc_id}"] = action[global_agent_idx].item()
                    actions_dict[f"worker_{dc_id}"] = 1
                actions_list.append(actions_dict)
            
            # Step environments
            try:
                obs_list, rewards_list, dones_list, trunc_list, info_list = env.step(actions_list)

                # SPEED-OPTIMIZED: Faster results processing
                episode_rewards = []
                episode_dones = []
                
                for env_idx in range(args.num_envs):
                    env_rewards = rewards_list[env_idx]
                    env_dones = dones_list[env_idx]
                    env_trunc = trunc_list[env_idx]
                    
                    if isinstance(env_rewards, dict):
                        per_agent_rewards = [env_rewards.get(f"manager_{dc_id}", 0.0) for dc_id in dc_ids]
                        env_reward = sum(per_agent_rewards) / len(per_agent_rewards)  # Faster than np.mean
                    else:
                        env_reward = float(env_rewards)
                    
                    episode_rewards.extend([env_reward] * num_agents)
                    
                    is_done = env_dones.get("__all__", False) if isinstance(env_dones, dict) else bool(env_dones)
                    is_trunc = env_trunc.get("__all__", False) if isinstance(env_trunc, dict) else bool(env_trunc)
                    episode_dones.extend([is_done or is_trunc] * num_agents)

                rewards_storage[step] = torch.tensor(episode_rewards, dtype=torch.float32, device=device)
                next_done = torch.tensor(episode_dones, dtype=torch.float32, device=device)
                total_episode_reward += sum(episode_rewards) / len(episode_rewards)

                # Update observations
                for env_idx in range(args.num_envs):
                    next_obs_dict = obs_list[env_idx]
                    for agent_idx in range(num_agents):
                        global_agent_idx = env_idx * num_agents + agent_idx
                        dc_id = dc_ids[agent_idx]
                        manager_key = f"manager_{dc_id}"
                        if manager_key in next_obs_dict:
                            next_obs[global_agent_idx] = torch.tensor(next_obs_dict[manager_key], dtype=torch.float32)

            except Exception as e:
                print(f"❌ Error in environment step: {e}")
                rewards_storage[step] = torch.zeros(total_agents, device=device, dtype=torch.float32)
                next_done = torch.zeros(total_agents, device=device, dtype=torch.float32)

        # SPEED-OPTIMIZED: Faster GAE computation
        with torch.no_grad():
            next_values = torch.zeros(total_agents, device=device, dtype=torch.float32)
            for env_idx in range(args.num_envs):
                start_idx = env_idx * num_agents
                end_idx = (env_idx + 1) * num_agents
                env_next_obs = next_obs[start_idx:end_idx].reshape(1, -1)
                env_next_value = critic(env_next_obs).squeeze()
                next_values[start_idx:end_idx] = env_next_value.expand(num_agents)
            
            advantages = torch.zeros_like(rewards_storage)
            last_gae_lambda = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_values_for_delta = next_values
                else:
                    nextnonterminal = 1.0 - dones_storage[t + 1]
                    next_values_for_delta = values_storage[t + 1]
                
                delta = rewards_storage[t] + algo_cfg["gamma"] * next_values_for_delta * nextnonterminal - values_storage[t]
                advantages[t] = last_gae_lambda = delta + algo_cfg["gamma"] * algo_cfg["gae_lambda"] * nextnonterminal * last_gae_lambda
            returns = advantages + values_storage

        # SPEED-OPTIMIZED: Faster PPO update with larger batches
        b_obs = obs_storage.view(-1, obs_dim)
        b_actions = actions_storage.view(-1)
        b_logprobs = logprobs_storage.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        
        # Global observations for critic
        global_obs_storage = torch.zeros(num_steps, args.num_envs, num_agents * obs_dim, device=device)
        for t in range(num_steps):
            for env_idx in range(args.num_envs):
                start_idx = env_idx * num_agents
                end_idx = (env_idx + 1) * num_agents
                global_obs_storage[t, env_idx] = obs_storage[t, start_idx:end_idx].view(-1)

        actor.train()
        critic.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # SPEED-OPTIMIZED: Fewer epochs for faster training
        update_epochs = max(algo_cfg["update_epochs"] - 2, 2)  # Reduce epochs for speed
        
        for epoch in range(update_epochs):
            b_inds = torch.randperm(num_steps * total_agents, device=device)
            
            # SPEED-OPTIMIZED: Larger mini-batches for faster processing
            minibatch_size = min(algo_cfg["minibatch_size"] * 2, len(b_inds) // 4)
            
            for start in range(0, len(b_inds), minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                new_logits = actor(b_obs[mb_inds])
                new_dist = torch.distributions.Categorical(logits=new_logits)
                new_logprob = new_dist.log_prob(b_actions[mb_inds])
                entropy = new_dist.entropy()

                # Critic inputs
                mb_global_obs_list = []
                for idx in mb_inds:
                    timestep = idx // total_agents
                    agent_in_timestep = idx % total_agents
                    env_idx = agent_in_timestep // num_agents
                    mb_global_obs_list.append(global_obs_storage[timestep, env_idx])
                
                mb_global_obs = torch.stack(mb_global_obs_list)
                new_value = critic(mb_global_obs).squeeze()

                # PPO losses
                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - algo_cfg["clip_coef"], 1 + algo_cfg["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - algo_cfg["ent_coef"] * entropy_loss + v_loss * algo_cfg["vf_coef"]
                
                total_loss += loss.item()
                num_batches += 1

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), algo_cfg["max_grad_norm"])
                optimizer.step()

        # Logging and progress
        avg_loss = total_loss / max(num_batches, 1)
        avg_reward = total_episode_reward / num_steps
        update_time = time.time() - update_start_time
        
        # SPEED-OPTIMIZED: Reduced logging frequency
        if update % 5 == 0:  # Log every 5 updates instead of every update
            writer.add_scalar("Loss/Total", avg_loss, global_step)
            writer.add_scalar("Rollout/EpisodeReward", avg_reward, global_step)
            writer.add_scalar("Performance/UpdateTime", update_time, global_step)
            writer.add_scalar("Performance/StepsPerSecond", num_steps / update_time, global_step)

        # Progress logging
        if update % 5 == 0:  # Reduced frequency
            print(f"⚡ Update {update}/{num_updates}: reward={avg_reward:.4f}, loss={avg_loss:.4f}, time={update_time:.1f}s")
        
        # SPEED-OPTIMIZED: Less frequent checkpointing
        if update % 20 == 0:
            checkpoint_path = os.path.join(ckpt_dir, f"checkpoint_update_{update}.pth")
            torch.save({
                'update': update,
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_reward': avg_reward,
                'args': vars(args)
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

        # SPEED-OPTIMIZED: Less frequent evaluation
        if update % 50 == 0:
            print(f"🧪 Running SPEED-OPTIMIZED evaluation at update {update}...")
            actor.eval()
            
            if hasattr(eval_env, 'obs_rms') and hasattr(env, 'obs_rms'):
                eval_env.obs_rms = env.obs_rms
                print("📊 Shared normalization stats with evaluation environment")
    
            
            total_eval_reward = 0.0
            for ep in range(1):  # REDUCED: 2 → 1 episode for speed
                eval_obs_list = eval_env.reset(seed=args.seed + ep)
                
                eval_ep_return = 0.0
                done = False
                steps = 0
                max_steps = 30  # REDUCED: 50 → 30 for faster evaluation
                
                while not done and steps < max_steps:
                    eval_obs_dict = eval_obs_list[0]
                    eval_obs_tensor = torch.zeros((num_agents, obs_dim), device=device, dtype=torch.float32)
                    
                    for agent_idx in range(num_agents):
                        dc_id = dc_ids[agent_idx]
                        manager_key = f"manager_{dc_id}"
                        if manager_key in eval_obs_dict:
                            eval_obs_tensor[agent_idx] = torch.tensor(eval_obs_dict[manager_key], dtype=torch.float32)
                    
                    with torch.no_grad():
                        action_logits = actor(eval_obs_tensor)
                        actions = torch.argmax(action_logits, dim=1)

                    actions_dict = {f"manager_{dc_id}": actions[i].item() for i, dc_id in enumerate(dc_ids)}
                    actions_dict.update({f"worker_{dc_id}": 1 for dc_id in dc_ids})

                    try:
                        eval_obs_list, eval_rewards_list, eval_dones_list, eval_trunc_list, _ = eval_env.step([actions_dict])
                        
                        eval_rewards = eval_rewards_list[0]
                        if isinstance(eval_rewards, dict):
                            eval_global_reward = sum([eval_rewards.get(f"manager_{dc_id}", 0.0) for dc_id in dc_ids]) / len(dc_ids)
                        else:
                            eval_global_reward = float(eval_rewards)
                        
                        eval_ep_return += eval_global_reward

                        eval_dones = eval_dones_list[0]
                        eval_trunc = eval_trunc_list[0]
                        done = (eval_dones.get("__all__", False) if isinstance(eval_dones, dict) else bool(eval_dones)) or \
                               (eval_trunc.get("__all__", False) if isinstance(eval_trunc, dict) else bool(eval_trunc))
                        
                        steps += 1
                    except Exception as e:
                        print(f"⚠️ Evaluation step error: {e}")
                        break

                total_eval_reward += eval_ep_return

            avg_eval_return = total_eval_reward
            writer.add_scalar("Eval/AverageReturn", avg_eval_return, global_step)
            print(f"📊 SPEED-OPTIMIZED Eval Return: {avg_eval_return:.4f}")

            if avg_eval_return > best_eval_return:
                best_eval_return = avg_eval_return
                print("🏆 New best evaluation return! Saving best model...")
                
                best_model_path = os.path.join(ckpt_dir, "best_model.pth")
                save_enhanced_checkpoint(
                    actor_state_dict=actor.state_dict(),
                    critic_state_dict=critic.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    norm_obs_rms=env.obs_rms,
                    save_path=best_model_path,
                    best_eval_return=best_eval_return,
                    update=update,
                    args=vars(args)
                )

    print(f"⚡ SPEED-OPTIMIZED training complete! Best eval return: {best_eval_return:.4f}")
    
    # Final save
    final_model_path = os.path.join(ckpt_dir, "final_model.pth")
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_eval_return': best_eval_return,
        'args': vars(args)
    }, final_model_path)
    print(f"💾 Final model saved: {final_model_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    writer.close()

def main():
    """Main entry point"""
    asyncio.run(train())

if __name__ == "__main__":
    main()