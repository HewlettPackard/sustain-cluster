# train_ppo_manager_llm_enhanced.py - FIXED VERSION
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange, tqdm
import logging
import os
import argparse
import datetime
import pandas as pd
import sys
import asyncio

from envs.sustaincluster_ma_env_llm import LLMIntegratedSustainClusterMAEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
# Import simple MLP networks
from rl_components.agent_net_simple import ManagerActorMLP, CentralizedCriticMLP
from rewards.registry_utils import get_reward_function
from rewards.predefined.composite_reward import CompositeReward
from utils.config_loader import load_yaml
from utils.config_logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
from utils.vec_normalize import VecNormalize

class RunningStats:
    def __init__(self, eps=1e-5):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x):
        x = float(x)
        self.count += 1
        last_mean = self.mean
        self.mean += (x - self.mean) / self.count
        self.var += (x - last_mean) * (x - self.mean)

    def normalize(self, x):
        std = max(np.sqrt(self.var / self.count), 1e-6)
        return (x - self.mean) / std
    
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
    parser = argparse.ArgumentParser(description="LLM-Enhanced PPO Manager Training")
    parser.add_argument("--sim-config", type=str, default="configs/env/sim_config_ma.yaml")
    parser.add_argument("--reward-config", type=str, default="configs/env/reward_config_manager_ci_only.yaml")
    parser.add_argument("--dc-config", type=str, default="configs/env/datacenters_ma.yaml")
    parser.add_argument("--algo-config", type=str, default="configs/env/ppo_algorithm_config.yaml")
    parser.add_argument("--tag", type=str, default="llm_enhanced", help="Run tag for LLM-enhanced training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-logger", type=str2bool, default=True, help="Enable logger")
    
    # NEW: LLM Integration Arguments
    parser.add_argument("--enable-llm", type=str2bool, default=True, help="Enable LLM advice integration")
    parser.add_argument("--llm-service-url", type=str, default="http://10.93.232.106:8000", help="LLM service URL")
    parser.add_argument("--llm-timeout", type=float, default=2.0, help="LLM request timeout in seconds")
    parser.add_argument("--llm-max-concurrent", type=int, default=16, help="Max concurrent LLM requests")
    parser.add_argument("--llm-history-window", type=int, default=10, help="LLM context history window")
    
    return parser.parse_args()

def save_simple_checkpoint(actor_state_dict, critic_state_dict, optimizer_state_dict, norm_obs_rms, save_path):
    """Saves a simple checkpoint for the PPO agent including normalization stats."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'actor_state_dict': actor_state_dict,
        'critic_state_dict': critic_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'obs_rms': norm_obs_rms,
    }, save_path)
    print(f"Saved best model checkpoint to {save_path}")

def make_llm_enhanced_env(sim_cfg_path, dc_cfg_path, reward_cfg_path, 
                         enable_llm=True, llm_service_url="http://10.93.232.106:8000",
                         llm_timeout=2.0, llm_max_concurrent=16, llm_history_window=10,
                         training_mode="manager",  # NEW: Specify training mode
                         logger=None):
    """Create LLM-enhanced environment with training mode specification"""
    sim_cfg_full = load_yaml(sim_cfg_path)
    sim_cfg = sim_cfg_full["simulation"]
    dc_cfg = load_yaml(dc_cfg_path)["datacenters"]
    reward_cfg = load_yaml(reward_cfg_path)["reward"]

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
        logger=logger,
    )

    # Build reward function
    if "components" in reward_cfg:
        if logger: logger.info("Building CompositeReward from config.")
        reward_fn = CompositeReward(components=reward_cfg["components"], 
                                  normalize=reward_cfg['normalize'], 
                                  freeze_stats_after_steps=reward_cfg['freeze_stats_after_steps'])
    elif "name" in reward_cfg:
        if logger: logger.info(f"Building single reward function: {reward_cfg['name']}")
        reward_name = reward_cfg["name"]
        reward_args = reward_cfg.get("args", {})
        reward_fn = get_reward_function(reward_name, **reward_args)
    else:
        raise ValueError("Reward configuration in YAML is invalid.")

    # Create LLM-enhanced environment with training mode specification
    return LLMIntegratedSustainClusterMAEnv(
        cluster_manager_ma=cluster, 
        start_time=start, 
        end_time=end,
        reward_fn=reward_fn, 
        logger=logger, 
        simple_obs_mode=True, 
        use_meta_tasks=True,
        # NEW: Specify training mode
        training_mode=training_mode,  # "manager", "worker", or "both"
        # LLM parameters
        enable_llm_advice=enable_llm,
        llm_service_url=llm_service_url,
        llm_timeout=llm_timeout,
        llm_max_concurrent=llm_max_concurrent,
        llm_history_window=llm_history_window
    )


# FIXED: Helper functions to handle async environment operations
async def async_reset_env(env, seed=None):
    """Async wrapper for environment reset"""
    if hasattr(env.venv, 'reset') and asyncio.iscoroutinefunction(env.venv.reset):
        return await env.venv.reset(seed=seed)
    else:
        return env.reset(seed=seed)

async def async_step_env(env, actions):
    """Async wrapper for environment step"""
    if hasattr(env.venv, 'step') and asyncio.iscoroutinefunction(env.venv.step):
        return await env.venv.step(actions)
    else:
        return env.step(actions)

async def train():
    """Enhanced training with LLM integration - FIXED VERSION"""
    # --- 1. Initialization and Configuration ---
    args = parse_args()
    run_name = f"PPO_MGR_LLM_{args.tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")
    logger = setup_logger(f"logs/{run_name}", enable_logger=args.enable_logger)
    algo_cfg = load_yaml(args.algo_config)["ppo_algorithm"]
    
    ckpt_dir = f"checkpoints/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Log LLM configuration
    if args.enable_llm:
        logger.info(f"LLM Integration Enabled:")
        logger.info(f"  Service URL: {args.llm_service_url}")
        logger.info(f"  Timeout: {args.llm_timeout}s")
        logger.info(f"  Max Concurrent: {args.llm_max_concurrent}")
        logger.info(f"  History Window: {args.llm_history_window}")
    else:
        logger.info("LLM Integration Disabled - Using standard observations")

    # --- 2. Environment Setup ---
    base_env = make_llm_enhanced_env(
        args.sim_config, args.dc_config, args.reward_config,
        enable_llm=args.enable_llm,
        llm_service_url=args.llm_service_url,
        llm_timeout=args.llm_timeout,
        llm_max_concurrent=args.llm_max_concurrent,
        llm_history_window=args.llm_history_window,
        training_mode="manager", 
        logger=logger
    )
    env = VecNormalize(venv=base_env)

    # Evaluation environment
    eval_base_env = make_llm_enhanced_env(
        args.sim_config, args.dc_config, args.reward_config,
        enable_llm=args.enable_llm,
        llm_service_url=args.llm_service_url,
        llm_timeout=args.llm_timeout,
        llm_max_concurrent=args.llm_max_concurrent,
        llm_history_window=args.llm_history_window,
        logger=logger
    )
    eval_env = VecNormalize(venv=eval_base_env)
    eval_env.eval()
    
    # Get observation and action dimensions
    first_mgr_id = f"manager_{env.venv._dc_ids[0]}"
    obs_dim = env.venv.observation_space(first_mgr_id).shape[0]
    action_dim = env.venv.action_space(first_mgr_id).n
    num_agents = env.venv.num_dcs

    logger.info(f"Environment created - Obs dim: {obs_dim}, Action dim: {action_dim}, Agents: {num_agents}")
    if args.enable_llm:
        base_obs_dim = obs_dim - 10  # Subtract LLM advice dimensions
        logger.info(f"LLM-enhanced obs: Base({base_obs_dim}) + LLM advice(10) = Total({obs_dim})")

    # --- 3. PPO Agent & Optimizer Setup ---
    actor = ManagerActorMLP(obs_dim, action_dim, hidden_dim=16).to(device)
    critic = CentralizedCriticMLP(num_agents=num_agents, single_agent_obs_dim=obs_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), 
                          lr=algo_cfg["learning_rate"], eps=1e-5)

    # --- 4. PPO Rollout Buffer Storage ---
    num_steps = algo_cfg["num_steps"]
    obs_storage = torch.zeros((num_steps, num_agents, obs_dim)).to(device)
    actions_storage = torch.zeros((num_steps, num_agents)).to(device)
    logprobs_storage = torch.zeros((num_steps, num_agents)).to(device)
    rewards_storage = torch.zeros((num_steps, num_agents)).to(device)
    dones_storage = torch.zeros((num_steps, num_agents)).to(device)
    values_storage = torch.zeros((num_steps, num_agents)).to(device)

    # --- 5. Main Training Loop ---
    global_step = 0
    num_updates = int(algo_cfg["total_timesteps"]) // (num_steps * num_agents)
    
    # FIXED: Handle reset properly - VecNormalize returns (obs, info) synchronously
    obs_dict, _ = env.reset(seed=args.seed)
    next_obs = torch.Tensor(np.stack([obs_dict[f"manager_{dc_id}"] for dc_id in env.venv._dc_ids])).to(device)
    next_done = torch.zeros(num_agents).to(device)
    current_episode_return = 0.0
    best_eval_return = -np.inf

    # Track LLM statistics
    llm_stats = {"requests": 0, "successes": 0, "failures": 0}

    for update in range(1, num_updates + 1):
        actor.eval()
        critic.eval()

        for step in range(num_steps):
            global_step += 1
            
            obs_storage[step] = next_obs
            dones_storage[step] = next_done

            with torch.no_grad():
                action_logits = actor(next_obs)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                logprob = dist.log_prob(action)
                global_obs = next_obs.reshape(1, -1)
                value = critic(global_obs).squeeze()

            values_storage[step] = value.expand(num_agents)
            actions_storage[step] = action
            logprobs_storage[step] = logprob

            actions_dict = {f"manager_{dc_id}": action[i].item() for i, dc_id in enumerate(env.venv._dc_ids)}
            actions_dict.update({f"worker_{dc_id}": 1 for dc_id in env.venv._dc_ids})
            
            # FIXED: Step environment - VecNormalize returns tuple synchronously
            next_obs_dict, rewards_dict, dones_dict, trunc_dict, info_dict = env.step(actions_dict)

            per_agent_rewards = [rewards_dict.get(f"manager_{dc_id}", 0.0) for dc_id in env.venv._dc_ids]
            global_reward = np.mean(per_agent_rewards)
            rewards_storage[step] = torch.tensor([global_reward] * num_agents).to(device)
            current_episode_return += global_reward
            
            is_done = dones_dict["__all__"]
            is_truncated = trunc_dict["__all__"]
            next_done = torch.tensor([is_done or is_truncated] * num_agents).to(device)

            # Track LLM statistics
            if args.enable_llm and "__global__" in info_dict and "llm_stats" in info_dict["__global__"]:
                llm_info = info_dict["__global__"]["llm_stats"]
                llm_stats["requests"] = llm_info.get("request_count", 0)
                llm_stats["successes"] = llm_info.get("success_count", 0)

            if is_done or is_truncated:
                writer.add_scalar("Rollout/Episode_Return", current_episode_return, global_step)
                if args.enable_llm and llm_stats["requests"] > 0:
                    llm_success_rate = llm_stats["successes"] / llm_stats["requests"]
                    writer.add_scalar("LLM/Success_Rate", llm_success_rate, global_step)
                    writer.add_scalar("LLM/Total_Requests", llm_stats["requests"], global_step)
                
                if logger: 
                    logger.info(f"global_step={global_step}, episodic_return={current_episode_return:.2f}")
                    if args.enable_llm and llm_stats["requests"] > 0:
                        logger.info(f"  LLM stats: {llm_stats['successes']}/{llm_stats['requests']} success rate")
                
                current_episode_return = 0.0
                # FIXED: Reset properly
                next_obs_dict, _ = env.reset(seed=args.seed+update*2+step//60)

            next_obs = torch.Tensor(np.stack([next_obs_dict[f"manager_{dc_id}"] for dc_id in env.venv._dc_ids])).to(device)
            
        # --- GAE and Update Phase ---
        with torch.no_grad():
            global_next_obs = next_obs.reshape(1, -1)
            next_value = critic(global_next_obs).squeeze()
            
            advantages = torch.zeros_like(rewards_storage).to(device)
            last_gae_lambda = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    next_values_for_delta = next_value.expand(num_agents)
                else:
                    nextnonterminal = 1.0 - dones_storage[t + 1].float()
                    next_values_for_delta = values_storage[t + 1]
                
                delta = rewards_storage[t] + algo_cfg["gamma"] * next_values_for_delta * nextnonterminal - values_storage[t]
                advantages[t] = last_gae_lambda = delta + algo_cfg["gamma"] * algo_cfg["gae_lambda"] * nextnonterminal * last_gae_lambda
            returns = advantages + values_storage

        # Flatten for training
        b_obs = obs_storage.reshape(-1, obs_dim)
        b_actions = actions_storage.reshape(-1)
        b_logprobs = logprobs_storage.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        global_obs_storage = obs_storage.permute(0, 1, 2).reshape(num_steps, -1)

        actor.train()
        critic.train()
        
        # PPO Update Epochs
        for epoch in range(algo_cfg["update_epochs"]):
            b_inds = np.arange(num_steps * num_agents)
            np.random.shuffle(b_inds)
            
            total_loss = 0.0
            pg_loss = 0.0
            v_loss = 0.0
            entropy_loss = 0.0
            
            for start in range(0, num_steps * num_agents, algo_cfg["minibatch_size"]):
                end = start + algo_cfg["minibatch_size"]
                mb_inds = b_inds[start:end]

                new_logits = actor(b_obs[mb_inds])
                new_dist = torch.distributions.Categorical(logits=new_logits)
                new_logprob = new_dist.log_prob(b_actions[mb_inds])
                entropy = new_dist.entropy()

                mb_timestep_inds = (mb_inds // num_agents)
                mb_global_obs = global_obs_storage[mb_timestep_inds]
                new_value = critic(mb_global_obs).squeeze()

                # Policy Loss
                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - algo_cfg["clip_coef"], 1 + algo_cfg["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                # Entropy Loss
                entropy_loss = entropy.mean()

                # Total Loss
                loss = pg_loss - algo_cfg["ent_coef"] * entropy_loss + v_loss * algo_cfg["vf_coef"]
                
                total_loss += loss.item()
                pg_loss += pg_loss.item()
                v_loss += v_loss.item()
                entropy_loss += entropy_loss.item()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), algo_cfg["max_grad_norm"])
                optimizer.step()
            
            # Average losses
            total_loss /= (num_steps * num_agents // algo_cfg["minibatch_size"])
            pg_loss /= (num_steps * num_agents // algo_cfg["minibatch_size"])
            v_loss /= (num_steps * num_agents // algo_cfg["minibatch_size"])
            entropy_loss /= (num_steps * num_agents // algo_cfg["minibatch_size"])

        # Logging after each update
        writer.add_scalar("Loss/Total", total_loss, global_step)
        writer.add_scalar("Loss/Policy", pg_loss, global_step)
        writer.add_scalar("Loss/Value", v_loss, global_step)
        writer.add_scalar("Loss/Entropy", entropy_loss, global_step)

        # Evaluation
        if update % algo_cfg.get("eval_frequency", 10) == 0:
            if logger: logger.info(f"--- Running evaluation at update {update}/{num_updates} ---")
            actor.eval()
            eval_env.obs_rms = env.obs_rms
            
            total_eval_reward = 0.0
            eval_llm_stats = {"requests": 0, "successes": 0}
            
            for ep in range(algo_cfg.get("eval_episodes", 5)):
                # FIXED: Evaluation reset
                eval_obs_dict, _ = eval_env.reset(seed=args.seed + ep)
                eval_ep_return = 0.0
                done = False
                
                while not done:
                    eval_obs_tensor = torch.Tensor(np.stack([eval_obs_dict[f"manager_{dc_id}"] for dc_id in eval_env.venv._dc_ids])).to(device)
                    with torch.no_grad():
                        action_logits = actor(eval_obs_tensor)
                        actions = torch.argmax(action_logits, dim=1)

                    actions_dict = {f"manager_{dc_id}": actions[i].item() for i, dc_id in enumerate(eval_env.venv._dc_ids)}
                    actions_dict.update({f"worker_{dc_id}": 1 for dc_id in eval_env.venv._dc_ids})

                    # FIXED: Evaluation step
                    eval_next_obs_dict, eval_rewards_dict, eval_dones_dict, eval_trunc_dict, eval_info = eval_env.step(actions_dict)

                    eval_global_reward = np.mean([eval_rewards_dict.get(f"manager_{dc_id}", 0.0) for dc_id in eval_env.venv._dc_ids])
                    eval_ep_return += eval_global_reward

                    # Track eval LLM stats
                    if args.enable_llm and "__global__" in eval_info and "llm_stats" in eval_info["__global__"]:
                        eval_llm_info = eval_info["__global__"]["llm_stats"]
                        eval_llm_stats["requests"] = eval_llm_info.get("request_count", 0)
                        eval_llm_stats["successes"] = eval_llm_info.get("success_count", 0)

                    done = eval_dones_dict["__all__"] or eval_trunc_dict["__all__"]
                    eval_obs_dict = eval_next_obs_dict

                total_eval_reward += eval_ep_return
                if logger: logger.info(f"  Eval Episode {ep+1} Return: {eval_ep_return:.2f}")

            avg_eval_return = total_eval_reward / algo_cfg.get("eval_episodes", 5)
            writer.add_scalar("Eval/AverageReturn", avg_eval_return, global_step)
            
            if args.enable_llm and eval_llm_stats["requests"] > 0:
                eval_llm_success_rate = eval_llm_stats["successes"] / eval_llm_stats["requests"]
                writer.add_scalar("Eval/LLM_Success_Rate", eval_llm_success_rate, global_step)
            
            if logger: 
                logger.info(f"  Average Evaluation Return: {avg_eval_return:.2f}")
                if args.enable_llm and eval_llm_stats["requests"] > 0:
                    logger.info(f"  Eval LLM stats: {eval_llm_stats['successes']}/{eval_llm_stats['requests']} success rate")

            # Save best model
            if avg_eval_return > best_eval_return:
                best_eval_return = avg_eval_return
                if logger: logger.info("  ** New best evaluation return! Saving model... **")
                
                best_model_path = os.path.join(ckpt_dir, "best_model.pth")
                save_simple_checkpoint(
                    actor_state_dict=actor.state_dict(),
                    critic_state_dict=critic.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    norm_obs_rms=env.obs_rms,
                    save_path=best_model_path
                )

    # Cleanup
    if hasattr(env.venv, 'close'):
        await env.venv.close()
    if hasattr(eval_env.venv, 'close'):
        await eval_env.venv.close()
    writer.close()

def main():
    """Main entry point"""
    # Run the async training function
    asyncio.run(train())

if __name__ == "__main__":
    main()