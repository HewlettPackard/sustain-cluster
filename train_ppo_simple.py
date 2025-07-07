# train_ppo_simple.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
import logging
import os
import argparse
import datetime
import pandas as pd

# --- Local Imports ---
from envs.sustaincluster_simple_env import SustainClusterSimpleEnv # <-- Import the new simple env
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rl_components.agent_net_simple import ManagerActorMLP, CentralizedCriticMLP # <-- Use the Centralized Critic
from rewards.registry_utils import get_reward_function
from rewards.predefined.composite_reward import CompositeReward # Assuming a simple reward can be wrapped
from utils.config_loader import load_yaml
from utils.config_logger import setup_logger
from torch.utils.tensorboard import SummaryWriter

# --- Argument Parsing and Helper Functions ---
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "1"): return True
    elif v.lower() in ("no", "false", "f", "0"): return False
    else: raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Simple Global Scheduler Training")
    parser.add_argument("--sim-config", type=str, default="configs/env/sim_config_ma.yaml")
    parser.add_argument("--reward-config", type=str, default="configs/env/reward_config_simple.yaml")
    parser.add_argument("--dc-config", type=str, default="configs/env/datacenters_ma.yaml")
    parser.add_argument("--algo-config", type=str, default="configs/env/ppo_algorithm_config.yaml")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-logger", type=str2bool, default=True, help="Enable logger")
    return parser.parse_args()

# --- Environment Creation ---
def make_env(sim_cfg_path, dc_cfg_path, reward_cfg_path, logger=None):
    sim_cfg_full = load_yaml(sim_cfg_path)
    sim_cfg = sim_cfg_full["simulation"]
    dc_cfg = load_yaml(dc_cfg_path)["datacenters"]
    reward_cfg = load_yaml(reward_cfg_path)["reward"]

    for cfg in dc_cfg:
        cfg.setdefault("simulation_year", sim_cfg["year"])
        
    start = pd.Timestamp(datetime.datetime(sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"],
                                           sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=sim_cfg["duration_days"])

    # This reward function might need to be adapted to not expect per-agent data
    if "name" in reward_cfg:
        reward_fn = get_reward_function(reward_cfg["name"], **reward_cfg.get("args", {}))
    else:
        raise ValueError("A named reward function is expected for the simple environment.")

    cluster = DatacenterClusterManagerMA(config_list=dc_cfg, **sim_cfg) # Pass sim_cfg directly
    
    return SustainClusterSimpleEnv(
        cluster_manager=cluster, start_time=start, end_time=end,
        reward_fn=reward_fn, logger=logger
    )

def train():
    # --- 1. Initialization and Configuration ---
    args = parse_args()
    run_name = f"PPO_CTDE_SIMPLE_{args.tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")
    logger = setup_logger(f"logs/{run_name}", enable_logger=args.enable_logger)
    algo_cfg = load_yaml(args.algo_config)["ppo_algorithm"]
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Environment and Agent Setup ---
    env = make_env(args.sim_config, args.dc_config, args.reward_config, logger)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    actor = ManagerActorMLP(obs_dim, action_dim).to(device)
    # Critic is now also decentralized (standard PPO)
    critic = CentralizedCriticMLP(num_agents=num_agents, single_agent_obs_dim=obs_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()),
                           lr=algo_cfg["learning_rate"], eps=1e-5)

    # --- 4. PPO Rollout Buffer Storage ---
    num_steps = algo_cfg["num_steps"]
    obs_storage = torch.zeros((num_steps, obs_dim)).to(device)
    actions_storage = torch.zeros((num_steps,)).to(device)
    logprobs_storage = torch.zeros((num_steps,)).to(device)
    rewards_storage = torch.zeros((num_steps,)).to(device)
    dones_storage = torch.zeros((num_steps,)).to(device)
    values_storage = torch.zeros((num_steps,)).to(device)

    # --- 5. Main Training Loop ---
    global_step = 0
    num_updates = int(algo_cfg["total_timesteps"]) // num_steps
    
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)
    
    pbar = trange(1, num_updates + 1)
    for update in pbar:
        actor.eval()
        critic.eval()
        
        # --- 5a. Rollout Phase ---
        for step in range(num_steps):
            global_step += 1
            obs_storage[step] = next_obs
            dones_storage[step] = next_done

            with torch.no_grad():
                dist = actor.get_action_dist(next_obs)
                action = dist.sample()
                logprob = dist.log_prob(action)
                value = critic(next_obs).squeeze()

            actions_storage[step] = action
            logprobs_storage[step] = logprob
            values_storage[step] = value

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            
            rewards_storage[step] = torch.tensor(reward).to(device).view(1)
            next_done = torch.tensor(terminated or truncated).to(device).view(1)
            next_obs = torch.Tensor(next_obs).to(device)

            if terminated or truncated:
                # Logging episode return would happen here if tracked
                writer.add_scalar("Rollout/Episode_Return", info.get("episode_return", 0), global_step)

        # --- 5b. GAE and Update Phase ---
        with torch.no_grad():
            next_value = critic(next_obs).squeeze()
            advantages = torch.zeros_like(rewards_storage).to(device)
            last_gae_lambda = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_storage[t + 1].float()
                    nextvalues = values_storage[t + 1]
                
                delta = rewards_storage[t] + algo_cfg["gamma"] * nextvalues * nextnonterminal - values_storage[t]
                advantages[t] = last_gae_lambda = delta + algo_cfg["gamma"] * algo_cfg["gae_lambda"] * nextnonterminal * last_gae_lambda
            returns = advantages + values_storage

        b_obs = obs_storage.reshape(-1, obs_dim)
        b_actions = actions_storage.reshape(-1)
        b_logprobs = logprobs_storage.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        actor.train()
        critic.train()

        for epoch in range(algo_cfg["update_epochs"]):
            b_inds = np.arange(num_steps)
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, algo_cfg["minibatch_size"]):
                end = start + algo_cfg["minibatch_size"]
                mb_inds = b_inds[start:end]

                new_dist = actor.get_action_dist(b_obs[mb_inds])
                new_logprob = new_dist.log_prob(b_actions[mb_inds])
                entropy = new_dist.entropy()
                new_value = critic(b_obs[mb_inds]).squeeze()

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

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), algo_cfg["max_grad_norm"])
                optimizer.step()

        # Logging
        pbar.set_description(f"Update {update}/{num_updates}, Value Loss: {v_loss.item():.4f}")
        writer.add_scalar("Loss/Policy", pg_loss.item(), global_step)
        writer.add_scalar("Loss/Value", v_loss.item(), global_step)
        writer.add_scalar("Policy/Entropy", entropy_loss.item(), global_step)

    # --- Final saving and cleanup ---
    env.close()
    writer.close()
    logger.info("Training finished.")

if __name__ == "__main__":
    train()