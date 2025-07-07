# train_ppo_manager_only.py

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

from envs.sustaincluster_ma_env import SustainClusterMAEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
# Import simple MLP networks
from rl_components.agent_net_simple import ManagerActorMLP, CentralizedCriticMLP
from rewards.registry_utils import get_reward_function
from rewards.predefined.composite_reward import CompositeReward
from utils.config_loader import load_yaml
from utils.config_logger import setup_logger
from torch.utils.tensorboard import SummaryWriter
from utils.vec_normalize import VecNormalize # <-- Add this import


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
    # Adjusted description for clarity
    parser = argparse.ArgumentParser(description="PPO Manager-Only SustainCluster Training")
    parser.add_argument("--sim-config", type=str, default="configs/env/sim_config_ma.yaml")
    parser.add_argument("--reward-config", type=str, default="configs/env/reward_config_manager_ci_only.yaml")
    parser.add_argument("--dc-config", type=str, default="configs/env/datacenters_ma.yaml")
    parser.add_argument("--algo-config", type=str, default="configs/env/ppo_algorithm_config.yaml")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag for manager-only training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-logger", type=str2bool, default=True, help="Enable logger")
    return parser.parse_args()


def save_simple_checkpoint(actor_state_dict, critic_state_dict, optimizer_state_dict, norm_obs_rms, save_path):
    """Saves a simple checkpoint for the PPO agent including normalization stats."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'actor_state_dict': actor_state_dict,
        'critic_state_dict': critic_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'obs_rms': norm_obs_rms, # <-- Add the stats
    }, save_path)
    print(f"Saved best model checkpoint to {save_path}")

# --- Environment Creation (can be adapted from your SAC script) ---
def make_simple_env(sim_cfg_path, dc_cfg_path, reward_cfg_path, writer=None, logger=None):
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

    
    # Check if the config is for a composite reward or a single reward
    if "components" in reward_cfg:
        # This is the old logic for composite rewards
        if logger: logger.info("Building CompositeReward from config.")
        reward_fn = CompositeReward(components=reward_cfg["components"], normalize=reward_cfg['normalize'], freeze_stats_after_steps=reward_cfg['freeze_stats_after_steps'])
    elif "name" in reward_cfg:
        # This is the new logic for single, named rewards
        if logger: logger.info(f"Building single reward function: {reward_cfg['name']}")
        reward_name = reward_cfg["name"]
        reward_args = reward_cfg.get("args", {}) # Get args, or an empty dict if not present
        
        # Use the utility to dynamically get the reward function by its registered name
        reward_fn = get_reward_function(reward_name, **reward_args)
    else:
        # If the config is malformed, raise an error
        raise ValueError("Reward configuration in YAML is invalid. Must contain either 'components' or 'name' key.")
    

    return SustainClusterMAEnv(cluster_manager_ma=cluster, start_time=start, end_time=end,
        reward_fn=reward_fn, logger = logger, simple_obs_mode=True, use_meta_tasks=True)

def train():
    # --- 1. Initialization and Configuration ---
    args = parse_args()
    run_name = f"PPO_MGR_ONLY_{args.tag}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")
    logger = setup_logger(f"logs/{run_name}", enable_logger=args.enable_logger)
    algo_cfg = load_yaml(args.algo_config)["ppo_algorithm"]
    
    ckpt_dir = f"checkpoints/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Environment Setup ---
    # Create the base environment first
    base_env = make_simple_env(args.sim_config, args.dc_config, args.reward_config, logger)
    # Wrap it with the normalization layer
    env = VecNormalize(venv=base_env)

    # For evaluation, create a separate wrapped environment
    eval_base_env = make_simple_env(args.sim_config, args.dc_config, args.reward_config, logger)
    eval_env = VecNormalize(venv=eval_base_env)
    eval_env.eval() # Set the eval env to not update its own stats
    
    first_mgr_id = f"manager_{env.venv._dc_ids[0]}"
    obs_dim = env.venv.observation_space(first_mgr_id).shape[0]
    action_dim = env.venv.action_space(first_mgr_id).n
    num_agents = env.venv.num_dcs

    # --- 3. PPO Agent & Optimizer Setup ---
    actor = ManagerActorMLP(obs_dim, action_dim, hidden_dim=16).to(device)
    critic = CentralizedCriticMLP(num_agents=num_agents, single_agent_obs_dim=obs_dim).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=algo_cfg["learning_rate"], eps=1e-5)

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
    
    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        
    obs_dict, _ = env.reset(seed=args.seed)
    next_obs = torch.Tensor(np.stack([obs_dict[f"manager_{dc_id}"] for dc_id in env.venv._dc_ids])).to(device)
    next_done = torch.zeros(num_agents).to(device)
    current_episode_return = 0.0

    best_eval_return = -np.inf

    for update in range(1, num_updates + 1):
        # Optional: Learning rate annealing could be added here
        
        actor.eval() # Set to eval mode for rollout
        critic.eval()

        for step in range(num_steps):
            global_step += 1
            
            # Store s_t and the done flag d_{t-1} corresponding to it
            obs_storage[step] = next_obs
            dones_storage[step] = next_done # This stores the "episode over" flag

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
            next_obs_dict, rewards_dict, dones_dict, trunc_dict, _ = env.step(actions_dict)

            per_agent_rewards = [rewards_dict.get(f"manager_{dc_id}", 0.0) for dc_id in env.venv._dc_ids]
            global_reward = np.mean(per_agent_rewards)
            rewards_storage[step] = torch.tensor([global_reward] * num_agents).to(device)
            current_episode_return += global_reward
            
            is_done = dones_dict["__all__"]
            is_truncated = trunc_dict["__all__"]
            
            next_done = torch.tensor([is_done or is_truncated] * num_agents).to(device)

            if is_done or is_truncated:
                writer.add_scalar("Rollout/Episode_Return", current_episode_return, global_step)
                if logger: logger.info(f"global_step={global_step}, episodic_return={current_episode_return:.2f}")
                current_episode_return = 0.0
                next_obs_dict, _ = env.reset(seed=args.seed+update*2+step//60)

            next_obs = torch.Tensor(np.stack([next_obs_dict[f"manager_{dc_id}"] for dc_id in env.venv._dc_ids])).to(device)
            
        # After the loop, the GAE calculation will use the collected storages.
        # The logic for the GAE calculation itself was mostly correct, as it uses `next_done`
        # for the very last step and `dones_storage` for the intermediate steps.

        # --- 5b. GAE and Update Phase ---
        # =======================================================================
        # === PRE-UPDATE SANITY CHECK ===========================================
        # =======================================================================
        print("--- Running Pre-Update Sanity Checks ---")
        
        # Check observations collected from the environment
        if torch.isnan(obs_storage).any() or torch.isinf(obs_storage).any():
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! CRITICAL ERROR: NaN or Inf found in obs_storage      !!!")
            print("!!! The environment is producing invalid observations.   !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Find exactly where the NaN is
            nan_locs = torch.isnan(obs_storage).nonzero()
            print("Locations of NaNs (step, agent, feature_idx):")
            print(nan_locs[:5]) # Print first 5 locations
            sys.exit(1) # Stop execution
        else:
            print("  [OK] obs_storage contains valid numbers.")

        # Check actions sampled by the policy
        if actions_storage.min() < 0 or actions_storage.max() >= action_dim:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! CRITICAL ERROR: Invalid action index found.        !!!")
            print(f"!!! Min action: {actions_storage.min()}, Max action: {actions_storage.max()} !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            sys.exit(1)
        else:
            print("  [OK] actions_storage contains valid indices.")
            
        # Check for NaNs in rewards, logprobs, values
        if torch.isnan(rewards_storage).any(): raise RuntimeError("NaNs in rewards_storage")
        if torch.isnan(logprobs_storage).any(): raise RuntimeError("NaNs in logprobs_storage")
        if torch.isnan(values_storage).any(): raise RuntimeError("NaNs in values_storage")
        print("  [OK] rewards, logprobs, and values are valid.")
        # =======================================================================
    
        # --- 5b. GAE and Update Phase ---
        with torch.no_grad():
            global_next_obs = next_obs.reshape(1, -1)
            next_value = critic(global_next_obs).squeeze()
            
            advantages = torch.zeros_like(rewards_storage).to(device)
            last_gae_lambda = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    next_values_for_delta = next_value.expand(num_agents) # FIX: Expand to match shape
                else:
                    nextnonterminal = 1.0 - dones_storage[t + 1].float()
                    next_values_for_delta = values_storage[t + 1]
                
                # The key is that `nextnonterminal` correctly reflects if V(s_{t+1}) is 0.
                delta = rewards_storage[t] + algo_cfg["gamma"] * next_values_for_delta * nextnonterminal - values_storage[t]
                
                # The GAE recursion ALSO needs to be reset on a terminal state.
                advantages[t] = last_gae_lambda = delta + algo_cfg["gamma"] * algo_cfg["gae_lambda"] * nextnonterminal * last_gae_lambda
            returns = advantages + values_storage

        # Flatten the batch for training
        b_obs = obs_storage.reshape(-1, obs_dim)
        b_actions = actions_storage.reshape(-1)
        b_logprobs = logprobs_storage.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        # === CTDE CHANGE: Create the global state batch for the critic ===
        # obs_storage has shape (num_steps, num_agents, obs_dim)
        # We need to reshape it to (num_steps * num_agents, num_agents * obs_dim)
        # This is tricky. A simpler way is to reshape it inside the minibatch loop.
        # Let's reshape obs_storage to (num_steps, num_agents * obs_dim) first.
        global_obs_storage = obs_storage.permute(0, 1, 2).reshape(num_steps, -1)
        # Then flatten for batching
        # === END OF CHANGE ===
        
        # === START OF DEBUGGING ASSERTIONS ===
        assert not torch.isnan(b_obs).any(), "NaNs found in observations batch!"
        assert not torch.isinf(b_obs).any(), "Infs found in observations batch!"

        assert not torch.isnan(b_actions).any(), "NaNs found in actions batch!"
        # Check if actions are within the valid range [0, 1, 2]
        assert b_actions.min() >= 0, f"Action minimum is out of bounds: {b_actions.min()}"
        assert b_actions.max() < action_dim, f"Action maximum is out of bounds: {b_actions.max()}"

        assert not torch.isnan(b_logprobs).any(), "NaNs found in logprobs batch!"
        assert not torch.isnan(b_advantages).any(), "NaNs found in advantages batch!"
        assert not torch.isnan(b_returns).any(), "NaNs found in returns batch!"

        print("Data integrity check passed before PPO update.")
        # === END OF DEBUGGING ASSERTIONS ===

        actor.train() # Set to train mode for update
        critic.train()
        
        # --- PPO Update Epochs ---
        for epoch in range(algo_cfg["update_epochs"]):
            b_inds = np.arange(num_steps * num_agents)
            np.random.shuffle(b_inds)
            
            # Log the errors
            total_loss = 0.0
            pg_loss = 0.0
            v_loss = 0.0
            entropy_loss = 0.0
            
            for start in range(0, num_steps * num_agents, algo_cfg["minibatch_size"]):
                end = start + algo_cfg["minibatch_size"]
                mb_inds = b_inds[start:end]

                # --- Calculate PPO Losses for the minibatch ---
                new_logits = actor(b_obs[mb_inds])
                new_dist = torch.distributions.Categorical(logits=new_logits)
                new_logprob = new_dist.log_prob(b_actions[mb_inds])
                entropy = new_dist.entropy()

                # --- Critic uses the global state ---
                # 1. Find which timesteps (from 0 to 2047) these transitions belong to.
                mb_timestep_inds = (mb_inds // num_agents)
                
                # 2. Gather the corresponding global states.
                mb_global_obs = global_obs_storage[mb_timestep_inds]
                
                new_value = critic(mb_global_obs).squeeze()

                # Policy Loss (Clipped Surrogate Objective)
                logratio = new_logprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages for stability
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

                # --- Optimization Step ---
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), algo_cfg["max_grad_norm"])
                optimizer.step()
            
            # Average losses over the minibatches
            total_loss /= (num_steps * num_agents // algo_cfg["minibatch_size"])
            pg_loss /= (num_steps * num_agents // algo_cfg["minibatch_size"])
            v_loss /= (num_steps * num_agents // algo_cfg["minibatch_size"])
            entropy_loss /= (num_steps * num_agents // algo_cfg["minibatch_size"])
            
            print(f"Update {update}/{num_updates}, Epoch {epoch+1}/{algo_cfg['update_epochs']}: "
                  f"Total Loss: {total_loss:.4f}, Policy Loss: {pg_loss:.4f}, "
                  f"Value Loss: {v_loss:.4f}, Entropy Loss: {entropy_loss:.4f}")

        # --- Logging after each update ---
        writer.add_scalar("Loss/Total", total_loss, global_step)
        writer.add_scalar("Loss/Policy", pg_loss.item(), global_step)
        writer.add_scalar("Loss/Value", v_loss.item(), global_step)
        writer.add_scalar("Loss/Entropy", entropy_loss.item(), global_step)

        # === START OF MODIFIED EVALUATION LOOP ===
        if update % algo_cfg.get("eval_frequency", 10) == 0:
            if logger: logger.info(f"--- Running evaluation at update {update}/{num_updates} (global step {global_step}) ---")
            actor.eval()
            eval_env.obs_rms = env.obs_rms
            
            total_eval_reward = 0.0
            for ep in range(algo_cfg.get("eval_episodes", 5)):
                # Use a consistent set of seeds for each evaluation run for fair comparison
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

                    eval_next_obs_dict, eval_rewards_dict, eval_dones_dict, eval_trunc_dict, _ = eval_env.step(actions_dict)

                    eval_global_reward = np.mean([eval_rewards_dict.get(f"manager_{dc_id}", 0.0) for dc_id in eval_env.venv._dc_ids])
                    eval_ep_return += eval_global_reward

                    done = eval_dones_dict["__all__"] or eval_trunc_dict["__all__"]
                    eval_obs_dict = eval_next_obs_dict

                total_eval_reward += eval_ep_return
                if logger: logger.info(f"  Eval Episode {ep+1} Return: {eval_ep_return:.2f}")

            avg_eval_return = total_eval_reward / algo_cfg.get("eval_episodes", 5)
            writer.add_scalar("Eval/AverageReturn", avg_eval_return, global_step)
            if logger: logger.info(f"  Average Evaluation Return: {avg_eval_return:.2f}")

            # === NEW: Logic to save the best model ===
            if avg_eval_return > best_eval_return:
                best_eval_return = avg_eval_return
                if logger: logger.info("  ** New best evaluation return! Saving model... **")
                
                best_model_path = os.path.join(ckpt_dir, "best_model.pth")
                save_simple_checkpoint(
                    actor_state_dict=actor.state_dict(),
                    critic_state_dict=critic.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    norm_obs_rms=env.obs_rms, # <-- Save the learned stats
                    save_path=best_model_path
                )
            # === END OF NEW LOGIC ===
        # === END OF MODIFIED EVALUATION LOOP ===

    env.close()
    eval_env.close()
    writer.close()

if __name__ == "__main__":
    train()