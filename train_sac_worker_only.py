import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import logging
import os
from collections import deque
import argparse
import datetime
import pandas as pd

from envs.sustaincluster_ma_env import SustainClusterMAEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
# Import only Worker networks
from rl_components.agent_net_ma import WorkerActor, WorkerCritic
# Import only Worker Replay Buffer
from rl_components.replay_buffer_ma import WorkerReplayBuffer
from rewards.predefined.composite_reward import CompositeReward
from utils.marl_utils import D_META_MANAGER, D_META_WORKER
from utils.checkpoint_manager_ma import save_checkpointMA
from utils.config_loader import load_yaml
from utils.config_logger import setup_logger
from torch.utils.tensorboard import SummaryWriter

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
    # Adjusted description for clarity
    parser = argparse.ArgumentParser(description="Worker-Only SustainCluster Training")
    parser.add_argument("--sim-config", type=str, default="configs/env/sim_config_ma.yaml")
    parser.add_argument("--reward-config", type=str, default="configs/env/reward_config.yaml")
    parser.add_argument("--dc-config", type=str, default="configs/env/datacenters_ma.yaml")
    parser.add_argument("--algo-config", type=str, default="configs/env/algorithm_config_ma.yaml")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag for worker-only training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-logger", type=str2bool, default=True, help="Enable logger")
    return parser.parse_args()


def make_env(sim_cfg_path, dc_cfg_path, reward_cfg_path, writer=None, logger=None):
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
        logger=logger,
    )

    reward_fn = CompositeReward(components=reward_cfg["components"], normalize=False)

    return SustainClusterMAEnv(cluster_manager_ma=cluster, start_time=start, end_time=end,
        reward_fn=reward_fn, logger = logger)

def train():
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Modified run ID to reflect worker-only training
    run_id = f"WKR_ONLY_{args.tag}_{timestamp}" if args.tag else f"WKR_ONLY_{timestamp}"

    log_dir = f"logs/train_{run_id}"
    tb_dir = f"runs/train_{run_id}"
    ckpt_dir = f"checkpoints/train_{run_id}"
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)
    logger = setup_logger(log_dir, enable_logger=args.enable_logger)

    algo_cfg = load_yaml(args.algo_config)["algorithm"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_seed = args.seed
    eval_seed = args.seed + 1000

    if algo_cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(algo_cfg["device"])

    if logger: logger.info(f"Using device: {device}")
    if logger: logger.info("### --- Starting WORKER-ONLY training session --- ###")


    env = make_env(args.sim_config, args.dc_config, args.reward_config, writer, logger)
    eval_env =  make_env(args.sim_config, args.dc_config, args.reward_config, writer, logger)

    obs_dict, _ = env.reset(seed = train_seed)

    D_GLOBAL = 4
    D_LOCAL_WORKER = obs_dict[f"worker_{env._dc_ids[0]}"]["obs_local_dc_i_for_worker"].shape[0]

    # --- ### --- WORKER-ONLY CHANGE --- ### ---
    # --- Network Initialization (Worker Only) ---

    wrk_actor   = WorkerActor(D_META_WORKER, D_LOCAL_WORKER, D_GLOBAL).to(device)
    wrk_critic  = WorkerCritic(D_META_WORKER, D_LOCAL_WORKER, D_GLOBAL).to(device)
    wrk_target_critic = WorkerCritic(D_META_WORKER, D_LOCAL_WORKER, D_GLOBAL).to(device)
    wrk_target_critic.load_state_dict(wrk_critic.state_dict())

    wrk_actor_opt = torch.optim.Adam(wrk_actor.parameters(), lr=float(algo_cfg["actor_learning_rate"]))
    wrk_critic_opt = torch.optim.Adam(wrk_critic.parameters(),lr=float(algo_cfg["critic_learning_rate"]))

    # --- Replay Buffer (Worker Only) ---

    wrk_buffer = WorkerReplayBuffer(capacity = algo_cfg["replay_buffer_size"],
                                   D_emb_meta_worker = D_META_WORKER,
                                   D_emb_local_worker = D_LOCAL_WORKER,
                                   D_global = D_GLOBAL)

   # --- Training Loop Variables ---
    stats = RunningStats()
    episode_reward, episode_steps = 0.0, 0
    best_eval_reward = float("-inf")
    episode_reward_buffer = deque(maxlen=10)
    best_avg_reward = float("-inf")
    pbar = trange(algo_cfg["total_steps"])
    # Only need worker losses
    critic_loss_w, actor_loss_w = None, None

    for global_step in pbar:

        # ### --- WORKER-ONLY CHANGE --- ###
        # --- Hardcoded Manager Action: Always Commit Locally (action=0) ---
        mgr_act = {dc: 0 for i, dc in enumerate(env._dc_ids)}
        obs_after_mgr = env.manager_step(mgr_act)

        # Get observations for the worker
        meta_w, local_w, glob_w = [], [], []
        for dc in env._dc_ids:
            o_wrk = obs_after_mgr[f"worker_{dc}"]
            meta_w.append(o_wrk["obs_worker_meta_task_i"])
            local_w.append(o_wrk["obs_local_dc_i_for_worker"])
            glob_w.append(o_wrk["global_context"])

        # Create tensors for worker policy
        meta_w  = torch.from_numpy(np.asarray(meta_w,  dtype=np.float32)).to(device)
        local_w = torch.from_numpy(np.asarray(local_w, dtype=np.float32)).to(device)
        glob_w  = torch.from_numpy(np.asarray(glob_w,  dtype=np.float32)).to(device)

        with torch.no_grad():
            act_w, _, _ = wrk_actor.sample_action(meta_w, local_w, glob_w)

        wrk_act = {dc: int(act_w[i].item()) for i, dc in enumerate(env._dc_ids)}
        env.worker_step(wrk_act)

        next_obs, rewards, dones, truncated, infos = env.env_step()
        done_flag = dones["__all__"] or truncated["__all__"]
        global_reward = rewards[next(iter(rewards))]
        stats.update(global_reward)
        norm_reward = stats.normalize(global_reward)

        # === store transition (Worker Only) ===
        for i, dc in enumerate(env._dc_ids):
            wrk_buffer.add(
                meta_w[i].cpu().numpy(), local_w[i].cpu().numpy(), glob_w[i].cpu().numpy(),
                int(act_w[i].item()),         norm_reward, done_flag,
                next_obs[f"worker_{dc}"]["obs_worker_meta_task_i"],
                next_obs[f"worker_{dc}"]["obs_local_dc_i_for_worker"],
                next_obs[f"worker_{dc}"]["global_context"],
            )
        obs_dict = next_obs
        episode_reward += global_reward
        episode_steps += 1

        if done_flag:
            avg_ep_reward = episode_reward/episode_steps if episode_steps > 0 else 0.0
            episode_reward_buffer.append(avg_ep_reward)
            writer.add_scalar("Reward/Episode", avg_ep_reward, global_step)
            if logger: logger.info(f"[Episode End] Step: {global_step}, Reward: {episode_reward:.2f}, Avg Reward: {avg_ep_reward:.2f}")
            pbar.write(f"Ep. Reward: {avg_ep_reward:.2f} (steps: {episode_steps})")
            obs_dict, _ = env.reset(seed=args.seed + global_step // 1000)
            episode_reward = 0
            episode_steps = 0

            if len(episode_reward_buffer) == 10:
                avg10_reward = np.mean(episode_reward_buffer)
                writer.add_scalar("Reward/Avg10", avg10_reward, global_step)
                pbar.write(f"Avg10 Reward: {avg10_reward:.2f}")
                if avg10_reward > best_avg_reward:
                    best_avg_reward = avg10_reward
                    # ### --- WORKER-ONLY CHANGE --- ###
                    # Only save worker models
                    models = {
                        "wrk_actor": wrk_actor,
                        "wrk_critic": wrk_critic,
                    }
                    optimizers = {
                        "wrk_actor_opt": wrk_actor_opt,
                        "wrk_critic_opt": wrk_critic_opt,
                    }
                    save_checkpointMA(
                        global_step,models=models,optimizers=optimizers,save_dir=ckpt_dir,is_best=True
                    )

        # ### --- WORKER-ONLY CHANGE --- ###
        # --- SAC Update (Worker Only) ---
        if global_step >= algo_cfg["warmup_steps"] and global_step % algo_cfg["update_frequency"] == 0:
            if len(wrk_buffer) >= algo_cfg["batch_size"]:
                (meta_wb, local_wb, glob_wb, act_wb, rew_wb, done_wb,
                next_meta_wb, next_local_wb, next_glob_wb) = \
                wrk_buffer.sample(algo_cfg["batch_size"])

                meta_wb      = meta_wb.to(device)
                local_wb     = local_wb.to(device)
                glob_wb      = glob_wb.to(device)
                act_wb       = act_wb.long().to(device)
                rew_wb       = rew_wb.to(device)
                done_wb      = done_wb.to(device)
                next_meta_wb = next_meta_wb.to(device)
                next_local_wb= next_local_wb.to(device)
                next_glob_wb = next_glob_wb.to(device)

                with torch.no_grad():
                    next_logits_w = wrk_actor(next_meta_wb, next_local_wb, next_glob_wb)
                    next_probs_w  = F.softmax(next_logits_w, dim=-1)
                    next_logp_w   = F.log_softmax(next_logits_w, dim=-1)
                    q1_t_w, q2_t_w= wrk_target_critic.forward_q_values(next_meta_wb, next_local_wb, next_glob_wb)
                    q_t_min_w     = torch.min(q1_t_w, q2_t_w)
                    v_next_w      = (next_probs_w * (q_t_min_w - algo_cfg["alpha"] * next_logp_w)).sum(dim=-1, keepdim=True)
                    q_target_w    = rew_wb + algo_cfg["gamma"] * (1 - done_wb) * v_next_w
                    q_target_w = q_target_w.squeeze(1)

                q1_pred_w, q2_pred_w = wrk_critic.q_for_action(meta_wb, local_wb, glob_wb, act_wb)
                critic_loss_w = 0.5 * (F.mse_loss(q1_pred_w, q_target_w) +
                F.mse_loss(q2_pred_w, q_target_w))

                wrk_critic_opt.zero_grad()
                critic_loss_w.backward()
                wrk_critic_opt.step()

                if global_step % algo_cfg["policy_update_frequency"] == 0:
                    logits_w = wrk_actor(meta_wb, local_wb, glob_wb)
                    probs_w  = F.softmax(logits_w, dim=-1)
                    logp_w   = F.log_softmax(logits_w, dim=-1)
                    q1_a_w, q2_a_w = wrk_critic.forward_q_values(meta_wb, local_wb, glob_wb)
                    q_min_w  = torch.min(q1_a_w, q2_a_w)
                    actor_loss_w = (probs_w * (algo_cfg["alpha"] * logp_w - q_min_w.detach())).sum(dim=-1).mean()

                    wrk_actor_opt.zero_grad()
                    actor_loss_w.backward()
                    wrk_actor_opt.step()

                    for p, tp in zip(wrk_critic.parameters(), wrk_target_critic.parameters()):
                        tp.data.mul_(1 - algo_cfg["tau"]).add_(algo_cfg["tau"] * p.data)
        
        if global_step % algo_cfg["log_interval"] == 0:
            if critic_loss_w is not None and actor_loss_w is not None:
                writer.add_scalar("Worker/Loss_Q", critic_loss_w.item(), global_step)
                writer.add_scalar("Worker/Loss_Policy", actor_loss_w.item(), global_step)

            if logger and all(v is not None for v in (critic_loss_w, actor_loss_w)):
                logger.info(
                    f"[{global_step}] "
                    f"WkrQ={critic_loss_w:.4f}  "
                    f"WkrP={actor_loss_w:.4f}"
                )
                pbar.set_description(
                    f"W_Q {critic_loss_w:.3f}  "
                    f"W_P {actor_loss_w:.3f}"
                )
        if global_step > 0 and global_step % algo_cfg["save_interval"] == 0:
            # ### --- WORKER-ONLY CHANGE --- ###
            models = {
                "wrk_actor": wrk_actor, "wrk_critic": wrk_critic
            }
            optimizers = {
                "wrk_actor_opt": wrk_actor_opt, "wrk_critic_opt": wrk_critic_opt
            }
            save_checkpointMA(global_step, models, optimizers, ckpt_dir, is_best=False)
            if logger:
                logger.info(f"[{global_step}] Periodic checkpoint saved.")
                pbar.write(f"saved checkpoint at step {global_step}")

        if global_step > 0 and global_step % algo_cfg["eval_frequency"] == 0:
            if logger:
                logger.info(f"Eval begins at step {global_step}")

            wrk_actor.eval()
            total_eval_reward = 0.0
            for ep in range(algo_cfg["eval_episodes"]):
                obs_dict, _ = eval_env.reset(seed=eval_seed + ep)
                ep_ret, done_flag = 0.0, False

                while not done_flag:
                    # ### --- WORKER-ONLY CHANGE --- ###
                    # --- Hardcoded Manager Action in Evaluation ---
                    mgr_act = {dc: 0 for i, dc in enumerate(eval_env._dc_ids)}
                    obs_after_mgr = eval_env.manager_step(mgr_act)

                    meta_w, local_w, glob_w = [], [], []
                    for dc in eval_env._dc_ids:
                        w = obs_after_mgr[f"worker_{dc}"]
                        meta_w.append(w["obs_worker_meta_task_i"])
                        local_w.append(w["obs_local_dc_i_for_worker"])
                        glob_w.append(w["global_context"])

                    meta_w  = torch.from_numpy(np.asarray(meta_w ,dtype=np.float32)).to(device)
                    local_w = torch.from_numpy(np.asarray(local_w,dtype=np.float32)).to(device)
                    glob_w  = torch.from_numpy(np.asarray(glob_w ,dtype=np.float32)).to(device)

                    with torch.no_grad():
                        # For evaluation, we can take the greedy action
                        logits_w = wrk_actor(meta_w, local_w, glob_w)
                        act_w = torch.argmax(logits_w, dim=-1)

                    wrk_act = {dc: act_w[i].item() for i, dc in enumerate(eval_env._dc_ids)}
                    eval_env.worker_step(wrk_act)

                    next_obs, rew_dict, dones_dict, trunc_dict, _ = eval_env.env_step()

                    ep_ret += rew_dict[next(iter(rew_dict))]
                    done_flag = dones_dict["__all__"] or trunc_dict["__all__"]
                    obs_dict = next_obs

                total_eval_reward += ep_ret
                if logger:
                    logger.info(f"Eval-Ep{ep+1}/{algo_cfg['eval_episodes']}  return={ep_ret:.2f}")

            avg_ret = total_eval_reward / algo_cfg["eval_episodes"]
            writer.add_scalar("Eval/AverageReturn", avg_ret, global_step)
            pbar.write(f"eval return = {avg_ret:.2f}")

            if avg_ret > best_eval_reward:
                best_eval_reward = avg_ret
                # ### --- WORKER-ONLY CHANGE --- ###
                models = {
                    "wrk_actor": wrk_actor, "wrk_critic": wrk_critic
                }
                optims = {
                    "wrk_actor_opt": wrk_actor_opt, "wrk_critic_opt": wrk_critic_opt
                }
                save_checkpointMA(global_step, models, optims,
                        ckpt_dir, filename="best_eval_ckpt.pth", is_best=True)
                pbar.write(f"New BEST model saved (avg return {avg_ret:.2f})")

            wrk_actor.train()

    writer.close()
    if logger: logger.info("Training finished.")

if __name__ == "__main__":
    train()