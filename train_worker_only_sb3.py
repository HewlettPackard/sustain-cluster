import argparse, datetime, os, sys, torch, numpy as np, pandas as pd
from pathlib import Path
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.sustaincluster_worker_simple_env import SustainClusterWorkerSimpleEnv
from envs.sustaincluster_worker_multi_env import SustainClusterWorkerMultiEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rewards.registry_utils import get_reward_function
from rewards.predefined.composite_reward import CompositeReward
from utils.config_loader import load_yaml

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo",        choices=["PPO", "A2C", "DQN"], default="PPO")
    p.add_argument("--sim_cfg",     default="configs/env/sim_config_ma.yaml")
    p.add_argument("--dc_cfg",      default="configs/env/datacenters_ma.yaml")
    p.add_argument("--reward_cfg",  default="configs/env/reward_config_worker_ci_only.yaml")
    p.add_argument("--train_cfg",   default="configs/env/ppo_algorithm_config.yaml")
    p.add_argument("--n_envs",      type=int, default=4)
    # p.add_argument("--total_steps", type=int, default=200000)
    # p.add_argument("--tensorboard", type=str2bool, default=True)
    return p.parse_args()

# --------------------------------------------------------------------- #
# Environment helpers                                                   #
# --------------------------------------------------------------------- #
def build_cluster(sim_cfg: dict, dc_cfg: list):
    return DatacenterClusterManagerMA(
        config_list       = dc_cfg,
        simulation_year   = sim_cfg["year"],
        tasks_file_path   = sim_cfg["workload_path"],
        cloud_provider    = sim_cfg["cloud_provider"],
        max_total_options = sim_cfg["max_total_options"],
        duration_days     = sim_cfg["duration_days"],
        logger            = None,
    )

def build_reward(rwd_cfg: dict):
    if "components" in rwd_cfg:
        return CompositeReward(
            components=rwd_cfg["components"],
            normalize=rwd_cfg.get("normalize", False),
            freeze_stats_after_steps=rwd_cfg.get("freeze_stats_after_steps"),
        )
    if "name" in rwd_cfg:
        return get_reward_function(rwd_cfg["name"], **rwd_cfg.get("args", {}))
    raise ValueError("Reward YAML must contain either 'components' or 'name'.")

def make_env(sim_yaml: str, dc_yaml: str, rwd_yaml: str):
    sim_cfg    = load_yaml(sim_yaml)["simulation"]
    dc_cfg     = load_yaml(dc_yaml)["datacenters"]
    reward_cfg = load_yaml(rwd_yaml)["reward"]

    for cfg in dc_cfg:
        cfg.setdefault("simulation_year", sim_cfg["year"])

    # Create aware timestamps
    start_dt = datetime.datetime(
        sim_cfg["year"], sim_cfg["month"], sim_cfg["init_day"],
        sim_cfg["init_hour"], 0, tzinfo=datetime.timezone.utc)
    start_ts = pd.Timestamp(start_dt)
    end_ts   = start_ts + pd.Timedelta(days=sim_cfg["duration_days"])

    # env = SustainClusterWorkerSimpleEnv(
    #     cluster_manager = build_cluster(sim_cfg, dc_cfg),
    #     start_time      = start_ts,
    #     end_time        = end_ts,
    #     reward_fn       = build_reward(reward_cfg),
    #     logger          = None,
    # )
    env = SustainClusterWorkerMultiEnv(
        cluster_manager = build_cluster(sim_cfg, dc_cfg),
        start_time      = start_ts,
        end_time        = end_ts,
        reward_fn       = build_reward(reward_cfg),
        logger          = None,
    )
   
    return Monitor(env)

# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #
def main():
    args = get_args()
    ppo_cfg = load_yaml(args.train_cfg)["ppo_algorithm"]
    run_name = f"train_worker_multi_SB3_{args.algo}_worker_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    tb_dir   = Path("runs") / run_name

    # Factory that *accepts* the rank argument SB3 passes but ignores it
    def env_fn(_rank: int = 0):
        return make_env(args.sim_cfg, args.dc_cfg, args.reward_cfg)

    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, seed=42)
    # ent_sched = linear_schedule(0.02)
    # algo_cls = {"PPO": PPO, "A2C": A2C, "DQN": DQN}[args.algo]
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["num_steps"],
        batch_size=ppo_cfg["minibatch_size"],
        n_epochs=ppo_cfg["update_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_coef"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        verbose=1,
        tensorboard_log=str(tb_dir),
        device="cpu",
    )
    total_timesteps = ppo_cfg["total_timesteps"]
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save(ckpt_dir / f"{args.algo}_model")
    vec_env.close()
    print(f" Training finished.  Model saved to {ckpt_dir}")

if __name__ == "__main__":
    main()
