import argparse, datetime, os, sys, torch, numpy as np, pandas as pd
from pathlib import Path
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.sustaincluster_worker_simple_env import SustainClusterWorkerSimpleEnv
from envs.sustaincluster_worker_multi_env import SustainClusterWorkerMultiEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rewards.registry_utils import get_reward_function
from rewards.predefined.composite_reward import CompositeReward
from utils.config_loader import load_yaml
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

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
    p.add_argument("--n_envs",      type=int, default=8, help="Number of parallel environments to run.")
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

def make_env(sim_yaml: str, dc_yaml: str, rwd_yaml: str, is_eval=False, fixed_seed=None):
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
        deterministic_reset=is_eval,
        fixed_seed=fixed_seed if is_eval else None,
    )
   
    return Monitor(env)

def make_eval_envs(sim_yaml, dc_yaml, rwd_yaml, seeds):
    def make_env_fn(seed):
        return lambda: make_env(sim_yaml, dc_yaml, rwd_yaml, is_eval=True, fixed_seed=seed)
    
    env_fns = [make_env_fn(seed) for seed in seeds]
    eval_env = SubprocVecEnv(env_fns)  # or SubprocVecEnv for heavier envs
    return eval_env

# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #
def main():
    args = get_args()
    ppo_cfg = load_yaml(args.train_cfg)["ppo_algorithm"]
    
    # --- 1. Setup Paths and Directories ---
    run_name = f"train_worker_multi_SB3_{args.algo}_worker_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    tb_dir   = Path("runs") / run_name
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 2. Create the Vectorized Training Environment ---
    # The `make_vec_env` helper from SB3 is perfect for this. It handles
    # creating multiple environments, seeding them correctly, and wrapping
    # them in a DummyVecEnv or SubprocVecEnv (if you specify it).
    vec_env = make_vec_env(
        lambda: make_env(args.sim_cfg, args.dc_cfg, args.reward_cfg),
        n_envs=args.n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,  # Use SubprocVecEnv for parallelism
    )
    # ent_sched = linear_schedule(0.02)
    # algo_cls = {"PPO": PPO, "A2C": A2C, "DQN": DQN}[args.algo]
    
    # --- 3. Create the Evaluation Environment and Callbacks ---
    print("Creating evaluation environment...")
    eval_seeds = [142, 143, 144, 145, 146]  # 5 deterministic seeds
    eval_env = make_eval_envs(args.sim_cfg, args.dc_cfg, args.reward_cfg, eval_seeds)
    
    # The EvalCallback will run the agent on the `eval_env` periodically.
    # It automatically logs the results to TensorBoard under an "eval" tab
    # and saves the best model found so far.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_dir / "best_model"),
        log_path=str(ckpt_dir / "eval_logs"),
        # Evaluate every `eval_freq` steps. This is per-environment, so we divide by n_envs.
        eval_freq=max(ppo_cfg.get("eval_frequency", 10000) // args.n_envs, 1),
        n_eval_episodes=1,
        deterministic=True, # Use greedy actions for evaluation
        render=False
    )
    
    # (Optional but Recommended) The CheckpointCallback saves the model at regular intervals.
    # This is useful for resuming training or if the script crashes.
    # checkpoint_callback = CheckpointCallback(
    #     save_freq=max(ppo_cfg.get("save_frequency", 20000) // args.n_envs, 1),
    #     save_path=str(ckpt_dir),
    #     name_prefix=f"{args.algo}_ckpt"
    # )

    # Combine the callbacks into a list. They will be executed in order.
    # The EvalCallback should generally come first.
    callback_list = [eval_callback]

    # --- 4. Define and Train the Model ---
    # n_steps (int) – The number of steps to run for each environment per update 
    # (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel) 
    # NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization) 
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["num_steps"]// args.n_envs,  # Divide by n_envs for parallelism
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
        device="cpu", # Change to "cuda" if needed
    )
    
    total_timesteps = ppo_cfg["total_timesteps"]
    
    print(f"--- Starting Training for {total_timesteps} timesteps ---")
    # Pass the list of callbacks to the learn method
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=callback_list
    )
    
    # --- 5. Final Save ---
    # The callbacks already saved the best and periodic models,
    # but it's good practice to save the final model as well.
    model.save(ckpt_dir / f"{args.algo}_final_model")
    vec_env.close()
    print(f"\n--- Training Finished ---")
    print(f"Best model saved to: {ckpt_dir / 'best_model.zip'}")
    print(f"Final model saved to: {ckpt_dir / f'{args.algo}_final_model.zip'}")

if __name__ == "__main__":
    main()
