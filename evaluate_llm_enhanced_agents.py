# evaluate_llm_enhanced_agents_fixed.py - FIXED FOR BOTH MANAGER AND WORKER EVALUATION

import sys
import os
import copy
import datetime
import logging
import random
import asyncio
import argparse
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

sns.set_theme(style="whitegrid")

# Local imports
from envs.sustaincluster_ma_env_llm import LLMIntegratedSustainClusterMAEnv
from envs.sustaincluster_worker_multi_env import SustainClusterWorkerMultiEnv
from envs.sustaincluster_ma_env_llm import LLMIntegratedWorkerEnv
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rl_components.agent_net_simple import ManagerActorMLP, WorkerActorMLP
from utils.config_loader import load_yaml
from rewards.registry_utils import get_reward_function
from rewards.predefined.composite_reward import CompositeReward

def parse_args():
    parser = argparse.ArgumentParser(description="FIXED: LLM-Enhanced Agent Evaluation for Manager/Worker")
    parser.add_argument("--sim-config", type=str, default="configs/env/sim_config_ma.yaml")
    parser.add_argument("--reward-config", type=str, default="configs/env/reward_config_manager_ci_only.yaml")
    parser.add_argument("--dc-config", type=str, default="configs/env/datacenters_ma.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--agent-type", type=str, choices=["manager", "worker"], required=True, 
                       help="Type of agent to evaluate")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--duration-days", type=int, default=7)
    
    # LLM evaluation options
    parser.add_argument("--enable-llm", type=bool, default=True, help="Enable LLM advice")
    parser.add_argument("--llm-service-url", type=str, default="http://10.93.232.106:8000")
    parser.add_argument("--llm-timeout", type=float, default=2.0)
    parser.add_argument("--llm-max-concurrent", type=int, default=16)
    parser.add_argument("--llm-history-window", type=int, default=10)
    
    # Comparison options
    parser.add_argument("--compare-modes", type=bool, default=True, help="Compare LLM vs non-LLM")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    
    return parser.parse_args()

def make_manager_eval_env(base_sim_cfg, base_dc_cfg, base_reward_cfg, duration_days, seed, 
                         enable_llm=True, llm_service_url="http://10.93.232.106:8000",
                         llm_timeout=2.0, llm_max_concurrent=16, llm_history_window=10,
                         simple_obs_mode=True, logger=None):
    """Creates MANAGER evaluation environment with optional LLM enhancement."""
    sim_cfg = copy.deepcopy(base_sim_cfg)
    dc_cfg = copy.deepcopy(base_dc_cfg)
    reward_cfg = copy.deepcopy(base_reward_cfg)

    sim_cfg["simulation"]["duration_days"] = duration_days

    # Build reward function
    if "components" in reward_cfg["reward"]:
        reward_fn = CompositeReward(components=reward_cfg["reward"]["components"], normalize=False)
    elif "name" in reward_cfg["reward"]:
        reward_name = reward_cfg["reward"]["name"]
        reward_args = reward_cfg["reward"].get("args", {})
        reward_fn = get_reward_function(reward_name, **reward_args)
    else:
        raise ValueError("Invalid reward configuration.")
    
    for cfg in dc_cfg["datacenters"]:
        cfg.setdefault("simulation_year", sim_cfg["simulation"]["year"])

    cluster = DatacenterClusterManagerMA(
        config_list=dc_cfg["datacenters"],
        simulation_year=sim_cfg["simulation"]["year"],
        tasks_file_path=sim_cfg["simulation"]["workload_path"],
        cloud_provider=sim_cfg["simulation"]["cloud_provider"],
        max_total_options=sim_cfg["simulation"]["max_total_options"],
        duration_days=sim_cfg["simulation"]["duration_days"],
        logger=logger,
    )

    start = pd.Timestamp(datetime.datetime(sim_cfg["simulation"]["year"], sim_cfg["simulation"]["month"], 
                                         sim_cfg["simulation"]["init_day"], sim_cfg["simulation"]["init_hour"], 
                                         0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=duration_days)

    env = LLMIntegratedSustainClusterMAEnv(
        cluster_manager_ma=cluster, 
        start_time=start, 
        end_time=end,
        reward_fn=reward_fn, 
        logger=logger, 
        simple_obs_mode=simple_obs_mode, 
        use_meta_tasks=True,
        training_mode="manager",  # Only manager agents
        # LLM parameters
        enable_llm_advice=enable_llm,
        llm_service_url=llm_service_url,
        llm_timeout=llm_timeout,
        llm_max_concurrent=llm_max_concurrent,
        llm_history_window=llm_history_window
    )
    return env

def make_worker_eval_env(base_sim_cfg, base_dc_cfg, base_reward_cfg, duration_days, seed, 
                        enable_llm=True, llm_service_url="http://10.93.232.106:8000",
                        llm_timeout=2.0, llm_max_concurrent=16, llm_history_window=10,
                        logger=None):
    """Creates WORKER evaluation environment with optional LLM enhancement."""
    sim_cfg = copy.deepcopy(base_sim_cfg)
    dc_cfg = copy.deepcopy(base_dc_cfg)
    reward_cfg = copy.deepcopy(base_reward_cfg)

    sim_cfg["simulation"]["duration_days"] = duration_days

    # Build reward function
    if "components" in reward_cfg["reward"]:
        reward_fn = CompositeReward(components=reward_cfg["reward"]["components"], normalize=False)
    elif "name" in reward_cfg["reward"]:
        reward_name = reward_cfg["reward"]["name"]
        reward_args = reward_cfg["reward"].get("args", {})
        reward_fn = get_reward_function(reward_name, **reward_args)
    else:
        raise ValueError("Invalid reward configuration.")
    
    for cfg in dc_cfg["datacenters"]:
        cfg.setdefault("simulation_year", sim_cfg["simulation"]["year"])

    cluster = DatacenterClusterManagerMA(
        config_list=dc_cfg["datacenters"],
        simulation_year=sim_cfg["simulation"]["year"],
        tasks_file_path=sim_cfg["simulation"]["workload_path"],
        cloud_provider=sim_cfg["simulation"]["cloud_provider"],
        max_total_options=sim_cfg["simulation"]["max_total_options"],
        duration_days=sim_cfg["simulation"]["duration_days"],
        logger=logger,
    )

    start = pd.Timestamp(datetime.datetime(sim_cfg["simulation"]["year"], sim_cfg["simulation"]["month"], 
                                         sim_cfg["simulation"]["init_day"], sim_cfg["simulation"]["init_hour"], 
                                         0, tzinfo=datetime.timezone.utc))
    end = start + datetime.timedelta(days=duration_days)

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
        llm_service_url=llm_service_url,
        llm_timeout=llm_timeout,
        llm_max_concurrent=llm_max_concurrent,
        llm_history_window=llm_history_window
    )
    
    return env

async def evaluate_manager_agent(env, actor, device, seed, duration_days, mode_name, logger):
    """Evaluate MANAGER agent in given environment mode."""
    logger.info(f"--- Evaluating MANAGER {mode_name} mode ---")
    
    # Reset environment
    obs_dict, _ = env.reset(seed=seed)
    
    num_steps = duration_days * 24 * 4
    all_step_infos = []
    total_reward = 0.0
    llm_stats = {"requests": 0, "successes": 0, "avg_confidence": 0.0}

    for step in tqdm(range(num_steps), desc=f"Simulating Manager {mode_name}"):
        # Prepare MANAGER observations
        obs_list = []
        for dc_id in env._dc_ids:
            manager_key = f"manager_{dc_id}"
            if manager_key in obs_dict:
                obs_list.append(obs_dict[manager_key])
            else:
                logger.error(f"Missing manager observation for {manager_key}")
                break
        
        if len(obs_list) != len(env._dc_ids):
            logger.error("Incomplete manager observations, ending evaluation")
            break
            
        obs_tensor = torch.from_numpy(np.stack(obs_list)).float().to(device)

        # Get MANAGER actions
        with torch.no_grad():
            logits = actor(obs_tensor)
            actions = torch.argmax(logits, dim=1)

        # Assemble action dictionary for MANAGER
        actions_dict = {f"manager_{dc_id}": actions[i].item() for i, dc_id in enumerate(env._dc_ids)}
        # Set worker actions to default (they're not being evaluated)
        actions_dict.update({f"worker_{dc_id}": 1 for dc_id in env._dc_ids})

        # Step environment
        next_obs, rew_dict, dones_dict, trunc_dict, info_dict = env.step(actions_dict)
        
        # Calculate reward
        if isinstance(rew_dict, dict):
            manager_rewards = [rew_dict.get(f"manager_{dc_id}", 0.0) for dc_id in env._dc_ids]
            step_reward = np.mean(manager_rewards)
        else:
            step_reward = float(rew_dict)
        total_reward += step_reward
        
        # Store step info
        all_step_infos.append(info_dict)
        
        # Track LLM statistics
        if "__global__" in info_dict and "llm_stats" in info_dict["__global__"]:
            llm_info = info_dict["__global__"]["llm_stats"]
            llm_stats["requests"] = llm_info.get("request_count", 0)
            llm_stats["successes"] = llm_info.get("success_count", 0)
        
        obs_dict = next_obs
        
        if dones_dict["__all__"] or trunc_dict["__all__"]:
            logger.info(f"Manager simulation ended early at step {step+1}")
            break

    # Calculate final statistics
    avg_reward = total_reward / len(all_step_infos) if all_step_infos else 0.0
    if llm_stats["requests"] > 0:
        llm_stats["success_rate"] = llm_stats["successes"] / llm_stats["requests"]
    else:
        llm_stats["success_rate"] = 0.0

    logger.info(f"Manager {mode_name} Results:")
    logger.info(f"  Total Steps: {len(all_step_infos)}")
    logger.info(f"  Average Reward: {avg_reward:.4f}")
    logger.info(f"  Total Reward: {total_reward:.4f}")
    if llm_stats["requests"] > 0:
        logger.info(f"  LLM Success Rate: {llm_stats['success_rate']:.2%}")

    return {
        "mode": mode_name,
        "agent_type": "manager",
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "num_steps": len(all_step_infos),
        "step_infos": all_step_infos,
        "llm_stats": llm_stats
    }

async def evaluate_worker_agent(env, actor, device, seed, duration_days, mode_name, logger):
    """Evaluate WORKER agent in given environment mode."""
    logger.info(f"--- Evaluating WORKER {mode_name} mode ---")
    
    # Reset environment
    obs, info = env.reset(seed=seed)
    
    num_steps = duration_days * 24 * 4
    all_step_infos = []
    total_reward = 0.0
    llm_stats = {"requests": 0, "successes": 0, "avg_confidence": 0.0}

    for step in tqdm(range(num_steps), desc=f"Simulating Worker {mode_name}"):
        # obs is already a matrix for workers (num_dcs, obs_dim)
        obs_tensor = torch.from_numpy(obs).float().to(device)

        # Get WORKER actions
        with torch.no_grad():
            logits = actor(obs_tensor)
            actions = torch.argmax(logits, dim=1)

        # Convert to numpy array for the environment
        action_array = actions.cpu().numpy()

        # Step environment
        next_obs, reward, terminated, truncated, info_dict = env.step(action_array)
        
        # Calculate reward
        step_reward = float(reward)
        total_reward += step_reward
        
        # Store step info
        all_step_infos.append(info_dict)
        
        # Track LLM statistics
        if "llm_stats" in info_dict:
            llm_info = info_dict["llm_stats"]
            llm_stats["requests"] = llm_info.get("request_count", 0)
            llm_stats["successes"] = llm_info.get("success_count", 0)
        
        obs = next_obs
        
        if terminated or truncated:
            logger.info(f"Worker simulation ended early at step {step+1}")
            break

    # Calculate final statistics
    avg_reward = total_reward / len(all_step_infos) if all_step_infos else 0.0
    if llm_stats["requests"] > 0:
        llm_stats["success_rate"] = llm_stats["successes"] / llm_stats["requests"]
    else:
        llm_stats["success_rate"] = 0.0

    logger.info(f"Worker {mode_name} Results:")
    logger.info(f"  Total Steps: {len(all_step_infos)}")
    logger.info(f"  Average Reward: {avg_reward:.4f}")
    logger.info(f"  Total Reward: {total_reward:.4f}")
    if llm_stats["requests"] > 0:
        logger.info(f"  LLM Success Rate: {llm_stats['success_rate']:.2%}")

    return {
        "mode": mode_name,
        "agent_type": "worker",
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "num_steps": len(all_step_infos),
        "step_infos": all_step_infos,
        "llm_stats": llm_stats
    }

def extract_detailed_metrics(step_infos, agent_type):
    """Extract detailed metrics from step information based on agent type."""
    flat_records = []
    
    if agent_type == "manager":
        # For manager environments, extract from multi-agent structure
        for t, step_info in enumerate(step_infos):
            raw_results = step_info.get("__global__", {}).get("raw_results", {})
            dc_infos = raw_results.get("datacenter_infos", {})
            
            for dc_id, dc_info_step in dc_infos.items():
                common = dc_info_step.get("__common__", {})
                sla_info = common.get("__sla__", {"met": 0, "violated": 0})
                
                record = {
                    "timestep": t,
                    "datacenter_id": dc_id,
                    "energy_cost_usd": common.get("energy_cost_USD", 0.0),
                    "energy_kwh": common.get("energy_consumption_kwh", 0.0),
                    "carbon_kg": common.get("carbon_emissions_kg", 0.0),
                    "price_usd_per_kwh": common.get("price_USD_kwh", 0.0),
                    "ci_g_per_kwh": common.get("ci", 0.0),
                    "external_temp_c": common.get("weather", 0.0),
                    "cpu_util_pct": common.get("cpu_util_percent", 0.0),
                    "gpu_util_pct": common.get("gpu_util_percent", 0.0),
                    "mem_util_pct": common.get("mem_util_percent", 0.0),
                    "running_tasks": common.get("running_tasks", 0),
                    "sla_met": sla_info.get("met", 0),
                    "sla_violated": sla_info.get("violated", 0),
                }
                flat_records.append(record)
    else:
        # For worker environments, extract from single-env structure
        for t, step_info in enumerate(step_infos):
            # Worker environments might have different structure
            record = {
                "timestep": t,
                "datacenter_id": 0,  # Aggregate for workers
                "total_reward": step_info.get("reward", 0.0),
                # Add more worker-specific metrics as available
            }
            flat_records.append(record)

    return pd.DataFrame(flat_records)

async def main():
    """FIXED: Main evaluation function supporting both manager and worker agents."""
    args = parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"logs/eval_{args.agent_type}_llm_comparison_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_path = os.path.join(args.output_dir, "evaluation.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"=== {args.agent_type.upper()} LLM-Enhanced Agent Evaluation ===")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Agent Type: {args.agent_type}")
    logger.info(f"Duration: {args.duration_days} days")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    # Load configurations
    base_sim_cfg_dict = load_yaml(args.sim_config)
    base_dc_cfg_dict = load_yaml(args.dc_config)
    base_reward_cfg_dict = load_yaml(args.reward_config)
    
    # Load model
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    device = torch.device("cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Create appropriate environment to determine dimensions
    if args.agent_type == "manager":
        base_env = make_manager_eval_env(
            base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict, 
            1, args.seed, enable_llm=args.enable_llm, simple_obs_mode=True, logger=logger
        )
        first_agent_id = f"manager_{base_env._dc_ids[0]}"
        obs_dim = base_env.observation_space(first_agent_id).shape[0]
        action_dim = base_env.action_space(first_agent_id).n
        base_env.close()
        
        # Create actor
        actor = ManagerActorMLP(obs_dim, action_dim, hidden_dim=16).to(device)
        
    else:  # worker
        base_env = make_worker_eval_env(
            base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict, 
            1, args.seed, enable_llm=args.enable_llm, logger=logger
        )
        obs_dim = base_env.observation_space.shape[1]  # Second dimension for workers
        action_dim = base_env.action_space.n
        base_env.close()
        
        # Create actor  
        actor = WorkerActorMLP(obs_dim, action_dim, hidden_dim=16).to(device)
    
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    logger.info(f"Loaded {args.agent_type} model: obs_dim={obs_dim}, action_dim={action_dim}")
    
    results = {}
    
    # Evaluate with LLM advice (if enabled)
    if args.enable_llm:
        logger.info(f"Creating LLM-enhanced {args.agent_type} environment...")
        
        if args.agent_type == "manager":
            llm_env = make_manager_eval_env(
                base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict,
                args.duration_days, args.seed, 
                enable_llm=True,
                llm_service_url=args.llm_service_url,
                llm_timeout=args.llm_timeout,
                llm_max_concurrent=args.llm_max_concurrent,
                llm_history_window=args.llm_history_window,
                simple_obs_mode=True, logger=logger
            )
            
            result_llm = await evaluate_manager_agent(
                llm_env, actor, device, args.seed, 
                args.duration_days, "LLM-Enhanced", logger
            )
        else:
            llm_env = make_worker_eval_env(
                base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict,
                args.duration_days, args.seed, 
                enable_llm=True,
                llm_service_url=args.llm_service_url,
                llm_timeout=args.llm_timeout,
                llm_max_concurrent=args.llm_max_concurrent,
                llm_history_window=args.llm_history_window,
                logger=logger
            )
            
            result_llm = await evaluate_worker_agent(
                llm_env, actor, device, args.seed, 
                args.duration_days, "LLM-Enhanced", logger
            )
        
        results["LLM-Enhanced"] = result_llm
        llm_env.close()
    
    # Evaluate without LLM advice (for comparison)
    if args.compare_modes or not args.enable_llm:
        logger.info(f"Creating standard {args.agent_type} environment (no LLM)...")
        
        if args.agent_type == "manager":
            standard_env = make_manager_eval_env(
                base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict,
                args.duration_days, args.seed, 
                enable_llm=False,  # Disable LLM
                simple_obs_mode=True, logger=logger
            )
            
            result_standard = await evaluate_manager_agent(
                standard_env, actor, device, args.seed, 
                args.duration_days, "Standard", logger
            )
        else:
            standard_env = make_worker_eval_env(
                base_sim_cfg_dict, base_dc_cfg_dict, base_reward_cfg_dict,
                args.duration_days, args.seed, 
                enable_llm=False,  # Disable LLM
                logger=logger
            )
            
            result_standard = await evaluate_worker_agent(
                standard_env, actor, device, args.seed, 
                args.duration_days, "Standard", logger
            )
        
        results["Standard"] = result_standard
        standard_env.close()
    
    # Print comparison summary
    print_summary_comparison(results)
    
    # Save detailed results
    for mode, result in results.items():
        df = extract_detailed_metrics(result["step_infos"], args.agent_type)
        csv_path = os.path.join(args.output_dir, f"detailed_results_{mode.lower().replace('-', '_')}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Detailed results saved: {csv_path}")
    
    # Save summary results
    summary_path = os.path.join(args.output_dir, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"=== {args.agent_type.upper()} LLM-Enhanced Agent Evaluation Summary ===\n\n")
        
        for mode, result in results.items():
            f.write(f"{mode} Mode Results:\n")
            f.write(f"  Agent Type: {result['agent_type']}\n")
            f.write(f"  Total Reward: {result['total_reward']:.4f}\n")
            f.write(f"  Average Reward per Step: {result['avg_reward']:.6f}\n")
            f.write(f"  Steps Completed: {result['num_steps']}\n")
            
            if result['llm_stats']['requests'] > 0:
                f.write(f"  LLM Success Rate: {result['llm_stats']['success_rate']:.2%}\n")
                f.write(f"  LLM Total Requests: {result['llm_stats']['requests']}\n")
            
            f.write("\n")
    
    logger.info(f"Evaluation summary saved: {summary_path}")
    logger.info(f"=== {args.agent_type.upper()} Evaluation Complete ===")

def print_summary_comparison(results_dict):
    """Print summary comparison table."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY COMPARISON")
    print("="*80)
    
    summary_data = []
    for mode, result in results_dict.items():
        summary = {
            "Mode": mode,
            "Agent Type": result['agent_type'],
            "Total Reward": f"{result['total_reward']:.2f}",
            "Avg Reward/Step": f"{result['avg_reward']:.4f}",
            "Steps Completed": result['num_steps']
        }
        
        if result['llm_stats']['requests'] > 0:
            summary["LLM Success Rate"] = f"{result['llm_stats']['success_rate']:.1%}"
            summary["LLM Requests"] = result['llm_stats']['requests']
        else:
            summary["LLM Success Rate"] = "N/A"
            summary["LLM Requests"] = 0
            
        summary_data.append(summary)
    
    # Print as table
    if summary_data:
        headers = list(summary_data[0].keys())
        
        # Print headers
        header_line = " | ".join(f"{h:>15}" for h in headers)
        print(header_line)
        print("-" * len(header_line))
        
        # Print data rows
        for row in summary_data:
            data_line = " | ".join(f"{str(row[h]):>15}" for h in headers)
            print(data_line)
    
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())