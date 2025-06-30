# envs/sustaincluster_ma_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List

from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rewards.base_reward import BaseReward
from utils.marl_utils import D_META_MANAGER, D_META_WORKER

class SustainClusterMAEnv(gym.Env):
    """
    A multi-agent Gymnasium environment for the SustainCluster-MA benchmark.

    This environment wraps the DatacenterClusterManagerMA and provides a standardized
    API for MARL algorithms. It manages the interactions for N DTA_Managers
    and N DTA_Workers, each identified by a unique agent ID.

    - Observation Space: A dictionary mapping agent IDs to their individual observation dicts.
    - Action Space: A dictionary mapping agent IDs to their individual action spaces.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 cluster_manager_ma: DatacenterClusterManagerMA,
                 start_time: pd.Timestamp,
                 end_time: pd.Timestamp,
                 reward_fn: BaseReward,
                 logger: Optional[Any] = None):
        """
        Initializes the multi-agent environment.

        Args:
            cluster_manager_ma (DatacenterClusterManagerMA): The simulation backend.
            start_time (pd.Timestamp): The starting time for simulation episodes.
            end_time (pd.Timestamp): The ending time for simulation episodes.
            reward_fn (BaseReward): A MARL-aware reward function that returns a global
                                    or per-agent reward.
            logger (Optional[Any]): A logger instance.
        """
        super().__init__()
        self.cluster_manager_ma = cluster_manager_ma
        self.logger = logger
        self.start_time = start_time
        self.end_time = end_time
        self.current_time = self.start_time
        self.time_step = pd.Timedelta(minutes=15)
        self.reward_fn = reward_fn

        self.num_dcs = self.cluster_manager_ma.num_dcs
        self._dc_ids = sorted(self.cluster_manager_ma.nodes.keys())

        # --- Define Agent IDs ---
        self.manager_agent_ids = [f"manager_{dc_id}" for dc_id in self._dc_ids]
        self.worker_agent_ids = [f"worker_{dc_id}" for dc_id in self._dc_ids]
        self.possible_agents = self.manager_agent_ids + self.worker_agent_ids
        self.agents = self.possible_agents.copy() # For tracking active agents

        # --- Define Observation and Action Spaces for a single agent of each type ---
        max_options = self.cluster_manager_ma.max_total_options
        d_option_feat = self.cluster_manager_ma.D_OPTION_FEAT

        # DTA_Manager Observation Space
        self._manager_obs_space = spaces.Dict({
            "obs_manager_meta_task_i": spaces.Box(low=-np.inf, high=np.inf, shape=(D_META_MANAGER,), dtype=np.float32),
            "obs_all_options_set_padded": spaces.Box(low=-np.inf, high=np.inf, shape=(max_options, d_option_feat), dtype=np.float32),
            "all_options_padding_mask": spaces.Box(low=0, high=1, shape=(max_options,), dtype=np.bool_),
            "global_context": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        })
        
        # DTA_Worker Observation Space
        # NOTE: D_LOCAL_WORKER_STATE must match the implementation in DatacenterNodeMA.prepare_worker_observation
        D_LOCAL_WORKER_STATE = 5 # Example: cpu_avail, gpu_avail, mem_avail, price, ci
        self._worker_obs_space = spaces.Dict({
            "obs_worker_meta_task_i": spaces.Box(low=-np.inf, high=np.inf, shape=(D_META_WORKER,), dtype=np.float32),
            "obs_local_dc_i_for_worker": spaces.Box(low=-np.inf, high=np.inf, shape=(D_LOCAL_WORKER_STATE,), dtype=np.float32),
            "global_context": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        })

        # DTA_Manager Action Space (selects one of the options)
        self._manager_action_space = spaces.Discrete(max_options)
        
        # DTA_Worker Action Space (0=Execute, 1=Defer)
        self._worker_action_space = spaces.Discrete(2)
        # For multi-agent compatibility, we can use these helper functions
        # Some MARL libraries (like PettingZoo) require these.
    def observation_space(self, agent: str) -> spaces.Space:
        agent_type = agent.split('_')[0]
        if agent_type == "manager":
            return self._manager_obs_space
        elif agent_type == "worker":
            return self._worker_obs_space
        else:
            raise ValueError(f"Unknown agent type for agent '{agent}'")

    def action_space(self, agent: str) -> spaces.Space:
        agent_type = agent.split('_')[0]
        if agent_type == "manager":
            return self._manager_action_space
        elif agent_type == "worker":
            return self._worker_action_space
        else:
            raise ValueError(f"Unknown agent type for agent '{agent}'")
    """
    def _get_observations(self) -> Dict[str, Any]:
        # Gathers observations for all active agents from the cluster manager.
        # Shared global context (time features)
        day_of_year = self.current_time.dayofyear
        hour_of_day = self.current_time.hour + self.current_time.minute / 60.0
        time_features = np.array([
            np.sin(2 * np.pi * day_of_year / 365.0), np.cos(2 * np.pi * day_of_year / 365.0),
            np.sin(2 * np.pi * hour_of_day / 24.0), np.cos(2 * np.pi * hour_of_day / 24.0)
        ], dtype=np.float32)

        # Get raw structured observations from manager
        manager_obs_raw = self.cluster_manager_ma._prepare_all_manager_observations(self.current_time)
        
        # Prepare final observation dict for all agents
        obs_dict = {}
        for dc_id, node in self.cluster_manager_ma.nodes.items():
            # Manager observation
            manager_id = f"manager_{dc_id}"
            obs_dict[manager_id] = manager_obs_raw[dc_id]
            obs_dict[manager_id]['global_context'] = time_features

            # Worker observation
            worker_id = f"worker_{dc_id}"
            worker_obs_parts = node.prepare_worker_observation(self.current_time)
            worker_obs_parts['global_context'] = time_features
            obs_dict[worker_id] = worker_obs_parts
        
        return obs_dict
    """
        
    def _get_time_features(self):
        day_of_year = self.current_time.dayofyear
        hour_of_day = self.current_time.hour + self.current_time.minute / 60.0
        return np.array([
            np.sin(2 * np.pi * day_of_year / 365.0), np.cos(2 * np.pi * day_of_year / 365.0),
            np.sin(2 * np.pi * hour_of_day / 24.0), np.cos(2 * np.pi * hour_of_day / 24.0)
        ], dtype=np.float32)

    def get_manager_observations(self):
        gctx  = self._get_time_features()
        mgr_o = self.cluster_manager_ma._prepare_all_manager_observations(
            self.current_time
        )
        return {
            f"manager_{dc}": {**mgr_o[dc], "global_context": gctx}
            for dc in self._dc_ids
        }

    def get_worker_observations(self):
        gctx = self._get_time_features()
        out  = {}
        for dc in self._dc_ids:
            w_obs = self.cluster_manager_ma.nodes[dc].prepare_worker_observation(
                self.current_time
            )
            w_obs["global_context"] = gctx
            out[f"worker_{dc}"] = w_obs
        return out

    def _get_observations(self):
        obs = self.get_manager_observations()
        obs.update(self.get_worker_observations())
        return obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        self.current_time = self.start_time
        
        # Reset the cluster manager with the seed
        # Use random parts of the year for training variability
        # Note: This logic could be more sophisticated if needed
        rng = np.random.default_rng(seed)
        init_day = rng.integers(0, 335) # Start somewhere in the first ~11 months
        init_hour = rng.integers(0, 24)
        
        self.cluster_manager_ma.reset(self.start_time.year, init_day, init_hour, seed)
        
        # The list of active agents is reset to all possible agents
        self.agents = self.possible_agents[:]
        
        if self.logger:
            self.logger.info(f"SustainClusterMAEnv reset. Start time: {self.current_time}, Seed: {seed}")
            
        return self._get_observations(), {agent_id: {} for agent_id in self.agents}

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Takes a step in the environment for all agents.

        Args:
            actions (Dict[str, Any]): A dictionary mapping agent IDs to their chosen actions.

        Returns:
            Tuple: A tuple containing (observations, rewards, terminations, truncations, infos).
        """
        # 1. Map actions from agent IDs to the format expected by cluster_manager_ma
        manager_actions_dict = {
            dc_id: actions[f"manager_{dc_id}"] for dc_id in self._dc_ids if f"manager_{dc_id}" in actions
        }
        worker_actions_dict = {
            dc_id: bool(actions[f"worker_{dc_id}"]) for dc_id in self._dc_ids if f"worker_{dc_id}" in actions
        }
        
        # 2. Step the simulation backend with the agent actions
        results = self.cluster_manager_ma.step_marl(
            self.current_time,
            manager_actions_dict,
            worker_actions_dict
        )

        # 3. Advance time
        self.current_time += self.time_step

        # 4. Get next observations for all agents
        next_observations = self._get_observations()
        
        # 5. Calculate rewards
        # For now, we assume a global reward signal that is the same for all agents.
        # The reward function is expected to parse the `results` dictionary.
        global_reward = self.reward_fn(cluster_info=results, current_time=self.current_time)
        rewards = {agent_id: global_reward for agent_id in self.agents}

        # 6. Check for termination conditions
        terminated = self.current_time >= self.end_time
        terminations = {agent_id: terminated for agent_id in self.agents}
        terminations["__all__"] = terminated

        # Truncation is the same as termination in this setup
        truncations = {agent_id: terminated for agent_id in self.agents}
        truncations["__all__"] = terminated
        
        # If terminated, the list of active agents becomes empty
        if terminated:
            self.agents = []

        # 7. Info dictionary (can be populated with more debug info if needed)
        infos = {agent_id: {} for agent_id in self.possible_agents}
        # Add a global info key if desired
        infos["__global__"] = {"raw_results": results}

        return next_observations, rewards, terminations, truncations, infos
    
    def collect_manager_obs(self) -> Dict[str, Any]:
        self.cluster_manager_ma.task_origination(self.current_time)
        return self.get_manager_observations()
    
    def manager_step(self, mgr_act: Dict[int, int]) -> Dict[str, Any]:
        """Route / migrate tasks """
        # generate new tasks for this tick
        if self.cluster_manager_ma.is_cluster_idle():
            return self.get_worker_observations()
        filtered = {dc: a for dc, a in mgr_act.items()
                    if self.cluster_manager_ma.nodes[dc].originating_tasks_queue}
        if filtered:
            self.cluster_manager_ma.step_manager(self.current_time, filtered)
        return self.get_worker_observations()
    
    def worker_step(self, wrk_act: Dict[int, bool])-> None:
        """Execute / defer tasks - *no* time advance."""
        any_queue = any(self.cluster_manager_ma.nodes[dc].worker_commitment_queue
                        for dc in self._dc_ids)
        if any_queue:
            self.cluster_manager_ma.step_worker(self.current_time, wrk_act)
    
    def env_step(self) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """15-min physical sim + reward."""
        results = self.cluster_manager_ma.step_physics(self.current_time)  
        self.current_time += self.time_step
        next_observations = self._get_observations()
        global_reward = self.reward_fn(cluster_info=results, current_time=self.current_time)
        rewards = {agent_id: global_reward for agent_id in self.agents}

        terminated = self.current_time >= self.end_time
        terminations = {agent_id: terminated for agent_id in self.agents}
        terminations["__all__"] = terminated
        truncations = terminations.copy()

        if terminated:
            self.agents = []

        infos = {agent_id: {} for agent_id in self.possible_agents}
        infos["__global__"] = {"raw_results": results}

        return next_observations, rewards, terminations, truncations, infos

    def render(self, mode='human'):
        """Renders the environment (placeholder)."""
        if mode == 'human':
            print(f"--- Timestep: {self.current_time} ---")
            for dc_id, node in self.cluster_manager_ma.nodes.items():
                print(f"  DC {dc_id} ({node.location}):")
                print(f"    Originating Queue: {len(node.originating_tasks_queue)} tasks")
                print(f"    Worker Queue: {len(node.worker_commitment_queue)} tasks")
                print(f"    Running Tasks: {len(node.physical_dc_model.running_tasks)}")
                print(f"    Available Cores: {node.physical_dc_model.available_cores:.1f}")
        else:
            super().render(mode=mode) # Raise error for other modes

    def close(self):
        """Clean up any resources."""
        pass
