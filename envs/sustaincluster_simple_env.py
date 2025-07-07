# envs/sustaincluster_simple_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from collections import deque

# Assuming these imports are correct from your project root
from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rewards.base_reward import BaseReward
from utils.meta_task_util import create_meta_task_from_group
from rl_components.task import Task

class SustainClusterSimpleEnv(gym.Env):
    """
    A simplified, single-agent version of the SustainCluster environment.

    This environment presents a single global scheduler agent with a sequence of
    "meta-tasks". At each step, the agent observes one meta-task and the
    current state of all datacenters, and must decide which datacenter to
    route that single meta-task to. The simulation clock only advances after
    all meta-tasks for a 15-minute period have been scheduled.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 cluster_manager: DatacenterClusterManagerMA,
                 start_time: pd.Timestamp,
                 end_time: pd.Timestamp,
                 reward_fn: BaseReward,
                 logger: Optional[Any] = None):
        """
        Initializes the simplified single-agent environment.
        """
        super().__init__()
        self.cluster_manager = cluster_manager
        self.logger = logger
        self.start_time = start_time
        self.end_time = end_time
        self.current_time = self.start_time
        self.time_step = pd.Timedelta(minutes=15)
        self.reward_fn = reward_fn # Note: The reward function might need simplification as well

        self.num_dcs = self.cluster_manager.num_dcs
        self._dc_ids = self.cluster_manager._canonical_dc_order

        # A queue to hold the meta-tasks that need decisions within a 15-min window
        self.pending_meta_tasks: deque[Task] = deque()

        # --- Define Observation and Action Spaces ---
        # The action is to choose one of the N datacenters for the current meta-task.
        self.action_space = spaces.Discrete(self.num_dcs)

        # The observation space is a single flat vector containing:
        # 1. Features of the current meta-task.
        # 2. Features of all possible destination datacenters.
        # 3. Global time features.
        
        # Define feature dimensions
        # Assuming the meta_task object has the same core features as a Task
        self.D_META_TASK_FEAT = 6 # e.g., cores, gpus, mem, bw, duration, sla_urgency
        self.D_DC_STATE_FEAT = self.cluster_manager.D_OPTION_FEAT # Reuse from cluster manager
        self.D_GLOBAL = 4 # Time features
        
        total_obs_dim = self.D_META_TASK_FEAT + (self.num_dcs * self.D_DC_STATE_FEAT) + self.D_GLOBAL
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )

    def _get_time_features(self) -> np.ndarray:
        """Calculates sine/cosine time features for the current time."""
        day_of_year = self.current_time.dayofyear
        hour_of_day = self.current_time.hour + self.current_time.minute / 60.0
        return np.array([
            np.sin(2 * np.pi * day_of_year / 365.0), np.cos(2 * np.pi * day_of_year / 365.0),
            np.sin(2 * np.pi * hour_of_day / 24.0), np.cos(2 * np.pi * hour_of_day / 24.0)
        ], dtype=np.float32)

    def _generate_meta_tasks(self):
        """
        Scans all datacenter nodes, aggregates their originating tasks into
        meta-tasks, and populates the `pending_meta_tasks` queue.
        """
        self.pending_meta_tasks.clear()
        for dc_id in self._dc_ids:
            node = self.cluster_manager.nodes[dc_id]
            task_group = list(node.originating_tasks_queue)
            
            if task_group:
                meta_task = create_meta_task_from_group(task_group, dc_id, self.current_time)
                if meta_task:
                    self.pending_meta_tasks.append(meta_task)
                    # Important: Clear the original queue as it has now been aggregated
                    node.originating_tasks_queue.clear()

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the flattened observation vector for the current state,
        which is defined by the meta-task at the front of the queue and the
        current state of the cluster.
        """
        # If no tasks are pending a decision, return a zero vector.
        if not self.pending_meta_tasks:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        current_meta_task = self.pending_meta_tasks[0]
        
        # 1. Extract features from the current meta-task
        time_to_deadline = (current_meta_task.sla_deadline - self.current_time).total_seconds() / 60.0
        sla_urgency = 1.0 / (time_to_deadline + 1e-6)
        
        meta_task_features = np.array([
            current_meta_task.cores_req,
            current_meta_task.gpu_req,
            current_meta_task.mem_req,
            current_meta_task.bandwidth_gb,
            current_meta_task.duration,
            sla_urgency
        ], dtype=np.float32)
        
        # 2. Get the state of all possible destination datacenters
        dc_states = []
        dc_info_map = self.cluster_manager._prepare_all_manager_observations(self.current_time)
        options_map = dc_info_map["valid_options_maps"][current_meta_task.origin_dc_id]
        
        for i in range(self.num_dcs):
            option_info = options_map[i]
            feature_vector = [option_info[key] for key in self.cluster_manager.DESTINATION_OPTION_FEATURE_ORDER]
            dc_states.append(np.array(feature_vector, dtype=np.float32))
        
        flat_dc_states = np.concatenate(dc_states)

        # 3. Get global time features
        time_features = self._get_time_features()

        # 4. Concatenate all parts into a single flat vector
        flat_obs = np.concatenate([
            meta_task_features,
            flat_dc_states,
            time_features
        ]).astype(np.float32)

        return flat_obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        self.current_time = self.start_time
        
        rng = np.random.default_rng(seed)
        init_day = rng.integers(0, 335)
        init_hour = rng.integers(0, 24)
        
        self.cluster_manager.reset(self.start_time.year, init_day, init_hour, seed)
        
        # Generate initial tasks and meta-tasks for t=0
        self.cluster_manager.task_origination(self.current_time)
        self._generate_meta_tasks()
        
        if self.logger: self.logger.info(f"SustainClusterSimpleEnv reset. Start time: {self.current_time}")
            
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Takes an action for the current meta-task, advances the simulation if
        the queue for the current 15-min window is cleared, and returns the transition.
        """
        if not self.pending_meta_tasks:
            # This case happens if the agent is asked to step but there are no tasks.
            # We just advance time by one 15-minute step.
            self.current_time += self.time_step
            self.cluster_manager.task_origination(self.current_time)
            self._generate_meta_tasks()
            
            terminated = self.current_time >= self.end_time
            return self._get_observation(), 0.0, terminated, terminated, {}

        # 1. Get the task we are making a decision for and remove it from the queue.
        meta_task_to_schedule = self.pending_meta_tasks.popleft()
        
        # 2. Determine the destination and route the task.
        destination_dc_id = self._dc_ids[action]
        self.cluster_manager.route_single_task(meta_task_to_schedule, destination_dc_id, self.current_time)
        
        # 3. Check if we need to advance the main simulation clock.
        # We only do this after all meta-tasks for the current 15-min window are scheduled.
        reward = 0.0
        info = {}

        if not self.pending_meta_tasks:
            # All decisions for this 15-min window are made. Now, run the full simulation step.
            # a) Step the worker logic (we can assume a simple "execute all" policy here)
            worker_actions = {dc_id: 1 for dc_id in self._dc_ids} # 1 = Execute Now
            self.cluster_manager.step_worker(self.current_time, worker_actions)
            
            # b) Step the physics simulation.
            results = self.cluster_manager.step_physics(self.current_time)
            
            # c) Calculate a global reward based on the outcome of this 15-min step.
            #    Note: A simplified reward function might be needed.
            reward = self.reward_fn(cluster_info=results) # Assumes reward_fn can return a single float
            info = {"__global__": {"raw_results": results}}
            
            # d) Advance the clock to the next 15-min interval.
            self.current_time += self.time_step
            
            # e) Generate new tasks and meta-tasks for the new timestep.
            self.cluster_manager.task_origination(self.current_time)
            self._generate_meta_tasks()

        # 4. Prepare return values.
        next_observation = self._get_observation()
        terminated = self.current_time >= self.end_time
        truncated = terminated # No separate truncation condition for now

        return next_observation, float(reward), terminated, truncated, info
