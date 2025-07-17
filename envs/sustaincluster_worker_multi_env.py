# envs/sustaincluster_worker_multi_env.py

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Deque, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rewards.base_reward import BaseReward
from rl_components.task import Task


class SustainClusterWorkerMultiEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(
        self,
        cluster_manager: DatacenterClusterManagerMA,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        reward_fn: BaseReward,
        logger: Optional[Any] = None,
        deterministic_reset: bool = False,
        fixed_seed: Optional[int] = None,
    ):
        """
        Initializes the multi-agent environment only for worker.
        """
        super().__init__()

        # External handles
        self.cluster_manager = cluster_manager
        self.reward_fn = reward_fn
        self.logger = logger

        # Episode timing
        self.start_time = start_time
        self.end_time = end_time
        self.simulation_duration = end_time - start_time

        self.current_time = start_time
        self.time_step = pd.Timedelta(minutes=15)

        # Datacenter list (fixed order)
        self.dc_ids: List[int] = self.cluster_manager._canonical_dc_order

        # Spaces
        self.action_space = spaces.MultiDiscrete([2] * len(self.dc_ids))

        self.D_TASK_FEAT = 7
        self.D_DC_FEAT = 5
        self.D_GLOBAL = 4
        obs_dim = self.D_TASK_FEAT + self.D_DC_FEAT + self.D_GLOBAL

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.dc_ids), obs_dim), dtype=np.float32
        )

        # Local buffers
        self.pending_tasks: Dict[int, Deque[Task]] = {
            dc: deque() for dc in self.dc_ids
        }
        
        self.deterministic_reset = deterministic_reset
        self.fixed_seed = fixed_seed

    
    def _get_time_features(self) -> np.ndarray:
        """Calculates sine/cosine time features for the current time."""
        day_of_year = self.current_time.dayofyear
        hour_of_day  = self.current_time.hour + self.current_time.minute / 60.0
        return np.array(
            [
                np.sin(2 * np.pi * day_of_year / 365.0),
                np.cos(2 * np.pi * day_of_year / 365.0),
                np.sin(2 * np.pi * hour_of_day / 24.0),
                np.cos(2 * np.pi * hour_of_day/ 24.0),
            ],
            dtype=np.float32,
        )

    def _sync_pending(self) -> bool:
        """Mirror each DC worker queue into local deques.
        Returns True if any queue is non-empty."""
        any_nonempty = False
        for dc in self.dc_ids:
            q = self.cluster_manager.nodes[dc].worker_commitment_queue
            self.pending_tasks[dc] = deque(q)
            any_nonempty |= bool(q)
        return any_nonempty

    def _get_observation(self, dc: int) -> np.ndarray:
        """Build observation vector for one DC."""
        q = self.pending_tasks[dc]
        if not q:
            return np.zeros(self.observation_space.shape[1], dtype=np.float32)

        current_meta_task = q[0]
        age = (self.current_time - current_meta_task.arrival_time).total_seconds() / 60.0
        time_to_deadline = (current_meta_task.sla_deadline - self.current_time).total_seconds() / 60.0
        sla_urgency = 1.0 / (time_to_deadline + 1e-6)

        task_vec = np.array(
            [
                current_meta_task.cores_req,
                current_meta_task.gpu_req,
                current_meta_task.mem_req,
                current_meta_task.bandwidth_gb,
                current_meta_task.duration,
                sla_urgency,
                age,
            ],
            dtype=np.float32,
        )
        dc_vec = self.cluster_manager.nodes[dc].prepare_worker_observation(self.current_time)["obs_local_dc_i_for_worker"]

        return np.concatenate([task_vec, dc_vec, self._get_time_features()])

    def _all_queues_empty(self) -> bool:
        return all(len(q) == 0 for q in self.pending_tasks.values())
    
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        self.current_time = self.start_time
        # Randomize offset inside data year
        # Set numpy random seed for reproducibility
        # if seed is not None:
            # np.random.seed(seed)
            # print(f"Resetting with seed: {seed}")
            
        
        if self.deterministic_reset and self.fixed_seed is not None:
            np.random.seed(self.fixed_seed)
            # month_day = 15  # Fixed values for deterministic reset
            # init_hour = 12
            month_day = np.random.randint(1, 31)
            init_hour = np.random.randint(0, 24)
            print(f"Deterministic environemnt reset with seed: {self.fixed_seed} at day {month_day} hour {init_hour}")
        else:
            if seed is not None:
                np.random.seed(seed)
                print(f"Resetting with seed: {seed}")
            month_day = np.random.randint(1, 31)
            init_hour = np.random.randint(0, 24)

        # Random init day in month 7 (day 30*7)
        # month_day = np.random.randint(1, 31)
        init_day = month_day + 30 * 6  # July is the 7th month, so we start from day 1 to 30
        # init_hour = np.random.randint(0, 24)

        self.cluster_manager.reset(self.start_time.year, init_day, init_hour, seed)

        # Transform self.start_time.year, init_day, init_hour to the self.current_time
        self.current_time = pd.Timestamp(year=self.start_time.year, month=7, day=month_day, hour=init_hour, minute=0, second=0)
        # Add TZ to self.current_time
        self.current_time = self.current_time.tz_localize('UTC')  # Assuming UTC
        self.start_time = self.current_time
        self.end_time = self.current_time + self.simulation_duration
        
        # Initial workload generation
        self.cluster_manager.task_origination(self.current_time)
        stay_local = {
            dc: self.cluster_manager._canonical_dc_order.index(dc)
            for dc in self.dc_ids
        }
        self.cluster_manager.step_manager(self.current_time, stay_local, {})
        self._sync_pending()

        obs = np.stack([self._get_observation(dc) for dc in self.dc_ids])
        return obs, {}
    

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        
        if self._all_queues_empty():
            self.current_time += self.time_step
            self.cluster_manager.task_origination(self.current_time)
            manager_actions = {
                dc: self.cluster_manager._canonical_dc_order.index(dc)
                for dc in self.dc_ids
            }
            self.cluster_manager.step_manager(self.current_time, manager_actions, {})
            self._sync_pending()
            obs_next = np.stack([self._get_observation(dc) for dc in self.dc_ids])

            terminated = (self.current_time >= self.end_time)
            truncated = False
            return obs_next, 0.0, terminated, truncated, {}
        committed_this_step = {dc: 0 for dc in self.dc_ids}
        for idx, dc in enumerate(self.dc_ids):
            self.cluster_manager.single_worker_dc_id = dc
            before_commit = len(self.cluster_manager.nodes[dc].worker_commitment_queue)
            self.cluster_manager.commit_tasks(
                self.current_time, int(actions[idx])
            )
            after_commit = len(self.cluster_manager.nodes[dc].worker_commitment_queue)
            committed_this_step[dc] = max(0, before_commit - after_commit)
        self._sync_pending()
       
        cluster_info = self.cluster_manager.step_physics(self.current_time)
        task_info = {}
        for dc, q in self.pending_tasks.items():
            if q:
                ages = [(self.current_time - task.arrival_time).total_seconds() / 60.0 for task in q]
                task_info[dc] = {
                    "age_min": min(ages),
                    "age_mean": np.mean(ages),
                    "queue_length": len(q),
                }
            else:
                task_info[dc] = {"age_min": 0.0, "age_mean": 0.0, "queue_length": 0}

        reward = float(self.reward_fn(cluster_info=cluster_info, task_info=task_info))

        self.current_time += self.time_step
        self.cluster_manager.task_origination(self.current_time)
        manager_actions = {
            dc: self.cluster_manager._canonical_dc_order.index(dc)
            for dc in self.dc_ids
        }
        self.cluster_manager.step_manager(self.current_time, manager_actions, {})
        self._sync_pending()

       
        obs_next = np.stack([self._get_observation(dc) for dc in self.dc_ids])
        terminated = (self.current_time >= self.end_time)
        # Extract terminations and truncations from the results dictionary
        # terminations = cluster_info.get('terminateds', {agent_id: False for agent_id in range(len(self.dc_ids))})
        # truncations = cluster_info.get('truncateds', {agent_id: False for agent_id in range(len(self.dc_ids))})
        
        # Also add the __all key to both terminations and truncations
        terminations = cluster_info['datacenter_infos'][1].get('terminateds')['__all__']
        truncations  = cluster_info['datacenter_infos'][1].get('truncateds')['__all__']
        info = {"raw_results": cluster_info,
                "committed_this_step": committed_this_step,
                "pending_after_step": {dc: len(self.pending_tasks[dc]) for dc in self.dc_ids},
                "task_info": task_info,
                "is_training": not self.deterministic_reset,}
        return obs_next, reward, terminations, truncations, info
    
    def render(self, mode: str = "human"):
        for dc in self.dc_ids:
            run = len(self.cluster_manager.nodes[dc].physical_dc_model.running_tasks)
            pend = len(self.pending_tasks[dc])
            print(
                f"[{self.current_time}] DC {dc} | Pending={pend:3d} | Running={run:3d}"
            )

    def close(self):
        pass
