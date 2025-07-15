# envs/sustaincluster_worker_simple_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from collections import deque

from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rewards.base_reward import BaseReward
from rl_components.task import Task


class SustainClusterWorkerSimpleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            cluster_manager: DatacenterClusterManagerMA,
            start_time: pd.Timestamp,
            end_time: pd.Timestamp,
            reward_fn: BaseReward,
            logger: Optional[Any] = None):

        super().__init__()
        self.cluster_manager = cluster_manager
        self.logger = logger
        self.start_time = start_time
        self.end_time = end_time
        self.current_time = start_time
        self.time_step = pd.Timedelta(minutes=15)
        self.reward_fn = reward_fn

        self.dc_ids = self.cluster_manager._canonical_dc_order
        self.local_dc_id = self.dc_ids[0]  # assuming single DC for simplicity

        self.action_space = spaces.Discrete(2)
        self.pending_tasks: deque[Task] = deque()

        self.D_TASK_FEAT = 6
        self.D_DC_FEAT = 5
        self.D_GLOBAL = 4
        obs_dim = self.D_TASK_FEAT + self.D_DC_FEAT + self.D_GLOBAL
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _get_time_features(self) -> np.ndarray:
        day_of_year = self.current_time.dayofyear
        hour_of_day = self.current_time.hour + self.current_time.minute / 60.0
        return np.array([
            np.sin(2 * np.pi * day_of_year / 365.0), np.cos(2 * np.pi * day_of_year / 365.0),
            np.sin(2 * np.pi * hour_of_day / 24.0), np.cos(2 * np.pi * hour_of_day / 24.0)
        ], dtype=np.float32)

    def _sync_pending_tasks(self) -> None:
        node = self.cluster_manager.nodes[self.local_dc_id]
        self.pending_tasks = deque(node.worker_commitment_queue)

        while not self.pending_tasks and self.current_time < self.end_time:
            self.current_time += self.time_step
            self.cluster_manager.task_origination(self.current_time)
            manager_actions = {
                dc_id: self.cluster_manager._canonical_dc_order.index(dc_id)
                for dc_id in self.dc_ids
            }
            self.cluster_manager.step_manager(self.current_time, manager_actions, {})
            self.local_dc_id = self._choose_local_dc()
            self.cluster_manager.single_worker_dc_id = self.local_dc_id
            node = self.cluster_manager.nodes[self.local_dc_id]
            self.pending_tasks = deque(node.worker_commitment_queue)

    def _get_observation(self) -> np.ndarray:
        if not self.pending_tasks:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        current_meta_task = self.pending_tasks[0]
        time_to_deadline = (current_meta_task.sla_deadline - self.current_time).total_seconds() / 60.0
        sla_urgency = 1.0 / (time_to_deadline + 1e-6)

        task_features = np.array([
            current_meta_task.cores_req,
            current_meta_task.gpu_req,
            current_meta_task.mem_req,
            current_meta_task.bandwidth_gb,
            current_meta_task.duration,
            sla_urgency
        ], dtype=np.float32)

        dc_features = self.cluster_manager.nodes[self.local_dc_id].prepare_worker_observation(self.current_time)[
            "obs_local_dc_i_for_worker"]
        time_features = self._get_time_features()

        return np.concatenate([task_features, dc_features, time_features])

    def _choose_local_dc(self) -> int:
        biggest = max(
            self.cluster_manager.nodes.items(),
            key=lambda kv: (len(kv[1].worker_commitment_queue), -kv[0])
        )
        return biggest[0]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_time = self.start_time
        rng = np.random.default_rng(seed)
        init_day = rng.integers(0, 335)
        init_hour = rng.integers(0, 24)

   
        self.cluster_manager.reset(self.start_time.year, init_day, init_hour, seed)

  
        self.cluster_manager.task_origination(self.current_time)


        manager_actions = {
            dc_id: self.cluster_manager._canonical_dc_order.index(dc_id)
            for dc_id in self.dc_ids
        }
        self.cluster_manager.step_manager(self.current_time, manager_actions, {})

        self.local_dc_id = self._choose_local_dc()
        self.cluster_manager.single_worker_dc_id = self.local_dc_id
        if self.logger:
         self.logger.info(f"[env.reset] Worker bound to DC {self.local_dc_id}")

    #
        self._sync_pending_tasks()

        return self._get_observation(), {}

    def step(self, action: int):
        if not self.pending_tasks:
            self.current_time += self.time_step
            self.cluster_manager.task_origination(self.current_time)
            manager_actions = {
                dc_id: self.cluster_manager._canonical_dc_order.index(dc_id)
                for dc_id in self.dc_ids
            }
            self.cluster_manager.step_manager(self.current_time, manager_actions, {})
            self.local_dc_id = self._choose_local_dc()
            self.cluster_manager.single_worker_dc_id = self.local_dc_id
            self._sync_pending_tasks()

        self.local_dc_id = self._choose_local_dc()
        self.cluster_manager.single_worker_dc_id = self.local_dc_id
        self.cluster_manager.commit_tasks(self.current_time, action)

        results = self.cluster_manager.step_physics(self.current_time)
        reward = float(self.reward_fn(cluster_info=results))

        self.current_time += self.time_step
        self.cluster_manager.task_origination(self.current_time)
        manager_actions = {
            dc_id: self.cluster_manager._canonical_dc_order.index(dc_id)
            for dc_id in self.dc_ids
        }
        self.cluster_manager.step_manager(self.current_time, manager_actions, {})

        self._sync_pending_tasks()
        obs_next = self._get_observation()

        terminated = truncated = self.current_time >= self.end_time
        return obs_next, reward, terminated, truncated, {"raw_results": results}

    def render(self, mode: str = "human"):
        node = self.cluster_manager.nodes[self.local_dc_id]
        running = len(node.physical_dc_model.running_tasks)
        avail_cores = node.physical_dc_model.available_cores
        print(f"[{self.current_time}] Pending={len(self.pending_tasks)} | Running={running} | AvailCores={avail_cores}")

    def close(self):
        pass