# rewards/predefined/manager_ci_reward.py

import numpy as np
from typing import Dict, Any

from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("manager_ci_proportional") # New name to reflect new logic
class ManagerCarbonProportionalReward(BaseReward):
    """
    A MARL-aware reward for the DTA_Manager that is proportional to the
    workload being routed.

    This reward calculates an immediate reward for each Manager agent based on
    the carbon intensity (CI) of its chosen destination, multiplied by a measure
    of the workload in its originating queue. This incentivizes the manager
    not just to pick green datacenters, but to send *more work* to the
    greenest ones.

    The reward is calculated as:
    `Reward = - (workload_measure * destination_ci / normalization_factor)`

    A lower total carbon impact results in a higher (less negative) reward.
    """
    def __init__(self,
                 ci_normalization_factor: float = 1000.0,
                 workload_measure: str = "num_tasks"):
        """
        Args:
            ci_normalization_factor (float): A factor to scale the final reward.
                                             e.g., 1000.0 scales gCO2 to kgCO2.
            workload_measure (str): How to measure the workload.
                                    Options: "num_tasks", "total_cpu", "total_gpu".
        """
        if workload_measure not in ["num_tasks", "total_cpu", "total_gpu"]:
            raise ValueError(f"Invalid workload_measure: {workload_measure}")

        self.ci_norm = ci_normalization_factor
        self.workload_measure = workload_measure
        self.last_reward: Dict[str, float] = {}

    def __call__(self,
                 cluster_info: Dict[str, Any],
                 manager_actions: Dict[int, int],
                 worker_actions: Dict[int, int],
                 valid_options_maps: Dict[int, Dict],
                 workload_magnitudes: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculates the workload-proportional CI-based reward for each DTA_Manager.
        Now uses the pre-calculated workload_magnitudes.
        """
        rewards = {}

        for dc_id, action_idx in manager_actions.items():
            manager_id = f"manager_{dc_id}"
            
            # 1. Get the workload magnitude from the new argument
            magnitudes = workload_magnitudes.get(dc_id, {})
            workload_magnitude = magnitudes.get(self.workload_measure, 0.0)

            if workload_magnitude == 0:
                rewards[manager_id] = 0.0
                continue

            # The rest of the logic remains the same
            options_map = valid_options_maps.get(dc_id)
            if not options_map or action_idx not in options_map:
                rewards[manager_id] = -1.0 * workload_magnitude
                continue
            
            chosen_destination_info = options_map[action_idx]
            destination_ci = chosen_destination_info.get("ci", self.ci_norm)

            carbon_impact = workload_magnitude * destination_ci
            reward = - (carbon_impact / self.ci_norm)
            rewards[manager_id] = reward

        # Workers get a neutral reward
        for dc_id in worker_actions.keys():
            rewards.setdefault(f"worker_{dc_id}", 0.0)

        # Obtain the total reward as the sum of individual rewards
        total_reward = sum(rewards.values())
        self.last_reward = total_reward
        return total_reward

    def __str__(self):
        return "ManagerCarbonProportionalReward"