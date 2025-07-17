# in rewards/predefined/worker_throughput_reward.py (or a new file)

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("worker_throughput")
class WorkerThroughputReward(BaseReward):
    """
    A reward function that incentivizes task completion (throughput) while
    penalizing for carbon emissions and SLA violations.

    This creates a clear economic trade-off for the agent:
    - It gets a large, positive reward for each task it successfully completes.
    - It receives a penalty based on the carbon emitted to complete those tasks.
    - It receives a large, sharp penalty for any task that misses its SLA.
    - It receives a small, continuous penalty for keeping tasks waiting in the queue.
    """
    def __init__(
        self,
        reward_per_task_completion: float = 1.0,
        carbon_penalty_factor: float = 1.0,
        sla_violation_penalty: float = 1.0,
        queue_age_penalty_factor: float = 0.1,
        normalization_stats: Dict[str, Dict[str, float]] = None
    ):
        # These are treated as WEIGHTS for the normalized components
        self.throughput_weight = reward_per_task_completion
        self.carbon_weight = carbon_penalty_factor
        self.sla_weight = sla_violation_penalty
        self.age_weight = queue_age_penalty_factor

        # Store normalization stats, with defaults to prevent errors
        self.stats = normalization_stats or {}
        self.eps = 1e-8 # Epsilon to avoid division by zero

    def _normalize(self, value: float, component_name: str) -> float:
        """Normalizes a value using its component's mean and std."""
        if component_name in self.stats and "mean" in self.stats[component_name] and "std" in self.stats[component_name]:
            mean = self.stats[component_name]["mean"]
            std = self.stats[component_name]["std"]
            return (value - mean) / (std + self.eps)
        # If no stats are provided for this component, return the raw value (no normalization)
        return value
    
    def __call__(self, **kwargs) -> float:
        cluster_info: Dict[str, Any] = kwargs.get("cluster_info", {})
        task_info: Dict[int, Dict[str, float]] = kwargs.get("task_info", {})

        # --- 1. Calculate RAW component values ---
        num_tasks_completed = sum(dc["__common__"].get("finished_tasks_count", 0) for dc in cluster_info.get("datacenter_infos", {}).values())
        raw_throughput_reward = num_tasks_completed

        total_kg_co2 = sum(dc["__common__"].get("carbon_emissions_kg", 0.0) for dc in cluster_info.get("datacenter_infos", {}).values())
        raw_carbon_penalty = -total_kg_co2

        num_sla_violations = sum(dc["__common__"].get("__sla__", {}).get("violated", 0) for dc in cluster_info.get("datacenter_infos", {}).values())
        raw_sla_penalty = -num_sla_violations

        total_queue_age_minutes = sum(dc_info.get("age_mean", 0.0) * dc_info.get("queue_length", 0) for dc_info in task_info.values())
        raw_aging_penalty = -total_queue_age_minutes

        # --- 2. Normalize each component ---
        norm_throughput = self._normalize(raw_throughput_reward, "throughput")
        norm_carbon = self._normalize(raw_carbon_penalty, "carbon")
        norm_sla = self._normalize(raw_sla_penalty, "sla")
        norm_aging = self._normalize(raw_aging_penalty, "aging") # Will use raw value if 'aging' not in stats
        
        # print(f"Raw values: throughput={raw_throughput_reward:.2f}, carbon={raw_carbon_penalty:.2f}, sla={raw_sla_penalty:.2f}, aging={raw_aging_penalty:.2f}")
        # print(f"Normalized values: throughput={norm_throughput:.2f}, carbon={norm_carbon:.2f}, sla={norm_sla:.2f}, aging={norm_aging:.2f}")

        # --- 3. Combine normalized components using weights ---
        final_reward = (
            self.throughput_weight * norm_throughput +
            self.carbon_weight * norm_carbon +
            self.sla_weight * norm_sla +
            self.age_weight * norm_aging
        )
        
        return float(final_reward)
        
        return float(final_reward)

    def __str__(self) -> str:
        return "NormalizedWorkerThroughputReward"