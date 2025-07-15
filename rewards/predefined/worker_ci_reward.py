from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward


@register_reward("worker_ci_sla_penalty")
class WorkerCISLAPenaltyReward(BaseReward):
    """Reward combining carbon emissions with SLA violation penalty."""

    def __init__(
        self,
        ci_norm: float = 1000.0,
        delay_penalty_coef: float = 0.01,
        sla_threshold_min: float = 10.0,
    ):
        """
        Parameters
        ----------
        ci_norm : float
            Scaling factor to normalize carbon emissions.
        delay_penalty_coef : float
            Penalty coefficient per minute exceeding SLA threshold.
        sla_threshold_min : float
            Threshold in minutes after which delay penalty begins.
        """
        self.ci_norm = ci_norm
        self.delay_penalty_coef = delay_penalty_coef
        self.sla_threshold_min = sla_threshold_min

    def __call__(self, **kwargs) -> float:
        cluster_info: Dict[str, Any] = kwargs.get("cluster_info", {})
        task_info: Dict[int, Dict[str, float]] = kwargs.get("task_info", {})

        # Carbon penalty
        total_kg = sum(
            dc["__common__"].get("carbon_emissions_kg", 0.0)
            for dc in cluster_info.get("datacenter_infos", {}).values()
        )
        carbon_term = -total_kg / self.ci_norm

        # SLA violation penalty
        sla_penalty = 0.0
        for dc_info in task_info.values():
            age_mean = dc_info["age_mean"]
            queue_length = dc_info["queue_length"]
        
            overtime = max(0.0, age_mean - self.sla_threshold_min)
            sla_penalty += queue_length * overtime

        delay_term = -self.delay_penalty_coef * sla_penalty / (len(task_info) + 1e-6)

        return carbon_term + delay_term

    def __str__(self) -> str:
        return "WorkerCISLAPenaltyReward"
