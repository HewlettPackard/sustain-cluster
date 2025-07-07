from rewards.base_reward import BaseReward
from rewards.registry_utils import register_reward

@register_reward("efficiency")
class EfficiencyReward(BaseReward):
    def __init__(self, normalize_factor: float = 1000.0):
        super().__init__()
        self.normalize_factor = normalize_factor
        
    # def __call__(self, cluster_info, current_tasks, current_time):
    def __call__(self, cluster_info, current_time):
        total_energy = sum(
            dc_info["__common__"].get("energy_consumption_kwh", 0)
            for dc_info in cluster_info["datacenter_infos"].values()
        )
        total_tasks = cluster_info.get("scheduled_tasks", 0)
        if total_tasks == 0:
            self.last_reward = 0.0
            return 0.0
        
        reward = -total_energy / total_tasks
        self.last_reward = reward
        return reward

    def get_last_value(self):
        return self.last_reward