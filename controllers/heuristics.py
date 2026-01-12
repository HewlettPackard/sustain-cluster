import numpy as np
from controllers.base import BaseController

class RandomController(BaseController):
    def __init__(self, cluster_manager):
        super().__init__(cluster_manager)
        self.name = "Random"
        self.num_dcs = len(cluster_manager.datacenters)

    def get_action(self, env, observation):
        return np.random.randint(0, self.num_dcs), {"type": "random"}

class LowestCarbonController(BaseController):
    def __init__(self, cluster_manager):
        super().__init__(cluster_manager)
        self.name = "Greedy-Carbon"

    def get_action(self, env, observation):
        dcs = list(self.cluster_manager.datacenters.values())
        ci_values = [dc.ci_manager.get_current_ci(norm=False) for dc in dcs]
        best_idx = int(np.argmin(ci_values))
        return best_idx, {"carbon": ci_values[best_idx]}

class LowestPriceController(BaseController):
    def __init__(self, cluster_manager):
        super().__init__(cluster_manager)
        self.name = "Greedy-Cost"

    def get_action(self, env, observation):
        dcs = list(self.cluster_manager.datacenters.values())
        price_values = [dc.price_manager.get_current_price() for dc in dcs]
        best_idx = int(np.argmin(price_values))
        return best_idx, {"price": price_values[best_idx]}