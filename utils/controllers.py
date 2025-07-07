import numpy as np
from typing import Dict, List
import torch

class BaseController:
    """Base class for all controllers (RL and Rule-Based)."""
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def get_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Takes the observation dictionary for all agents and returns a dictionary of actions.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError

class RandomController(BaseController):
    """A controller that takes random actions."""
    def get_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        actions = {}
        # The keys in obs_dict tell us which manager agents are active
        for agent_id in obs_dict.keys():
            if agent_id.startswith("manager"):
                actions[agent_id] = np.random.randint(0, self.num_actions)
        return actions

class LowestCarbonController(BaseController):
    """
    A Rule-Based Controller that sends each meta-task to the datacenter
    with the lowest current carbon intensity (CI).
    """
    def get_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        actions = {}
        if not obs_dict:
            return {}

        # The cluster state is the same for all agents, so we can extract it from the first one.
        first_agent_obs = next(iter(obs_dict.values()))
        
        # --- Extract the CI values from the flattened observation ---
        # This requires knowing the structure of your observation vector.
        # Structure: [meta_task(7), dc1_state(5), dc2_state(5), dc3_state(5), time(4)]
        # The 'ci' is the 4th feature in each 5-feature dc_state block.
        # Indices:
        # - DC1 CI: 7 + 3 = 10
        # - DC2 CI: 7 + 5 + 3 = 12
        # - DC3 CI: 7 + 10 + 3 = 20
        ci_indices = [10, 15, 20] # Assuming 3 DCs, 8 meta-task feats, 5 dc-state feats
        
        # This logic needs to be robust to changes in obs dimensions.
        # Let's assume D_META_MANAGER=8, D_DC_STATE_FEAT=5 for now.
        ci_values = [first_agent_obs[7 + i*5 + 3] for i in range(self.num_actions)]
        
        # Find the index of the datacenter with the minimum CI
        best_dc_index = np.argmin(ci_values)
        
        # All managers will make the same decision: send to the globally best DC.
        for agent_id in obs_dict.keys():
            if agent_id.startswith("manager"):
                actions[agent_id] = best_dc_index
                
        return actions

class LowestTemperatureController(BaseController):
    """
    A Rule-Based Controller that sends each meta-task to the datacenter
    with the lowest current external temperature.
    """
    def get_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        actions = {}
        if not obs_dict:
            return {}

        first_agent_obs = next(iter(obs_dict.values()))
        
        # --- Extract the Temperature values ---
        # Structure: [meta_task(8), dc1_state(5), dc2_state(5), dc3_state(5), time(4)]
        # The 'external_temperature' is the 5th feature (index 4) in each dc_state block.
        temp_indices = [8 + i*5 + 4 for i in range(self.num_actions)]
        temp_values = [first_agent_obs[idx] for idx in temp_indices]
        
        # Find the index of the datacenter with the minimum temperature
        best_dc_index = np.argmin(temp_values)
        
        for agent_id in obs_dict.keys():
            if agent_id.startswith("manager"):
                actions[agent_id] = best_dc_index
                
        return actions
    
    
class PPOController(BaseController):
    """A controller that uses a trained PPO actor network to get actions."""
    def __init__(self, actor_network, device):
        # The number of actions is inferred from the actor's output layer
        super().__init__(num_actions=actor_network.actor_net[-1].out_features)
        self.actor = actor_network
        self.device = device
        self.actor.eval() # Ensure actor is in evaluation mode

    def get_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        if not obs_dict:
            return {}

        # Stack the observations into a single batch tensor
        obs_list = [obs for agent_id, obs in obs_dict.items() if agent_id.startswith("manager")]
        if not obs_list:
            return {}
            
        obs_tensor = torch.from_numpy(np.stack(obs_list)).float().to(self.device)

        # Get actions from the actor network (greedy evaluation)
        with torch.no_grad():
            logits = self.actor(obs_tensor)
            actions_tensor = torch.argmax(logits, dim=1)
        
        # Create the action dictionary
        manager_agent_ids = [agent_id for agent_id in obs_dict.keys() if agent_id.startswith("manager")]
        actions_dict = {agent_id: actions_tensor[i].item() for i, agent_id in enumerate(manager_agent_ids)}
        
        return actions_dict