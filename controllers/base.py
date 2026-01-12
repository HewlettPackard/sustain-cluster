from abc import ABC, abstractmethod

class BaseController(ABC):
    def __init__(self, cluster_manager):
        self.cluster_manager = cluster_manager
        self.name = "Base"

    @abstractmethod
    def get_action(self, env, observation):
        """
        Input: 
            env: The environment instance (to access complex state if needed)
            observation: The gym observation vector
        Output: 
            int: The chosen DC index
            dict: Metadata (reasoning, retrieval info, etc.)
        """
        pass