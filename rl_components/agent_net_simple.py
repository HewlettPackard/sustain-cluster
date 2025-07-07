import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization for a linear layer, which is a robust default.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ManagerActorMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) based actor network for the DTA_Manager.
    
    This network takes a flattened observation vector and outputs logits for a
    categorical distribution over the discrete action space (the destination DCs).
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initializes the MLP actor.

        Args:
            obs_dim (int): The dimension of the flattened observation space.
            action_dim (int): The number of possible discrete actions (i.e., number of DCs).
            hidden_dim (int): The number of neurons in the hidden layers.
        """
        super().__init__()
        self.actor_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            # INSIGHT: Initialize the final layer with a very small standard deviation.
            # This makes the initial logits close to zero, resulting in a nearly
            # uniform (high-entropy) initial policy.
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass to get the action logits.

        Args:
            x (torch.Tensor): The flattened observation tensor.

        Returns:
            torch.Tensor: The logits for each action. Shape: (batch_size, action_dim).
        """
        return self.actor_net(x)

    def get_action_dist(self, obs: torch.Tensor):
        """
        Helper function to get the action distribution from an observation.
        """
        logits = self.forward(obs)
        # Creates a categorical distribution that can be used to sample actions
        # and calculate log-probabilities.
        return torch.distributions.Categorical(logits=logits)

    

class CentralizedCriticMLP(nn.Module):
    """
    A centralized critic network for the DTA_Managers.

    This network takes the concatenated observations of ALL manager agents as
    input to estimate a single, global state value, V(s_global). This allows
    it to learn a more accurate value function by considering the full system state.
    """
    def __init__(self, num_agents: int, single_agent_obs_dim: int, hidden_dim: int = 256):
        """
        Initializes the centralized MLP critic.

        Args:
            num_agents (int): The total number of manager agents in the system.
            single_agent_obs_dim (int): The observation dimension for one agent.
            hidden_dim (int): The number of neurons in the hidden layers.
        """
        super().__init__()
        
        # The input dimension is the combined size of all agents' observations.
        global_obs_dim = num_agents * single_agent_obs_dim
        
        self.critic_net = nn.Sequential(
            layer_init(nn.Linear(global_obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            # INSIGHT: The value function's final layer should be initialized normally.
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass to get the global state value.

        Args:
            x (torch.Tensor): The concatenated global observation tensor.
                               Shape: (batch_size, num_agents * single_agent_obs_dim).

        Returns:
            torch.Tensor: The estimated value of the global state. Shape: (batch_size, 1).
        """
        return self.critic_net(x)