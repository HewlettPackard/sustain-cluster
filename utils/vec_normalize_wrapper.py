# utils/vec_normalize_wrapper.py
import numpy as np
import torch
from utils.vec_normalize import RunningMeanStd

class VecNormalizeWrapper:
    """
    VecNormalize wrapper for SpeedOptimizedVectorizedEnv
    Adapts the original VecNormalize to work with the custom vectorized environment
    """
    def __init__(self, venv, gamma=0.99, epsilon=1e-8):
        self.venv = venv
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = True
        
        # Get observation dimensions from the vectorized environment
        self.num_envs = venv.num_envs
        self.num_agents = venv.num_agents
        self.obs_dim = venv.single_observation_space.shape[0]
        
        # Initialize running mean and std tracker for observations
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
        
        # Expose attributes that training script expects
        self.single_observation_space = venv.single_observation_space
        self.single_action_space = venv.single_action_space
        self.dc_ids = venv.dc_ids
        
    def normalize_observations(self, obs_dict_list):
        """Normalize observations for all environments"""
        normalized_obs_list = []
        
        for env_idx, obs_dict in enumerate(obs_dict_list):
            normalized_obs_dict = {}
            
            # Collect all manager observations for normalization
            manager_obs_batch = []
            manager_keys = []
            
            for key, obs in obs_dict.items():
                if key.startswith("manager_"):
                    manager_obs_batch.append(obs)
                    manager_keys.append(key)
                else:
                    # Keep non-manager observations as-is
                    normalized_obs_dict[key] = obs
            
            if manager_obs_batch:
                manager_obs_array = np.stack(manager_obs_batch)
                
                # Update running statistics if in training mode
                if self.training:
                    self.obs_rms.update(manager_obs_array)
                
                # Normalize observations
                normalized_manager_obs = (manager_obs_array - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
                normalized_manager_obs = np.clip(normalized_manager_obs, -10.0, 10.0)
                
                # Put normalized observations back
                for i, key in enumerate(manager_keys):
                    normalized_obs_dict[key] = normalized_manager_obs[i]
            
            normalized_obs_list.append(normalized_obs_dict)
        
        return normalized_obs_list
    
    def reset(self, seed=None):
        """Reset with normalization"""
        obs_dict_list = self.venv.reset(seed=seed)
        return self.normalize_observations(obs_dict_list)
    
    def step(self, actions_list):
        """Step with normalization"""
        obs_list, rewards_list, dones_list, truncs_list, infos_list = self.venv.step(actions_list)
        normalized_obs_list = self.normalize_observations(obs_list)
        return normalized_obs_list, rewards_list, dones_list, truncs_list, infos_list
    
    def train(self):
        """Set to training mode (updates running stats)"""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode (does not update running stats)"""
        self.training = False
    
    def close(self):
        """Close the underlying environment"""
        self.venv.close()
    
    def __getattr__(self, name):
        """Forward any missing attributes to the underlying environment"""
        return getattr(self.venv, name)