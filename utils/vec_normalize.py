# utils/vec_normalize.py

import numpy as np
import torch
import os

class RunningMeanStd:
    """Tracks the mean, variance, and count of a streaming data source."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class VecNormalize:
    """
    A wrapper for the multi-agent environment that normalizes observation vectors.
    """
    def __init__(self, venv, gamma=0.99, epsilon=1e-8):
        self.venv = venv # The original environment
        
        # It's good practice to expose some key attributes of the wrapped env directly.
        self.num_dcs = self.venv.num_dcs
        self.observation_space = self.venv.observation_space
        self.action_space = self.venv.action_space
        self._dc_ids = self.venv._dc_ids # Expose _dc_ids        
        
        self.num_agents = venv.num_dcs
        # Get the observation dimension from the original environment
        first_mgr_id = f"manager_{venv._dc_ids[0]}"
        self.obs_dim = venv.observation_space(first_mgr_id).shape[0]

        # Initialize the running mean and standard deviation tracker
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.epsilon = epsilon
        self.gamma = gamma
        self.training = True

    def step(self, actions):
        # Pass actions to the underlying environment
        obs_dict, rews_dict, dones_dict, trunc_dict, infos = self.venv.step(actions)
        
        # We only normalize the manager observations
        manager_obs = np.stack([obs_dict[f"manager_{dc_id}"] for dc_id in self.venv._dc_ids])
        
        # If in training mode, update the running statistics
        if self.training:
            self.obs_rms.update(manager_obs)
            
        # Normalize the observations
        normalized_manager_obs = (manager_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        # Clip observations to a reasonable range to prevent extreme values
        normalized_manager_obs = np.clip(normalized_manager_obs, -10.0, 10.0)
        
        # Re-insert the normalized observations back into the dictionary
        for i, dc_id in enumerate(self.venv._dc_ids):
            obs_dict[f"manager_{dc_id}"] = normalized_manager_obs[i]
            
        return obs_dict, rews_dict, dones_dict, trunc_dict, infos

    def reset(self, seed=None):
        obs_dict, info = self.venv.reset(seed=seed)
        
        manager_obs = np.stack([obs_dict[f"manager_{dc_id}"] for dc_id in self.venv._dc_ids])
        
        if self.training:
            self.obs_rms.update(manager_obs)
            
        normalized_manager_obs = (manager_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        normalized_manager_obs = np.clip(normalized_manager_obs, -10.0, 10.0)
        
        for i, dc_id in enumerate(self.venv._dc_ids):
            obs_dict[f"manager_{dc_id}"] = normalized_manager_obs[i]
            
        return obs_dict, info

    def train(self):
        """Set the wrapper to training mode (updates running stats)."""
        self.training = True

    def eval(self):
        """Set the wrapper to evaluation mode (does not update running stats)."""
        self.training = False

    def __getattr__(self, name):
        """
        If an attribute is not found on the wrapper, forward the request
        to the underlying environment. This makes the wrapper transparent.
        """
        # This will be called for any attribute not found on VecNormalize,
        # like `possible_agents` or any other custom methods or properties
        # you might have on SustainClusterMAEnv.
        return getattr(self.venv, name)
    
    def save(self, path):
        """Saves the running mean and std to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.obs_rms, path)
        
    def load(self, path):
        """Loads the running mean and std from a file."""
        self.obs_rms = torch.load(path)