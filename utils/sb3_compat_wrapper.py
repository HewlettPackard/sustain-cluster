import gymnasium as gym
import numpy as np
from typing import Dict

class SB3MultiAgentWrapper(gym.Wrapper):
    """
    A wrapper to make the multi-agent SustainClusterMAEnv compatible with
    single-agent RL libraries like Stable Baselines3.

    It flattens the multi-agent observation dictionary into a single vector and
    expects a single action, which it then distributes to the underlying agents.
    For PPO in a CTDE setup, this wrapper isn't perfect, but for a simple
    shared-policy approach, it's a great start.

    A better approach for CTDE with SB3 is more advanced (e.g., SB3-Contrib),
    but let's start with a simplified shared-policy view.

    This wrapper will present the environment as having:
    - A single observation space (the features of all agents concatenated).
    - A single action space (a MultiDiscrete space for all agent actions).
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_agents = self.env.num_dcs
        self.env_unwrapped = env # Get the underlying SustainClusterMAEnv
        
        # --- Define the new, single-agent observation space ---
        single_agent_obs_dim = self.env_unwrapped.observation_space("manager_1").shape[0]
        self.observation_space = gym.spaces.Box(
            low=-111, high=111, 
            shape=(self.num_agents * single_agent_obs_dim,), 
            dtype=np.float32
        )
        
        # --- Define the new, single-agent action space ---
        single_agent_action_dim = self.env_unwrapped.action_space("manager_1").n
        self.action_space = gym.spaces.MultiDiscrete([single_agent_action_dim] * self.num_agents)

        # Previous action space example:
        # DC1 (new task) -> [0.1, 0.7, 0.3] (actions for DC1)
        # DC2 (new task) -> [0.2, 0.7, 0.3] (actions for DC2)
        # DC3 (new task) -> [0.1, 0.8, 0.2] (actions for DC2)
        
        # New action space example:
        # top_level action matrix -> [[0.1, 0.7, 0.3], 
        #                             [0.2, 0.7, 0.3],
        #                             [0.1, 0.8, 0.2]]
        
        # One tranfer per timestep, e.g., [act1, act2, act3] for 3 datacenters
        # act1 means "transfer tasks from DC1 to DC2"
        # act2 means "transfer tasks from DC2 to DC3"
        # act3 means "transfer tasks from DC3 to DC1"
        
        #top_level action matrix -> [[Prob_source_DC1, Prob_source_DC1, Prob_source_DC1], 
        #                             [Prob_dest_DC1, Prob_dest_DC1, Prob_dest_DC1]]
        #self.action_space = gym.spaces.MultiDiscrete([single_agent_action_dim] * 2)

        
        # === NEW: Add trackers for episodic info ===
        self.ep_total_carbon = 0
        self.ep_total_energy = 0
        self.ep_sla_violations = 0
        self.ep_len = 0
        self.ep_cpu_utils = []

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        
        # === NEW: Reset trackers at the beginning of an episode ===
        self.ep_total_carbon = 0
        self.ep_total_energy = 0
        self.ep_sla_violations = 0
        self.ep_len = 0
        self.ep_cpu_utils = []
        
        return self._flatten_obs(obs_dict), info

    def step(self, action_array):
        # action_array is a numpy array from SB3, e.g., [act1, act2, act3]
        
        # Convert array back to the dict format the underlying env expects
        actions_dict = {f"manager_{dc_id}": action_array[i] for i, dc_id in enumerate(self.env_unwrapped._dc_ids)}
        actions_dict.update({f"worker_{dc_id}": 1 for dc_id in self.env_unwrapped._dc_ids})
        
        next_obs_dict, rewards_dict, dones_dict, trunc_dict, infos_dict = self.env.step(actions_dict)


        # === NEW: Accumulate metrics from the info dictionary ===
        raw_results = infos_dict.get("__global__", {}).get("raw_results", {})
        dc_infos = raw_results.get("datacenter_infos", {})
        
        step_carbon = 0
        step_energy = 0
        step_sla_violations = 0
        step_cpu_utils = []

        for dc_id, dc_info_step in dc_infos.items():
            common = dc_info_step.get("__common__", {})
            step_carbon += common.get("carbon_emissions_kg", 0.0)
            step_energy += common.get("energy_consumption_kwh", 0.0)
            step_sla_violations += common.get("__sla__", {}).get("violated", 0)
            step_cpu_utils.append(common.get("cpu_util_percent", 0.0))

        self.ep_total_carbon += step_carbon
        self.ep_total_energy += step_energy
        self.ep_sla_violations += step_sla_violations
        if step_cpu_utils:
            self.ep_cpu_utils.append(np.mean(step_cpu_utils))
        self.ep_len += 1
        # === END OF ACCUMULATION ===
        
        # Convert the results back to the single-agent format
        flat_next_obs = self._flatten_obs(next_obs_dict)
        global_reward = np.mean(list(rewards_dict.values()))
        done = dones_dict["__all__"]
        truncated = trunc_dict["__all__"]
        # print(f"Global reward: {global_reward}, Done: {done}, Truncated: {truncated}")
        
        # If the episode ends, add the episodic info to the info dict for SB3 callbacks
        # === NEW: Add episodic info to the info dict when the episode ends ===
        if done or truncated:
            avg_cpu_util = np.mean(self.ep_cpu_utils) if self.ep_cpu_utils else 0
            
            # The Monitor wrapper looks for this specific "episode" key
            infos_dict["episode"] = {
                "r": global_reward, # The Monitor uses this for the main reward plot
                "l": self.ep_len,       # Episode length
                "t": 0,                 # Time (not used by default logger)
                "total_carbon_kg": self.ep_total_carbon,
                "total_energy_kwh": self.ep_total_energy,
                "total_sla_violations": self.ep_sla_violations,
                "avg_cpu_util_pct": avg_cpu_util,
            }

        return flat_next_obs, global_reward, done, truncated, infos_dict
        
    def _flatten_obs(self, obs_dict: Dict) -> np.ndarray:
        """Flattens the multi-agent observation dict into a single vector."""
        # Ensure a consistent order
        obs_list = [obs_dict[f"manager_{dc_id}"] for dc_id in self.env_unwrapped._dc_ids]
        return np.concatenate(obs_list).astype(np.float32)