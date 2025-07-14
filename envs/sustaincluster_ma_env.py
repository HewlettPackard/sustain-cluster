import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rewards.base_reward import BaseReward
from utils.marl_utils import D_META_MANAGER, D_META_WORKER
from utils.meta_task_util import create_meta_task_from_group # You might not need this if aggregation is in marl_utils

class SustainClusterMAEnv(gym.Env):
    """
    A multi-agent Gymnasium environment for the SustainCluster-MA benchmark.
    ...
    This version includes a `simple_obs_mode` to produce flattened observation
    vectors for use with simpler MLP-based agents.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 cluster_manager_ma: DatacenterClusterManagerMA,
                 start_time: pd.Timestamp,
                 end_time: pd.Timestamp,
                 reward_fn: BaseReward,
                 logger: Optional[Any] = None,
                 simple_obs_mode: bool = False,
                 use_meta_tasks: bool = False): # === SIMPLIFICATION CHANGE ===
        """
        Initializes the multi-agent environment.
        ...
        Args:
            simple_obs_mode (bool): If True, the environment will produce a single
                                    flattened numpy array for each agent's observation.
                                    If False, it produces the complex dictionary-based
                                    observation suitable for attention networks.
        """
        super().__init__()
        self.cluster_manager_ma = cluster_manager_ma
        self.logger = logger
        self.start_time = start_time
        self.end_time = end_time
        self.simulation_duration = end_time - start_time
        self.current_time = self.start_time
        self.time_step = pd.Timedelta(minutes=15)
        self.reward_fn = reward_fn
        self.simple_obs_mode = simple_obs_mode # === SIMPLIFICATION CHANGE ===
        self.use_meta_tasks = use_meta_tasks

        self.num_dcs = self.cluster_manager_ma.num_dcs
        self._dc_ids = sorted(self.cluster_manager_ma.nodes.keys())

        # --- Define Agent IDs ---
        self.manager_agent_ids = [f"manager_{dc_id}" for dc_id in self._dc_ids]
        self.worker_agent_ids = [f"worker_{dc_id}" for dc_id in self._dc_ids]
        self.possible_agents = self.manager_agent_ids + self.worker_agent_ids
        self.agents = []

        # --- Define Observation and Action Spaces ---
        self._define_spaces()

        # --- Internal state for reward calculation ---
        self.last_manager_actions: Dict[int, int] = {}
        self.last_worker_actions: Dict[int, int] = {}
        self.last_valid_options_maps: Dict[int, Dict] = {}
        self.last_manager_workload_magnitudes: Dict[int, Dict[str, float]] = {}


    def _define_spaces(self):
        """Defines observation and action spaces based on the mode."""
        d_option_feat = self.cluster_manager_ma.D_OPTION_FEAT
        D_GLOBAL = 4

        # The action space is the same in both modes: choose one of N DCs.
        self._manager_action_space = spaces.Discrete(self.num_dcs)

        if self.simple_obs_mode:
            # For the simple mode, calculate the total dimension of the flattened vector.
            # This is exactly what you described: MetaTask + (NumDCs * DCState) + Global
            total_obs_dim = D_META_MANAGER + (self.num_dcs * d_option_feat) + D_GLOBAL
            self._manager_obs_space = spaces.Box(
                low=-100, high=100, shape=(total_obs_dim,), dtype=np.float32
            )
        else:
            # Original, complex observation space for attention networks
            max_options = self.cluster_manager_ma.max_total_options
            self._manager_obs_space = spaces.Dict({
                "obs_manager_meta_task_i": spaces.Box(low=-np.inf, high=np.inf, shape=(D_META_MANAGER,), dtype=np.float32),
                "obs_all_options_set_padded": spaces.Box(low=-np.inf, high=np.inf, shape=(max_options, d_option_feat), dtype=np.float32),
                "all_options_padding_mask": spaces.Box(low=0, high=1, shape=(max_options,), dtype=np.bool_),
                "global_context": spaces.Box(low=-1.0, high=1.0, shape=(D_GLOBAL,), dtype=np.float32)
            })


        # Worker space remains the same for now, but could also be simplified if needed.
        D_LOCAL_WORKER_STATE = 5
        self._worker_obs_space = spaces.Dict({
            "obs_worker_meta_task_i": spaces.Box(low=-np.inf, high=np.inf, shape=(D_META_WORKER,), dtype=np.float32),
            "obs_local_dc_i_for_worker": spaces.Box(low=-np.inf, high=np.inf, shape=(D_LOCAL_WORKER_STATE,), dtype=np.float32),
            "global_context": spaces.Box(low=-1.0, high=1.0, shape=(D_GLOBAL,), dtype=np.float32)
        })
        self._worker_action_space = spaces.Discrete(2)


    def observation_space(self, agent: str) -> spaces.Space:
        """Returns the observation space for a single agent."""
        if agent.startswith("manager"):
            return self._manager_obs_space
        elif agent.startswith("worker"):
            return self._worker_obs_space
        raise ValueError(f"Unknown agent type for agent '{agent}'")

    def action_space(self, agent: str) -> spaces.Space:
        """Returns the action space for a single agent."""
        if agent.startswith("manager"):
            return self._manager_action_space
        elif agent.startswith("worker"):
            return self._worker_action_space
        raise ValueError(f"Unknown agent type for agent '{agent}'")

    def _get_time_features(self) -> np.ndarray:
        """Calculates sine/cosine time features for the current time."""
        day_of_year = self.current_time.dayofyear
        hour_of_day = self.current_time.hour + self.current_time.minute / 60.0
        return np.array([
            np.sin(2 * np.pi * day_of_year / 365.0), np.cos(2 * np.pi * day_of_year / 365.0),
            np.sin(2 * np.pi * hour_of_day / 24.0), np.cos(2 * np.pi * hour_of_day / 24.0)
        ], dtype=np.float32)

    def _get_complex_observations(self) -> Dict[str, Any]:
        """Gathers the original, complex dictionary-based observations."""
        # This is the old _get_observations method
        full_manager_obs_data = self.cluster_manager_ma._prepare_all_manager_observations(self.current_time)
        
        self.last_valid_options_maps = full_manager_obs_data["valid_options_maps"]
        self.last_manager_workload_magnitudes = full_manager_obs_data["workload_magnitudes"]

        manager_obs_raw = full_manager_obs_data["observations"]
        gctx = self._get_time_features()
        
        obs_dict = {f"manager_{dc}": {**manager_obs_raw[dc], "global_context": gctx} for dc in self._dc_ids}
        
        for dc_id in self._dc_ids:
            worker_obs_parts = self.cluster_manager_ma.nodes[dc_id].prepare_worker_observation(self.current_time)
            worker_obs_parts['global_context'] = gctx
            obs_dict[f"worker_{dc_id}"] = worker_obs_parts
            
        return obs_dict
    
    def _get_simple_observations(self) -> Dict[str, np.ndarray]: # === SIMPLIFICATION CHANGE: New Method ===
        """Gathers and flattens observations into a single vector for each manager."""
        full_manager_obs_data = self.cluster_manager_ma._prepare_all_manager_observations(self.current_time)
        
        # Cache the helper data that defines the context of the current state.
        self.last_valid_options_maps = full_manager_obs_data["valid_options_maps"]
        self.last_manager_workload_magnitudes = full_manager_obs_data["workload_magnitudes"]

        # Prepare the final observation dictionary for the agents' policies.
        manager_obs_raw = full_manager_obs_data["observations"]
        gctx = self._get_time_features()
        
        obs_dict = {}
        
        # Get all DC option features just once
        # Assuming the options are always in the same canonical order [DC1, DC2, DC3]
        all_option_features = []
        # We can get this from the map for the first DC, as the remote parts are the same for all.
        options_map_for_first_dc = self.last_valid_options_maps[self._dc_ids[0]]
        for i in range(self.num_dcs):
            # Extract the feature vector for each DC option
            option_info = options_map_for_first_dc[i]
            # Use the canonical feature order defined in the cluster manager
            feature_vector = [option_info[key] for key in self.cluster_manager_ma.DESTINATION_OPTION_FEATURE_ORDER]
            all_option_features.append(np.array(feature_vector, dtype=np.float32))

        flat_all_options = np.concatenate(all_option_features)
        
        
        # Now, create the specific flat vector for each manager
        for dc_id in self._dc_ids:
            # 1. Get the manager's specific meta-task
            meta_task_vec = manager_obs_raw[dc_id]["obs_manager_meta_task_i"]
            
            # 2. Get the global context
            # (already have gctx)
            
            # 3. Get the features for all destination options
            # (already have flat_all_options)
            
            # 4. Concatenate everything into a single flat vector
            flat_obs = np.concatenate([
                meta_task_vec,
                gctx,
                flat_all_options
            ]).astype(np.float32)
            
            obs_dict[f"manager_{dc_id}"] = flat_obs
            
            # Add worker observations (still in dict format for now)
            worker_obs_parts = self.cluster_manager_ma.nodes[dc_id].prepare_worker_observation(self.current_time)
            worker_obs_parts['global_context'] = gctx
            obs_dict[f"worker_{dc_id}"] = worker_obs_parts
        
        return obs_dict

    def _get_observations(self) -> Dict[str, Any]:
        """
        Dispatcher method that gets observations in the format specified by the mode.
        """
        # This part is the same for both modes: get all the component data.
        full_manager_obs_data = self.cluster_manager_ma._prepare_all_manager_observations(self.current_time)
    
        self.last_valid_options_maps = full_manager_obs_data["valid_options_maps"]
        self.last_manager_workload_magnitudes = full_manager_obs_data["workload_magnitudes"]

        manager_obs_components = full_manager_obs_data["observations"]
        gctx = self._get_time_features()
        obs_dict = {}

        # Get the base state of all DCs just once (without the 'is_local' flag yet)
        # We can get this from the options map of any DC.
        options_map_template = self.last_valid_options_maps[self._dc_ids[0]]

        if self.simple_obs_mode:
            # === This block implements YOUR desired logic ===
            # First, get the states of all DCs just once.
            # Now, loop through each agent to build its unique observation
            for acting_dc_id in self._dc_ids:
                # 1. Get this agent's specific meta-task vector
                meta_task_vec = manager_obs_components[acting_dc_id]["obs_manager_meta_task_i"]
                
                # Reescale the meta-task vector if needed
                # Divide by 100.0 the second element of the meta-task vector. CPU Requirement
                meta_task_vec[1] /= 100.0
                
                # Divide by 100.0 the third element of the meta-task vector. Memor Requirement
                meta_task_vec[3] /= 100.0
                
                # --- START OF CORRECTED LOGIC ---
                # 2. Build the cluster state vector FROM THE PERSPECTIVE of this agent
                agent_specific_dc_states = []
                for i, dest_dc_id in enumerate(self._dc_ids):
                    # Get the generic features for the destination DC
                    option_info = options_map_template[i]
                    feature_vector = [option_info[key] for key in self.cluster_manager_ma.DESTINATION_OPTION_FEATURE_ORDER]
                    
                    # Now, correctly set the 'is_local' flag for this specific agent.
                    # The first feature in DESTINATION_OPTION_FEATURE_ORDER must be 'is_local'.
                    if acting_dc_id == dest_dc_id:
                        feature_vector[0] = 1.0 # This IS the local DC
                    else:
                        feature_vector[0] = 0.0 # This is a remote DC
                        
                    agent_specific_dc_states.append(np.array(feature_vector, dtype=np.float32))
                    
                flat_all_dc_states_for_this_agent = np.concatenate(agent_specific_dc_states)
                # --- END OF CORRECTED LOGIC ---

                # 3. Concatenate all parts into the final flat vector
                flat_obs = np.concatenate([
                    meta_task_vec,
                    flat_all_dc_states_for_this_agent,
                    gctx
                ]).astype(np.float32)
                
                obs_dict[f"manager_{acting_dc_id}"] = flat_obs

        else:
            # Original complex mode
            for dc_id in self._dc_ids:
                obs_dict[f"manager_{dc_id}"] = {**manager_obs_components[dc_id], "global_context": gctx}
                
        return obs_dict
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Resets the environment to an initial state and returns the first observation.
        """
        super().reset(seed=seed)
        self.current_time = self.start_time
        
        # Set numpy random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            
        
        # Random init day in month 7 (day 30*7)
        month_day = np.random.randint(1, 31)
        init_day = month_day + 30 * 6  # July is the 7th month, so we start from day 1 to 30
        init_hour = np.random.randint(0, 24)
        
        self.cluster_manager_ma.reset(self.start_time.year, init_day, init_hour, seed)
        self.agents = self.possible_agents[:] # Reset active agents
        
        # Transform self.start_time.year, init_day, init_hour to the self.current_time
        self.current_time = pd.Timestamp(year=self.start_time.year, month=7, day=month_day, hour=init_hour, minute=0, second=0)
        # Add TZ to self.current_time
        self.current_time = self.current_time.tz_localize('UTC')  # Assuming UTC
        self.start_time = self.current_time
        self.end_time = self.current_time + self.simulation_duration
        
        # CRITICAL: Generate tasks for the initial state (t=0) BEFORE getting the first observation.
        self.cluster_manager_ma.task_origination(self.current_time, use_meta_tasks=self.use_meta_tasks)
        
        if self.logger:
            self.logger.info(f"SustainClusterMAEnv reset. Start time: {self.current_time}, Seed: {seed}")
        # print(f"SustainClusterMAEnv reset. Start time: {self.current_time}, Seed: {seed} on sustaincluster_ma_env.py.")
            
        # Get the observation of the complete initial state.
        initial_obs = self._get_observations()
        return initial_obs, {agent_id: {} for agent_id in self.agents}

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        The single, unified step function. It takes all agent actions, processes
        the entire 15-minute timestep, and returns the complete transition tuple.
        """
        # --- 1. Store Actions and Apply Decisions for the Current Timestep `t` ---
        # The context for these actions (workload magnitudes, valid options) was
        # already cached when the agent requested the observation for this state.
        self.last_manager_actions = {int(k.split('_')[1]): v for k, v in actions.items() if k.startswith("manager")}
        self.last_worker_actions = {int(k.split('_')[1]): v for k, v in actions.items() if k.startswith("worker")}

        filtered_mgr_acts = {dc: act for dc, act in self.last_manager_actions.items() if self.last_manager_workload_magnitudes[dc]['num_tasks'] > 0}
        if filtered_mgr_acts:
            self.cluster_manager_ma.step_manager(self.current_time, filtered_mgr_acts, self.last_valid_options_maps)

        self.cluster_manager_ma.step_worker(self.current_time, self.last_worker_actions)
        
        # --- 2. Advance the Physical Simulation and Calculate Reward for `(s_t, a_t)` ---
        results = self.cluster_manager_ma.step_physics(self.current_time)
        results['nodes'] = self.cluster_manager_ma.nodes

        per_agent_rewards = self.reward_fn(
            cluster_info=results,
            manager_actions=self.last_manager_actions,
            worker_actions=self.last_worker_actions,
            valid_options_maps=self.last_valid_options_maps,
            workload_magnitudes=self.last_manager_workload_magnitudes
        )
        
        # If the reward function returns a float (global reward), convert to per-agent dict
        if isinstance(per_agent_rewards, float) or isinstance(per_agent_rewards, int):
            rewards = {agent_id: float(per_agent_rewards) for agent_id in self.agents}
        else:
            rewards = per_agent_rewards
        
        for agent_id in self.agents:
            if agent_id not in rewards:
                rewards[agent_id] = 0.0

        # --- 3. Transition to the Next State `s_{t+1}` ---
        self.current_time += self.time_step
        
        # Generate new tasks that arrive at the beginning of the next timestep.
        self.cluster_manager_ma.task_origination(self.current_time, use_meta_tasks=self.use_meta_tasks)

        # Get the complete observation for this new state, s_{t+1}.
        next_observations = self._get_observations()

        # --- 4. Prepare and Return Final Transition Tuple ---
        terminated = self.current_time >= self.end_time
        # terminations = {agent_id: terminated for agent_id in self.agents}
        # terminations["__all__"] = terminated
        # truncations = terminations.copy()
        
        # Extract terminations and truncations from the results dictionary
        terminations = results.get('terminateds', {agent_id: False for agent_id in self.agents})
        truncations = results.get('truncateds', {agent_id: False for agent_id in self.agents})
        
        # Also add the __all key to both terminations and truncations
        terminations["__all__"] = results['datacenter_infos'][1].get('terminateds')['__all__']
        truncations["__all__"]  = results['datacenter_infos'][1].get('truncateds')['__all__']

        if terminations["__all__"] or truncations["__all__"] or terminated:
            self.agents = []

        infos = {agent_id: {} for agent_id in self.possible_agents}
        infos["__global__"] = {"raw_results": results}

        return next_observations, rewards, terminations, truncations, infos

    def render(self, mode='human'):
        """Renders the environment."""
        if mode == 'human':
            print(f"--- Timestep: {self.current_time} ---")
            for dc_id, node in self.cluster_manager_ma.nodes.items():
                print(f"  DC {dc_id} ({node.location}):")
                print(f"    Originating Queue: {len(node.originating_tasks_queue)} tasks")
                print(f"    Worker Queue: {len(node.worker_commitment_queue)} tasks")
                print(f"    Running Tasks: {len(node.physical_dc_model.running_tasks)}")
                print(f"    Available Cores: {node.physical_dc_model.available_cores:.1f}")
        else:
            super().render(mode=mode)

    def close(self):
        """Clean up any resources."""
        pass
