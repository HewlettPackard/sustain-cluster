# sustaincluster_ma_env_llm_speed_optimized.py - SPEED PRIORITIZED VERSION
# Optimized for high-performance training with reliable LLM servers

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

from simulation.cluster_manager_ma import DatacenterClusterManagerMA
from rewards.base_reward import BaseReward
from utils.marl_utils import D_META_MANAGER, D_META_WORKER
from envs.sync_llm_client_speed_optimized import create_speed_optimized_sync_llm_client
from utils.marl_utils import D_META_MANAGER, D_META_WORKER

logger = logging.getLogger(__name__)

class SpeedOptimizedLLMIntegratedSustainClusterMAEnv(gym.Env):
    """
    SPEED-OPTIMIZED: Enhanced SustainCluster-MA environment with fast LLM integration
    Prioritizes speed while maintaining all core functionality
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 cluster_manager_ma: DatacenterClusterManagerMA,
                 start_time: pd.Timestamp,
                 end_time: pd.Timestamp,
                 reward_fn: BaseReward,
                 logger: Optional[Any] = None,
                 simple_obs_mode: bool = False,
                 use_meta_tasks: bool = False,
                 # Training mode specification
                 training_mode: str = "manager",
                 # SPEED-OPTIMIZED: LLM Integration Parameters
                 enable_llm_advice: bool = True,
                 llm_service_url: str = "http://10.93.232.106:8000",
                 llm_timeout: float = 2.0,  # REDUCED: 2.0s for speed
                 llm_max_concurrent: int = 64,  # INCREASED: for high throughput
                 llm_history_window: int = 8,  # REDUCED: 10 → 8 for speed
                 num_envs: int = 8):  # NEW: For thread pool optimization
        
        super().__init__()
        self.cluster_manager_ma = cluster_manager_ma
        self.logger = logger
        self.start_time = start_time
        self.end_time = end_time
        self.simulation_duration = end_time - start_time
        self.current_time = self.start_time
        self.time_step = pd.Timedelta(minutes=15)
        self.reward_fn = reward_fn
        self.simple_obs_mode = simple_obs_mode
        self.use_meta_tasks = use_meta_tasks
        self.num_envs = num_envs

        # Training mode specification
        self.training_mode = training_mode.lower()
        if self.training_mode not in ["manager", "worker", "both"]:
            raise ValueError(f"Invalid training_mode: {training_mode}")

        self.num_dcs = self.cluster_manager_ma.num_dcs
        self._dc_ids = sorted(self.cluster_manager_ma.nodes.keys())

        # SPEED-OPTIMIZED: LLM Integration
        self.enable_llm_advice = enable_llm_advice
        self.llm_client = None
        self.llm_service_url = llm_service_url
        self.llm_timeout = llm_timeout
        self.llm_max_concurrent = llm_max_concurrent
        self.llm_history_window = llm_history_window
        
        # SPEED-OPTIMIZED: Minimal caching for performance
        self.last_llm_advice = {}
        self.llm_request_count = 0
        self.llm_success_count = 0
        
        # LLM advice dimensions
        self.D_LLM_ADVICE = 10

        # Agent definitions
        self.manager_agent_ids = [f"manager_{dc_id}" for dc_id in self._dc_ids]
        self.worker_agent_ids = [f"worker_{dc_id}" for dc_id in self._dc_ids]
        self.possible_agents = self.manager_agent_ids + self.worker_agent_ids
        self.agents = []

        # Define observation and action spaces
        self._manager_action_space = None
        self._manager_obs_space = None
        self._worker_obs_space = None
        self._worker_action_space = None
        self._define_spaces()

        # Internal state for reward calculation
        self.last_manager_actions: Dict[int, int] = {}
        self.last_worker_actions: Dict[int, int] = {}
        self.last_valid_options_maps: Dict[int, Dict] = {}
        self.last_manager_workload_magnitudes: Dict[int, Dict[str, float]] = {}

        if self.enable_llm_advice:
            if self.logger:
                self.logger.info(f"🚀 SPEED-OPTIMIZED LLM integration enabled:")
                self.logger.info(f"  Training mode: {self.training_mode}")
                self.logger.info(f"  Service URL: {llm_service_url}")
                self.logger.info(f"  Timeout: {llm_timeout}s (fast)")
                self.logger.info(f"  Max concurrent: {llm_max_concurrent} (high)")
                self.logger.info(f"  History window: {llm_history_window} (optimized)")
                self.logger.info(f"  Thread optimization for {num_envs} environments")

    def _define_spaces(self):
        """Enhanced space definitions with LLM advice integration and dimension verification."""
        d_option_feat = self.cluster_manager_ma.D_OPTION_FEAT  # Should be 5
        D_GLOBAL = 4

        # Manager action space
        self._manager_action_space = spaces.Discrete(self.num_dcs)

        if self.simple_obs_mode:
            # Enhanced simple mode: Original obs + LLM advice
            base_obs_dim = D_META_MANAGER + (self.num_dcs * d_option_feat) + D_GLOBAL
            
            if self.enable_llm_advice:
                total_obs_dim = base_obs_dim + self.D_LLM_ADVICE
            else:
                total_obs_dim = base_obs_dim
                
            self._manager_obs_space = spaces.Box(
                low=-100, high=100, shape=(total_obs_dim,), dtype=np.float32
            )
            
            # ✅ FIX: Log dimension calculation for verification
            if self.logger:
                self.logger.info(f"Manager observation space dimensions:")
                self.logger.info(f"  Meta-task: {D_META_MANAGER}")
                self.logger.info(f"  DC options: {self.num_dcs} × {d_option_feat} = {self.num_dcs * d_option_feat}")
                self.logger.info(f"  Global context: {D_GLOBAL}")
                self.logger.info(f"  LLM advice: {self.D_LLM_ADVICE if self.enable_llm_advice else 0}")
                self.logger.info(f"  Total: {total_obs_dim}")
        else:
            # Enhanced complex mode: Add LLM advice to dictionary
            obs_dict = {
                "obs_manager_meta_task_i": spaces.Box(low=-np.inf, high=np.inf, shape=(D_META_MANAGER,), dtype=np.float32),
                "obs_all_options_set_padded": spaces.Box(low=-np.inf, high=np.inf, 
                                                    shape=(self.cluster_manager_ma.max_total_options, d_option_feat), dtype=np.float32),
                "all_options_padding_mask": spaces.Box(low=0, high=1, 
                                                    shape=(self.cluster_manager_ma.max_total_options,), dtype=np.bool_),
                "global_context": spaces.Box(low=-1.0, high=1.0, shape=(D_GLOBAL,), dtype=np.float32)
            }
            
            if self.enable_llm_advice:
                obs_dict["llm_advice"] = spaces.Box(low=-1.0, high=1.0, shape=(self.D_LLM_ADVICE,), dtype=np.float32)
                
            self._manager_obs_space = spaces.Dict(obs_dict)

        # Enhanced worker observation space with dimension verification
        D_TASK_FEAT = 7
        D_DC_FEAT = 5
        worker_base_obs_dim = D_TASK_FEAT + D_DC_FEAT + D_GLOBAL
        
        if self.enable_llm_advice:
            worker_total_obs_dim = worker_base_obs_dim + self.D_LLM_ADVICE
        else:
            worker_total_obs_dim = worker_base_obs_dim
            
        # Worker observation: (num_dcs, obs_dim) matrix format
        self._worker_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.num_dcs, worker_total_obs_dim), 
            dtype=np.float32
        )
        self._worker_action_space = spaces.Discrete(2)
        
        # ✅ FIX: Log worker dimension calculation
        if self.logger:
            self.logger.info(f"Worker observation space dimensions:")
            self.logger.info(f"  Task features per DC: {D_TASK_FEAT}")
            self.logger.info(f"  DC features per DC: {D_DC_FEAT}")
            self.logger.info(f"  Global context per DC: {D_GLOBAL}")
            self.logger.info(f"  LLM advice per DC: {self.D_LLM_ADVICE if self.enable_llm_advice else 0}")
            self.logger.info(f"  Total per DC: {worker_total_obs_dim}")
            self.logger.info(f"  Shape: ({self.num_dcs}, {worker_total_obs_dim})")

    def _initialize_llm_client(self):
        """SPEED-OPTIMIZED: Initialize the fast sync LLM client."""
        if not self.enable_llm_advice or self.llm_client is not None:
            return
            
        try:
            self.llm_client = create_speed_optimized_sync_llm_client(
                service_url=self.llm_service_url,
                timeout=self.llm_timeout,
                max_concurrent=self.llm_max_concurrent,
                max_retries=2,  # REDUCED for speed
                history_window=self.llm_history_window,
                num_envs=self.num_envs  # NEW: For thread optimization
            )
            self.llm_client.initialize()
            if self.logger:
                self.logger.info("✅ Speed-optimized sync LLM client initialized")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize speed-optimized LLM client: {e}")
            self.enable_llm_advice = False

    def _get_llm_advice(self, observations: Dict) -> Dict:
        """SPEED-OPTIMIZED: Get LLM advice with minimal overhead."""
        if not self.enable_llm_advice or self.llm_client is None:
            return self._get_fallback_llm_advice()

        try:
            self.llm_request_count += 1
            
            combined_advice = {
                "manager": {},
                "worker": {},
                "metadata": {}
            }
            
            # SPEED-OPTIMIZED: Only query relevant LLM based on training mode
            if self.training_mode in ["manager", "both"]:
                # Get manager advice
                manager_obs = {k: v for k, v in observations.items() if k.startswith("manager_")}
                if manager_obs:
                    manager_advice = self.llm_client.get_llm_advice_sync(
                        observations=manager_obs,
                        agent_type="manager", 
                        dc_ids=[str(dc_id) for dc_id in self._dc_ids],
                        use_context=True
                    )
                    combined_advice["manager"] = manager_advice.get("manager", {})
                    combined_advice["metadata"].update(manager_advice.get("metadata", {}))

            if self.training_mode in ["worker", "both"]:
                # Get worker advice
                worker_obs = {k: v for k, v in observations.items() if k.startswith("worker_")}
                if worker_obs:
                    worker_advice = self.llm_client.get_llm_advice_sync(
                        observations=worker_obs,
                        agent_type="worker",
                        dc_ids=[str(dc_id) for dc_id in self._dc_ids], 
                        use_context=True
                    )
                    combined_advice["worker"] = worker_advice.get("worker", {})
                    # Merge metadata
                    worker_metadata = worker_advice.get("metadata", {})
                    if worker_metadata:
                        combined_advice["metadata"].update(worker_metadata)
            
            self.llm_success_count += 1
            # SPEED-OPTIMIZED: Only cache if performance is good
            if combined_advice.get("metadata", {}).get("service_type") != "fallback":
                self.last_llm_advice = combined_advice
            
            return combined_advice
            
        except Exception as e:
            if self.logger and logger.isEnabledFor(logging.WARNING):
                self.logger.warning(f"Fast LLM advice request failed: {e}")
            return self._get_fallback_llm_advice()

    def _get_fallback_llm_advice(self) -> Dict:
        """SPEED-OPTIMIZED: Fast fallback LLM advice generation."""
        # SPEED-OPTIMIZED: Quick cache check
        if self.last_llm_advice:
            return self.last_llm_advice
            
        # SPEED-OPTIMIZED: Pre-computed fallback
        return {
            "manager": {
                str(dc_id): {
                    "suggested_action": 1,
                    "confidence": 0.3,  # Higher for better fallback performance
                    "reasoning_embedding": [0.5] * 8,
                    "context_used": False
                } for dc_id in self._dc_ids
            },
            "worker": {
                str(dc_id): {
                    "suggested_action": 1,
                    "confidence": 0.3,
                    "reasoning_embedding": [0.5] * 8,
                    "context_used": False
                } for dc_id in self._dc_ids
            },
            "metadata": {
                "manager_response_time_ms": 1.0,
                "worker_response_time_ms": 1.0,
                "service_type": f"speed_optimized_fallback_{self.training_mode}",
                "context_enhanced": False
            }
        }

    def _extract_llm_advice_vector(self, llm_advice: Dict, agent_id: str) -> np.ndarray:
        """SPEED-OPTIMIZED: Fast LLM advice vector extraction with dimension verification."""
        if not self.enable_llm_advice:
            return np.zeros(self.D_LLM_ADVICE, dtype=np.float32)
            
        agent_type, dc_id = agent_id.split("_", 1)
        
        # SPEED-OPTIMIZED: Fast training mode check
        if self.training_mode == "manager" and agent_type == "worker":
            return np.zeros(self.D_LLM_ADVICE, dtype=np.float32)
        elif self.training_mode == "worker" and agent_type == "manager":
            return np.zeros(self.D_LLM_ADVICE, dtype=np.float32)
        
        advice_data = llm_advice.get(agent_type, {}).get(dc_id, {})
        
        if not advice_data:
            return np.zeros(self.D_LLM_ADVICE, dtype=np.float32)
        
        # SPEED-OPTIMIZED: Direct extraction with validation
        suggested_action = advice_data.get("suggested_action", 1)
        confidence = advice_data.get("confidence", 0.3)
        reasoning_embedding = advice_data.get("reasoning_embedding", [0.5] * 8)
        
        # ✅ FIX: Ensure exactly 8 reasoning dimensions
        if len(reasoning_embedding) != 8:
            if len(reasoning_embedding) > 8:
                reasoning_embedding = reasoning_embedding[:8]
            else:
                reasoning_embedding = reasoning_embedding + [0.5] * (8 - len(reasoning_embedding))
        
        # ✅ FIX: Ensure exactly D_LLM_ADVICE dimensions
        llm_vector = np.array([suggested_action, confidence] + reasoning_embedding, dtype=np.float32)
        
        # Verify dimension
        if len(llm_vector) != self.D_LLM_ADVICE:
            if self.logger:
                self.logger.warning(f"LLM advice dimension mismatch: got {len(llm_vector)}, expected {self.D_LLM_ADVICE}")
            # Pad or truncate to correct size
            if len(llm_vector) < self.D_LLM_ADVICE:
                llm_vector = np.pad(llm_vector, (0, self.D_LLM_ADVICE - len(llm_vector)), constant_values=0.5)
            else:
                llm_vector = llm_vector[:self.D_LLM_ADVICE]
        
        return llm_vector

    def observation_space(self, agent: str) -> spaces.Space:
        """Returns the enhanced observation space for a single agent."""
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
        """SPEED-OPTIMIZED: Fast time feature calculation."""
        day_of_year = self.current_time.dayofyear
        hour_of_day = self.current_time.hour + self.current_time.minute / 60.0
        
        # SPEED-OPTIMIZED: Pre-compute constants
        day_factor = 2 * np.pi / 365.0
        hour_factor = 2 * np.pi / 24.0
        
        return np.array([
            np.sin(day_factor * day_of_year), np.cos(day_factor * day_of_year),
            np.sin(hour_factor * hour_of_day), np.cos(hour_factor * hour_of_day)
        ], dtype=np.float32)

    def _get_observations(self) -> Dict[str, Any]:
        """SPEED-OPTIMIZED: Fast observation gathering with LLM advice integration and verification."""
        # Get base observations
        full_manager_obs_data = self.cluster_manager_ma._prepare_all_manager_observations(self.current_time)
        
        self.last_valid_options_maps = full_manager_obs_data["valid_options_maps"]
        self.last_manager_workload_magnitudes = full_manager_obs_data["workload_magnitudes"]

        manager_obs_components = full_manager_obs_data["observations"]
        gctx = self._get_time_features()
        
        # Prepare base observations
        base_obs_dict = {}
        
        # Manager observations
        if self.simple_obs_mode:
            options_map_template = self.last_valid_options_maps[self._dc_ids[0]]
            
            for acting_dc_id in self._dc_ids:
                meta_task_vec = manager_obs_components[acting_dc_id]["obs_manager_meta_task_i"].copy()
                meta_task_vec[1] *= 0.01  # SPEED-OPTIMIZED: Use multiplication instead of division
                meta_task_vec[3] *= 0.01
                
                # ✅ FIX: Verify meta_task_vec dimensions
                if len(meta_task_vec) != D_META_MANAGER:
                    if self.logger:
                        self.logger.warning(f"Meta-task vector dimension mismatch: got {len(meta_task_vec)}, expected {D_META_MANAGER}")
                    # Pad or truncate
                    if len(meta_task_vec) < D_META_MANAGER:
                        meta_task_vec = np.pad(meta_task_vec, (0, D_META_MANAGER - len(meta_task_vec)), constant_values=0.0)
                    else:
                        meta_task_vec = meta_task_vec[:D_META_MANAGER]
                
                # SPEED-OPTIMIZED: Pre-allocate arrays
                d_option_feat = len(self.cluster_manager_ma.DESTINATION_OPTION_FEATURE_ORDER)
                agent_specific_dc_states = np.zeros((self.num_dcs, d_option_feat), dtype=np.float32)
                
                for i, dest_dc_id in enumerate(self._dc_ids):
                    option_info = options_map_template[i]
                    for j, key in enumerate(self.cluster_manager_ma.DESTINATION_OPTION_FEATURE_ORDER):
                        agent_specific_dc_states[i, j] = option_info[key]
                    agent_specific_dc_states[i, 0] = 1.0 if acting_dc_id == dest_dc_id else 0.0
                    
                flat_all_dc_states = agent_specific_dc_states.flatten()
                base_obs = np.concatenate([meta_task_vec, flat_all_dc_states, gctx]).astype(np.float32)
                
                # ✅ FIX: Verify base observation dimensions
                expected_base_dim = D_META_MANAGER + (self.num_dcs * d_option_feat) + len(gctx)
                if len(base_obs) != expected_base_dim:
                    if self.logger:
                        self.logger.error(f"Base observation dimension error for manager_{acting_dc_id}: "
                                        f"got {len(base_obs)}, expected {expected_base_dim}")
                
                base_obs_dict[f"manager_{acting_dc_id}"] = base_obs
        else:
            for dc_id in self._dc_ids:
                base_obs_dict[f"manager_{dc_id}"] = {**manager_obs_components[dc_id], "global_context": gctx}
        
        # Worker observations - use matrix format with dimension verification
        D_TASK_FEAT = 7
        D_DC_FEAT = 5
        expected_worker_base_dim = D_TASK_FEAT + D_DC_FEAT + len(gctx)
        worker_obs_matrix = np.zeros((self.num_dcs, expected_worker_base_dim), dtype=np.float32)
        
        for i, dc_id in enumerate(self._dc_ids):
            # Get base worker observation components
            worker_obs_parts = self.cluster_manager_ma.nodes[dc_id].prepare_worker_observation(self.current_time)
            
            # SPEED-OPTIMIZED: Direct array assignment with validation
            if "obs_worker_meta_task_i" in worker_obs_parts:
                task_vec = worker_obs_parts["obs_worker_meta_task_i"][:D_TASK_FEAT]
                if len(task_vec) < D_TASK_FEAT:
                    task_vec = np.pad(task_vec, (0, D_TASK_FEAT - len(task_vec)), constant_values=0.0)
            else:
                task_vec = np.zeros(D_TASK_FEAT, dtype=np.float32)
                
            if "obs_local_dc_i_for_worker" in worker_obs_parts:
                dc_vec = worker_obs_parts["obs_local_dc_i_for_worker"][:D_DC_FEAT]
                if len(dc_vec) < D_DC_FEAT:
                    dc_vec = np.pad(dc_vec, (0, D_DC_FEAT - len(dc_vec)), constant_values=0.0)
            else:
                dc_vec = np.zeros(D_DC_FEAT, dtype=np.float32)
            
            # SPEED-OPTIMIZED: Direct concatenation with verification
            worker_obs_matrix[i, :D_TASK_FEAT] = task_vec
            worker_obs_matrix[i, D_TASK_FEAT:D_TASK_FEAT+D_DC_FEAT] = dc_vec
            worker_obs_matrix[i, D_TASK_FEAT+D_DC_FEAT:] = gctx

        base_obs_dict["worker_matrix"] = worker_obs_matrix

        # SPEED-OPTIMIZED: Get LLM advice only for relevant agent type
        llm_advice = self._get_llm_advice(base_obs_dict)
        
        # Integrate LLM advice into observations with dimension verification
        enhanced_obs_dict = {}
        
        # Process manager observations
        for agent_id, base_obs in base_obs_dict.items():
            if agent_id.startswith("manager"):
                if self.enable_llm_advice:
                    llm_vector = self._extract_llm_advice_vector(llm_advice, agent_id)
                    
                    if self.simple_obs_mode:
                        enhanced_obs = np.concatenate([base_obs, llm_vector]).astype(np.float32)
                        
                        # ✅ FIX: Final dimension verification
                        expected_total_dim = len(base_obs) + self.D_LLM_ADVICE
                        if len(enhanced_obs) != expected_total_dim:
                            if self.logger:
                                self.logger.error(f"Enhanced observation dimension error for {agent_id}: "
                                                f"got {len(enhanced_obs)}, expected {expected_total_dim}")
                        
                        enhanced_obs_dict[agent_id] = enhanced_obs
                    else:
                        enhanced_obs_dict[agent_id] = {**base_obs, "llm_advice": llm_vector}
                else:
                    enhanced_obs_dict[agent_id] = base_obs
        
        # Process worker observations (matrix format) with dimension verification
        if "worker_matrix" in base_obs_dict:
            worker_base_matrix = base_obs_dict["worker_matrix"]
            
            if self.enable_llm_advice:
                # SPEED-OPTIMIZED: Vectorized LLM advice integration
                enhanced_worker_matrix = np.zeros((self.num_dcs, worker_base_matrix.shape[1] + self.D_LLM_ADVICE), dtype=np.float32)
                enhanced_worker_matrix[:, :worker_base_matrix.shape[1]] = worker_base_matrix
                
                for i, dc_id in enumerate(self._dc_ids):
                    llm_vector = self._extract_llm_advice_vector(llm_advice, f"worker_{dc_id}")
                    enhanced_worker_matrix[i, worker_base_matrix.shape[1]:] = llm_vector
                
                # ✅ FIX: Verify final worker matrix dimensions
                expected_worker_total_dim = worker_base_matrix.shape[1] + self.D_LLM_ADVICE
                if enhanced_worker_matrix.shape[1] != expected_worker_total_dim:
                    if self.logger:
                        self.logger.error(f"Enhanced worker matrix dimension error: "
                                        f"got {enhanced_worker_matrix.shape[1]}, expected {expected_worker_total_dim}")
                
                enhanced_obs_dict["worker_matrix"] = enhanced_worker_matrix
            else:
                enhanced_obs_dict["worker_matrix"] = worker_base_matrix
        
        return enhanced_obs_dict
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """SPEED-OPTIMIZED: Fast reset with LLM client initialization."""
        super().reset(seed=seed)
        self.current_time = self.start_time
        
        if seed is not None:
            np.random.seed(seed)
            
        # Random initialization  
        month_day = np.random.randint(1, 31)
        init_day = month_day + 30 * 6
        init_hour = np.random.randint(0, 24)
        
        self.cluster_manager_ma.reset(self.start_time.year, init_day, init_hour, seed)
        self.agents = self.possible_agents[:]
        
        # Update current time
        self.current_time = pd.Timestamp(year=self.start_time.year, month=7, day=month_day, 
                                        hour=init_hour, minute=0, second=0, tz='UTC')
        self.start_time = self.current_time
        self.end_time = self.current_time + self.simulation_duration
        
        # SPEED-OPTIMIZED: Fast LLM client initialization
        if self.enable_llm_advice:
            try:
                self._initialize_llm_client()
                if self.llm_client:
                    self.llm_client.reset_episode()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to initialize speed-optimized LLM client: {e}")
                self.enable_llm_advice = False

        # Generate initial tasks
        self.cluster_manager_ma.task_origination(self.current_time, use_meta_tasks=self.use_meta_tasks)
        
        initial_obs = self._get_observations()
        return initial_obs, {agent_id: {} for agent_id in self.agents}

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """SPEED-OPTIMIZED: Fast step with minimal LLM context overhead."""
        # Store actions
        self.last_manager_actions = {int(k.split('_')[1]): v for k, v in actions.items() if k.startswith("manager")}
        self.last_worker_actions = {int(k.split('_')[1]): v for k, v in actions.items() if k.startswith("worker")}

        # Apply decisions
        filtered_mgr_acts = {dc: act for dc, act in self.last_manager_actions.items() 
                           if self.last_manager_workload_magnitudes[dc]['num_tasks'] > 0}
        if filtered_mgr_acts:
            self.cluster_manager_ma.step_manager(self.current_time, filtered_mgr_acts, self.last_valid_options_maps)

        self.cluster_manager_ma.step_worker(self.current_time, self.last_worker_actions)
        
        # Physical simulation and rewards
        results = self.cluster_manager_ma.step_physics(self.current_time)
        results['nodes'] = self.cluster_manager_ma.nodes

        per_agent_rewards = self.reward_fn(
            cluster_info=results,
            manager_actions=self.last_manager_actions,
            worker_actions=self.last_worker_actions,
            valid_options_maps=self.last_valid_options_maps,
            workload_magnitudes=self.last_manager_workload_magnitudes
        )
        
        if isinstance(per_agent_rewards, (float, int)):
            rewards = {agent_id: float(per_agent_rewards) for agent_id in self.agents}
        else:
            rewards = per_agent_rewards
        
        for agent_id in self.agents:
            if agent_id not in rewards:
                rewards[agent_id] = 0.0

        # SPEED-OPTIMIZED: Fast LLM context update
        if self.enable_llm_advice and self.llm_client:
            try:
                trust_scores = {agent_id: 0.7 for agent_id in self.agents}
                self.llm_client.update_context(
                    observations={},  # SPEED-OPTIMIZED: Skip observation serialization
                    actions=actions,
                    rewards=rewards,
                    trust_scores=trust_scores,
                    llm_advice=self.last_llm_advice
                )
            except Exception:
                pass  # SPEED-OPTIMIZED: Silent failure for speed

        # Advance time and generate new tasks
        self.current_time += self.time_step
        self.cluster_manager_ma.task_origination(self.current_time, use_meta_tasks=self.use_meta_tasks)

        # Get next observations
        next_observations = self._get_observations()

        # Termination logic
        terminated = self.current_time >= self.end_time
        terminations = results.get('terminateds', {agent_id: False for agent_id in self.agents})
        truncations = results.get('truncateds', {agent_id: False for agent_id in self.agents})
        
        terminations["__all__"] = results['datacenter_infos'][1].get('terminateds', {}).get('__all__', terminated)
        truncations["__all__"] = results['datacenter_infos'][1].get('truncateds', {}).get('__all__', terminated)

        if terminations["__all__"] or truncations["__all__"] or terminated:
            self.agents = []

        infos = {agent_id: {} for agent_id in self.possible_agents}
        infos["__global__"] = {"raw_results": results}
        
        # SPEED-OPTIMIZED: Minimal LLM stats
        if self.enable_llm_advice and self.llm_client and logger.isEnabledFor(logging.DEBUG):
            stats = self.llm_client.get_stats()
            infos["__global__"]["llm_stats"] = {
                "success_rate": stats["success_rate"],
                "training_mode": self.training_mode
            }

        return next_observations, rewards, terminations, truncations, infos

    def close(self):
        """SPEED-OPTIMIZED: Fast cleanup."""
        if self.llm_client:
            try:
                self.llm_client.close()
            except:
                pass
            if self.logger:
                self.logger.info("Speed-optimized sync LLM client closed")

    def render(self, mode='human'):
        """SPEED-OPTIMIZED: Minimal rendering for performance."""
        if mode == 'human':
            print(f"--- Timestep: {self.current_time} (Training: {self.training_mode}) ---")
            if self.enable_llm_advice and self.llm_client:
                stats = self.llm_client.get_stats()
                success_rate = stats["success_rate"]
                print(f"  Fast LLM ({self.training_mode}): {success_rate:.1%} success")


# SPEED-OPTIMIZED: Worker environment wrapper
class SpeedOptimizedLLMIntegratedWorkerEnv(gym.Env):
    """
    SPEED-OPTIMIZED: Worker environment wrapper with fast LLM integration
    """
    
    def __init__(self, base_worker_env, enable_llm_advice=True, 
                 llm_service_url="http://10.93.232.106:8000",
                 llm_timeout=4.0, llm_max_concurrent=32, llm_history_window=8, num_envs=8):
        super().__init__()
        
        self.base_env = base_worker_env
        self.enable_llm_advice = enable_llm_advice
        self.num_envs = num_envs
        
        # SPEED-OPTIMIZED: LLM client setup
        self.llm_client = None
        self.llm_service_url = llm_service_url
        self.llm_timeout = llm_timeout
        self.llm_max_concurrent = llm_max_concurrent
        self.llm_history_window = llm_history_window
        
        # LLM advice dimensions
        self.D_LLM_ADVICE = 10
        
        # Update observation space to include LLM advice
        base_obs_shape = self.base_env.observation_space.shape
        if self.enable_llm_advice:
            enhanced_obs_dim = base_obs_shape[1] + self.D_LLM_ADVICE
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(base_obs_shape[0], enhanced_obs_dim), 
                dtype=np.float32
            )
        else:
            self.observation_space = self.base_env.observation_space
            
        self.action_space = self.base_env.action_space
        
        # SPEED-OPTIMIZED: Minimal caching
        self.last_llm_advice = {}
        self.llm_request_count = 0
        self.llm_success_count = 0

    def _initialize_llm_client(self):
        """SPEED-OPTIMIZED: Initialize fast sync LLM client."""
        if not self.enable_llm_advice or self.llm_client is not None:
            return
            
        try:
            self.llm_client = create_speed_optimized_sync_llm_client(
                service_url=self.llm_service_url,
                timeout=self.llm_timeout,
                max_concurrent=self.llm_max_concurrent,
                max_retries=3,
                history_window=self.llm_history_window,
                num_envs=self.num_envs
            )
            self.llm_client.initialize()
        except Exception as e:
            print(f"Failed to initialize speed-optimized worker LLM client: {e}")
            self.enable_llm_advice = False

    def _get_llm_advice(self, base_observations):
        """SPEED-OPTIMIZED: Fast LLM advice for worker agents."""
        if not self.enable_llm_advice or self.llm_client is None:
            return self._get_fallback_llm_advice()

        try:
            self.llm_request_count += 1
            
            # Convert matrix observations to dict format
            worker_obs_dict = {f"worker_{dc_id}": base_observations[i] 
                             for i, dc_id in enumerate(self.base_env.dc_ids)}
            
            worker_advice = self.llm_client.get_llm_advice_sync(
                observations=worker_obs_dict,
                agent_type="worker",
                dc_ids=[str(dc_id) for dc_id in self.base_env.dc_ids],
                use_context=True
            )
            
            self.llm_success_count += 1
            self.last_llm_advice = worker_advice
            return worker_advice
            
        except Exception:
            return self._get_fallback_llm_advice()

    def _get_fallback_llm_advice(self):
        """SPEED-OPTIMIZED: Fast fallback advice."""
        if self.last_llm_advice:
            return self.last_llm_advice
            
        return {
            "worker": {
                str(dc_id): {
                    "suggested_action": 1,
                    "confidence": 0.3,
                    "reasoning_embedding": [0.5] * 8,
                    "context_used": False
                } for dc_id in self.base_env.dc_ids
            },
            "metadata": {
                "service_type": "speed_optimized_fallback",
                "context_enhanced": False
            }
        }

    def _extract_llm_advice_vector(self, llm_advice, dc_id):
        """SPEED-OPTIMIZED: Fast LLM advice vector extraction with dimension verification."""
        if not self.enable_llm_advice:
            return np.zeros(self.D_LLM_ADVICE, dtype=np.float32)
            
        advice_data = llm_advice.get("worker", {}).get(str(dc_id), {})
        
        if not advice_data:
            return np.zeros(self.D_LLM_ADVICE, dtype=np.float32)
        
        # SPEED-OPTIMIZED: Direct extraction with validation
        suggested_action = advice_data.get("suggested_action", 1)
        confidence = advice_data.get("confidence", 0.3)
        reasoning_embedding = advice_data.get("reasoning_embedding", [0.5] * 8)
        
        # ✅ FIX: Ensure exactly 8 reasoning dimensions
        if len(reasoning_embedding) != 8:
            if len(reasoning_embedding) > 8:
                reasoning_embedding = reasoning_embedding[:8]
            else:
                reasoning_embedding = reasoning_embedding + [0.5] * (8 - len(reasoning_embedding))
        
        # ✅ FIX: Ensure exactly D_LLM_ADVICE dimensions
        llm_vector = np.array([suggested_action, confidence] + reasoning_embedding, dtype=np.float32)
        
        # Verify dimension
        if len(llm_vector) != self.D_LLM_ADVICE:
            print(f"⚠️ Worker LLM advice dimension mismatch: got {len(llm_vector)}, expected {self.D_LLM_ADVICE}")
            # Pad or truncate to correct size
            if len(llm_vector) < self.D_LLM_ADVICE:
                llm_vector = np.pad(llm_vector, (0, self.D_LLM_ADVICE - len(llm_vector)), constant_values=0.5)
            else:
                llm_vector = llm_vector[:self.D_LLM_ADVICE]
        
        return llm_vector

    def reset(self, **kwargs):
        """SPEED-OPTIMIZED: Fast reset with LLM advice integration."""
        if self.enable_llm_advice and self.llm_client is None:
            self._initialize_llm_client()

        base_obs, info = self.base_env.reset(**kwargs)
        
        if not self.enable_llm_advice:
            return base_obs, info
        
        llm_advice = self._get_llm_advice(base_obs)
        
        # SPEED-OPTIMIZED: Vectorized enhancement
        enhanced_obs = np.zeros((base_obs.shape[0], base_obs.shape[1] + self.D_LLM_ADVICE), dtype=np.float32)
        enhanced_obs[:, :base_obs.shape[1]] = base_obs
        
        for i, dc_id in enumerate(self.base_env.dc_ids):
            llm_vector = self._extract_llm_advice_vector(llm_advice, dc_id)
            enhanced_obs[i, base_obs.shape[1]:] = llm_vector
        
        return enhanced_obs, info

    def step(self, action):
        """SPEED-OPTIMIZED: Fast step with LLM advice integration."""
        base_obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        if not self.enable_llm_advice:
            return base_obs, reward, terminated, truncated, info
        
        llm_advice = self._get_llm_advice(base_obs)
        
        # SPEED-OPTIMIZED: Vectorized enhancement
        enhanced_obs = np.zeros((base_obs.shape[0], base_obs.shape[1] + self.D_LLM_ADVICE), dtype=np.float32)
        enhanced_obs[:, :base_obs.shape[1]] = base_obs
        
        for i, dc_id in enumerate(self.base_env.dc_ids):
            llm_vector = self._extract_llm_advice_vector(llm_advice, dc_id)
            enhanced_obs[i, base_obs.shape[1]:] = llm_vector
        
        # SPEED-OPTIMIZED: Fire-and-forget context update
        if self.llm_client:
            try:
                action_dict = {f"worker_{dc_id}": action[i] for i, dc_id in enumerate(self.base_env.dc_ids)}
                trust_scores = {f"worker_{dc_id}": 0.7 for dc_id in self.base_env.dc_ids}
                
                self.llm_client.update_context(
                    observations={},
                    actions=action_dict,
                    rewards={"worker": reward},
                    trust_scores=trust_scores,
                    llm_advice=llm_advice
                )
            except Exception:
                pass  # Silent failure for speed
        
        return enhanced_obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.base_env.render(mode)

    def close(self):
        """SPEED-OPTIMIZED: Fast close."""
        if self.llm_client:
            try:
                self.llm_client.close()
            except:
                pass
        self.base_env.close()