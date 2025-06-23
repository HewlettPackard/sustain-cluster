# simulation/cluster_manager_ma.py

import os
import zipfile
from collections import deque
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd

from envs.datacenter_node_ma import DatacenterNodeMA
from utils.workload_utils import extract_tasks_from_row
from data.network_cost.network_delay import get_transmission_delay
from utils.transmission_cost_loader import load_transmission_matrix
from utils.transmission_region_mapper import map_location_to_region
from rl_components.task import Task

class DatacenterClusterManagerMA:
    """
    Orchestrates the entire multi-agent simulation for SustainCluster-MA.

    This manager is responsible for:
    - Initializing and managing a collection of DatacenterNodeMA instances.
    - Driving the global 15-minute timestep loop.
    - Handling task origination and routing them to the correct DC node.
    - Managing inter-datacenter task transfers, including delays and costs.
    - Assembling the complex observations required by each DTA_Manager by
      querying all other nodes.
    - Collecting results and metrics from all nodes at each step.
    """
    def __init__(self, config_list: List[Dict], simulation_year: int, tasks_file_path: str,
                 cloud_provider: str, max_total_options: int, logger: Any = None):
        """
        Initializes the multi-agent cluster manager.

        Args:
            config_list (List[Dict]): A list of configuration dictionaries, one for each DC.
            simulation_year (int): The year for which to load environmental data.
            tasks_file_path (str): Path to the workload trace file (.pkl).
            cloud_provider (str): The cloud provider ('aws', 'gcp', 'azure') for network costs.
            max_total_options (int): The fixed size for the DTA_Manager's destination options
                                     set (for padding). Must be >= number of DCs.
            logger (Any, optional): A logger instance.
        """
        self.logger = logger
        self.nodes: Dict[int, DatacenterNodeMA] = {
            cfg['dc_id']: DatacenterNodeMA(cfg, self.logger) for cfg in config_list
        }
        self.num_dcs = len(self.nodes)
        
        # Create a single, globally consistent, sorted list of DC IDs.
        # This will be the "canonical" order for the entire simulation.
        self._canonical_dc_order: List[int] = sorted(self.nodes.keys())
        
        if max_total_options < self.num_dcs:
            raise ValueError(f"max_total_options ({max_total_options}) must be >= number of datacenters ({self.num_dcs})")
        self.max_total_options = max_total_options

        self.tasks_df = self._load_tasks(tasks_file_path)
        self.in_transit_tasks: deque[Tuple[pd.Timestamp, Task, int]] = deque() # (arrival_time, task, dest_dc_id)

        # Network models
        self.cloud_provider = cloud_provider
        self.transmission_cost_matrix = load_transmission_matrix(cloud_provider)
        
        # Define the fixed feature order for a destination option
        self.DESTINATION_OPTION_FEATURE_ORDER = [
            "is_local",
            "cpu_avail_pct",
            "gpu_avail_pct",
            "ci",
            "external_temperature" # New
        ]
        self.D_OPTION_FEAT = len(self.DESTINATION_OPTION_FEATURE_ORDER)

        if self.logger:
            self.logger.info(f"DatacenterClusterManagerMA initialized with {self.num_dcs} nodes. DTA_Manager option padding size: {self.max_total_options}.")

    def _load_tasks(self, tasks_file_path: str) -> pd.DataFrame:
        """Loads workload data from a .pkl file, with a fallback to unzip."""
        if os.path.exists(tasks_file_path):
            if self.logger: self.logger.info(f"Loading workload from: {tasks_file_path}")
            df = pd.read_pickle(tasks_file_path)
        else:
            zip_path = tasks_file_path.replace(".pkl", ".zip")
            if os.path.exists(zip_path):
                if self.logger: self.logger.info(f"Workload .pkl not found. Extracting from: {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(tasks_file_path))
                if not os.path.exists(tasks_file_path):
                    raise FileNotFoundError(f"Failed to find {tasks_file_path} after unzipping.")
                df = pd.read_pickle(tasks_file_path)
            else:
                raise FileNotFoundError(f"Workload file not found at {tasks_file_path} or {zip_path}")
        
        df['interval_15m'] = df['interval_15m'].dt.tz_convert('UTC')
        return df

    def reset(self, init_year: int, init_day: int, init_hour: int, seed: int):
        """Resets all nodes and clears in-transit tasks."""
        for i, node in enumerate(self.nodes.values()):
            # Use a different seed for each node to ensure varied internal randomness
            node.reset(init_year, init_day, init_hour, seed + i)
        self.in_transit_tasks.clear()

    def _get_newly_arrived_tasks(self, current_time_utc: pd.Timestamp) -> Dict[int, List[Task]]:
        """Loads and assigns origins to tasks for the current timestep."""
        adjusted_time = current_time_utc.replace(year=2020)
        tasks_for_time = self.tasks_df[self.tasks_df['interval_15m'] == adjusted_time]
        
        if tasks_for_time.empty:
            return {}

        row = tasks_for_time.iloc[0]
        # extract_tasks_from_row now assigns origins internally
        # We need to get the list of datacenter configurations to pass to the
        # origin assignment logic. We can get this from our `self.nodes`.
        all_dc_configs = [node.physical_dc_model.env_config for node in self.nodes.values()]
        
        # Call extract_tasks_from_row with the required context
        all_new_tasks = extract_tasks_from_row(
            row,
            datacenter_configs=all_dc_configs,  # Pass the configs
            current_time_utc=current_time_utc,  # Pass the current time
            logger=self.logger
            # You can also pass scale, task_scale, group_size here if you make them configurable
        )

        # Group tasks by their assigned origin DC ID
        tasks_by_origin = {}
        for task in all_new_tasks:
            origin_id = task.origin_dc_id
            if origin_id not in tasks_by_origin:
                tasks_by_origin[origin_id] = []
            tasks_by_origin[origin_id].append(task)
        
        return tasks_by_origin


    def _prepare_all_manager_observations(self, current_time_utc: pd.Timestamp) -> Dict[str, Any]:
        """
        Assembles the full, complex observation for every DTA_Manager.

        This method ensures a globally consistent order for destination options,
        which is critical for training a shared policy. Action index 'k' will
        always correspond to the k-th datacenter in a sorted list of all DCs.
        The agent uses the 'is_local' feature to identify its own datacenter
        within this globally consistent list.

        Returns:
            A dictionary containing two keys:
            - "observations": A dict mapping each manager's dc_id to its observation data.
            - "valid_options_maps": A helper dict for mapping action indices to destination info.
        """
        all_observations = {}
        all_valid_options_maps = {}

        # 1. Gather the current state of all DC nodes at once for efficiency.
        remote_query_states = {dc_id: node.get_state_for_remote_query() for dc_id, node in self.nodes.items()}

        # 2. For each DTA_Manager, construct its unique observation.
        for dc_id, node in self.nodes.items():
            # Get the meta-task vector for tasks originating at this node.
            meta_task_vector = node.prepare_manager_observation(current_time_utc)["obs_manager_meta_task_i"]

            # This list will hold the feature dictionaries for each destination option.
            options_list = []

            # 3. Build the options list using the canonical (sorted) order.
            # This ensures that for EVERY manager, the option at index `k`
            # always corresponds to the same physical datacenter.
            for dest_dc_id in self._canonical_dc_order:
                dest_node = self.nodes[dest_dc_id]
                
                # Determine if this destination is the manager's own local DC.
                # This feature is now critical for the agent's context.
                is_local_flag = 1.0 if dest_dc_id == dc_id else 0.0

                # Get the state of the destination DC.
                destination_state = remote_query_states[dest_dc_id]
                
                # Set transmission features to zero if local, otherwise calculate them.
                if is_local_flag == 1.0:
                    cost_per_gb = 0.0
                    delay_s = 0.0
                else:
                    # For remote options, calculate network cost and delay.
                    # Using a 1.0 GB placeholder for delay calculation per GB.
                    delay_s = get_transmission_delay(node.location, dest_node.location, self.cloud_provider, 1.0)
                    origin_region = map_location_to_region(node.location, self.cloud_provider)
                    dest_region = map_location_to_region(dest_node.location, self.cloud_provider)
                    cost_per_gb = self.transmission_cost_matrix.loc[origin_region, dest_region]

                # Assemble all features for this destination option into a dictionary.
                option_features = {
                    "is_local": is_local_flag,
                    "dc_id": dest_dc_id,
                    "location": dest_node.location,
                    **destination_state,  # Unpacks dict items like 'ci', 'cpu_avail_pct'
                    "transmission_cost_per_gb": cost_per_gb,
                    "transmission_delay_s_per_gb": delay_s,
                    "external_temperature": dest_node.weather_manager.get_current_temperature(norm=False)
                }
                options_list.append(option_features)

            # 4. Convert the list of feature dicts into a padded NumPy array for the NN.
            num_valid_options = len(options_list)
            padded_options_array = np.zeros((self.max_total_options, self.D_OPTION_FEAT), dtype=np.float32)
            
            for i, option_dict in enumerate(options_list):
                # Build the feature vector in the correct, predefined order.
                # Use .get(key, 0.0) for safety in case a key is missing.
                feature_vector = [option_dict.get(key, 0.0) for key in self.DESTINATION_OPTION_FEATURE_ORDER]
                padded_options_array[i] = np.array(feature_vector, dtype=np.float32)

            # 5. Create the padding mask for the attention layer.
            # `True` means the position is masked (invalid/padded).
            mask = np.ones(self.max_total_options, dtype=bool)
            mask[:num_valid_options] = False  # Mark the actual options as valid (not masked).

            # 6. Assemble the final observation dictionary for this specific manager.
            all_observations[dc_id] = {
                "obs_manager_meta_task_i": meta_task_vector,
                "obs_all_options_set_padded": padded_options_array,
                "all_options_padding_mask": mask,
            }
            
            # Also store the helper map for robust action application later.
            all_valid_options_maps[dc_id] = {i: opt_dict for i, opt_dict in enumerate(options_list)}

        # 7. Return the final payload containing observations and helper maps.
        return {
            "observations": all_observations,
            "valid_options_maps": all_valid_options_maps
        }

    def step_marl(self, current_time_utc: pd.Timestamp,
                  manager_actions: Dict[int, int],
                  worker_actions: Dict[int, bool]) -> Dict[str, Any]:
        """
        Orchestrates one full 15-minute timestep in the MARL environment.
        This follows the detailed step-by-step plan.
        """
        # A. Update time in all nodes' data managers
        for node in self.nodes.values():
            node.ci_manager.step()
            node.price_manager.step()
            node.weather_manager.step()

        # B. Task Origination
        newly_arrived_tasks = self._get_newly_arrived_tasks(current_time_utc)
        for dc_id, tasks in newly_arrived_tasks.items():
            if dc_id in self.nodes:
                self.nodes[dc_id].add_originating_tasks(tasks)
        
        # C. Apply DTA_Manager Decisions
        # The actions are provided as input to this function.
        for dc_id, manager_action_idx in manager_actions.items():
            node = self.nodes[dc_id]
            # Need to map the action index back to a destination DC ID
            # We assume the order of options was [local, remote1, remote2, ...]
            all_dc_ids = sorted(self.nodes.keys())
            if manager_action_idx == 0:
                chosen_dest_id = dc_id # Local
            else:
                remote_ids = [other_id for other_id in all_dc_ids if other_id != dc_id]
                chosen_dest_id = remote_ids[manager_action_idx - 1]

            tasks_to_transfer = node.apply_manager_decision(chosen_dest_id)
            for task in tasks_to_transfer:
                # Calculate actual delay based on task bandwidth
                delay_s = get_transmission_delay(node.location, self.nodes[task.dest_dc_id].location, self.cloud_provider, task.bandwidth_gb)
                arrival_time = current_time_utc + pd.Timedelta(seconds=delay_s)
                self.in_transit_tasks.append((arrival_time, task, task.dest_dc_id))
        
        # D. Process Arriving Transferred Tasks
        remaining_in_transit = deque()
        while self.in_transit_tasks:
            arrival_time, task, dest_dc_id = self.in_transit_tasks.popleft()
            if arrival_time <= current_time_utc:
                self.nodes[dest_dc_id].add_transferred_tasks([task])
            else:
                remaining_in_transit.append((arrival_time, task, dest_dc_id))
        self.in_transit_tasks = remaining_in_transit

        # E. Apply DTA_Worker Decisions
        for dc_id, worker_action_execute in worker_actions.items():
            self.nodes[dc_id].apply_worker_decision(worker_action_execute, current_time_utc)

        # F. Simulate Physical DC Operations
        all_dc_infos = {}
        for dc_id, node in self.nodes.items():
            all_dc_infos[dc_id] = node.step_physical_simulation(current_time_utc)
        
        # G. Collect and return results (reward calculation will be done by the env wrapper)
        return {"datacenter_infos": all_dc_infos} # Return the raw info for now
    
    def task_origination(self, current_time_utc: pd.Timestamp):
        for node in self.nodes.values():
            node.ci_manager.step()
            node.price_manager.step()
            node.weather_manager.step()

        # B. Task Origination
        newly_arrived_tasks = self._get_newly_arrived_tasks(current_time_utc)
        for dc_id, tasks in newly_arrived_tasks.items():
            if dc_id in self.nodes:
                self.nodes[dc_id].add_originating_tasks(tasks)


    def step_manager(self, current_time_utc: pd.Timestamp,
                     manager_actions: Dict[int, int],
                     valid_options_maps: Dict[int, Dict]) -> None:
    
        for dc_id, manager_action_idx in manager_actions.items():
            node = self.nodes[dc_id]
            
            # ### --- THE FIX IS HERE --- ###
            # --- Start of new, corrected logic ---

            # 1. Get the correct map for this specific manager
            options_map_for_dc = valid_options_maps.get(dc_id)

            # 2. Safety check: ensure the map and action index are valid
            if not options_map_for_dc or manager_action_idx not in options_map_for_dc:
                if self.logger:
                    self.logger.error(f"FATAL: Invalid action index {manager_action_idx} for DC {dc_id}. "
                                    f"This indicates a mismatch between agent's action space and "
                                    f"the observation. Skipping manager action for this DC.")
                continue

            # 3. Robustly get the chosen destination's information from the map
            chosen_option_info = options_map_for_dc[manager_action_idx]

            # 4. Extract the destination ID directly from the chosen option's info
            chosen_dest_id = chosen_option_info['dc_id']
            
            # ### --- END OF FIX --- ###

            tasks_to_transfer = node.apply_manager_decision(chosen_dest_id)
            for task in tasks_to_transfer:
                # Calculate actual delay based on task bandwidth
                delay_s = get_transmission_delay(node.location, self.nodes[task.dest_dc_id].location, self.cloud_provider, task.bandwidth_gb)
                arrival_time = current_time_utc + pd.Timedelta(seconds=delay_s)
                self.in_transit_tasks.append((arrival_time, task, task.dest_dc_id))
        
        remaining_in_transit = deque()
        while self.in_transit_tasks:
            arrival_time, task, dest_dc_id = self.in_transit_tasks.popleft()
            if arrival_time <= current_time_utc:
                self.nodes[dest_dc_id].add_transferred_tasks([task])
            else:
                remaining_in_transit.append((arrival_time, task, dest_dc_id))
        self.in_transit_tasks = remaining_in_transit

    
    def step_worker(self, current_time_utc: pd.Timestamp,
                  worker_actions: Dict[int, int]) -> Dict[str, Any]:
        
        for dc_id, worker_action_execute in worker_actions.items():
            self.nodes[dc_id].apply_worker_decision(worker_action_execute, current_time_utc)
    
    def step_physics(self, current_time_utc: pd.Timestamp) -> dict:

        all_dc_infos = {}
        for dc_id, node in self.nodes.items():
            all_dc_infos[dc_id] = node.step_physical_simulation(current_time_utc)
        return {"datacenter_infos": all_dc_infos}
    
    def is_cluster_idle(self) -> bool:
        if self.in_transit_tasks:
            return False
        for node in self.nodes.values():
            if (node.originating_tasks_queue or 
                node.worker_commitment_queue or
                node.physical_dc_model.running_tasks):

                return False
        return True
        
