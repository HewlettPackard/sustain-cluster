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
            "is_local", "worker_queue_len", "cpu_avail_pct", "gpu_avail_pct",
            "price", "ci", "transmission_cost_per_gb", "transmission_delay_s_per_gb"
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

        # hits = (self.tasks_df['interval_15m'] == adjusted_time).sum()
        # print(f"[DEBUG TASK_MATCH] cur={current_time_utc.isoformat()}  "
            # f"adj={adjusted_time.isoformat()}  hits={hits}")

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
        # print(f"[TASK_ORIGIN] {current_time_utc} ⇒ "
            #   f"{ {k: len(v) for k,v in tasks_by_origin.items()} }")
        
        return tasks_by_origin

    def _prepare_all_manager_observations(self, current_time_utc: pd.Timestamp) -> Dict[int, Dict]:
        """Assembles the full, complex observation for every DTA_Manager."""
        all_observations = {}
        
        # 1. Get the current state of all DCs for remote queries
        remote_query_states = {dc_id: node.get_state_for_remote_query() for dc_id, node in self.nodes.items()}

        # 2. For each DTA_Manager, construct its unique observation
        for dc_id, node in self.nodes.items():
            # local_obs_part = node.prepare_manager_observation(current_time_utc)

            # ----- new logic -----
            task_snapshot = list(node.originating_tasks_queue)
            local_obs_part = node.prepare_manager_observation(current_time_utc, task_snapshot)
            # ----- new logic -----

            options_list = []
            # ----- new logic -----
            all_dc_ids_sorted = sorted(self.nodes.keys())  # ensure stable order
            # ----- new logic -----
            """
            # Add local DC as the first option
            local_option_dict = local_obs_part['local_destination_option_features']
            # Make sure it has all keys, even if transmission is zero
            local_option_dict['transmission_cost_per_gb'] = 0.0
            local_option_dict['transmission_delay_s_per_gb'] = 0.0
            options_list.append(local_option_dict)
            
            # Add all other remote DCs as options
            for remote_dc_id, remote_node in self.nodes.items():
                if remote_dc_id == dc_id:
                    continue
                
                remote_state = remote_query_states[remote_dc_id]
                
                # We need a representative bandwidth to calculate delay. Let's use 1 GB.
                # The agent can learn to scale this.
                task_bw_placeholder = 1.0 
                delay_s = get_transmission_delay(node.location, remote_node.location, self.cloud_provider, task_bw_placeholder)
                
                origin_region = map_location_to_region(node.location, self.cloud_provider)
                dest_region = map_location_to_region(remote_node.location, self.cloud_provider)
                cost_per_gb = self.transmission_cost_matrix.loc[origin_region, dest_region]

                remote_option_features = {
                    "is_local": 0.0,
                    **remote_state,
                    "transmission_cost_per_gb": cost_per_gb,
                    "transmission_delay_s_per_gb": delay_s # Simplification for now
                }
                options_list.append(remote_option_features)
            """
            for other_dc_id in all_dc_ids_sorted:
                if other_dc_id == dc_id:
                    option_dict = local_obs_part['local_destination_option_features']
                    option_dict['transmission_cost_per_gb'] = 0.0
                    option_dict['transmission_delay_s_per_gb'] = 0.0
                else:
                    remote_state = remote_query_states[other_dc_id]
                    delay_s = get_transmission_delay(
                        node.location, self.nodes[other_dc_id].location, self.cloud_provider, 1.0)
                    cost_per_gb = self.transmission_cost_matrix.loc[
                        map_location_to_region(node.location, self.cloud_provider),
                        map_location_to_region(self.nodes[other_dc_id].location, self.cloud_provider)]
                    option_dict = {
                        "is_local": 0.0,
                        **remote_state,
                        "transmission_cost_per_gb": cost_per_gb,
                        "transmission_delay_s_per_gb": delay_s
                    }
                options_list.append(option_dict)

            # 3. Convert to padded NumPy array and create mask
            num_valid_options = len(options_list)
            padded_options_array = np.zeros((self.max_total_options, self.D_OPTION_FEAT), dtype=np.float32)
            mask = np.ones(self.max_total_options, dtype=bool) # True means MASKED/INVALID

            for i, option_dict in enumerate(options_list):
                padded_options_array[i] = np.array([option_dict[key] for key in self.DESTINATION_OPTION_FEATURE_ORDER], dtype=np.float32)
            
            mask[:num_valid_options] = False # First M options are valid

            # 4. Assemble the final observation dictionary for this manager
            all_observations[dc_id] = {
                "obs_manager_meta_task_i": local_obs_part["obs_manager_meta_task_i"],
                "obs_all_options_set_padded": padded_options_array,
                "all_options_padding_mask": mask,
                "valid_options_map": {i: opt_dict for i, opt_dict in enumerate(options_list)} # Helper for action application
            }
            
        return all_observations

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
                  manager_actions: Dict[int, int]) -> None:
    
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
        
        remaining_in_transit = deque()
        while self.in_transit_tasks:
            arrival_time, task, dest_dc_id = self.in_transit_tasks.popleft()
            if arrival_time <= current_time_utc:
                self.nodes[dest_dc_id].add_transferred_tasks([task])
            else:
                remaining_in_transit.append((arrival_time, task, dest_dc_id))
        self.in_transit_tasks = remaining_in_transit

    
    def step_worker(self, current_time_utc: pd.Timestamp,
                  worker_actions: Dict[int, int]) -> None:
        
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
        
