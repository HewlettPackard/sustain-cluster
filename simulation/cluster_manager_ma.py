import os
import zipfile
from collections import deque
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd

from envs.datacenter_node_ma import DatacenterNodeMA
from utils.workload_utils import extract_tasks_from_row
from data.network_cost.network_delay import get_transmission_delay
from utils.transmission_cost_loader import load_transmission_matrix
from utils.transmission_region_mapper import map_location_to_region
from rl_components.task import Task
from utils.meta_task_util import create_meta_task_from_group

class DatacenterClusterManagerMA:
    """
    Orchestrates the entire multi-agent simulation for SustainCluster-MA.
    ...
    """
    def __init__(self, config_list: List[Dict], simulation_year: int, tasks_file_path: str,
                 cloud_provider: str, max_total_options: int, duration_days: int, logger: Any = None):
        """
        Initializes the multi-agent cluster manager.
        """
        self.logger = logger
        
        # Add in the config list the simulation duration days
        for cfg in config_list:
            cfg['duration_days'] = duration_days
            
        self.nodes: Dict[int, DatacenterNodeMA] = {
            cfg['dc_id']: DatacenterNodeMA(cfg, self.logger) for cfg in config_list
        }
        self.num_dcs = len(self.nodes)
        
        self._canonical_dc_order: List[int] = sorted(self.nodes.keys())
        
        if max_total_options < self.num_dcs:
            raise ValueError(f"max_total_options ({max_total_options}) must be >= number of datacenters ({self.num_dcs})")
        self.max_total_options = max_total_options

        self.tasks_df = self._load_tasks(tasks_file_path)
        self.in_transit_tasks: deque[Tuple[pd.Timestamp, Task, int]] = deque()

        self.cloud_provider = cloud_provider
        self.transmission_cost_matrix = load_transmission_matrix(cloud_provider)
        self.single_worker_dc_id: Optional[int] = None 
        # === SIMPLIFICATION CHANGE: Ensure `is_local` is the first feature ===
        # This makes it easy for the agent to know which DC it is.
        self.DESTINATION_OPTION_FEATURE_ORDER = [
            "is_local",
            "cpu_avail_pct",
            "gpu_avail_pct",
            "ci",
            "external_temperature"
        ]
        self.D_OPTION_FEAT = len(self.DESTINATION_OPTION_FEATURE_ORDER)

        if self.logger:
            self.logger.info(f"DatacenterClusterManagerMA initialized with {self.num_dcs} nodes.")
        
        self.seed = 0  # Default

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
        if seed:
            self.seed = seed
        for i, node in enumerate(self.nodes.values()):
            # Use a different seed for each node to ensure varied internal randomness
            node.reset(init_year, init_day, init_hour, self.seed + i)
        self.in_transit_tasks.clear()

    def _get_newly_arrived_tasks(self, current_time_utc: pd.Timestamp, use_meta_tasks: bool) -> Dict[int, List[Task]]:
        """Loads and assigns origins to tasks for the current timestep."""
        # adjusted_time = current_time_utc.replace(year=2020)

        # hits = (self.tasks_df['interval_15m'] == adjusted_time).sum()
        # print(f"[DEBUG TASK_MATCH] cur={current_time_utc.isoformat()}  "
            # f"adj={adjusted_time.isoformat()}  hits={hits}")

        # 1. Create a naive timestamp by replacing the year.
        #    This will likely drop the timezone info.
        naive_adjusted_time = current_time_utc.replace(year=2020)
        
        # 2. Explicitly re-localize the new timestamp to UTC to make it timezone-aware.
        #    This ensures it has the same dtype as the DataFrame column.
        #    `tz_localize('UTC')` is the key operation here.
        if not isinstance(naive_adjusted_time, pd.Timestamp):
            adjusted_time = pd.Timestamp(naive_adjusted_time).tz_localize('UTC')
        else:
            adjusted_time = naive_adjusted_time
        
        # --- END OF FIX ---

        if self.logger:
            # Add a debug log to confirm the types match
            self.logger.debug(f"Matching Timestamps: adjusted_time='{adjusted_time}' (type: {type(adjusted_time)}), "
                            f"DataFrame dtype='{self.tasks_df['interval_15m'].dtype}'")
            
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
        
        if not use_meta_tasks:
            # If not using meta-tasks, return the individual tasks as before.
            return tasks_by_origin

        # --- NEW META-TASK LOGIC ---
        meta_tasks_by_origin = {}
        for dc_id, task_group in tasks_by_origin.items():
            if task_group:
                # Create one super-task to represent the entire group
                meta_task = create_meta_task_from_group(task_group, dc_id, current_time_utc)
                # The result is still a list, but it contains only one item.
                meta_tasks_by_origin[dc_id] = [meta_task] 
        # print(f"[DEBUG-TASK] time={current_time_utc}  "
        # f"hits={len(tasks_for_time)}")
        print("[TASK] t=", current_time_utc, "hits=", len(tasks_for_time))
        return meta_tasks_by_origin


    def _prepare_all_manager_observations(self, current_time_utc: pd.Timestamp) -> Dict[str, Any]:
        """
        Assembles the observation data for all DTA_Managers.

        This method is now the single source of truth for all data related to
        a manager's decision, supporting both complex and simple observation modes.
        """
        all_observations = {}
        valid_options_maps = {}
        workload_magnitudes_map = {}

        # 1. Get the current state of all DCs for remote queries. This is efficient.
        remote_query_states = {dc_id: node.get_state_for_remote_query() for dc_id, node in self.nodes.items()}

        # 2. For each DTA_Manager, construct its unique observation data.
        for dc_id, node in self.nodes.items():
            task_snapshot = list(node.originating_tasks_queue)
            local_obs_part = node.prepare_manager_observation(current_time_utc)
            workload_magnitudes_map[dc_id] = local_obs_part["workload_magnitude_dict"]
            
            options_list = []
            
            # --- START OF SIMPLIFICATION CHANGE ---
            # Loop through all possible destination DCs in a fixed, canonical order.
            for dest_dc_id in self._canonical_dc_order:
                is_local_flag = 1.0 if dest_dc_id == dc_id else 0.0
                
                # Get the state of the destination DC.
                dest_state = remote_query_states[dest_dc_id]
                
                # Get network info if it's a remote transfer.
                if is_local_flag == 1.0:
                    delay_s = 0.0
                    cost_per_gb = 0.0
                else:
                    delay_s = get_transmission_delay(
                        node.location, self.nodes[dest_dc_id].location, self.cloud_provider, 1.0)
                    cost_per_gb = self.transmission_cost_matrix.loc[
                        map_location_to_region(node.location, self.cloud_provider),
                        map_location_to_region(self.nodes[dest_dc_id].location, self.cloud_provider)]
                
                # Assemble the feature dictionary for this option.
                # This dict is used for both the valid_options_map and to build the feature vector.
                option_dict = {
                    "is_local": is_local_flag,
                    "cpu_avail_pct": dest_state["cpu_avail_pct"],
                    "gpu_avail_pct": dest_state["gpu_avail_pct"],
                    "ci": dest_state["ci"],
                    "external_temperature": dest_state["external_temperature"],
                    # We can add extra info to the map for the reward function if needed
                    "transmission_cost_per_gb": cost_per_gb,
                    "transmission_delay_s_per_gb": delay_s,
                    "dc_id": dest_dc_id # Crucial for mapping in the complex mode
                }
                # print(f"[DC {dc_id}] External Temperature: {dest_state['external_temperature']:.2f}°C")

                options_list.append(option_dict)
            # --- END OF SIMPLIFICATION CHANGE ---
            

            # 3. Convert to padded NumPy array and create mask (for complex/attention mode).
            num_valid_options = len(options_list)
            padded_options_array = np.zeros((self.max_total_options, self.D_OPTION_FEAT), dtype=np.float32)
            mask = np.ones(self.max_total_options, dtype=bool)

            for i, option_dict in enumerate(options_list):
                # Create the feature vector based on the canonical order.
                feature_vector = [option_dict[key] for key in self.DESTINATION_OPTION_FEATURE_ORDER]
                padded_options_array[i] = np.array(feature_vector, dtype=np.float32)
            
            mask[:num_valid_options] = False

            valid_options_maps[dc_id] = {i: opt_dict for i, opt_dict in enumerate(options_list)}
            
            # 4. Assemble the observation dictionary component for this manager.
            all_observations[dc_id] = {
                "obs_manager_meta_task_i": local_obs_part["obs_manager_meta_task_i"],
                "obs_all_options_set_padded": padded_options_array,
                "all_options_padding_mask": mask,
                # The simple, flattened observation will be constructed in the environment class.
                # This method just provides all the necessary component parts.
            }
            
            # Lets print the external temperature for debugging

        # Return a single dictionary containing all the data.
        return {
            "observations": all_observations,
            "valid_options_maps": valid_options_maps,
            "workload_magnitudes": workload_magnitudes_map,
        }
    
    def task_origination(self, current_time_utc: pd.Timestamp, use_meta_tasks: bool = False):
        # print(f"[CHECK-A] task_origination called  t={current_time_utc}")
        # print("[DEBUG-ORIG] tasks_df rows =", len(self.tasks_df))
        # REMOVE the loop that steps the managers. This is now done inside SustainDC.
        # for node in self.nodes.values():
        #     node.ci_manager.step()
        #     node.price_manager.step()
        #     node.weather_manager.step()

        # B. Task Origination
        newly_arrived_tasks = self._get_newly_arrived_tasks(current_time_utc, use_meta_tasks)
        for dc_id, tasks in newly_arrived_tasks.items():
            if dc_id in self.nodes:
                self.nodes[dc_id].add_originating_tasks(tasks)



    def step_manager(self, current_time_utc: pd.Timestamp,
                     manager_actions: Dict[int, int],
                     valid_options_maps: Dict[int, Dict[int, Dict]]) -> None:
    
        # The canonical order of all DC IDs, sorted once and used everywhere.
        # This is the key to consistent action mapping.
        # It's better to make this an instance variable, e.g., self._canonical_dc_order
        # initialized in __init__, but defining it here works too.
        all_dc_ids_sorted = self._canonical_dc_order

        for dc_id, manager_action_idx in manager_actions.items():
            node = self.nodes[dc_id]

            # 1. Safety check: ensure the action index is within the valid range.
            #    The action index must be less than the total number of DCs.
            if manager_action_idx >= len(all_dc_ids_sorted):
                if self.logger:
                    self.logger.error(
                        f"FATAL: Invalid action index {manager_action_idx} for DC {dc_id}. "
                        f"Index is out of bounds for the number of datacenters ({len(all_dc_ids_sorted)}). "
                        f"This indicates a network output or action space mismatch. Skipping."
                    )
                break

            # 2. Map the action index to the destination DC ID using the canonical sorted list.
            #    The observation for every manager was constructed using this same
            #    sorted order for the destination options. Therefore, action index `k`
            #    *always* corresponds to the k-th datacenter in this globally
            #    consistent list.
            chosen_dest_id = all_dc_ids_sorted[manager_action_idx]
            
            # apply_manager_decision will move tasks from the originating queue
            # to either the local worker queue (if chosen_dest_id == dc_id) or
            # return them for transfer.
            tasks_to_transfer = node.apply_manager_decision(chosen_dest_id)
            
            # If the decision was to transfer (i.e., chosen_dest_id is remote),
            # process the returned tasks.
            if tasks_to_transfer:
                destination_node = self.nodes[chosen_dest_id]
                for task in tasks_to_transfer:
                    delay_s = get_transmission_delay(
                        node.location, destination_node.location, self.cloud_provider, task.bandwidth_gb)
                    arrival_time = current_time_utc + pd.Timedelta(seconds=delay_s)
                    self.in_transit_tasks.append((arrival_time, task, chosen_dest_id))
        
        # Create a new, empty queue to hold tasks that are *still* in transit
        # after this check. This is an efficient way to filter the deque.
        remaining_in_transit = deque()

        # Process every task currently in the main in-transit queue.
        # The `while` loop continues until the queue is empty.
        while self.in_transit_tasks:
            # Efficiently remove the task from the front of the queue.
            arrival_time, task, dest_dc_id = self.in_transit_tasks.popleft()

            # This is the core logic: check if the task's calculated arrival time
            # is now in the past or present.
            if arrival_time <= current_time_utc:
                # The task has arrived. Deliver it to the destination node's
                # DTA_Worker queue for future processing.
                self.nodes[dest_dc_id].add_transferred_tasks([task])
            else:
                # The task has not arrived yet. Add it to our temporary list
                # of tasks that are still in-flight.
                remaining_in_transit.append((arrival_time, task, dest_dc_id))

        # After checking all tasks, update the main in-transit queue to only
        # contain the tasks that are still traveling. The tasks that have
        # arrived are now in their respective DC's worker queues.
        self.in_transit_tasks = remaining_in_transit
        # for dc_id, node in self.nodes.items():
            # print(f" [DEBUG-ROUTE] dc={dc_id} "
            # f"origin_q={len(node.originating_tasks_queue)} "
            # f"worker_q={len(node.worker_commitment_queue)}")

    
    def step_worker(self, current_time_utc: pd.Timestamp,
                  worker_actions: Dict[int, int]) -> None:
        
        for dc_id, worker_action_execute in worker_actions.items():
            self.nodes[dc_id].apply_worker_decision(worker_action_execute, current_time_utc)
    
    def step_physics(self, current_time_utc: pd.Timestamp) -> dict:

        all_dc_infos = {}
        for dc_id, node in self.nodes.items():
            terminateds, truncateds, info = node.step_physical_simulation(current_time_utc)
            
            # Add in info the termnateds and truncateds flags
            info['terminateds'] = terminateds
            info['truncateds'] = truncateds
            
            all_dc_infos[dc_id] = info
        return {"datacenter_infos": all_dc_infos}
    

    def is_cluster_idle(self) -> bool:
        """
        Checks if the entire cluster is inactive, meaning there are no tasks
        anywhere in the system (in transit, pending, or running).

        This is used to determine if a simulation episode can end early because
        all work has been completed.

        Returns:
            bool: True if the cluster is completely idle, False otherwise.
        """
        # 1. Check if there are any tasks currently being transferred over the network.
        #    If the in-transit queue is not empty, the cluster is still active.
        if self.in_transit_tasks:
            return False

        # 2. If the network is clear, check the state of each individual datacenter node.
        #    The loop will exit and return False as soon as the first active task is found.
        for node in self.nodes.values():
            # Check all possible locations a task could be within a single datacenter node:
            #   - originating_tasks_queue: Tasks waiting for the DTA_Manager to route them.
            #   - worker_commitment_queue: Tasks waiting for the DTA_Worker to execute/defer them.
            #   - running_tasks: Tasks currently being processed on the physical hardware.
            if (node.originating_tasks_queue or
                node.worker_commitment_queue or
                node.physical_dc_model.running_tasks):
                # If any of these queues or lists are not empty, the node is active,
                # and therefore the entire cluster is not idle.
                return False

        # 3. If the loop completes without finding any tasks in any node, and the
        #    in-transit queue was also empty, the cluster is truly idle.
        return True

    def route_single_task(self, task: Task, destination_dc_id: int, current_time_utc: pd.Timestamp):
        """
        Routes a single task (likely a meta-task) to a destination.
        This is a simplified version of step_manager for the single-agent env.
        """
        task.dest_dc_id = destination_dc_id
        origin_node = self.nodes[task.origin_dc_id]

        if destination_dc_id == task.origin_dc_id:
            # Local transfer: immediately add to the worker queue.
            origin_node.add_transferred_tasks([task])
        else:
            # Remote transfer: calculate delay and add to in-transit queue.
            destination_node = self.nodes[destination_dc_id]
            delay_s = get_transmission_delay(
                origin_node.location, destination_node.location, self.cloud_provider, task.bandwidth_gb)
            arrival_time = current_time_utc + pd.Timedelta(seconds=delay_s)
            self.in_transit_tasks.append((arrival_time, task, destination_dc_id))

        # After any routing decision, it's good practice to process any tasks
        # that might have arrived from previous steps.
        self._process_in_transit_tasks(current_time_utc)

    def _process_in_transit_tasks(self, current_time_utc: pd.Timestamp):
        """Helper to process the in-transit queue."""
        remaining_in_transit = deque()
        while self.in_transit_tasks:
            arrival_time, task, dest_dc_id = self.in_transit_tasks.popleft()
            if arrival_time <= current_time_utc:
                self.nodes[dest_dc_id].add_transferred_tasks([task])
            else:
                remaining_in_transit.append((arrival_time, task, dest_dc_id))
        self.in_transit_tasks = remaining_in_transit

    def commit_tasks(self, current_time_utc: pd.Timestamp, worker_action_execute: int) -> None:

        if self.single_worker_dc_id is None:
            raise RuntimeError(
                "single_worker_dc_id is not set"
            )

        node = self.nodes[self.single_worker_dc_id]
        node.apply_worker_decision(worker_action_execute, current_time_utc)
    