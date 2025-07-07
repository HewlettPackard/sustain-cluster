import random
import os
import zipfile
from collections import defaultdict
from envs.sustaindc.sustaindc_env import SustainDC
from utils.deprecated_task_assignment_strategies import (DistributeMostAvailable, DistributeRandom, DistributePriorityOrder, DistributeLowestPrice, DistributeLeastPending, 
                                                DistributeLowestCarbon, DistributeRoundRobin, DistributeLowestUtilization, BaseRBCStrategy, DistributeLocalOnly)

import numpy as np
import pandas as pd
from rl_components.task import Task

from utils.transmission_cost_loader import load_transmission_matrix
from utils.transmission_region_mapper import map_location_to_region
from utils.workload_utils import assign_task_origins, extract_tasks_from_row

class DatacenterClusterManager:
    """
    Backend simulator that manages multiple datacenters and handles task routing.

    This class can operate in either:
      - Rule-based mode: Uses predefined heuristics for task assignment.
      - RL mode: Waits for TaskSchedulingEnv to assign tasks via actions.

    It steps the internal simulation, updates datacenter resources,
    and computes detailed info at each timestep.
    """
    def __init__(self, config_list, simulation_year, init_day, init_hour, strategy="priority_order", 
                 tasks_file_path=None, shuffle_datacenter_order=True, cloud_provider="gcp", logger=None):
        """
        Initializes multiple datacenters using SustainDC and loads tasks.

        Args:
            config_list (list): List of datacenter configuration dictionaries.
            simulation_year (int): Simulation year to use.
            init_day (int): Initial day of the simulation.
            init_hour (int): Initial hour of the simulation.
            strategy (str): Task distribution strategy.
            tasks_file_path (str, optional): Path to the tasks pickle file.
        """
        # Inject simulation parameters into each datacenter's configuration
        for config in config_list:
            config['simulation_year'] = simulation_year
            config['init_day'] = init_day
            config['init_hour'] = init_hour
            
            # Convert CPU/GPU/MEM to estimated MW
            total_cores = config["total_cores"]
            total_gpus = config["total_gpus"]
            total_mem = config["total_mem"]

            cpu_power = 6 * total_cores      # max_watts per core
            gpu_power = 500 * total_gpus     # max_watts per GPU
            mem_power = 2.5 * total_mem      # max_watts per GB

            total_power_watt = cpu_power + gpu_power + mem_power
            capacity_mw = total_power_watt / 1e6

            config["datacenter_capacity_mw"] = capacity_mw  # Inject into env_config
    
        self.datacenters = {f"DC{i+1}": SustainDC(config) for i, config in enumerate(config_list)}
        self.simulation_year = simulation_year
        self.init_day = init_day
        self.init_hour = init_hour
        self.shuffle_datacenter_order = shuffle_datacenter_order
        
        self.cloud_provider = cloud_provider
        self.transmission_matrix = load_transmission_matrix(cloud_provider)
        self.logger = logger

        # Load tasks if a file path is provided; otherwise, self.tasks remains None
        # Load tasks with fallback unzip logic
        if tasks_file_path:
            self.tasks = self._load_or_extract_tasks_file(tasks_file_path)
        else:
            raise ValueError("No tasks file path provided. Please provide a valid path to the tasks pickle file.")


        # Strategies for RBC task assignment
        self.strategy = strategy
        self.strategy_map = {
                            "most_available": DistributeMostAvailable(),
                            "random": DistributeRandom(),
                            "priority_order": DistributePriorityOrder(),
                            "least_pending": DistributeLeastPending(),
                            "lowest_carbon": DistributeLowestCarbon(),
                            "round_robin": DistributeRoundRobin(),
                            "lowest_price": DistributeLowestPrice(),
                            "lowest_utilization": DistributeLowestUtilization(),
                            "local_only": DistributeLocalOnly(),
                        }

    def reset(self, seed=None):
        """
        Reset all datacenters in the cluster.

        This method ensures that each datacenter starts fresh for a new episode.
        """
        # for dc_name, dc in self.datacenters.items():
            # print(f"Resetting {dc_name}...")
            # dc.reset()  # Call reset on each SustainDC instance
        # Set the random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            # print(f"Random seed set to {seed} for reproducibility.")
        rng = np.random.default_rng(seed)
        # Limit day to ±7 days around the initial value, wrapping around the year (0-364)
        low_day = max(0, self.init_day - 7)
        high_day = min(364, self.init_day + 7)
        self.random_init_day = int(rng.integers(low_day, high_day + 1))  # +1 because high is exclusive

        self.random_year = self.simulation_year
        # You can keep full 0–23 hour range, or restrict it too if needed
        self.random_init_hour = int(rng.integers(0, 24))

        if self.shuffle_datacenter_order:
            items = list(self.datacenters.items())
            np.random.shuffle(items)
            self.datacenters = dict(items)


        for dc_name, dc in self.datacenters.items():
            # print(f"Resetting {dc_name} with UTC start: Day {self.random_init_day}, Hour {self.random_init_hour}")
            dc.reset(init_year=self.random_year, init_day=self.random_init_day, init_hour=self.random_init_hour, seed=seed)
            
        
        # Reset stateful strategies
        for strategy_obj in self.strategy_map.values():
            if isinstance(strategy_obj, BaseRBCStrategy): # Check if it's one of our classes
                strategy_obj.reset()
        # print("All datacenters have been reset.")


    def get_tasks_for_timestep(self, current_time):
        """
        Returns the list of Task objects for the given timestep.

        Args:
            current_time (pd.Timestamp): Current simulation time (UTC).

        Returns:
            list: List of Task objects with their origin assigned.
        """
        logger = self.logger
        if self.tasks is None:
            return []
        # Replace year in current_time with 2020
        adjusted_time = current_time.replace(year=2020)

        tasks_for_time = self.tasks[self.tasks['interval_15m'] == adjusted_time]
        tasks_list = []

        if not tasks_for_time.empty:
            # We'll just take the first row in that timeslot
            row = tasks_for_time.iloc[0]
            tasks_list = extract_tasks_from_row(
                row,
                scale=1,
                datacenter_configs=self.get_config_list(),
                current_time_utc=current_time,
                logger=logger,  # pass logger for debug
                task_scale=5,  # Pass the task scale factor
                group_size=1 # Pass the grouping factor
            )

        if logger:
            logger.info(f"get_tasks_for_timestep: Found {len(tasks_list)} tasks at time {current_time}")
        return tasks_list
    
    def distribute_workload(self, task, current_time, logger):
        if self.strategy == "manual_rl":
            # Let the RL agent handle task assignment
            return None
        elif self.strategy in self.strategy_map:
            assigned_dc_id = self.strategy_map[self.strategy](task, self.datacenters, logger)
            assigned_dc = next((dc for dc in self.datacenters.values() if dc.dc_id == assigned_dc_id), None)
            task.dest_dc = assigned_dc
            task.dest_dc_id = assigned_dc.dc_id
            if not assigned_dc:
                logger.warning(f"[{current_time}] Warning! Task {task.job_name} could not be assigned and remains in queue.")
            return assigned_dc_id
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        
    def step(self, current_time, logger=None):
        """
        Advance all datacenters by one simulation step (15 minutes).
        
        Depending on the selected strategy:
            - If strategy == "manual_rl": tasks are already assigned by the agent.
            - Otherwise: use a rule-based strategy to assign tasks before stepping DCs.
        
        Returns:
            dict containing:
                - energy_usage: energy consumption per DC
                - datacenter_infos: detailed info per DC
                - task_distribution: where each task was routed (only in rule-based mode)
                - resource_usage: CPU/GPU/MEM usage per DC
                - pending_task_count: number of pending tasks per DC
        """

        if logger:
            logger.info(f"===== TIMESTEP {current_time} =====")

        # STEP 1. Update simulation clock for all datacenters
        for dc in self.datacenters.values():
            dc._update_current_time_task(current_time)

        # Initialize the results dictionary
        results = {
            "energy_usage": {},
            "datacenter_infos": {},
            "task_distribution": defaultdict(list),  # Only used in rule-based mode
            "resource_usage": defaultdict(list),
            "pending_task_count": {}
            }

        # STEP 2. (Optional) Rule-based task routing
        if self.strategy != "manual_rl":
            # STEP 2.1: Load incoming tasks for this timestep
            tasks = self.get_tasks_for_timestep(current_time)
            if logger:
                logger.info(f"[{current_time}] New tasks fetched: {len(tasks)}")

            # STEP 2.2: Assign tasks to datacenters using rule-based strategy
            dc_id_to_name = {dc.dc_id: name for name, dc in self.datacenters.items()}

            for task in tasks:
                assigned_dc_id = self.distribute_workload(task, current_time, logger)
                if assigned_dc_id is not None:
                    dc_name = dc_id_to_name.get(assigned_dc_id)
                    if dc_name is None: # Safety check
                        if logger: logger.error(f"Could not find dc_name for assigned_dc_id {assigned_dc_id}")
                        continue # Skip this task if mapping fails
                    self.datacenters[dc_name].pending_tasks.append(task)
                    task.dest_dc_id = assigned_dc_id
                    task.dest_dc = self.datacenters[dc_name]
                    results["task_distribution"][dc_name].append(task)

                    if logger:
                        logger.info(f"[{current_time}] Task {task.job_name} assigned -> {dc_name} (ID: {assigned_dc_id}) via RBC")
        else:
            # Manual RL mode: tasks have already been enqueued
            if logger:
                logger.info(f"[{current_time}] Using manual RL mode — skipping internal task assignment")

        # STEP 3. Step each datacenter environment
        for dc_name, dc in self.datacenters.items():
            # STEP 3.1: Advance internal simulation (release finished, schedule pending)
            obs, rew, terminateds, truncateds, info = dc.step(current_time, logger)

            # STEP 3.2: Record energy usage
            results["energy_usage"][dc_name] = info['agent_bat']['bat_total_energy_with_battery_KWh']

            # STEP 3.3: Record full info dict
            results["datacenter_infos"][dc_name] = info

            # STEP 3.4: Track number of pending tasks
            results["pending_task_count"][dc_name] = len(dc.pending_tasks)

            # STEP 3.5: Compute resource utilization snapshot
            resource_snapshot = {
                "cpu": (dc.total_cores - dc.available_cores) / dc.total_cores * 100,
                "gpu": (dc.total_gpus - dc.available_gpus) / dc.total_gpus * 100,
                "mem": (dc.total_mem_GB - dc.available_mem) / dc.total_mem_GB * 100,
            }
            results["resource_usage"][dc_name].append(resource_snapshot)

            if logger:
                logger.info(f"[{current_time}] {dc_name} - Usage: CPU={resource_snapshot['cpu']:.2f}%, "
                            f"GPU={resource_snapshot['gpu']:.2f}%, MEM={resource_snapshot['mem']:.2f}%")
                logger.info(f"[{current_time}] {dc_name} - Running Tasks: {len(dc.running_tasks)}, "
                            f"Pending Tasks: {len(dc.pending_tasks)}")

        # === STEP 4: Compute total transmission metrics ===
        transmission_cost_total = 0.0
        transmission_energy_total = 0.0
        transmission_emissions_total = 0.0

        for dc_name, dc in self.datacenters.items():
            dc_info = results["datacenter_infos"][dc_name]
            routed_tasks = dc_info["__common__"].get("routed_tasks_this_step", [])

            for task in routed_tasks:
                origin_loc = self.get_dc_location(task.origin_dc_id)
                dest_loc = self.get_dc_location(task.dest_dc_id)

                origin_region = map_location_to_region(origin_loc, self.cloud_provider)
                dest_region = map_location_to_region(dest_loc, self.cloud_provider)

                if origin_region and dest_region:
                    try:
                        cost_per_gb = self.transmission_matrix.loc[origin_region, dest_region]
                        transmission_cost = cost_per_gb * task.bandwidth_gb
                        transmission_cost_total += transmission_cost
                        
                        # === Compute transmission energy and emissions (Simpler model) ===
                        kwh_per_gb = 0.06  # fixed intensity (KWh/GB) Extracted from https://onlinelibrary.wiley.com/doi/10.1111/jiec.12630
                        energy_kwh_transmision = task.bandwidth_gb * kwh_per_gb
                        task.origin_dc = next(dc for dc in self.datacenters.values() if dc.dc_id == task.origin_dc_id)
                        ci_origin = task.origin_dc.ci_manager.get_current_ci(norm=False) / 1000  # in kgCO2/kWh

                        # If there is no transmision (origin datacenter is the same as destination)
                        if task.origin_dc_id == task.dest_dc_id:
                            energy_kwh_transmision = 0.0
                            ci_origin = 0.0
                        
                        transmission_emissions = energy_kwh_transmision * ci_origin
                        transmission_energy_total += energy_kwh_transmision
                        transmission_emissions_total += transmission_emissions
                            
                        if logger:
                            origin_dc = task.origin_dc  # already assigned above
                            logger.info(
                                f"[{current_time}] Task '{task.job_name}' | "
                                f"From {origin_dc.location} (ID={task.origin_dc_id}, Region={origin_region}) → "
                                f"To {dest_loc} (ID={task.dest_dc_id}, Region={dest_region}) | "
                                f"Bandwidth: {task.bandwidth_gb:.2f} GB | "
                                f"Transmission Cost Rate: ${cost_per_gb:.2f}/GB | "
                                f"Transmission Cost: ${transmission_cost:.4f} | "
                                f"Transmission Energy Used: {energy_kwh_transmision:.4f} kWh | "
                                f"Origin CI: {ci_origin:.4f} kgCO2/kWh | "
                                f"Transmission CO2 Emissions: {transmission_emissions:.4f} kgCO₂"
                            )

                    except KeyError:
                        print(f"[WARNING] Transmission cost not found between {origin_region} and {dest_region}")
                
                else:
                    if logger:
                        logger.warning(
                            f"[{current_time}] Unknown region mapping for {origin_loc} (ID={task.origin_dc_id}) "
                            f"or {dest_loc} (ID={task.dest_dc_id})"
                        )
                    raise ValueError(
                        f"Unknown region mapping for {origin_loc} (ID={task.origin_dc_id}) or {dest_loc} (ID={task.dest_dc_id})"
                    )

        results["transmission_cost_total_usd"] = transmission_cost_total
        results["transmission_energy_total_kwh"] = transmission_energy_total
        results["transmission_emissions_total_kg"] = transmission_emissions_total


        # FINAL STEP. Return aggregated results from all datacenters
        return results

    def get_dc_location(self, dc_id):
        for dc in self.datacenters.values():
            if dc.dc_id == dc_id:
                return dc.location
        return None


    def get_config_list(self):
        return [dc.env_config for dc in self.datacenters.values()]

    def _load_or_extract_tasks_file(self, tasks_file_path: str) -> pd.DataFrame:
        if os.path.exists(tasks_file_path):
            if hasattr(self, "logger") and self.logger:
                self.logger.info(f"Loading workload from: {tasks_file_path}")
            df = pd.read_pickle(tasks_file_path)
        else:
            zip_path = tasks_file_path.replace(".pkl", ".zip")
            if os.path.exists(zip_path):
                if hasattr(self, "logger") and self.logger:
                    self.logger.info(f"Workload .pkl not found. Extracting zip: {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(tasks_file_path))

                if not os.path.exists(tasks_file_path):
                    raise FileNotFoundError(f"Unzipped file not found: {tasks_file_path}")

                df = pd.read_pickle(tasks_file_path)
            else:
                raise FileNotFoundError(
                    f"Workload file not found: {tasks_file_path} or zip version: {zip_path}"
                )

        # Ensure timestamps are in UTC
        df['interval_15m'] = df['interval_15m'].dt.tz_convert('UTC')
        return df

