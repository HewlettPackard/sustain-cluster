# envs/datacenter_node_ma.py

from collections import deque
from typing import List, Deque, Dict, Any, Optional
import numpy as np
import pandas as pd

# Assuming the following import paths are correct from your project root
from envs.sustaindc.sustaindc_env import SustainDC
from utils.managers import CI_Manager, ElectricityPrice_Manager, Weather_Manager
from utils.marl_utils import (
    aggregate_tasks_for_manager,
    aggregate_tasks_for_worker,
    MANAGER_META_TASK_FEATURE_ORDER,
    WORKER_META_TASK_FEATURE_ORDER
)
from rl_components.task import Task


class DatacenterNodeMA:
    """
    Represents a single, intelligent datacenter node in the SustainCluster-MA framework.

    This class encapsulates all logic for one datacenter, including:
    - Its two-level agent queues (Manager and Worker).
    - The underlying physical simulation model (re-using SustainDC).
    - Local environmental data managers.
    - Methods to prepare observations for its agents and apply their decisions.
    """

    def __init__(self, dc_config: Dict[str, Any], logger: Optional[Any] = None):
        """
        Initializes a DatacenterNodeMA instance.

        Args:
            dc_config (Dict[str, Any]): The configuration dictionary for this specific
                                        datacenter, typically from datacenters.yaml.
            logger (Optional[Any]): A logger instance for logging events.
        """
        self.dc_id: int = dc_config['dc_id']
        self.location: str = dc_config['location']
        self.logger = logger

        # --- Queues for the two-level agent hierarchy ---
        self.originating_tasks_queue: Deque[Task] = deque()
        self.worker_commitment_queue: Deque[Task] = deque()

        # --- Internal Physical Model (re-using existing SustainDC) ---
        # SustainDC will handle resource tracking (cores, gpus, mem) and the
        # detailed physics simulation of power, cooling, etc.
        self.physical_dc_model = SustainDC(dc_config)

        # --- Local Environmental Data Managers ---
        # These are initialized here but will be reset with the correct
        # start time by the global Cluster Manager.
        self.ci_manager = CI_Manager(
            location=self.location,
            simulation_year=dc_config['simulation_year'],
            timezone_shift=dc_config['timezone_shift']
        )
        self.price_manager = ElectricityPrice_Manager(
            location=self.location,
            simulation_year=dc_config['simulation_year'],
            timezone_shift=dc_config['timezone_shift']
        )
        self.weather_manager = Weather_Manager(
            location=self.location,
            simulation_year=dc_config['simulation_year'],
            timezone_shift=dc_config['timezone_shift']
        )
        
        if self.logger:
            self.logger.info(f"DatacenterNodeMA for DC {self.dc_id} ({self.location}) initialized.")

    def reset(self, init_year: int, init_day: int, init_hour: int, seed: int):
        """
        Resets the state of the datacenter node for a new episode.

        Args:
            init_year (int): The simulation year.
            init_day (int): The starting day of the year (0-364).
            init_hour (int): The starting hour of the day (0-23).
            seed (int): The random seed for this episode.
        """
        # 1. Clear all MARL-specific queues
        self.originating_tasks_queue.clear()
        self.worker_commitment_queue.clear()

        # 2. Reset the underlying physical DC model
        self.physical_dc_model.reset(init_year, init_day, init_hour, seed)

        # 3. Reset all environmental data managers to the correct start time
        self.ci_manager.reset(init_day=init_day, init_hour=init_hour, seed=seed)
        self.price_manager.reset(init_day=init_day, init_hour=init_hour, seed=seed)
        self.weather_manager.reset(init_day=init_day, init_hour=init_hour, seed=seed)
        
        if self.logger:
            self.logger.info(f"DatacenterNodeMA {self.dc_id} reset to Day {init_day}, Hour {init_hour}.")

    def add_originating_tasks(self, tasks: List[Task]):
        """Adds newly generated tasks to this DC's originating queue."""
        self.originating_tasks_queue.extend(tasks)

    def add_transferred_tasks(self, tasks: List[Task]):
        """Adds tasks that have arrived via transfer to the worker's queue."""
        self.worker_commitment_queue.extend(tasks)

    # --- Observation Preparation ---

    def prepare_manager_observation(self, current_time_utc: pd.Timestamp, task_snapshot: Optional[List[Task]] = None) -> Dict[str, Any]:
        """
        Prepares the local components of the observation for this DC's DTA_Manager.
        The ClusterManagerMA will combine this with remote DC info.
        """

        # meta_task_vector = aggregate_tasks_for_manager(self.originating_tasks_queue, current_time_utc)
        # ----- new logic -----
        if task_snapshot is None:
            task_snapshot = list(self.originating_tasks_queue)
        meta_task_vector = aggregate_tasks_for_manager(task_snapshot, current_time_utc)
        # ----- new logic -----

        # These features describe this DC as a *potential destination* for its own tasks
        local_option_features = {
            "is_local": 1.0, # Flag to indicate this is the local option
            "worker_queue_len": float(len(self.worker_commitment_queue)),
            "cpu_avail_pct": self.physical_dc_model.available_cores / self.physical_dc_model.total_cores,
            "gpu_avail_pct": self.physical_dc_model.available_gpus / self.physical_dc_model.total_gpus,
            "price": self.price_manager.get_current_price(),
            "ci": self.ci_manager.get_current_ci(norm=False),
            "transmission_cost": 0.0,
            "transmission_delay_s": 0.0
        }
        
        return {
            "obs_manager_meta_task_i": meta_task_vector,
            "local_destination_option_features": local_option_features
        }

    def prepare_worker_observation(self, current_time_utc: pd.Timestamp) -> Dict[str, np.ndarray]:
        """
        Prepares the full observation vector for this DC's DTA_Worker.
        """
        meta_task_vector = aggregate_tasks_for_worker(self.worker_commitment_queue, current_time_utc)

        local_dc_state_vector = np.array([
            self.physical_dc_model.available_cores / self.physical_dc_model.total_cores,
            self.physical_dc_model.available_gpus / self.physical_dc_model.total_gpus,
            self.physical_dc_model.available_mem / self.physical_dc_model.total_mem_GB,
            self.price_manager.get_current_price(),
            self.ci_manager.get_current_ci(norm=False),
            # TODO: Add battery/HVAC state here later if needed (e.g., self.physical_dc_model.bat_env.get_battery_soc())
        ], dtype=np.float32)

        return {
            "obs_worker_meta_task_i": meta_task_vector,
            "obs_local_dc_i_for_worker": local_dc_state_vector
        }

    # --- Action Application ---

    def apply_manager_decision(self, chosen_destination_dc_id: int) -> List[Task]:
        """
        Processes the DTA_Manager's decision. Moves tasks from the originating queue
        to the local worker queue or returns them for remote transfer.

        Returns:
            List[Task]: A list of tasks to be transferred. Empty if destination is local.
        """
        tasks_to_process = list(self.originating_tasks_queue)
        self.originating_tasks_queue.clear() # Decision is for the whole meta-task

        if chosen_destination_dc_id == self.dc_id:
            # Commit to local worker queue
            self.worker_commitment_queue.extend(tasks_to_process)
            if self.logger:
                self.logger.info(f"[DC {self.dc_id}] DTA_Manager committed {len(tasks_to_process)} tasks to local DTA_Worker.")
            return [] # No tasks to transfer
        else:
            # Mark tasks for remote transfer and return them
            for task in tasks_to_process:
                task.dest_dc_id = chosen_destination_dc_id
            if self.logger:
                self.logger.info(f"[DC {self.dc_id}] DTA_Manager routing {len(tasks_to_process)} tasks to remote DC {chosen_destination_dc_id}.")
            return tasks_to_process

    def apply_worker_decision(self, action_execute_now: bool, current_time_utc: pd.Timestamp):
        """
        Processes the DTA_Worker's decision (Execute Now 1 vs. Defer Locally 0).

        Returns:
            List[Task]: The list of tasks that were successfully scheduled to run.
        """
        if not action_execute_now:
            # Defer locally: tasks simply remain in the worker_commitment_queue.
            if self.logger:
                self.logger.info(f"[DC {self.dc_id}] DTA_Worker deferred {len(self.worker_commitment_queue)} tasks.")
            return [] # No tasks were attempted

        tasks_to_attempt_scheduling = list(self.worker_commitment_queue)
        self.worker_commitment_queue.clear() # We will re-add any that fail to schedule

        newly_scheduled_tasks = []
        unschedulable_tasks = []

        for task in tasks_to_attempt_scheduling:
            # We must adapt how the physical model handles scheduling.
            # Here we assume a refactored `try_to_schedule_task` that returns bool
            # and doesn't automatically re-queue on failure.
            
            # --- Modification to physical_dc_model might be needed ---
            # Original `try_to_schedule_task` might auto-re-queue. We need to control that.
            # For now, let's assume we can check `can_schedule` first.
            
            if self.physical_dc_model.can_schedule(task):
                # Allocate resources
                self.physical_dc_model.available_cores -= task.cores_req
                self.physical_dc_model.available_gpus -= task.gpu_req
                self.physical_dc_model.available_mem -= task.mem_req
                task.start_time = current_time_utc
                task.finish_time = current_time_utc + pd.Timedelta(minutes=task.duration)
                self.physical_dc_model.running_tasks.append(task)
                newly_scheduled_tasks.append(task)
            else:
                unschedulable_tasks.append(task)

        # Re-add tasks that couldn't be scheduled to the front of the queue for next time
        if unschedulable_tasks:
            self.worker_commitment_queue.extendleft(reversed(unschedulable_tasks))
            if self.logger:
                self.logger.warning(f"[DC {self.dc_id}] DTA_Worker tried to execute {len(tasks_to_attempt_scheduling)} tasks, but {len(unschedulable_tasks)} could not be scheduled due to resource limits.")

        if self.logger and newly_scheduled_tasks:
            self.logger.info(f"[DC {self.dc_id}] DTA_Worker executed {len(newly_scheduled_tasks)} tasks.")
            
        return newly_scheduled_tasks

    def step_physical_simulation(self, current_time_utc: pd.Timestamp) -> Dict[str, Any]:
        """
        Calls the underlying physical DC model's step to update power, cooling, etc.
        and release resources from completed tasks.
        
        Returns:
            Dict[str, Any]: The info dictionary from the physical DC step.
        """
        # The SustainDC.step() method handles the core physics simulation.
        # It needs the current time to check for completed tasks.
        # It also needs the current workload, which is now implicitly set by the
        # list of running_tasks within the physical_dc_model.
        
        # We need to ensure the physical_dc_model's internal `step` can be called
        # with just the time, as the workload is now managed externally by the MARL agents.
        # Let's assume the existing `SustainDC.step` can be adapted or is suitable.
        
        # Update the internal clock of the physical model before stepping it.
        # This is crucial for releasing completed tasks.
        self.physical_dc_model._update_current_time_task(current_time_utc)
        
        # `action_dict` for the physical model is now for local components (e.g., HVAC)
        # For now, we pass an empty dict to use default behavior.
        local_action_dict = {}
        _, _, _, _, info = self.physical_dc_model.step(local_action_dict, self.logger)
        
        return info

    # --- Helper methods for remote queries ---
    def get_state_for_remote_query(self) -> Dict[str, float]:
        """
        Returns a concise dictionary of this DC's state, intended to be observed
        by other DTA_Managers when considering this DC as a destination.
        """
        return {
            "worker_queue_len": float(len(self.worker_commitment_queue)),
            "cpu_avail_pct": self.physical_dc_model.available_cores / self.physical_dc_model.total_cores,
            "gpu_avail_pct": self.physical_dc_model.available_gpus / self.physical_dc_model.total_gpus,
            "price": self.price_manager.get_current_price(),
            "ci": self.ci_manager.get_current_ci(norm=False)
        }