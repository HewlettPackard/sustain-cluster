import random
import numpy as np
import pandas as pd
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from rewards.base_reward import BaseReward
from torch.utils.tensorboard import SummaryWriter
from data.network_cost.network_delay import get_transmission_delay

class TaskSchedulingEnv(gym.Env):
    """
    Simplified IJCAI Version:
    1. Single Action per Step (One decision for the whole batch).
    2. Geographical Only (No Deferral).
    3. Aggregated Workload (All tasks treated as one resource request).
    """
    def __init__(self, cluster_manager, start_time, end_time, 
                 reward_fn: BaseReward, writer: SummaryWriter = None,
                 sim_config: dict = None, initial_seed_for_resets=None):
        
        super().__init__()
        self.cluster_manager = cluster_manager
        self.logger = getattr(self.cluster_manager, "logger", None)
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = pd.Timedelta(minutes=15)
        self.current_time = self.start_time
        self.reward_fn = reward_fn 
        self.writer = writer

        # Queues
        self.pending_tasks = [] 
        # Note: We removed deferred_tasks list as deferral is disabled
        self.in_transit_tasks = []
        self.current_tasks = []

        self.global_step = 0
        self.num_dcs = len(self.cluster_manager.datacenters)
        self.base_seed = initial_seed_for_resets if initial_seed_for_resets is not None else random.randint(0, 1_000_000)
        self.current_episode_count = 0

        # --- FORCED SIMPLIFICATIONS ---
        self.single_action_mode = True
        self.disable_defer_action = True
        
        # --- Observation Space ---
        # 1. Global Time (4: sin/cos day, sin/cos hour)
        # 2. Aggregated Workload State (5: Total Tasks, Total CPU, Total GPU, Avg Duration, Min Deadline)
        # 3. DC States (5 * Num_DCs: CPU_util, GPU_util, Mem_util, Carbon, Price)
        self.num_aggregated_task_features = 5
        self.obs_dim_aggregated = 4 + self.num_aggregated_task_features + (5 * self.num_dcs)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim_aggregated,), dtype=np.float32
        )

        # --- Action Space ---
        # Action is simply: Which DC ID to send the batch to? (0 to N-1)
        self.action_space = spaces.Discrete(self.num_dcs)
        
        if self.logger:
            self.logger.info(f"Simplified Env Initialized. Obs Dim: {self.obs_dim_aggregated}, Actions: {self.num_dcs} (DCs Only)")

    def _aggregate_task_observations(self, current_tasks_list: list) -> np.ndarray:
        """
        Create the 'Giant Task' observation.
        """
        # 1. Global Time Features
        day_of_year = self.current_time.dayofyear
        hour_of_day = self.current_time.hour + self.current_time.minute / 60.0
        time_features = np.array([
            np.sin(2 * np.pi * day_of_year / 365.0),
            np.cos(2 * np.pi * day_of_year / 365.0),
            np.sin(2 * np.pi * hour_of_day / 24.0),
            np.cos(2 * np.pi * hour_of_day / 24.0)
        ], dtype=np.float32)

        # 2. Aggregated Workload Features
        # We SUM the requirements to represent the total load of the batch
        num_tasks = float(len(current_tasks_list))
        if num_tasks > 0:
            total_cores_req = sum([task.cores_req for task in current_tasks_list])
            total_gpu_req = sum([task.gpu_req for task in current_tasks_list])
            # We average duration and deadline to give a sense of urgency/length
            avg_duration = np.mean([task.duration for task in current_tasks_list])
            min_time_to_deadline = np.min([
                max(0.0, (task.sla_deadline - self.current_time).total_seconds() / 60.0)
                for task in current_tasks_list
            ])
        else:
            total_cores_req = 0.0
            total_gpu_req = 0.0
            avg_duration = 0.0
            min_time_to_deadline = 0.0

        aggregated_task_features = np.array([
            num_tasks, total_cores_req, total_gpu_req, avg_duration, min_time_to_deadline
        ], dtype=np.float32)

        # 3. DC State Features
        dc_state_features = []
        for dc in self.cluster_manager.datacenters.values():
            dc_state_features.extend([
                dc.available_cores / dc.total_cores if dc.total_cores > 0 else 0,
                dc.available_gpus / dc.total_gpus if dc.total_gpus > 0 else 0,
                dc.available_mem / dc.total_mem_GB if dc.total_mem_GB > 0 else 0,
                float(dc.ci_manager.get_current_ci(norm=False)/1000.0),  # carbon intensity
                float(dc.price_manager.get_current_price())/100.0,       # energy price
            ])
        
        dc_state_features = np.array(dc_state_features, dtype=np.float32)

        # Combine
        return np.concatenate([time_features, aggregated_task_features, dc_state_features])

    def _get_obs(self):
        return self._aggregate_task_observations(self.current_tasks)

    def reset(self, seed=None, options=None):
        if seed is None:
            current_reset_seed = self.base_seed + self.current_episode_count
        else:
            current_reset_seed = seed
        
        self.current_episode_count += 1
        super().reset(seed=current_reset_seed)
        random.seed(current_reset_seed)
        
        self.current_time = self.start_time
        self.cluster_manager.reset(seed=current_reset_seed)

        self.in_transit_tasks.clear()
        self._load_new_tasks()
        self.global_step = 0

        return self._get_obs(), {}

    def step(self, action):
        """
        Action is an integer [0, Num_DCs - 1]. 
        0 = Send batch to DC_1
        1 = Send batch to DC_2
        ...
        """
        # 1. Process Incoming Tasks (from transit)
        remaining_in_transit = []
        for arrival_time, task, dc_name in self.in_transit_tasks:
            if arrival_time <= self.current_time:
                self.cluster_manager.datacenters[dc_name].pending_tasks.append(task)
            else:
                remaining_in_transit.append((arrival_time, task, dc_name))
        self.in_transit_tasks = remaining_in_transit

        # 2. Execute Action (Distribution)
        dc_list_values = list(self.cluster_manager.datacenters.values())
        
        # Ensure action is valid
        selected_dc_idx = int(action)
        assert 0 <= selected_dc_idx < self.num_dcs, f"Invalid Action ID: {selected_dc_idx}"
        
        dest_dc_chosen = dc_list_values[selected_dc_idx]
        
        processed_count = 0
        
        # Apply this decision to ALL current tasks (The "Batch")
        for task in self.current_tasks:
            processed_count += 1
            
            # SLA Check (if task expired while waiting to be batched)
            if self.current_time > task.sla_deadline:
                # Fallback: force to origin
                origin_dc_obj = next(dc for dc in dc_list_values if dc.dc_id == task.origin_dc_id)
                origin_dc_obj.pending_tasks.append(task)
                task.dest_dc_id = origin_dc_obj.dc_id
                task.dest_dc = origin_dc_obj
                continue

            # Assign to the Agent's Chosen DC
            task.dest_dc_id = dest_dc_chosen.dc_id
            task.dest_dc = dest_dc_chosen
            
            # Calculate Transmission Delay
            origin_loc = self.cluster_manager.get_dc_location(task.origin_dc_id)
            dest_loc = dest_dc_chosen.location
            delay_s = get_transmission_delay(origin_loc, dest_loc, self.cluster_manager.cloud_provider, task.bandwidth_gb)
            
            # Send to transit
            arrival_ts = self.current_time + pd.to_timedelta(delay_s, unit='s')
            dest_dc_name = next(name for name, dc_obj in self.cluster_manager.datacenters.items() if dc_obj.dc_id == task.dest_dc_id)
            self.in_transit_tasks.append((arrival_ts, task, dest_dc_name))

        # 3. Advance Simulation
        results = self.cluster_manager.step(self.current_time, logger=self.logger)
        
        # 4. Calculate Reward
        if self.reward_fn:
            reward = self.reward_fn(cluster_info=results, current_tasks=self.current_tasks, current_time=self.current_time)
        else: 
            reward = 0.0

        # 5. Logging
        if self.writer and self.reward_fn:
            self.writer.add_scalar(f"Reward/Total", reward, self.global_step)
            # Log components if available...

        # ==============================================================================
        # 5. METRICS EXTRACTION (UPDATED)
        # ==============================================================================
        
        # A. Transmission Metrics (Global)
        trans_carbon = results.get("transmission_emissions_total_kg", 0.0)
        trans_energy = results.get("transmission_energy_total_kwh", 0.0)
        trans_cost   = results.get("transmission_cost_total_usd", 0.0)

        # B. Operational Metrics (Aggregated from all DCs)
        op_carbon = 0.0
        op_energy = 0.0
        op_cost = 0.0
        total_water_usage = 0.0
        
        # PUE Calculation components
        total_facility_power_kw = 0.0
        total_ite_power_kw = 0.0

        # Performance Metrics
        step_sla_met = 0
        step_sla_violated = 0
        step_wait_time_sum = 0.0
        step_finished_count = 0

        for dc_name, dc_info in results['datacenter_infos'].items():
            # 1. Access Sub-Dictionaries
            agent_dc = dc_info.get('agent_dc', {})
            common = dc_info.get('__common__', {})
            
            # 2. Operational Costs & Sustainability
            # Use __common__ for kg consistency (agent_bat often has grams)
            op_carbon += common.get('carbon_emissions_kg', 0.0)
            op_energy += common.get('energy_consumption_kwh', 0.0)
            op_cost   += common.get('energy_cost_USD', 0.0)
            
            # 3. Water (Sustainability)
            # Extracted from 'agent_dc': 'dc_water_usage'
            total_water_usage += agent_dc.get('dc_water_usage', 0.0)

            # 4. PUE Components (Efficiency)
            # Extracted from 'agent_dc'
            total_facility_power_kw += agent_dc.get('dc_total_power_kW', 0.0)
            total_ite_power_kw      += agent_dc.get('dc_ITE_total_power_kW', 0.0)

            # 5. Performance (SLA)
            sla_stats = common.get('__sla__', {'met': 0, 'violated': 0})
            step_sla_met += sla_stats['met']
            step_sla_violated += sla_stats['violated']
            
            # 6. Latency Calculation
            # We iterate over the actual objects to calculate wait time
            finished_tasks = common.get('tasks_finished_this_step_objects', [])
            for task in finished_tasks:
                step_finished_count += 1
                # Wait time = Start Time - Creation Time (approx latency)
                # Or Duration + Wait Intervals. 
                # Let's use the helper method if available, or calculate simple duration
                wait_duration = (task.start_time - task.arrival_time).total_seconds() / 60.0 # Minutes
                step_wait_time_sum += max(0.0, wait_duration)

        # C. Calculate Derived Metrics
        # Avoid division by zero for PUE
        current_pue = 1.0
        if total_ite_power_kw > 0:
            current_pue = total_facility_power_kw / total_ite_power_kw

        avg_latency = 0.0
        if step_finished_count > 0:
            avg_latency = step_wait_time_sum / step_finished_count

        # ==============================================================================
        
        self.global_step += 1
        self.current_time += self.time_step
        self._load_new_tasks()

        done = self.current_time >= self.end_time
        truncated = done

        # D. Populate Info Dictionary
        info = {
            # Sustainability
            "total_carbon_kg": op_carbon + trans_carbon,
            "operational_carbon_kg": op_carbon,
            "transmission_carbon_kg": trans_carbon,
            "total_water_liters": total_water_usage,
            
            # Efficiency / Cost
            "total_energy_kwh": op_energy + trans_energy,
            "total_cost_usd": op_cost + trans_cost,
            "cluster_pue": current_pue,
            
            # Performance
            "sla_violated": step_sla_violated,
            "sla_met": step_sla_met,
            "avg_latency_min": avg_latency,
            
            # Debug
            "batch_size": processed_count,
            "selected_dc": dest_dc_chosen.dc_id
        }
        
        # Logging
        if self.writer:
            self.writer.add_scalar("Sustainability/Total_Carbon", info["total_carbon_kg"], self.global_step)
            self.writer.add_scalar("Efficiency/PUE", info["cluster_pue"], self.global_step)
            self.writer.add_scalar("Cost/Total_USD", info["total_cost_usd"], self.global_step)

        return self._get_obs(), reward, done, truncated, info

    def _load_new_tasks(self):
        """
        Load tasks for the current timestep.
        Since deferral is disabled, we simply fetch the new arrivals.
        """
        new_tasks = []
        if self.cluster_manager.strategy == "manual_rl":
            new_tasks = self.cluster_manager.get_tasks_for_timestep(self.current_time)
        
        self.current_tasks = new_tasks