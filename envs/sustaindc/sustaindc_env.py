import os
import sys
import random
import datetime
from collections import deque
from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from utils.make_envs import make_bat_fwd_env, make_dc_env, make_ls_env
from utils.managers import CI_Manager, ElectricityPrice_Manager, Time_Manager, Weather_Manager
from utils.utils_cf import get_energy_variables, get_init_day, obtain_paths
from rl_components.agent_net import ActorNet
from utils.running_stats import RunningStats # Import RunningStats class

from ..env_config import EnvConfig

MAX_WAIT_TIMESTEPS = 4 * 8  # 8 hours, with 15-minute intervals = 32 timesteps

# --- Action Mapping ---
HVAC_ACTION_MAPPING = {0: -1.0, 1: 0.0, 2: 1.0}
DEFAULT_HVAC_SETPOINT = 22.0
MIN_HVAC_SETPOINT = 18.0
MAX_HVAC_SETPOINT = 27.0

class SustainDC(gym.Env):
    def __init__(self, env_config):
        '''
        Initialize the SustainDC environment.

        Args:
            env_config (dict): Dictionary containing parameters as defined in 
                               EnvConfig above.
        '''
        super().__init__()

        # Initialize the environment config
        env_config = EnvConfig(env_config)
        self.env_config = env_config

        self.dc_id = env_config['dc_id']
        self.network_cost_per_gb = env_config.get('network_cost_per_gb', 0.0)  # Default to 0 if missing

        # Create environments and agents
        self.agents = env_config['agents']
        self.rbc_agents = env_config.get('rbc_agents', [])
        
        self.location = env_config['location']
                
        self.max_bat_cap_Mw = env_config['max_bat_cap_Mw']
        
        self.datacenter_capacity_mw = env_config.get('datacenter_capacity_mw', 1.0)
        self.total_cores = env_config['total_cores']
        self.total_gpus = env_config['total_gpus']
        self.total_mem_GB = env_config['total_mem']
        
        self.dc_config_file = env_config['dc_config_file']
        self.timezone_shift = env_config['timezone_shift']
        self.duration_days = env_config['duration_days']
        
        # Assign month according to worker index, if available
        if hasattr(env_config, 'worker_index'):
            self.month = int((env_config.worker_index - 1) % 12)
        else:
            self.month = env_config.get('month')

        self.evaluation_mode = env_config['evaluation']

        self._agent_ids = set(self.agents)
        
        n_vars_energy, n_vars_battery = 0, 0  # For partial observability (for p.o.)
        n_vars_ci = 8

        # **✅ Pick a random simulation year for variability**
        self.simulation_year = None 

        # Set available resources equal to total at initialization
        self.available_cores = self.total_cores
        self.available_gpus = self.total_gpus
        self.available_mem = self.total_mem_GB

        # self.ls_env = make_ls_env(month=self.month, test_mode=self.evaluation_mode, n_vars_ci=n_vars_ci, 
        #                           n_vars_energy=n_vars_energy, n_vars_battery=n_vars_battery, queue_max_len=1000)
        self.dc_env, _ = make_dc_env(month=self.month, location=self.location, max_bat_cap_Mw=self.max_bat_cap_Mw, use_ls_cpu_load=True, 
                                             datacenter_capacity_mw=self.datacenter_capacity_mw, dc_config_file=self.dc_config_file, add_cpu_usage=False, total_cores=self.total_cores, total_gpus=self.total_gpus, dc_memory_GB=self.total_mem_GB)
        self.bat_env = make_bat_fwd_env(month=self.month, max_bat_cap_Mwh=self.dc_env.ranges['max_battery_energy_Mwh'], 
                                        max_dc_pw_MW=10, 
                                        dcload_max=10,
                                        dcload_min=1,
                                        n_fwd_steps=n_vars_ci, init_day=0)

        # self.bat_env.dcload_max = self.dc_env.power_ub_kW / 4  # Assuming 15 minutes timestep. Kwh
        # self.bat_env.dcload_min = self.dc_env.power_lb_kW / 4  # Assuming 15 minutes timestep. Kwh
        
        self._obs_space_in_preferred_format = True
        
        self.observation_space = []
        self.action_space = []

        # Running & Pending Task Queues
        self.running_tasks = []
        self.pending_tasks = deque()
        self.current_time_task = 0
        
        # HVAC agent
        self.use_rl_hvac = env_config.get('use_rl_hvac', False)
        self.hvac_controller_type = env_config.get('hvac_controller_type', 'none').lower()
        self.hvac_policy_path = env_config.get('hvac_policy_path', None)
        self.hvac_policy = None
        self.hvac_obs_dim = None
        self.hvac_act_dim = 3 # Discrete actions
        self.hvac_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hvac_obs_stats = None # For normalization

        # --- Load HVAC Policy if Configured ---
        if self.use_rl_hvac and self.hvac_policy_path and self.hvac_controller_type != 'none':
            if os.path.exists(self.hvac_policy_path):
                try:
                    print(f"[DC {self.dc_id}] Loading HVAC policy ({self.hvac_controller_type}) from: {self.hvac_policy_path}")
                    checkpoint = torch.load(self.hvac_policy_path, map_location=self.hvac_device)

                    self.hvac_obs_dim = checkpoint.get('hvac_obs_dim')
                    if self.hvac_obs_dim is None: raise ValueError("Checkpoint must contain 'hvac_obs_dim'")

                    # Assuming ActorNet is used for discrete policy in both SAC/PPO checkpoints
                    self.hvac_policy = ActorNet(self.hvac_obs_dim, self.hvac_act_dim, 64).to(self.hvac_device)
                    policy_state_dict_key = 'actor_state_dict' # Standard key expected
                    if policy_state_dict_key not in checkpoint: raise ValueError(f"Checkpoint missing '{policy_state_dict_key}'")
                    self.hvac_policy.load_state_dict(checkpoint[policy_state_dict_key])
                    self.hvac_policy.eval()

                    # Load observation normalization stats if available
                    if 'obs_stats_state' in checkpoint and checkpoint['obs_stats_state'] is not None:
                        self.hvac_obs_stats = RunningStats(shape=(self.hvac_obs_dim,))
                        self.hvac_obs_stats.set_state(checkpoint['obs_stats_state'])
                        print(f"[DC {self.dc_id}] Loaded observation stats for HVAC policy.")
                    else:
                        print(f"[DC {self.dc_id}] WARNING: No observation stats found in HVAC checkpoint. Policy will use raw observations.")

                    print(f"[DC {self.dc_id}] Successfully loaded HVAC policy.")

                except Exception as e:
                     print(f"[DC {self.dc_id}] ERROR loading HVAC policy from {self.hvac_policy_path}: {e}. Reverting to default HVAC.")
                     self.use_rl_hvac = False; self.hvac_policy = None; self.hvac_obs_stats = None
            else:
                print(f"[DC {self.dc_id}] WARNING: use_rl_hvac is True but policy path not found: {self.hvac_policy_path}. Using default HVAC action.")
                self.use_rl_hvac = False
        elif self.use_rl_hvac:
             print(f"[DC {self.dc_id}] WARNING: use_rl_hvac is True but hvac_policy_path or type not provided/invalid. Using default HVAC action.")
             self.use_rl_hvac = False

    def _create_dc_environment(self, env_config):
        """
        Creates the internal datacenter environment.
        """
        return SustainDC(env_config)
    
    def can_schedule(self, task):
        return (task.cores_req <= self.available_cores and
                task.gpu_req <= self.available_gpus and
                task.mem_req <= self.available_mem)
    
    def release_resources(self, current_time, logger):
        """
        Releases resources from completed tasks and logs their completion.
        """
        finished_tasks = [task for task in self.running_tasks if task.finish_time <= current_time]
        
        for task in finished_tasks:
            self.available_cores += task.cores_req
            self.available_gpus += task.gpu_req
            self.available_mem += task.mem_req
            task.sla_met = task.finish_time <= task.sla_deadline

            if logger:
                logger.info(f"[{current_time}] Task {task.job_name} finished, resources released. "
                            f"DC{self.dc_id} resources available: {self.available_cores:.3f} cores, "
                            f"{self.available_gpus:.3f} GPUs, {self.available_mem:.3f} MEM.")

        self.running_tasks = [task for task in self.running_tasks if task.finish_time > current_time]
        
        return finished_tasks


    def try_to_schedule_task(self, task, current_time, logger):
        """
        Attempts to schedule a task if sufficient resources are available.
        
        If the task is successfully scheduled:
            - Resources are allocated.
            - The task is moved to the running queue.
            - Logs the scheduling details.

        If resources are insufficient:
            - The task remains in the pending queue.
            - The system tracks how long the task has been waiting.
            - If the wait time exceeds MAX_WAIT_INTERVALS, the task is dropped.

        Returns:
            - `True` if the task was scheduled.
            - `False` if the task was re-added to the pending queue.
        """
        def log_info(msg):
            if logger:
                logger.info(msg)

        def log_warn(msg):
            if logger:
                logger.warning(msg)
            
        if self.can_schedule(task):
            # **Allocate resources**
            self.available_cores -= task.cores_req
            self.available_gpus -= task.gpu_req
            self.available_mem -= task.mem_req
            task.start_time = current_time
            task.finish_time = current_time + pd.Timedelta(minutes=task.duration)

            # **Compute network cost**
            # TODO: Implement network cost calculation based on task properties
            network_cost = -1

            # **Move task to running queue**
            self.running_tasks.append(task)

            log_info(f"[{current_time}] Task {task.job_name} scheduled successfully on DC{self.dc_id}. "
                    f"(CPU: {task.cores_req:.2f}, GPU: {task.gpu_req:.2f}, MEM: {task.mem_req:.2f}, "
                    f"Bandwidth: {task.bandwidth_gb:.4f}GB). "
                    f"Remaining: {self.available_cores:.2f} cores, {self.available_gpus:.2f} GPUs, "
                    f"{self.available_mem:.2f} GB MEM.")

            log_info(f"[{current_time}] Task {task.job_name} started and will finish at {task.finish_time}.")
            return True
        else:
            # **Task couldn't be scheduled - track wait time**
            task.increment_wait_intervals()

            # --- Always re-add the task to the pending queue ---
            self.pending_tasks.append(task) # Re-add task to queue for next cycle
            log_info(f"[{current_time}] Task {task.job_name} moved to pending queue of DC{self.dc_id} (wait: {task.wait_intervals}).")
            return False
            
    def set_seed(self, seed=None):
        """Set the random seed for the environment."""
        if seed is None:
            seed = seed or 1
            np.random.seed(seed)
            random.seed(seed)
            if torch is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            self._seed_spaces()

    def _seed_spaces(self):
        """Seed the action and observation spaces."""
        if hasattr(self, 'action_space') and hasattr(self.action_space, 'seed'):
            self.action_space.seed(self.seed)
        if hasattr(self, 'observation_space') and hasattr(self.observation_space, 'seed'):
            self.observation_space.seed(self.seed)

    def reset(self, init_year=None, init_day=None, init_hour=None, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Environment options.

        Returns:
            states (dict): Dictionary of states.
            infos (dict): Dictionary of infos.
        """
        self.set_seed(seed)
        # Reset resources to initial values
        self.available_cores = self.total_cores
        self.available_gpus = self.total_gpus
        self.available_mem = self.total_mem_GB

        self.running_tasks.clear()
        self.pending_tasks.clear()
        self.current_time_task = 0
        
        
        # Reset HVAC setpoint state
        self.current_crac_setpoint = DEFAULT_HVAC_SETPOINT

        # **Reinitialize the managers with new paths**
        self.simulation_year = init_year
        self.t_m = Time_Manager(init_day, timezone_shift=self.timezone_shift, duration_days=self.duration_days)
        self.ci_manager = CI_Manager(location=self.location, simulation_year=self.simulation_year, timezone_shift=self.timezone_shift)
        self.weather_manager = Weather_Manager(location=self.location, simulation_year=self.simulation_year, timezone_shift=self.timezone_shift)
        self.price_manager = ElectricityPrice_Manager(location=self.location, simulation_year=self.simulation_year, timezone_shift=self.timezone_shift)

        # Reset termination and reward flags for all agents
        self.ls_terminated = self.dc_terminated = self.bat_terminated = False
        self.ls_truncated = self.dc_truncated = self.bat_truncated = False

        # Adjust based on local timezone
        local_init_day = (init_day + int((init_hour + self.timezone_shift) / 24)) % 365
        local_init_hour = (init_hour + self.timezone_shift) % 24
        self.current_hour = local_init_hour
        
        t_i = self.t_m.reset(init_day=local_init_day, init_hour=local_init_hour, seed=seed)
        temp, norm_temp, wet_bulb, norm_wet_bulb = self.weather_manager.reset(init_day=local_init_day, init_hour=local_init_hour, seed=seed)
        ci_i, ci_i_future, ci_i_denorm = self.ci_manager.reset(init_day=local_init_day, init_hour=local_init_hour, seed=seed)
        price_i = self.price_manager.reset(init_day=local_init_day, init_hour=local_init_hour, seed=seed)

        # Reset sub-environments
        try:
            self.dc_env.set_ambient_temp(temp, wet_bulb)
            _, self.dc_info = self.dc_env.reset(seed=seed) # Pass seed
            bat_s, self.bat_info = self.bat_env.reset(seed=seed) # Pass seed
        except Exception as e:
            print(f"ERROR during sub-env reset in DC {self.dc_id}: {e}")
            self.dc_info = {}; self.bat_info = {} # Default empty on error
                
        current_cpu_workload = 0.0  #self.workload_m.get_current_workload()
        current_gpu_workload = 0.0 
        current_mem_workload = 0.0
        self.dc_env.update_workloads(current_cpu_workload, current_mem_workload, current_gpu_workload)

        # Update ci in the battery environment
        self.bat_env.update_ci(ci_i_denorm, ci_i_future[0])

        # States should be a dictionary with agent names as keys and their observations as values
        states = {}
        self.infos = {}
        # Update states and infos considering the agents defined in the environment config self.agents.
        if "agent_ls" in self.agents:
            states["agent_ls"] = self.ls_state
        if "agent_dc" in self.agents:
            states["agent_dc"] = self.dc_state
        if "agent_bat" in self.agents:
            states["agent_bat"] = self.bat_state

        # Prepare the infos dictionary with common and individual agent information
        self.infos = {
            'agent_dc': self.dc_info,
            'agent_bat': self.bat_info,
            '__common__': {
                'time': t_i,
                'cpu_workload': current_cpu_workload,
                'gpu_workload': current_gpu_workload,
                'memory_utilization': current_mem_workload,
                'weather': temp,
                'ci': ci_i,
                'ci_future': ci_i_future,
            }
        }
        
        
        # available_actions = None
        self._last_cpu_workload = 0.0
        self._last_gpu_workload = 0.0
        
        return states, self.infos
    
    def step(self, action_dict, logger):
        """
        Step the environment.

        Args:
            action_dict: Dictionary of actions of each agent defined in self.agents.
  
        Returns:
            obs (dict): Dictionary of observations/states.
            rews (dict): Dictionary of rewards.
            terminated (dict): Dictionary of terminated flags.
            truncated (dict): Dictionary of truncated flags.
            infos (dict): Dictionary of infos.
        """
        obs, rew, terminateds, truncateds, info = {}, {}, {}, {}, {}
        terminateds["__all__"] = False
        truncateds["__all__"] = False
        
        # Perform actions for each agent and update their respective environments
        self._perform_actions(action_dict)
    
        # Step the managers (time, workload, weather, CI) (t+1)
        day, hour, t_i, manager_truncated = self.t_m.step()
        # workload = self.workload_m.step()
        temp, norm_temp, wet_bulb, norm_wet_bulb = self.weather_manager.step()
        ci_i, ci_i_future, ci_i_denorm = self.ci_manager.step()
        price_i = self.price_manager.step()
        
        # New logic for the task execution scheduler
        self.current_hour = hour

        # **1. Release resources from completed tasks**
        finished_tasks = self.release_resources(self.current_time_task, logger)

        num_tasks_assigned = 0
        routed_tasks_this_step = []
        
        # **2. Try scheduling pending tasks (FIFO order)**
        for _ in range(len(self.pending_tasks)):  # Process each task once
            task = self.pending_tasks.popleft()  # Take the first task from the queue
            scheduled = self.try_to_schedule_task(task, self.current_time_task, logger)  # Let schedule_task() handle scheduling or re-queuing
            if scheduled:
                num_tasks_assigned += 1
                routed_tasks_this_step.append(task)


        # **3. Log resource utilization**
        used_cores = round(self.total_cores, 6) - round(self.available_cores, 6)
        used_gpu = round(self.total_gpus, 6) - round(self.available_gpus, 6)
        used_mem = round(self.total_mem_GB, 6) - round(self.available_mem, 6)

        # Convert used_cores => HPC environment usage
        # e.g. self.dc_env.set_shifted_wklds(used_cores / self.total_cores) or similar
        # HPC environment can produce the final usage metrics
        # I need a value between 0 and 1 for the workload.
        # At this time, we are only focused on the cpu usage.
        cpu_workload = used_cores / self.total_cores
        gpu_workload = used_gpu / self.total_gpus
        mem_util = used_mem / self.total_mem_GB
        # print(f"[{self.current_time_task}] DC:{self.dc_id} Running: {len(self.running_tasks)}, Pending: {len(self.pending_tasks)}")
        if logger:
            logger.info(f"[{self.current_time_task}] DC:{self.dc_id} Running: {len(self.running_tasks)}, Pending: {len(self.pending_tasks)}, CPU Utilization: {cpu_workload:.2%}, GPU Utilization: {gpu_workload:.2%}, MEM Utilization: {mem_util:.2%}")

        # Update environment states with new values from managers
        self._update_environments(cpu_workload, gpu_workload, mem_util, temp, wet_bulb, ci_i_denorm, ci_i_future, day, hour)

        # Create observations for the next step based on updated environment states
        # Populate observation dictionary based on updated states
        obs = self._populate_observation_dict()

        # Update the self.infos dictionary, similar to how it's done in the reset method
        self.infos = {
            'agent_dc': self.dc_info,
            'agent_bat': self.bat_info,
            '__common__': {
                'time': t_i,
                # 'workload': workload,
                'weather': temp,
                'ci': ci_i_denorm,
                'price_USD_kwh': price_i,
                'routed_tasks_this_step': routed_tasks_this_step,
            }
        }
        
        sla_stats = {
                    "met": 0,
                    "violated": 0,
                    }

        for task in finished_tasks:
            if task.sla_met:
                sla_stats["met"] += 1
            else:
                sla_stats["violated"] += 1
        
        # Calculate percentages
        cpu_util = used_cores / self.total_cores
        gpu_util = used_gpu / self.total_gpus
        mem_util = used_mem / self.total_mem_GB

        # Compute number of tasks
        running_count = len(self.running_tasks)
        pending_count = len(self.pending_tasks)

        # Estimate energy cost
        energy_kwh = self.bat_info.get("bat_total_energy_with_battery_KWh", 0)
        energy_cost_USD = energy_kwh * price_i/1000  # Assuming price_i is in $ per kWh
        energy_ci = self.bat_info.get("bat_avg_CI", 0)  # Average carbon intensity in gCO₂eq/kWh
        # Estimate CO₂ emissions, ci_i is in gCO₂eq/kWh
        carbon_emissions_kg = energy_kwh * energy_ci / 1000  # Convert gCO₂eq to kgCO₂eq

        # Add to common info
        self.infos["__common__"].update({
            "cpu_util_percent": cpu_util * 100,
            "gpu_util_percent": gpu_util * 100,
            "mem_util_percent": mem_util * 100,

            "running_tasks": running_count,
            "pending_tasks": pending_count,
            "tasks_assigned": num_tasks_assigned,

            # "dc_total_power_kW": self.dc_info.get("dc_total_power_kW", None),
            "energy_consumption_kwh": energy_kwh,
            "tasks_finished_this_step_objects": finished_tasks, # Add the list of objects
            "finished_tasks_count": len(finished_tasks),

            "energy_cost_USD": energy_cost_USD,
            "carbon_emissions_kg": carbon_emissions_kg,
            "__sla__": sla_stats,
            "hvac_setpoint_c": self.current_crac_setpoint

        })
        
        # update the truncated and terminated flags
        if manager_truncated:
            truncateds["__all__"] = True
            print(f"[DC {self.dc_id}] Environment truncated at time {self.current_time_task}.")

        return obs, rew, terminateds, truncateds, self.infos


    def _perform_actions(self, action_dict):
        """
        Performs actions for internal agents (DC, Battery).
        Uses loaded RL policy for HVAC if configured.
        Note: action_dict is currently ignored as sub-agents use fixed/RL logic here.
        """
        # --- Determine Target HVAC Setpoint ---
        target_hvac_setpoint = self.current_crac_setpoint # Default to maintain

        if self.use_rl_hvac and self.hvac_policy:
            norm_obs_vector = self._get_hvac_observation()
            if norm_obs_vector is not None:
                obs_tensor = torch.FloatTensor(norm_obs_vector).unsqueeze(0).to(self.hvac_device)
                with torch.no_grad():
                    logits = self.hvac_policy(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    discrete_action = dist.sample().item()

                # Translate discrete action to setpoint change using mapping
                setpoint_delta = HVAC_ACTION_MAPPING.get(discrete_action, 0.0) # Default to 0 change if action invalid
                target_hvac_setpoint = self.current_crac_setpoint + setpoint_delta

            else: # Handle observation error
                 print(f"[DC {self.dc_id}] Using default HVAC action due to observation error.")
                 target_hvac_setpoint = DEFAULT_HVAC_SETPOINT # Revert to default on error? Or just maintain? Let's maintain.
                 # target_hvac_setpoint = self.current_crac_setpoint
        else:
            # Use default fixed setpoint if RL HVAC is not enabled/loaded
            target_hvac_setpoint = DEFAULT_HVAC_SETPOINT

        # Clip target setpoint to valid range
        target_hvac_setpoint = np.clip(target_hvac_setpoint, MIN_HVAC_SETPOINT, MAX_HVAC_SETPOINT)

        # Update the state *before* stepping the DC environment
        self.current_crac_setpoint = target_hvac_setpoint

        # --- Step the internal datacenter environment with the chosen ABSOLUTE setpoint ---
        # dc_env (dc_gymenv) step expects the absolute setpoint
        try:
            _, _, self.dc_terminated, self.dc_truncated, self.dc_info = self.dc_env.step(self.current_crac_setpoint)
        except Exception as e:
            print(f"ERROR during dc_env.step in DC {self.dc_id}: {e}")
            # Handle error state? Maybe set default info?
            self.dc_info = self.dc_info or {} # Ensure dc_info exists


        # --- Battery Action (Fixed) ---
        DO_NOTHING_BATTERY = 2
        try:
            dc_total_power_kw = self.dc_info.get('dc_total_power_kW', 0) # Use .get for safety
            self.bat_env.set_dcload(dc_total_power_kw / 1e3)
            self.bat_state, _, self.bat_terminated, self.bat_truncated, self.bat_info = self.bat_env.step(DO_NOTHING_BATTERY)
        except Exception as e:
             print(f"ERROR during bat_env.step in DC {self.dc_id}: {e}")
             self.bat_info = self.bat_info or {} # Ensure bat_info exists



    def _update_environments(self, cpu_workload, gpu_workload, mem_util, temp, wet_bulb, ci_i_denorm, ci_i_future, current_day, current_hour):
        """ Update the internal environment states based on the manager's outputs. """
        # Store loads for potential use in next step's HVAC observation
        self._last_cpu_workload = cpu_workload
        self._last_gpu_workload = gpu_workload
        # NOTE: mem_util is not currently used in default HVAC obs, but store if needed

        try:
            self.dc_env.set_ambient_temp(temp, wet_bulb)
            # Pass GPU/Mem loads to the internal dc_env update if it uses them
            self.dc_env.update_workloads(cpu_workload, mem_util, gpu_workload)
        except Exception as e:
            print(f"ERROR during dc_env update in DC {self.dc_id}: {e}")

        try:
            self.bat_env.update_ci(ci_i_denorm, ci_i_future[0])
        except Exception as e:
             print(f"ERROR during bat_env update in DC {self.dc_id}: {e}")


    def _populate_observation_dict(self):
        """Generate the observation dictionary for all agents."""
        obs = {}
        if "agent_ls" in self.agents:
            obs['agent_ls'] = self.ls_state
        if "agent_dc" in self.agents:
            obs['agent_dc'] = self.dc_state
        if "agent_bat" in self.agents:
            obs['agent_bat'] = self.bat_state
        return obs

    
    def state(self):
        """
        Get the state of the environment.

        Returns:
            np.ndarray: State of the environment.
        """
        print('Calling the method state() of SustainDC')
        states = tuple(
            self.scenario.observation(  # pylint: disable=no-member
                self.world.agents[self._index_map[agent]], self.world  # pylint: disable=no-member
            ).astype(np.float32)
            for agent in self.possible_agents  # pylint: disable=no-member
        )
        return np.concatenate(states, axis=None)

    def _update_current_time_task(self, current_time):
        """
        Update the current time of the task.
        """
        self.current_time_task = current_time

    def _get_hvac_observation(self):
        """ Constructs the observation vector for the loaded HVAC policy. """
        if not self.use_rl_hvac or self.hvac_policy is None or self.hvac_obs_dim is None:
            print("DEBUG: _get_hvac_observation called but RL HVAC not active/loaded.")
            return None # Should not be called if not active

        # --- Construct observation based on the expected features ---
        try:
            # Features need to be available *before* dc_env.step is called
            ambient_temp = self.weather_manager._current_temp # Access current temp from manager
            # Loads are from the *previous* step's update
            cpu_load = self._last_cpu_workload
            gpu_load = self._last_gpu_workload
            # Use the *current* setpoint (before the action modifies it for the *next* step)
            setpoint_state = self.current_crac_setpoint
            # Get time features from the time manager
            current_hour = self.t_m.hour
            sin_hour = np.sin(2 * np.pi * current_hour / 24.0)
            cos_hour = np.cos(2 * np.pi * current_hour / 24.0)

            # --- Ensure order matches training and hvac_obs_dim ---
            # Example assuming obs_dim = 6: [sinH, cosH, ambT, cpuL, gpuL, currentSP]
            obs_list = [
                sin_hour, cos_hour, ambient_temp, cpu_load, gpu_load, setpoint_state
            ]

            # Add more features if hvac_obs_dim > 6, e.g.:
            # if self.hvac_obs_dim > 6:
            #     obs_list.append(self.ci_manager.get_current_ci(norm=True)) # Example: Normalized CI
            # if self.hvac_obs_dim > 7:
            #     obs_list.append(self.price_manager.get_current_price() / 1000) # Example: Normalized Price

            # --- Verification and Formatting ---
            if len(obs_list) != self.hvac_obs_dim:
                raise ValueError(f"Observation construction mismatch! Expected {self.hvac_obs_dim}, got {len(obs_list)} features.")

            raw_obs = np.array(obs_list, dtype=np.float32)

            # --- Apply Normalization if stats were loaded ---
            if self.hvac_obs_stats:
                norm_obs = self.hvac_obs_stats.normalize(raw_obs)
                return norm_obs
            else:
                # Return raw observation if no normalization stats available
                # Note: This might lead to poor performance if policy was trained with normalization
                return raw_obs

        except AttributeError as e:
             print(f"ERROR in _get_hvac_observation: Missing attribute, likely manager not ready? {e}")
             return None # Indicate error
        except Exception as e:
            print(f"ERROR creating HVAC observation: {e}")
            return None # Indicate error
        
    def get_current_carbon_intensity(self, norm=False):
        """Helper method to get CI from the internal manager."""
        if hasattr(self, 'ci_manager'):
            return self.ci_manager.get_current_ci(norm=norm)
        else:
            # Return a default high value or raise error if manager not initialized
            print(f"Warning: CI Manager not available for DC {self.dc_id}")
            return 1000.0 # Example default high value