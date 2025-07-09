from typing import Optional, Tuple
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces
from collections import deque

import envs.sustaindc.datacenter_model as DataCenter

class dc_gymenv(gym.Env):
    
    def __init__(self, observation_variables : list,
                       dc_memory_GB : float,
                       observation_space : spaces.Box,
                       action_variables: list,
                       action_space : spaces.Discrete,
                       action_mapping: dict,
                       ranges : dict,  # this data frame should be time indexed for the code to work
                       add_cpu_usage : bool,
                       add_gpu_usage : bool,  # Added GPU usage parameter
                       min_temp : float,
                       max_temp : float,
                       action_definition : dict,
                       DC_Config : dict,
                       seed : int = 123,
                       episode_length_in_time : pd.Timedelta = None,  # can be 1 week in minutes eg pd.Timedelta('7days')
                       ):
        """Creates the data center environment

        Args:
            observation_variables (list[str]): The partial list of variables that will be evaluated inside this evironment.The actual
                                                gym space may include other variables like sine cosine of hours, day of year, cpu usage,
                                                carbon intensity and battery state of charge.
            dc_memory_GB (float): The DRAM memory in a datacenter
            observation_space (spaces.Box): The gym observations space following gymnasium standard
            action_variables (list[str]): The list of action variables for the environment. It is used to create the info dict returned by
                                        the environment
            action_space (spaces.Discrete): The gym action space following gymnasium standard
            action_mapping (dict): A mapping from agent discrete action choice to actual delta change in setpoint. The mapping is defined in
                                    utils.make_pyeplus_env.py
            ranges (dict[str,list]): The upper and lower bounds on the observation_variables
            add_cpu_usage (bool): Whether to include CPU usage in the observation space
            add_gpu_usage (bool): Whether to include GPU usage in the observation space
            max_temp (float): The maximum temperature allowed for the CRAC setpoint
            min_temp (float): The minimum temperature allowed for the CRAC setpoint
            action_definition (dict): A mapping of the action name to the default or initialized value. Specified in utils.make_pyeplus_env.py
            episode_length_in_time (pd.Timedelta, optional): The maximum length after which the done flag should be True. Defaults to None. 
                                                            Setting none causes done to be True after data set is exausted.
        """
        super().__init__()

        self.observation_variables = observation_variables
        self.observation_space = observation_space
        self.action_variables = action_variables
        self.action_space = action_space
        self.action_mapping = action_mapping
        self.dc_memory_GB = dc_memory_GB
        self.ranges = ranges
        self.seed = seed
        self.add_cpu_usage = add_cpu_usage
        self.add_gpu_usage = add_gpu_usage  # Added GPU usage flag
        self.ambient_temp = 20
        self.scale_obs = False
        self.obs_max = []
        self.obs_min = []
        self.DC_Config = DC_Config
                
        # Initialize data center model with GPU support if available
        gpu_config = None
        if hasattr(self.DC_Config, 'RACK_GPU_CONFIG'):
            gpu_config = self.DC_Config.RACK_GPU_CONFIG
            
        self.dc = DataCenter.DataCenter_ITModel(num_racks=self.DC_Config.NUM_RACKS,
                                                dc_memory_GB = self.dc_memory_GB,
                                                rack_supply_approach_temp_list=self.DC_Config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                                                rack_CPU_config=self.DC_Config.RACK_CPU_CONFIG,
                                                rack_GPU_config=gpu_config,  # Add GPU config
                                                max_W_per_rack=self.DC_Config.MAX_W_PER_RACK,
                                                DC_ITModel_config=self.DC_Config)
        
        # Check if the data center has GPUs
        self.has_gpus = self.dc.has_gpus
        
        self.CRAC_Fan_load, self.CRAC_cooling_load, self.Compressor_load, self.CW_pump_load, self.CT_pump_load = None, None, None, None, None
        # self.HVAC_load = self.ranges['Facility Total HVAC Electricity Demand Rate(Whole Building)'][0]
        self.rackwise_cpu_pwr, self.rackwise_itfan_pwr, self.rackwise_memory_power, self.rackwise_gpu_pwr, self.rackwise_outlet_temp = [], [], [], [], []
        self.cpu_load_frac = 0.5
        self.gpu_load_frac = 0.5
        self.mem_load_frac = 0.5
        self.bat_SoC = 300*1e3  # all units are SI
        
        self.raw_curr_state = None
        self.raw_next_state = None
        self.raw_curr_stpt = action_definition['cooling setpoints']['initial_value']
        self.max_temp = max_temp
        self.min_temp = min_temp
        
        self.consecutive_actions = 0
        self.last_action = None
        self.action_scaling_factor = 1  # Starts with a scale factor of 1
        
        # IT + HVAC
        # self.power_lb_kW = (self.ranges['Facility Total Building Electricity Demand Rate(Whole Building)'][0] + 
        #                    self.ranges['Facility Total HVAC Electricity Demand Rate(Whole Building)'][0]) / 1e3
        # self.power_ub_kW = (self.ranges['Facility Total Building Electricity Demand Rate(Whole Building)'][1] + 
        #                    self.ranges['Facility Total HVAC Electricity Demand Rate(Whole Building)'][1] ) / 1e3


    
    def reset(self, *, seed=None, options=None):
        """
        Reset `dc_gymenv` to initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Environment options.

        Returns:
            raw_curr_state (List[float]): Current state of the environmment
            {} (dict): A dictionary that containing additional information about the environment state
        """

        super().reset(seed=self.seed)

        self.CRAC_Fan_load, self.CRAC_cooling_load, self.Compressor_load, self.CW_pump_load, self.CT_pump_load = None, None, None, None, None
        # self.HVAC_load = self.ranges['Facility Total HVAC Electricity Demand Rate(Whole Building)'][0]
        self.rackwise_cpu_pwr, self.rackwise_itfan_pwr, self.rackwise_gpu_pwr, self.rackwise_outlet_temp, self.rackwise_memory_power = [], [], [], [], []
        self.water_usage = None
        
        # self.raw_curr_state = self.get_obs()
        
        self.consecutive_actions = 0
        self.last_action = None
        self.action_scaling_factor = 1  # Starts with a scale factor of 1
        
        self.info = {
            'dc_ITE_total_power_kW': 0,
            'dc_CT_total_power_kW': 0,
            'dc_Compressor_total_power_kW': 0,
            'dc_HVAC_total_power_kW': 0,
            'dc_total_power_kW': 0,
            'dc_crac_setpoint_delta': 16,
            'dc_crac_setpoint': 16,
            'dc_cpu_workload_fraction': 1,
            'dc_gpu_workload_fraction': 1 if self.has_gpus else 0,  # Added GPU workload
            'dc_mem_workload_fraction': 1,
            'dc_int_temperature': 16,
            'dc_exterior_ambient_temp': 16,
            'dc_CW_pump_power_kW': 0,
            'dc_CT_pump_power_kW': 0,
            'dc_water_usage': 0,
        }
        
        if self.scale_obs:
            return self.normalize(self.raw_curr_state), self.info
        return None, self.info
    
    def step(self, raw_curr_stpt):
        """
        Makes an environment step in`dc_gymenv.

        Args:
            action_id (int): Action to take.

        Returns:
            observations (List[float]): Current state of the environmment
            reward (float): reward value.
            done (bool): A boolean value signaling the if the episode has ended.
            info (dict): A dictionary that containing additional information about the environment state
        """
        self.raw_curr_stpt = raw_curr_stpt  # Set a fixed CRAC setpoint to 18 C
    
        # Prepare load percentages for all racks
        ITE_load_pct_list = [self.cpu_load_frac*100 for i in range(self.DC_Config.NUM_RACKS)]
        mem_load_pct_list = [self.mem_load_frac*100 for i in range(self.DC_Config.NUM_RACKS)]
        
        # Prepare GPU load if GPUs are present
        GPU_load_pct_list = None
        if self.has_gpus:
            GPU_load_pct_list = [self.gpu_load_frac*100 for i in range(self.DC_Config.NUM_RACKS)]

        # Calculate power with GPU support
        result = self.dc.compute_datacenter_IT_load_outlet_temp(
            ITE_load_pct_list=ITE_load_pct_list, 
            CRAC_setpoint=self.raw_curr_stpt,
            GPU_load_pct_list=GPU_load_pct_list,
            MEMORY_load_pct_list=mem_load_pct_list
        )
        
        # Unpack result based on whether it includes GPU power
        if len(result) == 5:  # Includes GPU power
            self.rackwise_cpu_pwr, self.rackwise_itfan_pwr, rackwise_memory_power, self.rackwise_gpu_pwr, self.rackwise_outlet_temp = result
        else:  # Original version without GPU
            self.rackwise_cpu_pwr, self.rackwise_itfan_pwr, rackwise_memory_power, self.rackwise_outlet_temp = result
            self.rackwise_gpu_pwr = [0] * len(self.rackwise_cpu_pwr)
            
        avg_CRAC_return_temp = DataCenter.calculate_avg_CRAC_return_temp(
            rack_return_approach_temp_list=self.DC_Config.RACK_RETURN_APPROACH_TEMP_LIST,
            rackwise_outlet_temp=self.rackwise_outlet_temp
        )
        
        # Calculate total power including GPU if present
        data_center_total_ITE_Load = sum(self.rackwise_cpu_pwr) + sum(self.rackwise_itfan_pwr) + sum(self.rackwise_gpu_pwr) + sum(rackwise_memory_power)
        
        self.CRAC_Fan_load, self.CT_Cooling_load, self.CRAC_Cooling_load, self.Compressor_load, self.CW_pump_load, self.CT_pump_load = DataCenter.calculate_HVAC_power(
            CRAC_setpoint=self.raw_curr_stpt,
            avg_CRAC_return_temp=avg_CRAC_return_temp,
            ambient_temp=self.ambient_temp,
            data_center_full_load=data_center_total_ITE_Load,  # Use total load including GPU
            DC_Config=self.DC_Config
        )
        
        # print(f"Ambient Temp inside dc_gym for the calculation: {self.ambient_temp}")
        self.HVAC_load = self.CT_Cooling_load + self.Compressor_load

        # Set the additional attributes for the cooling tower water usage calculation
        self.dc.hot_water_temp = avg_CRAC_return_temp  # °C
        self.dc.cold_water_temp = self.raw_curr_stpt  # °C
        self.dc.wet_bulb_temp = self.wet_bulb  # °C from weather data

        # Calculate the cooling tower water usage
        self.water_usage = self.dc.calculate_cooling_tower_water_usage()

        # calculate reward
        self.reward = 0
                
        # calculate self.raw_next_state
        # self.raw_next_state = self.get_obs()
        
        # Update info dictionary with GPU information
        self.info = {
            'dc_ITE_total_power_kW': data_center_total_ITE_Load / 1e3,
            'dc_CT_total_power_kW': self.CT_Cooling_load / 1e3,
            'dc_Compressor_total_power_kW': self.Compressor_load / 1e3,
            'dc_HVAC_total_power_kW': (self.CT_Cooling_load + self.Compressor_load) / 1e3,
            'dc_total_power_kW': (data_center_total_ITE_Load + self.CT_Cooling_load + self.Compressor_load) / 1e3,
            'dc_crac_setpoint': self.raw_curr_stpt,
            'dc_cpu_workload_fraction': self.cpu_load_frac,
            'dc_gpu_workload_fraction': self.gpu_load_frac if self.has_gpus else 0,  # Added GPU workload
            'dc_int_temperature': np.mean(self.rackwise_outlet_temp),
            'dc_exterior_ambient_temp': self.ambient_temp,
            'dc_CW_pump_power_kW': self.CW_pump_load,
            'dc_CT_pump_power_kW': self.CT_pump_load,
            'dc_water_usage': self.water_usage,
        }
        
        # Done and truncated are managed by the main class
        truncated = False
        done = False 
        
        # Return processed/unprocessed state to agent
        if self.scale_obs:
            return self.normalize(self.raw_next_state), self.reward, done, truncated, self.info
        return None, self.reward, done, truncated, self.info

    def normalize(self, obs):
        """
        Normalizes the observation.
        """
        return np.float32((obs-self.obs_min)/self.obs_delta)

    def get_obs(self):
        """
        Returns the observation at the current time step.

        Returns:
            observation (List[float]): Current state of the environmment.
        """
        zone_air_therm_cooling_stpt = self.min_temp  # in C, default for reset state
        if self.raw_curr_stpt is not None:
            zone_air_therm_cooling_stpt = self.raw_curr_stpt
        
        zone_air_temp = self.obs_min[2]  # in C, default for reset state
        if self.rackwise_outlet_temp:
            zone_air_temp = sum(self.rackwise_outlet_temp)/len(self.rackwise_outlet_temp)

        # 'Facility Total HVAC Electricity Demand Rate(Whole Building)'  ie 'HVAC POWER'
        hvac_power = self.HVAC_load

        # Calculate 'Facility Total Building Electricity Demand Rate(Whole Building)' i.e. 'IT POWER'
        it_power = 0

        # Add CPU power if available
        if self.rackwise_cpu_pwr:
            it_power += sum(self.rackwise_cpu_pwr)

        # Add IT fan power if available
        if hasattr(self, 'rackwise_itfan_pwr') and self.rackwise_itfan_pwr:
            it_power += sum(self.rackwise_itfan_pwr)

        # Add GPU power if available
        if self.rackwise_gpu_pwr:
            it_power += sum(self.rackwise_gpu_pwr)

        # If no power components were available, use the fallback value
        if it_power == 0:

            it_power = self.ranges['Facility Total Building Electricity Demand Rate(Whole Building)'][0]

        # Basic observation list
        obs = [self.ambient_temp, zone_air_therm_cooling_stpt, zone_air_temp, hvac_power, it_power]            

        return obs

    def update_workloads(self, cpu_load, mem_load, gpu_load):
        """
        Updates the current CPU, GPU amd MEMORY utilization. Fraction between 0.0 and 1.0
        """
        if 0.0 > cpu_load or cpu_load > 1.0:
            print('CPU load out of bounds')
        assert 0.0 <= cpu_load <= 1.0, 'CPU load out of bounds'
        self.cpu_load_frac = cpu_load
        if 0.0 > gpu_load or gpu_load > 1.0:
            print('GPU load out of bounds')
        assert 0.0 <= gpu_load <= 1.0, 'GPU load out of bounds'
        self.gpu_load_frac = gpu_load
        if 0.0 > mem_load or mem_load > 1.0:
            print('Memory load out of bounds')
        assert 0.0 <= mem_load <= 1.0, 'Memory load out of bounds'
        self.mem_load_frac = mem_load
    
    def set_ambient_temp(self, ambient_temp, wet_bulb):
        """
        Updates the external temperature.
        """
        self.ambient_temp = ambient_temp
        self.wet_bulb = wet_bulb
        
    def set_bat_SoC(self, bat_SoC):
        """
        Updates the battery state of charge.
        """
        self.bat_SoC = bat_SoC
