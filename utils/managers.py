import os
import json
import numpy as np
import pandas as pd


class CoherentNoise:
    """Class to add coherent noise to the data.

        Args:
            base (List[float]): Base data
            weight (float): Weight of the noise to be added
            desired_std_dev (float, optional): Desired standard deviation. Defaults to 0.1.
            scale (int, optional): Scale. Defaults to 1.
    """
    def __init__(self, base, weight, desired_std_dev=0.1, scale=1):
        """Initialize CoherentNoise class

        Args:
            base (List[float]): Base data
            weight (float): Weight of the noise to be added
            desired_std_dev (float, optional): Desired standard deviation. Defaults to 0.1.
            scale (int, optional): Scale. Defaults to 1.
        """
        self.base = base
        self.weight = weight
        self.desired_std_dev = desired_std_dev
        self.scale = scale
        
    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)


    def generate(self, n_steps):
        """
        Generate coherent noise 

        Args:
            n_steps (int): Length of the data to generate.

        Returns:
            numpy.ndarray: Array of generated coherent noise.
        """
        steps = self.rng.normal(loc=0, scale=self.scale, size=n_steps)
        random_walk = np.cumsum(self.weight * steps)
        normalized_noise = (random_walk / np.std(random_walk)) * self.desired_std_dev
        return self.base + normalized_noise


# Function to normalize a value v given a minimum and a maximum
def normalize(v, min_v, max_v):
    """Function to normalize values

    Args:
        v (float): Value to be normalized
        min_v (float): Lower limit
        max_v (float): Upper limit

    Returns:
        float: Normalized value
    """
    return (v - min_v)/(max_v - min_v)

# Function to generate cosine and sine values for a given hour and day
def sc_obs(current_hour, current_day):
    """Generate sine and cosine of the hour and day

    Args:
        current_hour (int): Current hour of the day
        current_day (int): Current day of the year

    Returns:
        List[float]: Sine and cosine of the hour and day
    """
    # Normalize and round the current hour and day
    two_pi = np.pi * 2

    norm_hour = round(current_hour/24, 3) * two_pi
    norm_day = round((current_day)/365, 3) * two_pi
    
    # Calculate cosine and sine values for the current hour and day
    cos_hour = np.cos(norm_hour)*0.5 + 0.5
    sin_hour = np.sin(norm_hour)*0.5 + 0.5
    cos_day = np.cos(norm_day)*0.5 + 0.5
    sin_day = np.sin(norm_day)*0.5 + 0.5
    
    return [cos_hour, sin_hour, cos_day, sin_day]

class Time_Manager():
    """
    Class to manage the time dimension over an episode and handle termination
    based on simulation duration.

    Args:
        init_day (int, optional): Initial day of the year (0-364). Defaults to 0.
        timezone_shift (int, optional): Timezone shift in hours from UTC. Defaults to 0.
        duration_days (int, optional): Maximum duration of an episode in days.
                                       If None, the episode runs indefinitely (or until
                                       another termination condition is met). Defaults to None.
    """
    def __init__(self, init_day=0, timezone_shift=0, duration_days=None):
        """Initialize the Time_Manager class."""
        self.start_day_config = init_day # Store the configured initial day
        self.timezone_shift = timezone_shift
        self.duration_days = duration_days
        self.timestep_per_hour = 4 # 15-minute timesteps

        # Internal state variables initialized in reset()
        self.day = 0
        self.hour = 0
        self.current_timestep_in_year = 0 # Overall timestep within the year
        self.episode_step_counter = 0
        self.max_episode_timesteps = None

    def reset(self, init_day=None, init_hour=None, seed=None):
        """
        Resets the time manager to a specific day and hour, and resets the
        episode step counter.

        Args:
            init_day (int, optional): Day to start from (0-364). Defaults to the value
                                      provided during initialization.
            init_hour (int, optional): Hour to start from (0-23). Defaults to 0, adjusted
                                       by timezone_shift if not specified otherwise.
            seed (int, optional): Random seed (not used directly here but kept for consistency).

        Returns:
            list: Sine and cosine features for the initial hour and day.
        """
        if seed is not None:
            # We don't use rng here directly, but good practice if needed later
            pass

        # Use provided init day/hour or default to configured start day / 0 hour
        self.day = init_day if init_day is not None else self.start_day_config
        # Default to 0 hour unless specified, timezone_shift is not applied here anymore
        # The user should provide the desired starting hour directly.
        self.hour = init_hour if init_hour is not None else 0

        # Ensure day and hour are within valid ranges
        self.day = int(self.day) % 365 # Wrap around year if needed
        self.hour = int(self.hour) % 24

        # Calculate the absolute timestep within the year
        self.current_timestep_in_year = int(self.day * 24 * self.timestep_per_hour + self.hour * self.timestep_per_hour)

        # Reset episode duration tracking
        self.episode_step_counter = 0
        if self.duration_days is not None:
            self.max_episode_timesteps = self.duration_days * 24 * self.timestep_per_hour
        else:
            self.max_episode_timesteps = None # Run indefinitely

        # Return initial time features
        return sc_obs(self.hour, self.day)


    def step(self):
        """
        Advances the time by one timestep (15 minutes) and checks for episode termination
        based on duration.

        Returns:
            int: Current day of the year (0-364).
            float: Current hour of the day (0.0 - 23.75).
            list: Sine and cosine features for the current hour and day.
            bool: Done flag (True if episode duration reached, False otherwise).
        """
        # Increment counters
        self.current_timestep_in_year += 1
        self.episode_step_counter += 1

        # Advance time
        self.hour += 1.0 / self.timestep_per_hour
        if self.hour >= 24.0:
            self.hour = 0.0
            self.day += 1
            if self.day >= 365: # Wrap day around the year
                self.day = 0
                # Also wrap the absolute timestep if needed, although less critical usually
                self.current_timestep_in_year = 0

        # Check for termination based on duration
        done = False
        if self.max_episode_timesteps is not None:
            if self.episode_step_counter >= self.max_episode_timesteps:
                done = True

        # Return current day, hour, time features, and done flag
        return self.day, self.hour, sc_obs(self.hour, self.day), done


# Class to manage carbon intensity data
class CI_Manager():
    """Manager of the carbon intensity data.

    Args:
        filename (str, optional): Filename of the carbon intensity data. Defaults to ''.
        location (str, optional): Location identifier. Defaults to 'NYIS'.
        init_day (int, optional): Initial day of the episode. Defaults to 0.
        future_steps (int, optional): Number of steps of the CI forecast. Defaults to 4.
        weight (float, optional): Weight value for coherent noise. Defaults to 0.1.
        desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 5.
        timezone_shift (int, optional): Shift for the timezone. Defaults to 0.
    """
    def __init__(self, location, simulation_year=2020, init_day=0, future_steps=4, weight=0.1, desired_std_dev=5, timezone_shift=0):
        """Initialize the CI_Manager class.

        Args:
            filename (str, optional): Filename of the carbon intensity data. Defaults to ''.
            location (str, optional): Location identifier. Defaults to 'NYIS'.
            init_day (int, optional): Initial day of the episode. Defaults to 0.
            future_steps (int, optional): Number of steps of the CI forecast. Defaults to 4.
            weight (float, optional): Weight value for coherent noise. Defaults to 0.1.
            desired_std_dev (float, optional): Desired standard deviation for coherent noise. Defaults to 5.
            timezone_shift (int, optional): Shift for the timezone. Defaults to 0.
        """
        self.location = location
        self.simulation_year = simulation_year
        self.init_day = init_day
        self.future_steps = future_steps
        self.weight = weight
        self.desired_std_dev = desired_std_dev
        self.timezone_shift = timezone_shift

        self.timestep_per_hour = 4
        self.time_steps_day = 24 * self.timestep_per_hour
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Obtain the parent directory of the current script
        parent_dir = os.path.dirname(script_dir)
        
        filename = f"data/carbon_intensity/{location}/{simulation_year}/{location}_{simulation_year}_hourly.csv"
        
        # Join the parent directory with the filename
        filename = os.path.join(parent_dir, filename)
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"CI file not found: {filename}")

        df = pd.read_csv(filename)
        df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
        df.set_index("Datetime (UTC)", inplace=True)

        if timezone_shift != 0:
            df.index = df.index + pd.Timedelta(hours=timezone_shift)

        if "Carbon Intensity gCO₂eq/kWh (direct)" not in df.columns:
            raise ValueError("Expected column missing: 'Carbon Intensity gCO₂eq/kWh (direct)'")

        self.carbon_data = df["Carbon Intensity gCO₂eq/kWh (direct)"].values
        # assert len(self.carbon_data) == 8760, "Expected 8760 hourly values."

        self._interpolate()
        self.coherent_noise = CoherentNoise(base=0, weight=weight, desired_std_dev=desired_std_dev)

    def _interpolate(self):
        x = range(len(self.carbon_data))
        x_interp = np.linspace(0, len(self.carbon_data), len(self.carbon_data) * self.timestep_per_hour)
        smooth = np.interp(x_interp, x, self.carbon_data)
        self.carbon_smooth = np.roll(smooth, -1 * self.timezone_shift * self.timestep_per_hour)
        self.original_data = self.carbon_smooth.copy()

    def reset(self, init_day=None, init_hour=None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.coherent_noise.seed(seed)
        
        day = init_day if init_day is not None else self.init_day
        hour = init_hour if init_hour is not None else 0
        self.time_step = day * self.time_steps_day + hour * self.timestep_per_hour

        self.carbon_smooth = self.original_data
        max_30d = np.max(self.carbon_smooth[self.time_step:self.time_step + 30 * self.time_steps_day])
        min_30d = np.min(self.carbon_smooth[self.time_step:self.time_step + 30 * self.time_steps_day])
        self.norm_carbon = (self.carbon_smooth - min_30d) / (max_30d - min_30d + 1e-8)

        return self._get_state()

    def step(self):
        self.time_step += 1
        if self.time_step >= len(self.carbon_smooth):
            self.time_step = self.init_day * self.time_steps_day
        return self._get_state()

    def _get_state(self):
        c = self.carbon_smooth[self.time_step]
        norm = self.norm_carbon[self.time_step]
        forecast = self.norm_carbon[self.time_step + 1:self.time_step + 1 + self.future_steps]
        return norm, forecast, c

    def get_current_ci(self, norm=True):
        if norm:
            return self.norm_carbon[self.time_step]
        else:
            return self.carbon_smooth[self.time_step]


    def get_forecast_ci(self):
        return self.norm_carbon[self.time_step + 1:self.time_step + 1 + self.future_steps]

    def get_n_past_ci(self, n):
        return self.norm_carbon[max(0, self.time_step - n):self.time_step]

def load_weather_data(weather_file):
    """
    Reads weather data from a JSON file and converts it into a Pandas DataFrame.
    
    Args:
        weather_file (str): Path to the weather JSON file.
    
    Returns:
        pd.DataFrame: DataFrame containing hourly weather data.
    """
    with open(weather_file, 'r') as f:
        weather_data = json.load(f)

    # Extract timestamps and data variables
    hourly_data = weather_data["hourly"]
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly_data["time"]),
        "temperature_2m": hourly_data["temperature_2m"],
        "relative_humidity_2m": hourly_data["relative_humidity_2m"],
        "cloudcover": hourly_data["cloudcover"],
        "windspeed_10m": hourly_data["windspeed_10m"]
    })

    return df

# Class to manage weather data

class Weather_Manager():
    def __init__(self, location='US-NY-NYIS', simulation_year=2023, init_day=0, weight=0.02, desired_std_dev=0.75, timezone_shift=0, elevation=27.0, debug=False):
        self.location = location
        self.simulation_year = simulation_year
        self.init_day = init_day
        self.weight = weight
        self.desired_std_dev = desired_std_dev
        self.timezone_shift = timezone_shift
        self.elevation = elevation
        self.debug = debug

        self.timestep_per_hour = 4
        self.time_steps_day = self.timestep_per_hour * 24
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Obtain the parent directory of the current script
        parent_dir = os.path.dirname(script_dir)
        
        filename = f"data/weather/{self.location}/{self.simulation_year}.json"
        
        # Join the parent directory with the filename
        filename = os.path.join(parent_dir, filename)

        # Load weather data
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Weather data file not found: {filename}")
        self.weather_df = self._load_weather_data(filename)

        # Interpolate to simulation timestep (15 min)
        self._interpolate_weather_data()

        # Prepare noise and base arrays
        self.coherent_noise = CoherentNoise(base=0, weight=self.weight, desired_std_dev=self.desired_std_dev)

        self.original_temp_data = self.temperature_data.copy()
        self.original_wb_data = self.wet_bulb_data.copy()

    def _load_weather_data(self, file_path):
        with open(file_path, 'r') as f:
            raw = json.load(f)

        df = pd.DataFrame({
            "time": pd.to_datetime(raw["hourly"]["time"]),
            "temperature_2m": raw["hourly"]["temperature_2m"],
            "relative_humidity_2m": raw["hourly"]["relative_humidity_2m"],
            "cloudcover": raw["hourly"]["cloudcover"],
            "windspeed_10m": raw["hourly"]["windspeed_10m"],
        })

        # Apply timezone shift (shift in hours)
        df["time"] = df["time"] + pd.Timedelta(hours=self.timezone_shift)
        df = df.sort_values("time").reset_index(drop=True)

        return df

    def _interpolate_weather_data(self):
        x = np.arange(len(self.weather_df))
        new_x = np.linspace(0, len(x), len(x) * self.timestep_per_hour)

        self.temperature_data = np.interp(new_x, x, self.weather_df["temperature_2m"])
        self.humidity_data = np.interp(new_x, x, self.weather_df["relative_humidity_2m"])
        self.cloudcover_data = np.interp(new_x, x, self.weather_df["cloudcover"])
        self.windspeed_data = np.interp(new_x, x, self.weather_df["windspeed_10m"])

        # Wet bulb estimation (placeholder): here you can use psychrolib if needed
        self.wet_bulb_data = self.temperature_data * 0.9  # Placeholder

        # Normalize temperature and wet bulb for 30-day local window
        self.min_temp, self.max_temp = 0, 45
        self.min_wb_temp, self.max_wb_temp = 0, 45
        self.norm_temp_data = normalize(self.temperature_data, self.min_temp, self.max_temp)
        self.norm_wet_bulb_data = normalize(self.wet_bulb_data, self.min_wb_temp, self.max_wb_temp)

    # Function to reset the time step and return the weather at the first time step
    def reset(self, init_day=None, init_hour=None, seed=None):
        """Reset Weather_Manager to a specific initial day and hour.

        Args:
            init_day (int, optional): Day to start from. If None, defaults to the initial day set during initialization.
            init_hour (int, optional): Hour to start from. If None, defaults to 0.

        Returns:
            tuple: Temperature at current step, normalized temperature at current step, wet bulb temperature at current step, normalized wet bulb temperature at current step.
        """

        self.time_step = (init_day if init_day is not None else self.init_day) * self.time_steps_day + (init_hour if init_hour is not None else 0) * self.timestep_per_hour
        
        if not self.debug:
            # Add noise to the temperature data using the CoherentNoise
            rng = np.random.default_rng(seed)
            self.coherent_noise.seed(seed)
            coh_noise = self.coherent_noise.generate(len(self.original_temp_data))
            # print(f'TODO: check the generated coherent noise: {coh_noise[:3]} and the original temperature data: {self.original_temp_data[:3]} and the wet bulb data: {self.original_wb_data[:3]}' )
            self.temperature_data = self.original_temp_data + coh_noise
            self.wet_bulb_data = self.original_wb_data + coh_noise
            
            num_roll_days = rng.integers(0, 14) # Random roll the temperature some days.
            self.temperature_data =  np.roll(self.temperature_data, num_roll_days*self.timestep_per_hour*24)
            self.wet_bulb_data =  np.roll(self.wet_bulb_data, num_roll_days*self.timestep_per_hour*24)

            self.temperature_data = np.clip(self.temperature_data, self.min_temp, self.max_temp)
            max_30_days = np.max(self.temperature_data[self.time_step:30*self.time_steps_day + self.time_step])
            min_30_days = np.min(self.temperature_data[self.time_step:30*self.time_steps_day + self.time_step])
            self.norm_temp_data = (self.temperature_data - min_30_days) / (max_30_days - min_30_days)
            
            self.wet_bulb_data = np.clip(self.wet_bulb_data, self.min_wb_temp, self.max_wb_temp)
            max_30_days = np.max(self.wet_bulb_data[self.time_step:30*self.time_steps_day + self.time_step])
            min_30_days = np.min(self.wet_bulb_data[self.time_step:30*self.time_steps_day + self.time_step])
            self.norm_wet_bulb_data = self.wet_bulb_data
            
        else:
            # Use a fixed temperature for debugging and wet bulb temperature
            self.temperature_data = np.ones_like(self.temperature_data) * 30
            self.norm_temp_data = np.ones_like(self.norm_temp_data) * 0.5
            self.wet_bulb_data = np.ones_like(self.wet_bulb_data) * 25
            
        self._current_temp = self.temperature_data[self.time_step]
        self._next_temp = self.temperature_data[self.time_step + 1]
        self._current_norm_temp = self.norm_temp_data[self.time_step]
        self._next_norm_temp = self.norm_temp_data[self.time_step + 1]
        self._current_wet_bulb = self.wet_bulb_data[self.time_step]
        self._current_norm_wet_bulb = self.norm_wet_bulb_data[self.time_step]
        
        return self._current_temp, self._current_norm_temp, self._current_wet_bulb, self._current_norm_wet_bulb

    # Function to advance the time step and return the weather at the new time step
    def step(self):
        """Step on the Weather_Manager

        Returns:
            float: Temperature a current step
            float: Normalized temperature a current step
        """
                
        self.time_step += 1
        
        # If it tries to read further, restart from the initial index
        if self.time_step >= len(self.temperature_data):
            self.time_step = self.init_day*self.time_steps_day
            
        self._current_temp = self.temperature_data[self.time_step]
        self._next_temp = self.temperature_data[self.time_step + 1]
        self._current_norm_temp = self.norm_temp_data[self.time_step]
        self._next_norm_temp = self.norm_temp_data[self.time_step + 1]
        self._current_wet_bulb = self.wet_bulb_data[self.time_step]
        self._current_norm_wet_bulb = self.norm_wet_bulb_data[self.time_step]
            
        return self._current_temp, self._current_norm_temp, self._current_wet_bulb, self._current_norm_wet_bulb
    
    def get_current_temperature(self, norm=True):
        if norm:
            return self._current_norm_temp
        else:
            return self._current_temp
    
    def get_next_temperature(self):
        return self._next_norm_temp
    
    def get_n_next_temperature(self, n):
        return self.norm_temp_data[self.time_step+1:self.time_step+1+n]
    
    def get_current_wet_bulb(self):
        return self._current_wet_bulb


class ElectricityPrice_Manager:
    def __init__(self, location, simulation_year, timezone_shift=0):
        self.location = location
        self.simulation_year = simulation_year
        self.timezone_shift = timezone_shift
        self.prices = None
        self.index = 0
        self._load_data()

    def _load_data(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Obtain the parent directory of the current script
        parent_dir = os.path.dirname(script_dir)
        
        filename = f"data/electricity_prices/standardized/{self.location}/{self.simulation_year}/{self.location}_electricity_prices_{self.simulation_year}.csv"
        
        # Join the parent directory with the filename
        filename = os.path.join(parent_dir, filename)

        # print(f"Loading electricity prices from {file_path}")
        df = pd.read_csv(filename)

        # Convert UTC timestamp to datetime and sort to ensure order
        df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])
        # df = df.sort_values("Datetime (UTC)")

        # Store as numpy array of prices
        self.original_prices = df["Price (USD/MWh)"].values
        
        # Remove outliers using IQR
        Q1 = np.percentile(self.original_prices, 15)
        Q3 = np.percentile(self.original_prices, 85)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        self.original_prices = np.clip(self.original_prices, lower, upper)

        # There should be 8760 entries (or 8784 in leap year)
        assert len(self.original_prices) in [8760, 8784], f"Unexpected number of rows: {len(self.original_prices)}"
        
        self._interpolate_prices()

    def _interpolate_prices(self):
        self.timestep_per_hour = 4  # 15 min steps
        x = np.arange(len(self.original_prices))
        new_x = np.linspace(0, len(x) - 1, len(x) * self.timestep_per_hour)
        self.prices = np.interp(new_x, x, self.original_prices)

    def reset(self, init_day, init_hour, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.index = init_day * 24 * self.timestep_per_hour + init_hour * self.timestep_per_hour
        return self.get_current_price()

    def get_current_price(self):
        return self.prices[self.index]

    def get_future_prices(self, n=1):
        end = min(self.index + n, len(self.prices))
        return self.prices[self.index:end]

    def get_past_prices(self, n=1):
        start = max(self.index - n, 0)
        return self.prices[start:self.index]

    def step(self):
        self.index = (self.index + 1) % len(self.prices)
        return self.get_current_price()
