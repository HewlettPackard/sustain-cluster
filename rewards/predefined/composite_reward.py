from rewards.registry_utils import get_reward_function, register_reward
from rewards.base_reward import BaseReward
import numpy as np # Import numpy
from collections import defaultdict

@register_reward("composite") # Ensure registration if not already done
class CompositeReward(BaseReward):
    """
    Combines multiple reward components with weights.
    Supports optional normalization using running statistics, which can be
    frozen after a specified number of steps.
    """
    def __init__(self,
                 components: dict,
                 normalize: bool = True,
                 freeze_stats_after_steps: int = None, # Steps after which stats are frozen
                 epsilon: float = 1e-8):
        """
        Args:
            components (dict): Dictionary defining reward components, weights, and args.
                               Example: {"name": {"weight": w, "args": {...}}}
            normalize (bool): If True, normalize components using stats before weighting.
            freeze_stats_after_steps (Optional[int]): If normalize is True, specifies the
                                    number of __call__ executions after which the running
                                    mean/std statistics are frozen and used as fixed
                                    normalization factors. If None, stats keep updating.
            epsilon (float): Small value added to std dev denominator for numerical stability.
        """
        super().__init__() # Call parent __init__
        self.use_normalization = normalize # Renamed attribute for clarity
        self.freeze_steps = freeze_stats_after_steps
        self.epsilon = np.float64(epsilon) # Use float64 for stats

        # Running statistics storage (mean, sum_sq_diff, count)
        # M2 = sum of squares of differences from the current mean (for Welford's algorithm)
        self.running_stats = defaultdict(lambda: {"mean": np.float64(0.0), "M2": np.float64(0.0), "count": np.float64(0.0)})
        # Storage for frozen statistics
        self.frozen_stats = defaultdict(lambda: {"mean": np.float64(0.0), "std": np.float64(1.0)})

        self.steps_since_init = 0
        self.stats_frozen = False

        self.components = []
        self.last_unnormalized_values = {}  # Track latest raw component values

        # Instantiate sub-components
        for name, cfg in components.items():
            weight = cfg.get("weight", 1.0)
            args = cfg.get("args", {}) # Sub-component args (e.g., penalty_per_violation)
            # Get the reward function class from the registry
            reward_fn = get_reward_function(name, **args)
            self.components.append({"name": name, "weight": weight, "func": reward_fn})

            # Initialize stats entries even if not normalizing initially
            # This ensures keys exist if normalization is turned on later or frozen stats are used.
            _ = self.running_stats[name]
            _ = self.frozen_stats[name]


    def _update_running_stats_welford(self, name, value):
        """ Updates running mean and variance using Welford's online algorithm. """
        stats = self.running_stats[name]
        value = np.float64(value) # Ensure float64

        stats["count"] += 1
        delta = value - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = value - stats["mean"] # New delta based on updated mean
        stats["M2"] += delta * delta2

    def _calculate_std(self, name):
        """ Calculates standard deviation from running stats. """
        stats = self.running_stats[name]
        if stats["count"] < 2:
            return np.float64(1.0) # Cannot compute variance with < 2 samples
        variance = stats["M2"] / (stats["count"]) # Population variance, or use count-1 for sample
        return np.sqrt(np.maximum(variance, self.epsilon)) # Add epsilon for stability


    def _freeze_statistics(self):
        """ Calculates final mean/std and freezes them. """
        print(f"--- Freezing reward normalization statistics after {self.steps_since_init} steps ---")
        for name, stats in self.running_stats.items():
            mean = stats["mean"]
            std = self._calculate_std(name)
            self.frozen_stats[name]["mean"] = mean
            self.frozen_stats[name]["std"] = std if std > self.epsilon else np.float64(1.0) # Avoid std near zero
            print(f"  Component '{name}': Mean={mean:.4f}, Std={self.frozen_stats[name]['std']:.4f} (Count={stats['count']})")
        self.stats_frozen = True


    def __call__(self, **kwargs):
        """ Calculates the composite reward. """
        self.steps_since_init += 1
        total_reward = np.float64(0.0)
        self.last_unnormalized_values.clear()

        # Check if it's time to freeze statistics
        if self.use_normalization and not self.stats_frozen and self.freeze_steps is not None and self.steps_since_init > self.freeze_steps:
            self._freeze_statistics()

        for component in self.components:
            name = component["name"]
            weight = component["weight"]
            reward_func = component["func"]

            # Calculate raw value from the sub-component
            raw_value = np.float64(reward_func(**kwargs))
            self.last_unnormalized_values[name] = float(raw_value) # Store original value

            normalized_value = raw_value
            if self.use_normalization:
                if not self.stats_frozen:
                    # Update running stats during the initial phase (or always if freeze_steps is None)
                    self._update_running_stats_welford(name, raw_value)
                    # Normalize using current running stats
                    current_mean = self.running_stats[name]["mean"]
                    current_std = self._calculate_std(name)
                    normalized_value = (raw_value - current_mean) / (current_std if current_std > self.epsilon else np.float64(1.0))
                else:
                    # Use frozen statistics for normalization
                    frozen_mean = self.frozen_stats[name]["mean"]
                    frozen_std = self.frozen_stats[name]["std"] # Already has epsilon check
                    normalized_value = (raw_value - frozen_mean) / frozen_std

            # Add weighted (normalized or raw) value to total
            total_reward += weight * normalized_value

        self.last_reward = float(total_reward) # Store final reward as standard float
        return self.last_reward

    def get_last_components(self):
        """ Returns the dictionary of last *unnormalized* component values. """
        return self.last_unnormalized_values


'''
Example of composite reward configuration:

components={
    "energy_price": {
        "weight": 0.5,
        "args": {"normalize_factor": 100000}
    },
    "carbon_cost": {
        "weight": 0.3,
        "args": {"normalize_factor": 100}
    },
    "sla_penalty": {
        "weight": 0.2,
        "args": {"penalty_per_violation": 5.0}
    }
}


'''