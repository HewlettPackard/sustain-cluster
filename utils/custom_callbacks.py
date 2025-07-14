# utils/custom_callbacks.py

import os
from stable_baselines3.common.callbacks import EvalCallback

class SaveVecNormalizeCallback(EvalCallback):
    """
    A custom callback that derives from EvalCallback and saves the statistics
    of a VecNormalize wrapper whenever a new best model is found.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        # First, run the regular EvalCallback's logic
        # This will check if it's time for an evaluation, run it,
        # and if the model is a new best, it will save it and set self.is_best = True
        continue_training = super()._on_step()
        
        # If the parent callback decided to stop training, we obey
        if not continue_training:
            return False

        # If a new best model was just saved by the parent callback...
        if self.last_mean_reward > self.best_mean_reward:
            # self.is_best is a flag set by the parent EvalCallback when a new best model is saved.
            # We can use it as a trigger. Or a better way is to check the reward directly.
            print("New best mean reward! Saving VecNormalize stats...")
            
            # The model is saved to self.best_model_save_path
            # We will save the stats in the same directory.
            stats_path = os.path.join(os.path.dirname(self.best_model_save_path), "vec_normalize.pkl")
            
            # self.training_env is the VecNormalize wrapper from the main training loop
            self.training_env.save(stats_path)
            print(f"VecNormalize stats saved to {stats_path}")

        return True