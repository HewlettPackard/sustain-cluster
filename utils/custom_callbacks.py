import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

class CustomEvalCallback(EvalCallback):
    """
    A custom callback that extends EvalCallback to log the mean 'carbon_emissions_kg'
    from the info dictionary during evaluation.
    
    It calculates the average total carbon emissions per episode across all evaluation
    environments and logs it to TensorBoard as 'eval/mean_carbon_emissions_kg'.
    """
    def __init__(self, *args, **kwargs):
        super(CustomEvalCallback, self).__init__(*args, **kwargs)
        # We need a place to store the running carbon total for each parallel eval env
        self.episode_carbon_emissions = np.zeros(0) 
        # A list to store the final carbon total for each completed episode
        self.all_episode_carbon_emissions = []

    def _on_rollout_start(self) -> None:
        """
        This method is called before a new evaluation rollout.
        We use it to reset our metric accumulators.
        """
        super()._on_rollout_start()
        # Initialize or reset our custom metric accumulators
        self.episode_carbon_emissions = np.zeros(self.eval_env.num_envs)
        self.all_episode_carbon_emissions = []

    def _on_step(self) -> bool:
        """
        This method is called by the model after each call to `env.step()` during evaluation.
        We use it to inspect the `info` dictionary and accumulate our metric.
        """
        # First, let the parent class do its work. This handles rewards, checking for
        # episode completion (dones), and returns whether to continue training.
        continue_training = super()._on_step()
        if continue_training is False:
            return False

        # `self.locals["infos"]` is a list of info dicts, one for each parallel environment
        # Loop here only if we are on the evaluation phase
        if self.locals["infos"][0]['is_training']:
            return True  # No evaluation environment, skip
        
        for i in range(self.eval_env.num_envs):
            info = self.locals["infos"][i]
            
            # Sum carbon emissions from all datacenters in this step for this env
            step_carbon = 0.0
            # The data is nested, so we use .get() for safe access
            if 'raw_results' in info:
                cluster_info = info.get('raw_results', {})
                dc_infos = cluster_info.get('datacenter_infos', {})
                for dc_id, dc_info in dc_infos.items():
                    common_info = dc_info.get('__common__', {})
                    step_carbon += common_info.get('carbon_emissions_kg', 0.0)
            
            self.episode_carbon_emissions[i] += step_carbon

            # Check if the episode for this environment is done
            if self.locals["dones"][i]:
                # An episode is finished, so we save its total accumulated carbon
                self.all_episode_carbon_emissions.append(self.episode_carbon_emissions[i])
                # Reset the counter for this environment, as it will start a new episode
                self.episode_carbon_emissions[i] = 0

        return True

    def _on_rollout_end(self) -> None:
        """
        This method is called at the end of an evaluation rollout.
        We use it to calculate the mean of our metric and log it.
        """
        # The parent class handles saving the best model and logging the mean reward.
        # We call it first to ensure that behavior is preserved.
        super()._on_rollout_end()

        # Now, calculate and log our custom metric to TensorBoard
        if self.all_episode_carbon_emissions:
            mean_carbon = np.mean(self.all_episode_carbon_emissions)
            self.logger.record("eval/mean_carbon_emissions_kg", mean_carbon)
            
        # The parent's _on_rollout_end() calls self.logger.dump(), so we don't need to.
        # Clear the list for the next evaluation cycle.
        self.all_episode_carbon_emissions = []