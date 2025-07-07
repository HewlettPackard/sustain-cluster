import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np
import os # <-- Add this import at the top of your file

# --- Hyperparameters ---
gamma = 0.99
lr_actor = 0.0003
lr_critic = 0.001
epochs = 10
eps_clip = 0.2
gae_lambda = 0.95
T_horizon = 2048
minibatch_size = 64
max_updates = 3000

# --- NEW: Evaluation Hyperparameters ---
eval_episodes = 10
eval_frequency = 5 # Evaluate every 20 updates
acrobot_success_threshold = -100.0 # Define what a "successful" episode is for Acrobot

# --- Actor-Critic Network Definition ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)

    def act(self, state, deterministic=False):
        # Add a 'deterministic' flag for evaluation
        shared_output = self.shared_layers(state)
        action_logits = self.actor_head(shared_output)
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            # For evaluation, we take the most likely action
            action = torch.argmax(action_logits, dim=-1)
        else:
            # For training, we sample from the distribution
            action = dist.sample()
            
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        shared_output = self.shared_layers(state)
        action_logits = self.actor_head(shared_output)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic_head(shared_output)
        return action_logprobs, state_values, dist_entropy

# --- Memory Buffer ---
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# --- NEW: Evaluation Function with Video Recording ---
def evaluate_policy(policy, env_name, eval_episodes, video_folder, update_num):
    # Create a directory for the videos if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)
    
    # Create an environment for evaluation
    eval_env = gym.make(env_name, render_mode="rgb_array") # <-- Set render_mode for video
    
    # Wrap the environment with the RecordVideo wrapper
    # This will save a video of the first episode in this evaluation run
    eval_env = gym.wrappers.RecordVideo(
        eval_env, 
        video_folder=video_folder, 
        name_prefix=f"ppo-acrobot-update-{update_num}", # Names the video file based on the update number
        episode_trigger=lambda x: x == 0  # Record only the first episode of each evaluation run
    )
    
    total_rewards = 0
    successes = 0
    
    for _ in range(eval_episodes):
        state, _ = eval_env.reset(options={"low": -0.5, "high": 0.5})
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # Use deterministic actions for evaluation
            action, _ = policy.act(state_tensor, deterministic=True)
            state, reward, terminated, truncated, _ = eval_env.step(action.item())
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards += episode_reward
        if reward > -1:
            successes += 1
            
    avg_reward = total_rewards / eval_episodes
    completion_rate = successes / eval_episodes
    eval_env.close()
    
    return avg_reward, completion_rate

# --- Main Training Logic ---
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

print(f"State Dimension: {state_dim}, Action Dimension: {action_dim}")

policy = ActorCritic(state_dim, action_dim)
optimizer = torch.optim.Adam([
    {'params': policy.shared_layers.parameters(), 'lr': lr_actor},
    {'params': policy.actor_head.parameters(), 'lr': lr_actor},
    {'params': policy.critic_head.parameters(), 'lr': lr_critic}
])

memory = Memory()
mse_loss = nn.MSELoss()

time_step = 0
total_reward_collected = 0
state, _ = env.reset()

for i_update in range(1, max_updates + 1):
    # 2. DATA COLLECTION LOOP
    for t in range(T_horizon):
        time_step += 1
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # Use stochastic actions for training exploration
        action, logprob = policy.act(state_tensor, deterministic=False)
        
        memory.states.append(state_tensor)
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        
        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        total_reward_collected += reward

        if done:
            state, _ = env.reset(options={"low": -0.5, "high": 0.5})

    # 3. COMPUTE ADVANTAGES (GAE) and REWARDS-TO-GO
    # (This section remains the same)
    rewards = torch.tensor(memory.rewards, dtype=torch.float32)
    is_terminals = torch.tensor(memory.is_terminals, dtype=torch.float32)

    with torch.no_grad():
        old_states = torch.squeeze(torch.stack(memory.states, dim=0))
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0))
        _, state_values, _ = policy.evaluate(old_states, old_actions)
        state_values = state_values.squeeze()

    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    
    with torch.no_grad():
        if memory.is_terminals[-1]:
            next_value = torch.tensor(0.0)
        else:
            last_state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, next_value, _ = policy.evaluate(last_state_tensor, old_actions[-1])
            next_value = next_value.squeeze()

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - is_terminals[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - is_terminals[t]
            next_val = state_values[t + 1]
        
        delta = rewards[t] + gamma * next_val * next_non_terminal - state_values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
    rewards_to_go = advantages + state_values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach()
    
    # 4. PPO UPDATE LOOP WITH MINIBATCH SGD
    # (This section remains the same)
    for _ in range(epochs):
        indices = torch.randperm(T_horizon)
        for start in range(0, T_horizon, minibatch_size):
            end = start + minibatch_size
            minibatch_indices = indices[start:end]

            mb_states = old_states[minibatch_indices]
            mb_actions = old_actions[minibatch_indices]
            mb_logprobs = old_logprobs[minibatch_indices]
            mb_advantages = advantages[minibatch_indices]
            mb_rewards_to_go = rewards_to_go[minibatch_indices]

            logprobs, state_values, dist_entropy = policy.evaluate(mb_states, mb_actions)
            ratios = torch.exp(logprobs - mb_logprobs)
            
            surr1 = ratios * mb_advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * mb_advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = mse_loss(state_values.squeeze(), mb_rewards_to_go)
            entropy_loss = -0.01 * dist_entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss + entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Logging training progress
    avg_reward_per_timestep_batch = total_reward_collected / T_horizon
    print(f"Update #{i_update} | Timesteps: {time_step} | Avg Reward (Training Batch): {avg_reward_per_timestep_batch:.3f}")
    total_reward_collected = 0

    # 5. PERIODIC EVALUATION
    if i_update % eval_frequency == 0:
        video_directory = "videos" # Define the folder to save videos
        avg_eval_reward, completion_rate = evaluate_policy(
            policy, 
            "CartPole-v1", 
            eval_episodes, 
            video_directory, 
            i_update # Pass the current update number for the filename
        )
        print("------------------------------------------------------------")
        print(f"EVALUATION: Avg Reward over {eval_episodes} episodes: {avg_eval_reward:.2f}")
        print(f"EVALUATION: Completion Rate: {completion_rate*100:.2f}%")
        print(f"EVALUATION: Video saved for this evaluation run in '{video_directory}/' folder.")
        print("------------------------------------------------------------")

    memory.clear()

env.close()