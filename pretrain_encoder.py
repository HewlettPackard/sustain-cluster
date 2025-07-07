# pretrain_encoder.py
#%%
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import numpy as np
import pandas as pd
from typing import Tuple
from rl_components.meta_task_encoder import MetaTaskEncoder, MetaTaskDecoder, TaskAutoencoder
from typing import Optional

# 1. --- Data Generation ---
# Let's define the order and dimension of our task feature vectors.
def generate_synthetic_task_data(num_samples: int,
                                 max_tasks_per_group: int,
                                 task_feature_dim: int # <-- FIX: Added argument
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic dataset of task groups for pre-training the MetaTaskEncoder.

    For each sample, it creates a group of tasks with a random size (from 1 to
    max_tasks_per_group) and populates their features with values drawn from
    realistic, random distributions.

    Args:
        num_samples (int): The total number of task groups to generate (e.g., 100,000).
        max_tasks_per_group (int): The maximum number of tasks in a single group. This
                                   determines the padding size of the output arrays.

    Returns:
        A tuple containing:
        - all_tasks_padded (np.ndarray): A padded array of task features.
            Shape: (num_samples, max_tasks_per_group, TASK_FEAT_DIM)
        - all_masks (np.ndarray): A boolean array indicating padding. `True` means
            the entry is padding and should be ignored.
            Shape: (num_samples, max_tasks_per_group)
    """
    # Initialize the final output arrays with zeros and default mask to all True (padded)
    all_tasks_padded = np.zeros((num_samples, max_tasks_per_group, task_feature_dim), dtype=np.float32)
    all_masks = np.ones((num_samples, max_tasks_per_group), dtype=bool)

    # For each sample (i.e., each task group) we want to generate...
    for i in range(num_samples):
        # 1. Determine the number of real tasks in this group (from 1 to max).
        num_real_tasks = np.random.randint(1, max_tasks_per_group + 1)

        # 2. Generate features for all tasks in this group at once for efficiency.
        #    We use different distributions to make the data more realistic.

        # Cores: Skewed distribution (log-normal), most tasks are small, some are huge.
        cores = np.random.lognormal(mean=1.0, sigma=1.5, size=num_real_tasks) + 1
        
        # GPUs: Many tasks use 0 GPUs, some use 1, a few use more.
        # We'll use a choice with skewed probabilities.
        gpu_probs = [0.6, 0.3, 0.08, 0.02] # Probabilities for 0, 1, 2, 4 GPUs
        gpus = np.random.choice([0, 1, 2, 4], size=num_real_tasks, p=gpu_probs)
        # Add small fractional noise to simulate shared GPU usage
        gpus = gpus.astype(np.float32) + np.random.uniform(0, 0.1, size=num_real_tasks)
        gpus[gpus < 0.1] = 0 # Clamp tiny values to zero

        # Memory: Related to the number of cores.
        mem = cores * np.random.uniform(2, 8, size=num_real_tasks) + np.random.uniform(0, 16)
        
        # Duration: Most tasks are short, some are very long (log-normal).
        duration = np.random.lognormal(mean=3.0, sigma=1.0, size=num_real_tasks) * 15 # In minutes

        # Bandwidth: Can be anything, let's use a uniform distribution.
        bandwidth = np.random.uniform(0.1, 50, size=num_real_tasks)

        
        # 3. Assemble the feature matrix for this group.
        task_group_features = np.stack([
            cores,
            gpus,
            mem,
            duration,
            bandwidth
        ], axis=1) # Stack along the feature dimension

        # This will now correctly match the task_feature_dim argument
        if task_group_features.shape[1] != task_feature_dim:
            raise ValueError("Mismatch between generated features and task_feature_dim")
        
        # 4. Place the generated features into the padded output array.
        all_tasks_padded[i, :num_real_tasks, :] = task_group_features
        
        # 5. Update the mask for this sample. The first `num_real_tasks` are not padding.
        all_masks[i, :num_real_tasks] = False

    return all_tasks_padded, all_masks

#%%

# FIX: Define the feature order once, globally, to ensure consistency.
TASK_FEATURE_ORDER = [
    "cores_req",
    "gpu_req",
    "mem_req",
    "duration_mins",
    "bandwidth_gb"
]
TASK_FEAT_DIM = len(TASK_FEATURE_ORDER)

# --- Small test run to verify data generation ---
print("--- Running small test generation ---")
test_num_samples = 10
test_max_tasks = 5
print(f"Generating {test_num_samples} samples with max {test_max_tasks} tasks each and {TASK_FEAT_DIM} features per task...")

tasks_data, masks_data = generate_synthetic_task_data(
    test_num_samples, test_max_tasks, TASK_FEAT_DIM # Pass the dim
)

print("\nShape of the tasks data array:", tasks_data.shape)
print("Shape of the mask array:", masks_data.shape)


print("\n--- Example Sample 0 ---")
print("Task Features (padded):")
# Print with fixed precision for readability
with np.printoptions(precision=2, suppress=True):
    print(tasks_data[0])
print("\nMask (True means padding):")
print(masks_data[0])

# Find out how many real tasks are in the first sample
num_real_tasks_in_sample_0 = int(np.sum(~masks_data[0])) # Invert mask to count False values
print(f"\nThere are {num_real_tasks_in_sample_0} real tasks in Sample 0.")
print("Their features are:")
with np.printoptions(precision=2, suppress=True):
    print(tasks_data[0, :num_real_tasks_in_sample_0, :])
    
#%%
# 2. --- Hyperparameters & Instantiation ---
print("\n--- Starting Main Pre-training ---")

# Hyperparameters
EMBEDDING_DIM = 256  # This will be the output dimension of the encoder
MAX_TASKS_IN_QUEUE = 20
NUM_SAMPLES = 10000000
BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 1e-4

# IMPROVEMENT: Add device handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate the large dataset for training
print(f"Generating {NUM_SAMPLES} samples for training...")
task_data, mask_data = generate_synthetic_task_data(
    NUM_SAMPLES, MAX_TASKS_IN_QUEUE, TASK_FEAT_DIM # Pass the dim
)

# --- START OF NEW NORMALIZATION LOGIC ---
print("Normalizing dataset...")

# We only calculate stats on the real data, not the padding
# Flatten the batch and task dimensions to get a long list of tasks
# Shape: (num_samples * max_tasks, task_feat_dim)
flat_tasks = task_data.reshape(-1, TASK_FEAT_DIM)
# Use the mask to select only the real tasks
flat_mask = mask_data.flatten()
real_tasks = flat_tasks[~flat_mask]

# Calculate mean and std dev for each feature column
# These will be vectors of shape (TASK_FEAT_DIM,)
mean = np.mean(real_tasks, axis=0)
std = np.std(real_tasks, axis=0)
std[std < 1e-8] = 1.0 # Avoid division by zero for features with no variance

# Save these stats! The RL agent will need them later to normalize its inputs.
normalization_stats = {'mean': mean.tolist(), 'std': std.tolist()}
import json
with open('task_feature_normalization_stats.json', 'w') as f:
    json.dump(normalization_stats, f)
print("Saved normalization stats to task_feature_normalization_stats.json")

# Normalize the entire padded dataset using broadcasting
# (This will also "normalize" the zero-padding, which is fine)
normalized_task_data = (task_data - mean) / std
# --- END OF NEW NORMALIZATION LOGIC ---


dataset = TensorDataset(torch.from_numpy(normalized_task_data).float(), torch.from_numpy(mask_data).bool())
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Data generation complete.")

# Instantiate the models and move them to the selected device
encoder = MetaTaskEncoder(task_feature_dim=TASK_FEAT_DIM, embedding_dim=EMBEDDING_DIM).to(device)
decoder = MetaTaskDecoder(task_feature_dim=TASK_FEAT_DIM, embedding_dim=EMBEDDING_DIM).to(device)
autoencoder = TaskAutoencoder(encoder, decoder).to(device)

optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
# Use 'none' reduction to handle masking correctly
loss_fn = torch.nn.MSELoss(reduction='none')

# --- 3. Training Loop ---
print("\nStarting training loop...")
autoencoder.train()
for epoch in range(EPOCHS):
    total_loss = 0
    
    # Use a progress bar for the dataloader
    from tqdm import tqdm
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch_tasks, batch_masks in pbar:
        # IMPROVEMENT: Move data to the selected device
        batch_tasks = batch_tasks.to(device)
        batch_masks = batch_masks.to(device)
        
        optimizer.zero_grad()
        
        reconstructed_batch = autoencoder(batch_tasks, batch_masks)
        
        # Calculate loss only on the non-padded parts of the sequence
        unmasked_loss = loss_fn(reconstructed_batch, batch_tasks)
        
        # Invert mask for loss calculation (we want loss where mask is False)
        loss_mask = ~batch_masks
        masked_loss = unmasked_loss * loss_mask.unsqueeze(-1)
        
        # Average the loss over the number of real (unmasked) tasks, not the padded length
        # Add a small epsilon to avoid division by zero if a batch is all padding (unlikely)
        loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
    avg_epoch_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} complete. Average Reconstruction Loss: {avg_epoch_loss:.6f}")
#%%
# 4. --- Save the Trained Encoder ---
print("\nTraining complete.")
save_path = "pretrained_meta_task_encoder.pth"
# After training, save only the encoder's weights. This is the part you'll use for your RL agent.
torch.save(autoencoder.encoder.state_dict(), save_path)
print(f"Successfully saved pre-trained encoder weights to: {save_path}")
#%%
# In pretrain_encoder.py, after the training loop and saving the model...

# %%
# --- 5. Validation and Visualization ---
print("\n--- Running Validation and Visualization ---")

# Ensure the model is in evaluation mode
autoencoder.eval()

# --- Load the saved normalization stats ---
import json
with open('task_feature_normalization_stats.json', 'r') as f:
    stats = json.load(f)
mean = np.array(stats['mean'])
std = np.array(stats['std'])
print("Loaded normalization stats for de-normalizing the output.")

# --- Generate a small, separate validation dataset ---
# Using a different seed to ensure it's not the same as the training data
np.random.seed(999) 
NUM_VIZ_SAMPLES = 5
print(f"Generating {NUM_VIZ_SAMPLES} new samples for visualization...")
viz_task_data, viz_mask_data = generate_synthetic_task_data(
    NUM_VIZ_SAMPLES, MAX_TASKS_IN_QUEUE, TASK_FEAT_DIM
)

# Normalize this new data using the *same* stats from the training set
normalized_viz_task_data = (viz_task_data - mean) / std

# Convert to tensors and move to device
viz_tasks_tensor = torch.from_numpy(normalized_viz_task_data).float().to(device)
viz_masks_tensor = torch.from_numpy(viz_mask_data).bool().to(device)


# --- Run inference and de-normalize ---
with torch.no_grad():
    # Get the reconstructed (and still normalized) output
    reconstructed_normalized_tensor = autoencoder(viz_tasks_tensor, viz_masks_tensor)

# Move reconstructed tensor back to CPU and convert to numpy
reconstructed_normalized_data = reconstructed_normalized_tensor.cpu().numpy()

# De-normalize the reconstructed data to bring it back to the original scale
reconstructed_data = (reconstructed_normalized_data * std) + mean


# --- Print Side-by-Side Comparison ---
print("\n--- Reconstruction Quality Examples ---")
for i in range(NUM_VIZ_SAMPLES):
    num_real_tasks = int(np.sum(~viz_mask_data[i]))
    
    print(f"\n--- Sample {i+1} (Contains {num_real_tasks} tasks) ---")
    
    # Get the real task data for this sample (un-normalized)
    original = viz_task_data[i, :num_real_tasks, :]
    # Get the reconstructed task data for this sample (de-normalized)
    reconstruction = reconstructed_data[i, :num_real_tasks, :]
    
    # Use pandas for a beautiful side-by-side print
    df_orig = pd.DataFrame(original, columns=TASK_FEATURE_ORDER)
    df_recon = pd.DataFrame(reconstruction, columns=[f"{col}_recon" for col in TASK_FEATURE_ORDER])
    
    comparison_df = pd.concat([df_orig, df_recon], axis=1)
    
    print("Original vs. Reconstructed (in original scale):")
    # Set display options for better printing
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 2,
                           ):
        print(comparison_df)


# %%
# --- Plotting the Reconstruction Error ---
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Generating Error Visualization Plot ---")

# Calculate the absolute percentage error for all real tasks in the viz set
# Flatten the arrays to work with a long list of tasks
flat_original = viz_task_data.reshape(-1, TASK_FEAT_DIM)
flat_reconstruction = reconstructed_data.reshape(-1, TASK_FEAT_DIM)
flat_mask = viz_mask_data.flatten()

# Select only the real tasks
real_original = flat_original[~flat_mask]
real_reconstruction = flat_reconstruction[~flat_mask]

# Calculate Absolute Percentage Error: |(recon - orig) / orig| * 100
# Add a small epsilon to avoid division by zero
epsilon = 1e-6
abs_pct_error = np.abs((real_reconstruction - real_original) / (real_original + epsilon)) * 100

# Put into a DataFrame for easy plotting with Seaborn
error_df = pd.DataFrame(abs_pct_error, columns=TASK_FEATURE_ORDER)

# Use a boxplot to show the distribution of errors for each feature
plt.figure(figsize=(12, 7))
sns.boxplot(data=error_df)
plt.title("Distribution of Absolute Percentage Reconstruction Error per Feature", fontsize=16)
plt.ylabel("Absolute Percentage Error (%)")
plt.xlabel("Task Feature")
plt.yscale('log') # Use a log scale as errors can vary widely
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.show()
# %%
