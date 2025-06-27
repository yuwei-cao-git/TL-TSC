import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil

# Adjusted parameters as per your request
folder_path = "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_superpixel"
output_dir = "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/dataset"
species_names = ['AB', 'PO', 'SW', 'BW', 'BF', 'CE', 'MR', 'PW', 'MH', 'OR', 'PR']
target_split = [0.7, 0.15, 0.15]
tolerance = 0.01
num_classes = len(species_names)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load Superpixel Data
print("Loading superpixel data...")
superpixel_files = glob(os.path.join(folder_path, "*.npz"))

# Initialize lists to store labels and file names
labels = []
file_names = []

for file_path in superpixel_files:
    data = np.load(file_path, allow_pickle=True)
    label = data["label"]  # Assuming shape: (num_classes,)
    labels.append(label)
    file_names.append(os.path.basename(file_path))

# Convert labels to a NumPy array
labels = np.array(labels)  # Shape: (num_superpixels, num_classes)

# Step 2: Calculate Overall Species Proportions
print("Calculating overall species proportions...")
species_sums = np.sum(labels, axis=0)  # Shape: (num_classes,)
total_sum = np.sum(species_sums)
overall_proportions = species_sums / total_sum

# Step 3: Define Target Split Ratios and Tolerance (already set above)


# Step 4: Adapt the Iterative Splitting Function
def check_balance(labels, indices_list, target_split, tolerance):
    """
    Check if the species proportions are within the ±tolerance for the target split.
    """
    # Calculate total species sums
    total_species_sums = np.sum(labels, axis=0)

    # Initialize list to hold proportions for each split
    split_proportions = []

    for indices in indices_list:
        split_sums = np.sum(labels[indices], axis=0)
        split_proportions.append(split_sums / total_species_sums)

    # Check if all proportions are within the tolerance range
    for i, proportion in enumerate(split_proportions):
        if not np.all(np.abs(proportion - target_split[i]) <= tolerance):
            return False
    return True


def iterative_split_superpixels(
    labels, target_split=[0.7, 0.15, 0.15], max_iter=5000, tolerance=0.01
):
    """
    Iteratively split the superpixel dataset until species proportions are balanced within the given tolerance.
    """
    num_samples = labels.shape[0]
    indices = np.arange(num_samples)

    for i in range(max_iter):
        # Perform random splitting
        train_val_indices, test_indices = train_test_split(
            indices, test_size=target_split[2], random_state=i
        )
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=target_split[1] / (target_split[0] + target_split[1]),
            random_state=i,
        )

        # Check if the split is balanced
        if check_balance(
            labels, [train_indices, val_indices, test_indices], target_split, tolerance
        ):
            print(f"Balanced split found after {i+1} iterations.")
            return train_indices, val_indices, test_indices

    raise ValueError(
        f"Could not find a balanced split within the given tolerance after {max_iter} iterations."  # Balanced split found after 26 iterations.
    )


# Step 5: Perform the Iterative Splitting
print("Performing iterative splitting to balance the dataset...")
train_indices, val_indices, test_indices = iterative_split_superpixels(
    labels, target_split=target_split, tolerance=tolerance
)

print(f"Training samples: {len(train_indices)}")
print(f"Validation samples: {len(val_indices)}")
print(f"Testing samples: {len(test_indices)}")

# Step 6: Save the Superpixel File Names for Each Split
print("Saving the superpixel file names for each split...")
# Map indices to file names
train_files = [file_names[i] for i in train_indices]
val_files = [file_names[i] for i in val_indices]
test_files = [file_names[i] for i in test_indices]


def save_file_names(file_list, file_path):
    with open(file_path, "w") as f:
        f.write("\n".join(file_list))


save_file_names(train_files, os.path.join(output_dir, "train_superpixels.txt"))
save_file_names(val_files, os.path.join(output_dir, "val_superpixels.txt"))
save_file_names(test_files, os.path.join(output_dir, "test_superpixels.txt"))

# Step 7: Verify and Visualize the Species Proportions in Each Split
print("Calculating species proportions for each split...")


def calculate_split_proportions(labels, indices):
    split_sums = np.sum(labels[indices], axis=0)
    total_sums = np.sum(labels, axis=0)
    split_proportions = split_sums / total_sums
    return split_proportions


# Calculate proportions
train_proportions = calculate_split_proportions(labels, train_indices)
val_proportions = calculate_split_proportions(labels, val_indices)
test_proportions = calculate_split_proportions(labels, test_indices)

# Prepare data for depth map (heatmap)
print("Preparing data for depth map visualization...")

# Combine the proportions into a DataFrame
df_proportions = pd.DataFrame(
    {
        "Species": species_names,
        "Train": train_proportions,
        "Validation": val_proportions,
        "Test": test_proportions,
    }
)
df_proportions.set_index("Species", inplace=True)

# Transpose the DataFrame for heatmap
df_heatmap = df_proportions

# Plot the heatmap
print("Creating the depth map (heatmap) of species proportions...")
plt.figure(figsize=(10, 6))
sns.heatmap(
    df_heatmap, annot=True, cmap="YlGnBu", cbar_kws={"label": "Proportion"}, fmt=".4f"
)

plt.title("Species Proportions in Superpixel Dataset Splits")
plt.xlabel("Species")
plt.ylabel("Dataset Split")
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Save the heatmap
heatmap_path = os.path.join(output_dir, "species_proportions_heatmap.png")
plt.tight_layout()
plt.savefig(heatmap_path)

print(f"Heatmap saved to {heatmap_path}")

print("All steps completed successfully!")

# After saving the split file names and visualizations

# Move the .npz files into subfolders according to the splits
print("Moving .npz files into subfolders according to the splits...")


def move_files(file_list, src_folder, dst_folder):
    for file_name in file_list:
        src_file = os.path.join(src_folder, file_name)
        dst_file = os.path.join(dst_folder, file_name)
        if os.path.exists(src_file):
            shutil.move(src_file, dst_file)
        else:
            print(f"Warning: File {src_file} does not exist and cannot be moved.")


# Read file names (you can reuse the variables if they're still in scope)
train_files = [file_names[i] for i in train_indices]
val_files = [file_names[i] for i in val_indices]
test_files = [file_names[i] for i in test_indices]

# Create subfolders
train_folder = os.path.join(folder_path, "train")
val_folder = os.path.join(folder_path, "val")
test_folder = os.path.join(folder_path, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Move files
move_files(train_files, folder_path, train_folder)
move_files(val_files, folder_path, val_folder)
move_files(test_files, folder_path, test_folder)

print("Files have been successfully moved to their respective folders.")
