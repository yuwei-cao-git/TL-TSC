import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

# ==============================
# 1. CONFIGURATION
# ==============================

dataset_name = "ovf"  # For logging only

# For Dataset ovf
species_names = ['ash', 'poplar', 'spruce', 'birch', 'fir', 'cedar', 'maple', 'pine', 'oak']
species_to_genus = {
    'ash': 'hardwood',
    'poplar': 'poplar',
    'spruce': 'spruce',
    'birch': 'hardwood',
    'fir': 'fir',
    'cedar': 'cedar',
    'maple': 'hardwood',
    'pine': 'pine',
    'oak': 'hardwood',
}
# After grouping, only these genus will remain (order is important!)
genus_order = ['hardwood', 'poplar', 'spruce', 'birch', 'fir', 'cedar']

# === Input/Output Paths ===
src_folder = "/mnt/g/ovf/ovf_superpixel_dataset/tile_128"
output_dir = "/mnt/g/ovf/ovf_superpixel_dataset/tile_128"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# 2. FUNCTIONS
# ==============================

def group_to_genus(label, per_pixel_labels, species_list, species_to_genus, genus_order):
    genus_labels = np.zeros(len(genus_order), dtype=label.dtype)
    genus_per_pixel = np.zeros((len(genus_order),) + per_pixel_labels.shape[1:], dtype=per_pixel_labels.dtype)
    genus_to_idx = {g: i for i, g in enumerate(genus_order)}
    for idx, sp in enumerate(species_list):
        genus = species_to_genus.get(sp)
        if genus is not None and genus in genus_to_idx:
            genus_idx = genus_to_idx[genus]
            genus_labels[genus_idx] += label[idx]
            genus_per_pixel[genus_idx] += per_pixel_labels[idx]
    return genus_labels, genus_per_pixel

def check_balance(labels, indices_list, target_split, tolerance):
    total_species_sums = np.sum(labels, axis=0)
    split_proportions = []
    for indices in indices_list:
        split_sums = np.sum(labels[indices], axis=0)
        split_proportions.append(split_sums / total_species_sums)
    for i, proportion in enumerate(split_proportions):
        if not np.all(np.abs(proportion - target_split[i]) <= tolerance):
            return False
    return True

def iterative_split_superpixels(labels, target_split=[0.7,0.15,0.15], max_iter=5000, tolerance=0.02):
    num_samples = labels.shape[0]
    indices = np.arange(num_samples)
    for i in range(max_iter):
        train_val_indices, test_indices = train_test_split(
            indices, test_size=target_split[2], random_state=i
        )
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=target_split[1] / (target_split[0] + target_split[1]),
            random_state=i,
        )
        if check_balance(labels, [train_indices, val_indices, test_indices], target_split, tolerance):
            print(f"Balanced split found after {i+1} iterations.")
            return train_indices, val_indices, test_indices
    raise ValueError(f"Could not find a balanced split within the given tolerance after {max_iter} iterations.")

def save_genus_files(indices, split_name, file_paths, dst_folder, species_list, species_to_genus, genus_order):
    split_folder = os.path.join(dst_folder, split_name)
    os.makedirs(split_folder, exist_ok=True)
    for idx in indices:
        file_path = file_paths[idx]
        file_name = os.path.basename(file_path)
        data = np.load(file_path, allow_pickle=True)
        genus_label, genus_per_pixel = group_to_genus(
            data["label"], data["per_pixel_labels"], species_list, species_to_genus, genus_order
        )
        np.savez_compressed(
            os.path.join(split_folder, 'ovf_coarser', file_name),
            superpixel_images=data["superpixel_images"],
            point_cloud=data["point_cloud"],
            label=genus_label,
            per_pixel_labels=genus_per_pixel,
            nodata_mask=data["nodata_mask"]
        )

# ==============================
# 3. MAIN PIPELINE
# ==============================

print(f"=== {dataset_name.upper()} : Genus Grouping & Split ===")
print("Scanning files...")
superpixel_files = sorted(glob(os.path.join(src_folder, "**/ovf_genus/*.npz")))
file_names = [os.path.basename(f) for f in superpixel_files]
labels_genus = []

print("Grouping all samples to genus level...")
for file_path in superpixel_files:
    data = np.load(file_path, allow_pickle=True)
    label = data["label"]  # (num_species,)
    per_pixel_labels = data["per_pixel_labels"]
    genus_label, genus_per_pixel = group_to_genus(
        label, per_pixel_labels, species_names, species_to_genus, genus_order
    )
    labels_genus.append(genus_label)
labels_genus = np.array(labels_genus)

# Now do the iterative split for balanced genus proportions
print("Performing iterative balanced split...")
target_split = [0.7, 0.15, 0.15]
tolerance = 0.01
train_indices, val_indices, test_indices = iterative_split_superpixels(labels_genus, target_split, tolerance=tolerance)

print(f"Saving .npz files to train/val/test (genus grouped)...")
# file_names = [os.path.basename(f) for f in superpixel_files]  # <-- you can remove this line
save_genus_files(train_indices, "train", superpixel_files, output_dir, species_names, species_to_genus, genus_order)
save_genus_files(val_indices, "val", superpixel_files, output_dir, species_names, species_to_genus, genus_order)
save_genus_files(test_indices, "test", superpixel_files, output_dir, species_names, species_to_genus, genus_order)
print("Done!")

# Optional: Save file lists for reproducibility
def save_file_names(file_list, file_path):
    with open(file_path, "w") as f:
        f.write("\n".join(file_list))

save_file_names([os.path.basename(superpixel_files[i]) for i in train_indices], os.path.join(output_dir, "train_files.txt"))
save_file_names([os.path.basename(superpixel_files[i]) for i in val_indices], os.path.join(output_dir, "val_files.txt"))
save_file_names([os.path.basename(superpixel_files[i]) for i in test_indices], os.path.join(output_dir, "test_files.txt"))

print("File lists saved.")