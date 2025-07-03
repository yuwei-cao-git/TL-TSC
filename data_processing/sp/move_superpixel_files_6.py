import os
import shutil

# Define paths
folder_path = "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed/20m/superpixel"
output_dir = "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed/20m/superpixel/dataset"

# Paths to the split files
train_files_path = os.path.join(output_dir, "train_superpixels.txt")
val_files_path = os.path.join(output_dir, "val_superpixels.txt")
test_files_path = os.path.join(output_dir, "test_superpixels.txt")


# Function to read file names from text files
def read_file_names(file_path):
    with open(file_path, "r") as f:
        file_names = f.read().splitlines()
    return file_names


# Read file names from the split files
print("Reading file names from split files...")
train_files = read_file_names(train_files_path)
val_files = read_file_names(val_files_path)
test_files = read_file_names(test_files_path)

# Create subfolders in folder_path
print("Creating subfolders for train, validation, and test splits...")
train_folder = os.path.join(folder_path, "train")
val_folder = os.path.join(folder_path, "val")
test_folder = os.path.join(folder_path, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


# Function to move files
def move_files(file_list, src_folder, dst_folder):
    for file_name in file_list:
        src_file = os.path.join(src_folder, file_name)
        dst_file = os.path.join(dst_folder, file_name)
        if os.path.exists(src_file):
            shutil.move(src_file, dst_file)
        else:
            print(f"Warning: File {src_file} does not exist and cannot be moved.")


# Move files to the respective subfolders
print("Moving training files...")
move_files(train_files, folder_path, train_folder)

print("Moving validation files...")
move_files(val_files, folder_path, val_folder)

print("Moving testing files...")
move_files(test_files, folder_path, test_folder)

print("Files have been successfully moved to their respective folders.")
