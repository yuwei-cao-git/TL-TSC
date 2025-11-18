import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil
import argparse

# Species mapping per dataset
SPECIES_DICT = {
    "wrf": ["Sb", "La", "Pj", "Bw", "Pt", "Bf", "Cw", "Sw"],
    "rmf": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],
    "ovf": ["AB", "PO", "MR", "BF", "CE", "PW", "MH", "BW", "SW", "OR", "PR"],
}


def load_labels(folder_path):
    superpixel_files = glob(os.path.join(folder_path, "*", "*.npz"))
    labels, file_names = [], []
    for file_path in superpixel_files:
        data = np.load(file_path, allow_pickle=True)
        labels.append(data["label"])
        file_names.append(os.path.relpath(file_path, folder_path))
    return np.array(labels), file_names


def check_balance(labels, indices_list, target_split, tolerance):
    total_species_sums = np.sum(labels, axis=0)
    split_proportions = []
    for indices in indices_list:
        split_sums = np.sum(labels[indices], axis=0)
        split_proportions.append(split_sums / total_species_sums)
    return all(
        np.all(np.abs(p - target_split[i]) <= tolerance)
        for i, p in enumerate(split_proportions)
    )


def iterative_split_superpixels(
    labels, target_split=[0.7, 0.15, 0.15], max_iter=5000, tolerance=0.01
):
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
        if check_balance(
            labels, [train_indices, val_indices, test_indices], target_split, tolerance
        ):
            print(f"Balanced split found after {i+1} iterations.")
            return train_indices, val_indices, test_indices
    raise ValueError(f"Could not find a balanced split within {max_iter} iterations.")


def calculate_split_proportions(labels, indices):
    split_sums = np.sum(labels[indices], axis=0)
    total_sums = np.sum(labels, axis=0)
    return split_sums / total_sums


def save_file_names(file_list, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write("\n".join(file_list))


def move_files_to_split(file_list, src_folder, dst_folder):
    """
    Copy .npz files listed in file_list from src_folder to dst_folder.
    file_list may contain relative paths (e.g., 'train/1234.npz').
    """
    os.makedirs(dst_folder, exist_ok=True)
    for rel_path in file_list:
        src_file = os.path.join(src_folder, rel_path)
        dst_file = os.path.join(dst_folder, os.path.basename(rel_path))
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
        else:
            print(f"Warning: {src_file} does not exist.")


def plot_prop(df_proportions, output_dir):
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df_proportions,
        annot=True,
        cmap="YlGnBu",
        cbar_kws={"label": "Proportion"},
        fmt=".4f",
    )
    plt.xlabel("Dataset Split")
    plt.ylabel("Species")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "species_proportions_heatmap.png")
    plt.savefig(heatmap_path)
    print(f"Heatmap saved to {heatmap_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split superpixel dataset into train/val/test with balanced species proportions"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wrf",
        choices=list(SPECIES_DICT.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Tolerance for species proportion balance",
    )
    parser.add_argument(
        "--move_files",
        action="store_true",
        help="Whether to copy .npz files into split folders",
    )
    args = parser.parse_args()

    dataset = args.dataset
    species_names = SPECIES_DICT[dataset]

    folder_path = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/{dataset}/processed/{dataset}_superpixel_dataset"
    output_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/{dataset}/processed/dataset_split"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading superpixel data...")
    labels, file_names = load_labels(folder_path)

    print("Performing iterative split to balance the dataset...")
    train_indices, val_indices, test_indices = iterative_split_superpixels(
        labels, tolerance=args.tolerance
    )

    print(
        f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    train_files = [file_names[i] for i in train_indices]
    val_files = [file_names[i] for i in val_indices]
    test_files = [file_names[i] for i in test_indices]

    # Save file names
    save_file_names(train_files, os.path.join(output_dir, "train_superpixels.txt"))
    save_file_names(val_files, os.path.join(output_dir, "val_superpixels.txt"))
    save_file_names(test_files, os.path.join(output_dir, "test_superpixels.txt"))

    # Plot heatmap
    df_proportions = pd.DataFrame(
        {
            "Species": species_names,
            "Train": calculate_split_proportions(labels, train_indices),
            "Validation": calculate_split_proportions(labels, val_indices),
            "Test": calculate_split_proportions(labels, test_indices),
        }
    ).set_index("Species")

    plot_prop(df_proportions, output_dir)

    # Move files to split folders
    if args.move_files:
        print("Copying .npz files into split folders...")
        move_files_to_split(train_files, folder_path, os.path.join(output_dir, "train"))
        move_files_to_split(val_files, folder_path, os.path.join(output_dir, "val"))
        move_files_to_split(test_files, folder_path, os.path.join(output_dir, "test"))
        print("Files copied successfully.")

    # python split_superpixels.py --dataset wrf --move_files
