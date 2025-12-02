#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Split superpixel dataset into train/val/test with balanced species proportions,
optionally subsample the training set, save file lists, plot proportions,
and (optionally) copy .npz files into split folders.

Example:
    python superpixel_resplit_5.py --dataset wrf --subsample_rate 0.3 --move_files
"""

import os
from glob import glob
import shutil
import argparse

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Species mapping per dataset
SPECIES_DICT = {
    "wrf": ["Sb", "La", "Pj", "Bw", "Pt", "Bf", "Cw", "Sw"],
    "rmf": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],
    "ovf": ["AB", "PO", "MR", "BF", "CE", "PW", "MH", "BW", "SW", "OR", "PR"],
}


def load_labels(folder_path: str):
    """
    Load label vectors from all .npz superpixel files in folder_path (recursively).

    Expects each .npz to contain an array 'label' with shape (num_species,).

    Returns
    -------
    labels : np.ndarray
        2D array of shape (num_samples, num_species), dtype=float.
    file_names : list[str]
        List of relative file paths (relative to folder_path) corresponding to labels.
    """
    superpixel_files = glob(os.path.join(folder_path, "**", "*.npz"), recursive=True)
    superpixel_files = sorted(superpixel_files)

    labels = []
    file_names = []
    bad_files = []

    print(f"Found {len(superpixel_files)} .npz files. Loading labels...")
    for file_path in tqdm(superpixel_files):
        try:
            with np.load(file_path, allow_pickle=True) as data:
                if "label" not in data:
                    print(f"[WARN] 'label' not found in {file_path}, skipping.")
                    bad_files.append(file_path)
                    continue

                labels.append(np.array(data["label"], dtype=float))
                file_names.append(os.path.relpath(file_path, folder_path))

        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
            bad_files.append(file_path)
            continue

    if len(labels) == 0:
        raise RuntimeError("No valid labels loaded. Check your .npz files.")

    labels = np.stack(labels).astype(float)

    print(f"\nLoaded {labels.shape[0]} label arrays with {labels.shape[1]} species.")
    if bad_files:
        print(f"{len(bad_files)} bad files skipped, e.g.:")
        for bf in bad_files[:10]:
            print("  -", bf)

    return labels, file_names


def check_balance(labels, indices_list, target_split, tolerance):
    """
    Check if the species proportions in each split are within tolerance
    of the desired dataset-level proportions (train/val/test).

    labels : (num_samples, num_species) float array
    indices_list : list of arrays [train_indices, val_indices, test_indices]
    target_split : list [train_ratio, val_ratio, test_ratio]
    tolerance : float
    """
    total_species_sums = np.sum(labels, axis=0)  # per species

    split_proportions = []
    for indices in indices_list:
        split_sums = np.sum(labels[indices], axis=0)
        split_proportions.append(split_sums / total_species_sums)

    return all(
        np.all(np.abs(p - target_split[i]) <= tolerance)
        for i, p in enumerate(split_proportions)
    )


def iterative_split_superpixels(
    labels, target_split=(0.7, 0.15, 0.15), max_iter=5000, tolerance=0.01
):
    """
    Find a train/val/test split that balances species proportions across splits.

    labels : (num_samples, num_species) float array
    """
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
            print(f"Balanced split found after {i + 1} iterations.")
            return train_indices, val_indices, test_indices

    raise ValueError(f"Could not find a balanced split within {max_iter} iterations.")


def calculate_split_proportions(labels, indices):
    """
    Compute per-species proportions for a split.

    labels : (num_samples, num_species)
    indices : indices belonging to the split
    """
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
    file_list contains relative paths (e.g., 'subdir/1234.npz').
    """
    os.makedirs(dst_folder, exist_ok=True)
    for rel_path in file_list:
        src_file = os.path.join(src_folder, rel_path)
        dst_file = os.path.join(dst_folder, os.path.basename(rel_path))
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
        else:
            print(f"[WARN] {src_file} does not exist, skipping.")


def plot_prop(df_proportions, output_dir):
    """
    Plot a heatmap of species proportions for Train/Val/Test splits.
    """
    df_proportions = df_proportions.astype(float)

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
    plt.close()
    print(f"Heatmap saved to {heatmap_path}")


def subsample_balanced_train(
    labels, train_indices, frac=0.3, max_iter=1000, tolerance=0.01, random_state=0
):
    """
    Subsample a fraction of the training samples while approximately preserving
    the species distribution of the full training set.

    Parameters
    ----------
    labels : np.ndarray
        2D array (num_samples, num_species), e.g. label counts/proportions per superpixel.
    train_indices : array-like
        Indices of the original training samples.
    frac : float
        Fraction of training samples to keep, e.g. 0.3 for 30%.
    max_iter : int
        Maximum number of random tries.
    tolerance : float
        Allowed absolute deviation for each species proportion.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    sub_train_indices : np.ndarray
        Subset of `train_indices` of length ~ frac * len(train_indices).
    """
    rng = np.random.RandomState(random_state)

    train_indices = np.array(train_indices)
    n_train_full = len(train_indices)
    n_sub = int(np.round(frac * n_train_full))

    if n_sub < 1:
        raise ValueError("frac is too small; resulting subset would be empty.")

    # Species distribution in the full training set
    train_sum = labels[train_indices].sum(axis=0)  # shape (num_species,)
    train_dist = train_sum / train_sum.sum()

    for i in range(max_iter):
        # Random subset of the training samples
        cand_indices = rng.choice(train_indices, size=n_sub, replace=False)

        cand_sum = labels[cand_indices].sum(axis=0)
        cand_dist = cand_sum / cand_sum.sum()

        if np.all(np.abs(cand_dist - train_dist) <= tolerance):
            print(
                f"Balanced subsample of train found after {i + 1} iterations "
                f"with frac={frac}."
            )
            return np.sort(cand_indices)

    raise ValueError(
        f"Could not find a balanced subsample within tolerance after {max_iter} iterations. "
        f"Try increasing tolerance or max_iter."
    )


def main():
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
        help="Tolerance for species proportion balance in the split",
    )
    parser.add_argument(
        "--move_files",
        action="store_true",
        help="Whether to copy .npz files into split folders",
    )
    parser.add_argument(
        "--subsample_rate",
        type=float,
        default=1.0,
        help="Fraction of training data to keep (e.g. 0.3 for 30%%).",
    )
    args = parser.parse_args()

    dataset = args.dataset
    species_names = SPECIES_DICT[dataset]

    folder_path = (
        f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/"
        f"{dataset}/processed/{dataset}_superpixel_dataset"
    )
    output_dir = (
        f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/"
        f"{dataset}/processed/{dataset}_sp_{int(args.subsample_rate*100)}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("Loading superpixel data...")
    labels, file_names = load_labels(folder_path)

    if labels.shape[1] != len(species_names):
        print(
            f"[WARN] labels have {labels.shape[1]} species, but SPECIES_DICT "
            f"lists {len(species_names)}. Check consistency."
        )

    print("Performing iterative split to balance the dataset...")
    train_indices, val_indices, test_indices = iterative_split_superpixels(
        labels, tolerance=args.tolerance
    )

    print(
        f"Original Train: {len(train_indices)}, "
        f"Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    # Build initial proportions dataframe (full train/val/test)
    df_proportions = pd.DataFrame(
        {
            "Species": species_names,
            "Train": calculate_split_proportions(labels, train_indices),
            "Validation": calculate_split_proportions(labels, val_indices),
            "Test": calculate_split_proportions(labels, test_indices),
        }
    ).set_index("Species")

    # Optional subsampling of training data
    if args.subsample_rate < 1.0:
        train_indices = subsample_balanced_train(
            labels,
            train_indices,
            frac=args.subsample_rate,
            max_iter=2000,
            tolerance=0.02,
            random_state=42,
        )
        # Recompute train proportions after subsampling
        df_proportions["Train"] = calculate_split_proportions(labels, train_indices)

    # Prepare file lists
    train_files = [file_names[i] for i in train_indices]
    val_files = [file_names[i] for i in val_indices]
    test_files = [file_names[i] for i in test_indices]

    # Save file names
    save_file_names(train_files, os.path.join(output_dir, "train_superpixels.txt"))
    save_file_names(val_files, os.path.join(output_dir, "val_superpixels.txt"))
    save_file_names(test_files, os.path.join(output_dir, "test_superpixels.txt"))

    # Plot proportions
    plot_prop(df_proportions, output_dir)

    # Optionally move/copy files
    if args.move_files:
        print("Copying .npz files into split folders...")
        move_files_to_split(train_files, folder_path, os.path.join(output_dir, "train"))
        move_files_to_split(val_files, folder_path, os.path.join(output_dir, "val"))
        move_files_to_split(test_files, folder_path, os.path.join(output_dir, "test"))
        print("Files copied successfully.")


if __name__ == "__main__":
    main()
