import os
import argparse
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGS FOR EACH DATASET
# ============================================================

DATASET_CONFIG = {
    "wrf": {
        "species_names": ["SB", "LA", "PJ", "BW", "PT", "BF", "CW", "SW"],
        "species_to_genus": {
            "SB": "SB",
            "LA": "LA",
            "PJ": "PJ",
            "BW": "BW",
            "BF": "BF",
            "PT": "poplar",
            "CW": "cedar",
            "SW": "SW",
        },
        "genus_order": ["poplar", "SW", "BW", "BF", "cedar", "PJ", "SB", "LA"],
        "src_pattern": "**/wrf_sp/*.npz",
        "output_subfolder": "wrf_msp",
    },
    "rmf": {
        "species_names": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],
        "species_to_genus": {
            "PO": "poplar",
            "SW": "SW",
            "BW": "BW",
            "BF": "BF",
            "CE": "cedar",
            "PT": "poplar",
            "PJ": "PJ",
            "SB": "SB",
            "LA": "LA",
        },
        "genus_order": ["poplar", "SW", "BW", "BF", "cedar", "PJ", "SB", "LA"],
        "src_pattern": "**/rmf_sp/*.npz",
        "output_subfolder": "rmf_msp", 
    },
}

# ============================================================
# FUNCTIONS
# ============================================================


def group_to_coarser(
    label, per_pixel_labels, species_list, species_to_genus, genus_order
):
    genus_labels = np.zeros(len(genus_order), dtype=label.dtype)
    genus_per_pixel = np.zeros(
        (len(genus_order),) + per_pixel_labels.shape[1:], dtype=per_pixel_labels.dtype
    )
    genus_to_idx = {g: i for i, g in enumerate(genus_order)}

    for idx, sp in enumerate(species_list):
        genus = species_to_genus.get(sp)
        if genus is None or genus not in genus_to_idx:
            continue
        gidx = genus_to_idx[genus]
        genus_labels[gidx] += label[idx]
        genus_per_pixel[gidx] += per_pixel_labels[idx]

    return genus_labels, genus_per_pixel


def check_balance(labels, indices_list, target_split, tolerance):
    total_sums = np.sum(labels, axis=0)
    for i, split_indices in enumerate(indices_list):
        split_sums = np.sum(labels[split_indices], axis=0)
        prop = split_sums / total_sums
        if not np.all(np.abs(prop - target_split[i]) <= tolerance):
            return False
    return True


def iterative_split_superpixels(
    labels, target_split=[0.7, 0.15, 0.15], max_iter=5000, tolerance=0.01
):
    num_samples = labels.shape[0]
    indices = np.arange(num_samples)

    for seed in range(max_iter):
        train_val, test = train_test_split(
            indices, test_size=target_split[2], random_state=seed
        )
        train, val = train_test_split(
            train_val,
            test_size=target_split[1] / (target_split[0] + target_split[1]),
            random_state=seed,
        )

        if check_balance(labels, [train, val, test], target_split, tolerance):
            print(f"Balanced split found after {seed + 1} iterations.")
            return train, val, test

    raise RuntimeError(f"Could not find balanced split after {max_iter} attempts.")


def save_split_files(indices, split_name, file_paths, dst_folder, cfg):
    split_folder = os.path.join(dst_folder, split_name)
    if cfg["output_subfolder"]:
        split_folder = os.path.join(split_folder, cfg["output_subfolder"])
    os.makedirs(split_folder, exist_ok=True)

    species_list = cfg["species_names"]
    species_to_genus = cfg["species_to_genus"]
    genus_order = cfg["genus_order"]

    for idx in indices:
        fp = file_paths[idx]
        fname = os.path.basename(fp)
        data = np.load(fp, allow_pickle=True)

        genus_label, genus_pixel = group_to_coarser(
            data["label"],
            data["per_pixel_labels"],
            species_list,
            species_to_genus,
            genus_order,
        )

        out_path = os.path.join(split_folder, fname)
        np.savez_compressed(
            out_path,
            superpixel_images=data["superpixel_images"],
            point_cloud=data["point_cloud"],
            label=genus_label,
            per_pixel_labels=genus_pixel,
            nodata_mask=data["nodata_mask"],
        )


def save_file_list(name_list, file_path):
    with open(file_path, "w") as f:
        f.write("\n".join(name_list))


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ovf", "rmf", "wrf"], required=True)
    parser.add_argument("--src", required=True, help="Input root folder")
    parser.add_argument("--out", required=True, help="Output folder")
    args = parser.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    print(f"=== Processing {args.dataset.upper()} Dataset ===")

    # Find files
    pattern = os.path.join(args.src, cfg["src_pattern"])
    files = sorted(glob(pattern))
    print(f"Found {len(files)} samples.")

    # Group labels into genus
    print("Grouping labels...")
    labels_genus = []
    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        glabel, _ = group_to_coarser(
            data["label"],
            data["per_pixel_labels"],
            cfg["species_names"],
            cfg["species_to_genus"],
            cfg["genus_order"],
        )
        labels_genus.append(glabel)

    labels_genus = np.array(labels_genus)

    # Split
    print("Performing balanced split...")
    train_idx, val_idx, test_idx = iterative_split_superpixels(labels_genus)

    # Save NPZ files
    print("Saving data...")
    save_split_files(train_idx, "train", files, args.out, cfg)
    save_split_files(val_idx, "val", files, args.out, cfg)
    save_split_files(test_idx, "test", files, args.out, cfg)

    # Save file lists
    save_file_list(
        [os.path.basename(files[i]) for i in train_idx],
        os.path.join(args.out, "train_files.txt"),
    )
    save_file_list(
        [os.path.basename(files[i]) for i in val_idx],
        os.path.join(args.out, "val_files.txt"),
    )
    save_file_list(
        [os.path.basename(files[i]) for i in test_idx],
        os.path.join(args.out, "test_files.txt"),
    )

    print("Done!")


if __name__ == "__main__":
    main()

    """
    # how to use: 
    python data_processing/process_genus.py \
        --dataset rmf \
        --src /mnt/g/rmf/rmf_superpixel_dataset/tile_128 \
        --out /mnt/g/rmf/rmf_superpixel_dataset/tile_128
    python data_processing/mapping_species.py \
        --dataset wrf \
        --src /mnt/g/wrf/wrf_superpixel_dataset/tile_128 \
        --out /mnt/g/wrf/wrf_superpixel_dataset/tile_128
    """
