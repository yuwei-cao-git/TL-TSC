import os
import argparse
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from mapping_species import (
    group_to_coarser,
    iterative_split_superpixels,
    save_split_files,
    save_file_list,
)

# ============================================================
# CONFIGS FOR EACH DATASET
# ============================================================

DATASET_CONFIG = {
    "ovf": {
        "species_names": [
            "AB",
            "PO",
            "MR",
            "BF",
            "CE",
            "PW",
            "MH",
            "BW",
            "SW",
            "OR",
            "PR",
        ],
        "species_to_genus": {
            "AB": "ash",
            "PO": "poplar",
            "SW": "spruce",
            "BW": "birch",
            "BF": "fir",
            "CE": "cedar",
            "MR": "maple",
            "PW": "pine",
            "MH": "maple",
            "OR": "oak",
            "PR": "pine",
        },
        "genus_order": [
            "ash",
            "poplar",
            "spruce",
            "birch",
            "fir",
            "cedar",
            "maple",
            "pine",
            "oak",
        ],
        "src_pattern": "**/ovf_2s_sp/*.npz",
        "output_subfolder": "ovf_2s_genus",
    },
    "wrf": {
        "species_names": ["SB", "LA", "PJ", "BW", "PT", "BF", "CW", "SW"],
        "species_to_genus": {
            "SW": "spruce",
            "BW": "birch",
            "BF": "fir",
            "CW": "cedar",
            "PT": "poplar",
            "PJ": "pine",
            "SB": "spruce",
            "LA": "larch",
        },
        "genus_order": ["poplar", "spruce", "birch", "fir", "cedar", "pine", "larch"],
        "src_pattern": "**/wrf_sp/*.npz",
        "output_subfolder": "",  # RMF uses raw split folder
    },
}

# 2. import functions from mapping_species

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
    python process_genus.py \
        --dataset ovf \
        --src /mnt/g/ovf/ovf_superpixel_dataset_v2/tile_128 \
        --out /mnt/g/ovf/ovf_superpixel_dataset_v2/tile_128
    python process_genus.py \
        --dataset rmf \
        --src /mnt/g/rmf/rmf_superpixel_dataset/tile_128 \
        --out /mnt/g/rmf/rmf_genus
    python process_genus.py \
        --dataset wrf \
        --src /mnt/g/wrf/wrf_superpixel_dataset/tile_128 \
        --out /mnt/g/wrf/wrf_genus
    """
