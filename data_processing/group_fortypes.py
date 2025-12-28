import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
from .mapping_species import (
    group_to_coarser,
    iterative_split_superpixels,
    save_split_files
)

# ======================================================
# 1. DATASET CONFIG (augment-driven)
# ======================================================

DATASETS = {
    "rmf": {
        "species_names": ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"],
        "species_to_fortypes": {
            "PO": "hardwood",
            "SW": "spruce",
            "SB": "spruce",
            "BW": "hardwood",
            "BF": "conifer",
            "CE": "conifer",
            "PJ": "pine",
            "PT": "hardwood",
            "LA": "conifer",
        },
        "fortype_order": ["hardwood", "conifer", "pine", "spruce"],
        "subfolder": "rmf_sp",
        "save_prefix": "rmf_4class",
    },
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
        "species_to_fortypes": {
            "AB": "hardwood",
            "PO": "poplar",
            "SW": "conifer",
            "BW": "hardwood",
            "BF": "conifer",
            "CE": "conifer",
            "MR": "hardwood",
            "MH": "hardwood",
            "PW": "pine",
            "PR": "pine",
            "OR": "hardwood",
        },
        "fortype_order": ["hardwood", "poplar", "conifer", "pine"],
        "subfolder": "ovf_sp",
        "save_prefix": "ovf_4class",
    },
}

# 2. import functions from mapping_species

# ======================================================
# 3. MAIN
# ======================================================


def main(args):

    cfg = DATASETS[args.dataset]

    print(f"\n=== Processing dataset: {args.dataset.upper()} ===")
    print("Scanning NPZ files...")

    src_folder = os.path.join(
        args.root, args.dataset, f"{args.dataset}_superpixel_dataset/tile_128"
    )
    out_folder = src_folder
    os.makedirs(out_folder, exist_ok=True)

    search_pattern = f"**/{cfg['subfolder']}/*.npz"
    files = sorted(glob(os.path.join(src_folder, search_pattern)))

    print(f"Found {len(files)} files.")

    # ---- Group Each File to fortype ----
    fortype_labels = []
    print("Computing fortype labels for all samples...")
    for fp in tqdm(files):
        d = np.load(fp, allow_pickle=True)
        g, _ = group_to_coarser(
            d["label"],
            d["per_pixel_labels"],
            cfg["species_names"],
            cfg["species_to_fortypes"],
            cfg["fortype_order"],
        )
        fortype_labels.append(g)

    fortype_labels = np.array(fortype_labels)

    # ---- Balanced split ----
    print("\nGenerating balanced splits...")
    train_idx, val_idx, test_idx = iterative_split_superpixels(
        fortype_labels,
        target=args.split,
        tolerance=args.tolerance,
        max_iter=args.max_iter,
    )

    # ---- Save Files ----
    print("\nSaving files...")
    save_split_files(train_idx, "train", files, out_folder, cfg)
    save_split_files(val_idx, "val", files, out_folder, cfg)
    save_split_files(test_idx, "test", files, out_folder, cfg)

    print("\n✓ Done!\n")


# ======================================================
# 4. CLI ENTRY POINT
# ======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["rmf", "ovf"], required=True)
    parser.add_argument(
        "--root",
        default="/mnt/g",
        help="Base folder containing rmf/ and ovf/ directories.",
    )
    parser.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--max_iter", type=int, default=20000)

    args = parser.parse_args()
    main(args)

    # python fortype_split.py --dataset rmf
    # python fortype_split.py --dataset ovf
