import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm

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


# ======================================================
# 2. GENERIC FUNCTIONS
# ======================================================


def group_to_fortypes(
    label, per_pixel_labels, species_list, species_to_fortypes, fortype_order
):
    fortype_labels = np.zeros(len(fortype_order), dtype=label.dtype)
    fortype_perpixel = np.zeros(
        (len(fortype_order),) + per_pixel_labels.shape[1:], dtype=per_pixel_labels.dtype
    )

    fortype_to_idx = {g: i for i, g in enumerate(fortype_order)}

    for i, sp in enumerate(species_list):
        g = species_to_fortypes.get(sp)
        if g in fortype_to_idx:
            g_idx = fortype_to_idx[g]
            fortype_labels[g_idx] += label[i]
            fortype_perpixel[g_idx] += per_pixel_labels[i]

    return fortype_labels, fortype_perpixel


def check_balance(labels, index_groups, target_split, tolerance):
    total = np.sum(labels, axis=0)
    for i, idx in enumerate(index_groups):
        proportion = np.sum(labels[idx], axis=0) / total
        if not np.all(np.abs(proportion - target_split[i]) <= tolerance):
            return False
    return True


def iterative_split(labels, target=[0.7, 0.15, 0.15], max_iter=20000, tolerance=0.01):
    n = len(labels)
    indices = np.arange(n)

    for seed in range(max_iter):
        train_val, test = train_test_split(
            indices, test_size=target[2], random_state=seed
        )
        train, val = train_test_split(
            train_val, test_size=target[1] / (target[0] + target[1]), random_state=seed
        )

        if check_balance(labels, [train, val, test], target, tolerance):
            print(f"✓ Balanced split found at iteration {seed}")
            return train, val, test

    raise RuntimeError("Balanced split not found within max iterations.")


def save_split_files(indices, split_name, file_paths, dst, cfg):
    folder = os.path.join(dst, split_name, cfg["save_prefix"])
    os.makedirs(folder, exist_ok=True)

    for idx in indices:
        fp = file_paths[idx]
        name = os.path.basename(fp)

        data = np.load(fp, allow_pickle=True)
        glabel, gperpixel = group_to_fortypes(
            data["label"],
            data["per_pixel_labels"],
            cfg["species_names"],
            cfg["species_to_fortypes"],
            cfg["fortype_order"],
        )

        np.savez_compressed(
            os.path.join(folder, name),
            superpixel_images=data["superpixel_images"],
            point_cloud=data["point_cloud"],
            label=glabel,
            per_pixel_labels=gperpixel,
            nodata_mask=data["nodata_mask"],
        )


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
        g, _ = group_to_fortypes(
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
    train_idx, val_idx, test_idx = iterative_split(
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
