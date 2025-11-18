import os
import argparse
import numpy as np
import rasterio
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import geopandas as gpd
import laspy


# --------------------------------------------------
# PART 1 — Convert each season’s tiles → compressed .npz
# --------------------------------------------------
def process_season_tile(args):
    tile_path, output_dir = args
    tile_name = os.path.splitext(os.path.basename(tile_path))[0]

    with rasterio.open(tile_path) as src:
        tile_image = src.read()  # (bands, H, W)
        nodata_mask = src.read_masks(1) == 0

    out_path = os.path.join(output_dir, f"{tile_name}.npz")
    np.savez_compressed(out_path, tile_image=tile_image, nodata_mask=nodata_mask)
    return tile_name


def process_season(season_dir, output_dir, num_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    tile_paths = sorted(glob.glob(os.path.join(season_dir, "*.tif")))

    print(f"Processing season directory: {season_dir}")
    args = [(p, output_dir) for p in tile_paths]

    with Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_season_tile, args),
            total=len(args),
            desc=f"{os.path.basename(season_dir)}",
        ):
            pass


# --------------------------------------------------
# PART 2 — Combine all seasons + superpixel for each tile
# --------------------------------------------------
def combine_one_tile(args):
    tile_name, season_npz_dirs, superpixel_npz_dir, output_dir = args

    tile_images = []
    masks = []

    # Load seasonal .npz
    for season_dir in season_npz_dirs:
        path = os.path.join(season_dir, f"{tile_name}.npz")
        if not os.path.exists(path):
            print(f"[WARN] Missing {tile_name} in {season_dir}")
            return None

        data = np.load(path)
        tile_images.append(data["tile_image"])
        masks.append(data["nodata_mask"])

    # Load superpixel .npz
    sp_path = os.path.join(superpixel_npz_dir, f"{tile_name}.npz")
    if not os.path.exists(sp_path):
        print(f"[WARN] Missing superpixel for {tile_name}")
        return None

    sp_data = np.load(sp_path)
    superpixel_mask = sp_data["superpixel_mask"]
    label_array = sp_data["label_array"]
    nodata_super = sp_data["nodata_mask"]

    combined_mask = np.logical_or.reduce(masks + [nodata_super])

    out = os.path.join(output_dir, f"{tile_name}_combined.npz")
    np.savez_compressed(
        out,
        tile_images=tile_images,
        label_array=label_array,
        superpixel_mask=superpixel_mask,
        nodata_mask=combined_mask,
    )
    return tile_name


def combine_all_tiles(season_npz_dirs, superpixel_npz_dir, output_dir, num_workers=8):
    os.makedirs(output_dir, exist_ok=True)

    example_dir = season_npz_dirs[0]
    tile_paths = sorted(glob.glob(os.path.join(example_dir, "*.npz")))
    tile_names = [os.path.splitext(os.path.basename(t))[0] for t in tile_paths]

    args = [
        (name, season_npz_dirs, superpixel_npz_dir, output_dir) for name in tile_names
    ]

    print("Combining seasons → combined tiles")
    with Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(combine_one_tile, args),
            total=len(args),
            desc="CombineTiles",
        ):
            pass


# --------------------------------------------------
# PART 3 — Generate Superpixel Dataset (images + lidar)
# --------------------------------------------------
def load_point_cloud(path):
    pc = laspy.read(path)
    pts = np.vstack((pc.x, pc.y, pc.z)).T
    cls = pc.classification
    return pts[cls != 0]


def get_polygon_labels(polygon_path):
    gdf = gpd.read_file(polygon_path)
    gdf["POLYID"] = gdf["POLYID"].astype(int)

    mapping = {}
    for _, row in gdf.iterrows():
        s = row["perc_specs"].replace("[", "").replace("]", "").split(",")
        mapping[row["POLYID"]] = [float(i) for i in s]
    return mapping


def process_one_combined_tile(args):
    tile_path, point_cloud_dir, label_dict, output_dir = args
    tile_name = os.path.basename(tile_path).replace("_combined.npz", "")

    data = np.load(tile_path, allow_pickle=True)
    tile_images = data["tile_images"]
    label_array = data["label_array"]
    superpixel_mask = data["superpixel_mask"]
    nodata_mask = data["nodata_mask"]

    H, W = superpixel_mask.shape
    polyids = np.unique(superpixel_mask)
    polyids = polyids[polyids != 0]

    for polyid in polyids:
        sp_mask = superpixel_mask == polyid
        valid = sp_mask & (~nodata_mask)

        if not valid.any():
            continue

        # Prepare padded arrays
        num_seasons = len(tile_images)
        num_channels = tile_images[0].shape[0]
        num_classes = label_array.shape[0]

        superpixel_img = np.zeros(
            (num_seasons, num_channels, 128, 128), dtype=np.uint16
        )
        per_pixel_labels = np.zeros((num_classes, 128, 128), dtype=np.float32)
        padded_nodata = np.ones((128, 128), dtype=bool)
        padded_nodata[valid] = False

        for s in range(num_seasons):
            for c in range(num_channels):
                padded = np.zeros((128, 128), dtype=tile_images[s].dtype)
                padded[valid] = tile_images[s][c][valid]
                superpixel_img[s, c] = padded

        for cl in range(num_classes):
            padded = np.zeros((128, 128), dtype=label_array.dtype)
            padded[valid] = label_array[cl][valid]
            per_pixel_labels[cl] = padded

        laz_path = os.path.join(point_cloud_dir, f"{polyid}.laz")
        if not os.path.exists(laz_path):
            continue

        pc = load_point_cloud(laz_path)
        if pc.size == 0:
            continue

        if polyid not in label_dict:
            continue

        out_path = os.path.join(output_dir, f"{polyid}.npz")
        np.savez_compressed(
            out_path,
            superpixel_images=superpixel_img,
            point_cloud=pc,
            label=label_dict[polyid],
            per_pixel_labels=per_pixel_labels,
            nodata_mask=padded_nodata,
        )


def generate_superpixel_dataset(
    combined_dir, point_cloud_dir, polygon_path, split_file, output_dir, num_workers=8
):

    os.makedirs(output_dir, exist_ok=True)
    label_map = get_polygon_labels(polygon_path)

    with open(split_file, "r") as f:
        tile_names = [line.strip().replace(".tif", "") for line in f]

    tile_paths = [os.path.join(combined_dir, f"{t}_combined.npz") for t in tile_names]
    tile_paths = [p for p in tile_paths if os.path.exists(p)]

    args = [(p, point_cloud_dir, label_map, output_dir) for p in tile_paths]

    print(f"Generating superpixel dataset for split: {split_file}")
    with Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_one_combined_tile, args),
            total=len(args),
            desc="Superpixel",
        ):
            pass


# --------------------------------------------------
# CLI ENTRYPOINT
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name, e.g., wrf"
    )
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    args = parser.parse_args()

    dataset = args.dataset
    workers = args.workers

    base = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/{dataset}/processed/{dataset}_s2"

    seasons = ["spring", "summer", "fall", "winter"]

    # ---------------------- STEP 1 ----------------------
    # Process each season → compressed npz
    for season in seasons:
        season_dir = f"{base}/{season}/tiles_128"
        out_dir = f"{base}/{season}/compressed"
        process_season(season_dir, out_dir, workers)

    # ---------------------- STEP 2 ----------------------
    season_npz_dirs = [f"{base}/{s}/compressed" for s in seasons]
    superpixel_npz_dir = f"{base}/fall/superpixel"
    combined_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/{dataset}/processed/{dataset}_img_combined"
    combine_all_tiles(season_npz_dirs, superpixel_npz_dir, combined_dir, workers)

    # ---------------------- STEP 3 ----------------------
    splits = ["train", "val", "test"]
    polygon_path = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/{dataset}/processed/{dataset}_plots/superpixel_plots_Tilename.gpkg"
    point_cloud_dir = f"/mnt/g/{dataset}/superpxiel_plots/"

    for split in splits:
        split_file = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/{dataset}/processed/dataset/{split}_tiles.txt"
        output_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/{dataset}/processed/{dataset}_superpixel_dataset/{split}"

        generate_superpixel_dataset(
            combined_dir, point_cloud_dir, polygon_path, split_file, output_dir, workers
        )

    # python process_dataset.py --dataset wrf --workers 16