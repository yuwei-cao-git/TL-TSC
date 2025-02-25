import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point
from tqdm import tqdm
import os
from pathlib import Path
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from resample_pts import farthest_point_sampling
import laspy
from pyproj import CRS

# Configuration
TILE_SIZE = 32
SPECIES_COUNT = 22
IMG_PATHS = {
    "s2_2020_spring": "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_img/ovf_s2_10m_2020_spring.tif",
    "s2_2020_summer": "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_img/ovf_s2_10m_2020_summer.tif",
    "s2_2020_fall": "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_img/ovf_s2_10m_2020_fall.tif",
    "s2_2020_winter": "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_img/ovf_s2_10m_2020_winter.tif",
    "dem": "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_img/ovf_spldem_10m.tif",
}
LABEL_RASTER_PATH = r"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_fri/masked/ovf_label_10m.tif"
LAS_FILES_DIR = r"/mnt/g/ovf/raw_laz"
OUTPUT_DIR = r"/mnt/g/ovf/dataset/train"
MAX_POINTS = 7168  # Max points to sample per plot
NODATA_IMG = -np.inf
NODATA_LABEL = -1
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_plot_labels(specs_perc):
    # parse specs_perc
    specs_perc = specs_perc.replace("[", "")
    specs_perc = specs_perc.replace("]", "")
    specs_perc = specs_perc.split(",")
    label = [float(i) for i in specs_perc]  # convert items in label to float
    # polyid_to_label[polyid] = np.array(specs_perc, dtype=float)
    return label


def process_pixel(pixel_label):
    """Validate individual pixel and return mask value"""
    # Check if all bands are -1 (no data)
    if np.all(pixel_label == -1):
        return False
    # Check sum is approximately 1.0 with tolerance
    label_sum = np.sum(pixel_label)
    return np.isclose(label_sum, 1.0, rtol=1e-5, atol=1e-5)


def resample_points_within_polygon(pts, max_pts):
    # Number of points to sample
    num_points = max_pts

    # Randomly sample x, y, and z within the specified bounds
    if pts.shape[0] == 0:
        return None
    else:
        if pts.shape[0] >= max_pts:
            use_idx = farthest_point_sampling(pts, num_points)
            pts = pts[use_idx, :]
        else:
            use_idx = np.random.choice(pts.shape[0], num_points, replace=True)
            pts = pts[use_idx, :]
        xyz = pts
        xyz_min = np.amin(xyz, axis=0, keepdims=True)
        xyz_max = np.amax(xyz, axis=0, keepdims=True)
        xyz_center = (xyz_min + xyz_max) / 2
        xyz_center[0][-1] = xyz_min[0][-1]
        xyz = xyz - xyz_center
        pts = np.hstack((pts, xyz))
        return pts


def sample_points_within_polygon(las_file_path, polygon, max_pts):
    extracted_points = []
    inFile = laspy.read(las_file_path)
    minx, miny, maxx, maxy = polygon.bounds

    points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()

    height_filtered_points = points[points[:, 2] > 2]
    mask = (
        (height_filtered_points[:, 0] >= minx)
        & (height_filtered_points[:, 0] <= maxx)
        & (height_filtered_points[:, 1] >= miny)
        & (height_filtered_points[:, 1] <= maxy)
    )
    candidate_points = height_filtered_points[mask]

    for point in candidate_points:
        if polygon.contains(Point(point[0], point[1])):
            extracted_points.append(point)
    extracted_points = np.array(extracted_points)
    return resample_points_within_polygon(extracted_points, max_pts)


def process_plot(plot, plot_6661, plot_fid, las_files_directory, max_pts):
    """Process a single plot into training samples"""
    results = {}
    centroid = plot.geometry.centroid

    # 2. Process label raster
    with rasterio.open(LABEL_RASTER_PATH) as src:
        px_label, py_label = src.index(centroid.x, centroid.y)
        window_label = Window(
            col_off=px_label - TILE_SIZE // 2,
            row_off=py_label - TILE_SIZE // 2,
            width=TILE_SIZE,
            height=TILE_SIZE,
        )
        label_tile = src.read(window=window_label, boundless=True)
        results["pixel_labels"] = label_tile.transpose(1, 2, 0).astype(
            np.float32
        )  # Convert entire array upfront  # HWC format

    # 3. Create validity mask
    valid_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    for i in range(TILE_SIZE):
        for j in range(TILE_SIZE):
            valid_mask[i, j] = process_pixel(results["pixel_labels"][i, j])

    # Skip if valid pixels < 13 (10000m2)
    if np.sum(valid_mask) < 13:
        print(f"Skipping plot {plot_fid} - no valid pixels")
        return None

    else:
        # 5. Add validity mask to results
        results["valid_mask"] = valid_mask.astype(np.uint8)  # Save as binary (0 or 1)
        # 1. Process imagery
        for name, path in IMG_PATHS.items():
            with rasterio.open(path) as src:
                # Convert plot centroid to pixel coordinates
                px, py = src.index(centroid.x, centroid.y)
                # Calculate window bounds
                window = Window(
                    col_off=px - TILE_SIZE // 2,
                    row_off=py - TILE_SIZE // 2,
                    width=TILE_SIZE,
                    height=TILE_SIZE,
                )

                # Read tile and handle out-of-bounds
                tile = src.read(window=window, boundless=True, fill_value=NODATA_IMG)
                results[f"img_{name}"] = tile.transpose(1, 2, 0)  # HWB format

        # 4. Apply mask to S2 and labels
        for name in IMG_PATHS:
            img_data = results[f"img_{name}"]
            img_data[~valid_mask] = NODATA_IMG
            results[f"img_{name}"] = img_data

        results["pixel_labels"][~valid_mask] = NODATA_LABEL

        # 6. Add plot-level labels
        results["plot_label"] = np.array(plot["perc_specs"])

        # 7. Add point cloud data
        tilename = plot_6661["Tilename"]
        polygon = plot_6661.geometry.buffer(11.28)
        las_file_path = os.path.join(las_files_directory, f"{tilename}.laz")
        if os.path.exists(las_file_path):
            point_cloud = sample_points_within_polygon(las_file_path, polygon, max_pts)
            if point_cloud is None:
                pass
            else:
                results["point_cloud"] = point_cloud  # From process_polygon
                # 9. Save as compressed numpy file
                output_path = os.path.join(OUTPUT_DIR, f"plot_{plot_fid}.npz")
                np.savez_compressed(output_path, **results)
                return output_path
        else:
            print(f"LAS file for {tilename} not found.")
            return None


def main_workflow(plots_file):
    """End-to-end processing workflow"""
    plots = gpd.read_file(plots_file)
    print(f"Loaded {len(plots)} plots")

    las_crs = CRS.from_epsg(6661)
    plots_6661 = plots.to_crs(las_crs)  # Convert polygon to assumed LAS CRS

    # Process plots with valid point clouds
    final_results = Parallel(n_jobs=4)(
        delayed(process_plot)(plot, plot_6661, idx, LAS_FILES_DIR, MAX_POINTS)
        for (idx, plot), (_, plot_6661) in tqdm(
            zip(plots.iterrows(), plots_6661.iterrows()), total=len(plots)
        )
    )

    print(f"Processed {len([r for r in final_results if r])} valid plots")


# Run the pipeline
if __name__ == "__main__":
    main_workflow(
        plots_file="/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/plots/plot_train_prom10_rem100_Tilename_2958.gpkg"
    )
    """
    {
        's2_spring': (32, 32, bands),  # -1 for invalid pixels
        's2_summer': (32, 32, bands),
        'pixel_labels': (32, 32, 22),  # -1 for invalid pixels
        'valid_mask': (32, 32),        # Binary mask (1 = valid, 0 = invalid)
        'plot_label': (22,),           # Original plot composition
        'point_cloud': (N, 3+)         # XYZ + attributes
    }
    data = np.load('plot_0.npz')
    print(data.files)
    # ['s2_spring', 's2_summer', 'pixel_labels', 'valid_mask', 'plot_label', 'point_cloud']

    # Access mask
    valid_mask = data['valid_mask']  # (32, 32) binary array
    
    import matplotlib.pyplot as plt

    plt.imshow(data['valid_mask'], cmap='gray')
    plt.title("Valid Pixel Mask")
    plt.show()
    
    s2_spring = data['s2_spring']  # (32, 32, bands)
    valid_pixels = s2_spring[data['valid_mask'] == 1]  # Filter valid pixels
    """
