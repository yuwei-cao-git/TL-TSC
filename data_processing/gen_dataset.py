import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point
from tqdm import tqdm
import os
from joblib import Parallel, delayed
from resample_pts import farthest_point_sampling
import laspy
from pyproj import CRS
from pts_utils import normalize_point_cloud, center_point_cloud

# Configuration
TILE_SIZE = 128
IMG_PATHS = {
    "s2_2020_spring": "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_img/ovf_s2_10m_2020_spring.tif",
    "s2_2020_summer": "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_img/ovf_s2_10m_2020_summer.tif",
    "s2_2020_fall": "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_img/ovf_s2_10m_2020_fall.tif",
    "s2_2020_winter": "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_img/ovf_s2_10m_2020_winter.tif",
    "dem": "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_img/ovf_spldem_10m.tif",
}
LABEL_RASTER_PATH = os.path.abspath(
    "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_fri/masked/ovf_label_10m.tif"
)
LAS_FILES_DIR = r"/mnt/g/ovf/raw_laz"
OUTPUT_DIR = r"/mnt/g/ovf/ovf_tl_dataset/tile_128/test"
MAX_POINTS = 7168  # Max points to sample per plot
NODATA_IMG = 0
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
    # Convert data type
    pixel_label = pixel_label.astype(np.float32)
    # Check if all bands are -1 (no data)
    if np.all(pixel_label == -1.0) or np.all(pixel_label == 0):
        return False
    # Check sum is 1.0
    label_sum = np.sum(pixel_label)
    return label_sum == 1.0


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

        xyz_normalized = normalize_point_cloud(pts)
        xyz_cercered = center_point_cloud(pts)
        pts = np.hstack((xyz_normalized, xyz_cercered))
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


def process_plot(plot, plot_6661, plot_fid, label_path, las_files_directory, max_pts):
    """Process a single plot into training samples"""
    results = {}
    centroid = plot.geometry.centroid

    # 2. Process label raster
    with rasterio.open(label_path) as src:
        row_label, col_label = src.index(centroid.x, centroid.y)
        # Calculate the window boundaries
        col_off = max(0, col_label - TILE_SIZE // 2)
        row_off = max(0, row_label - TILE_SIZE // 2)
        width = min(TILE_SIZE, src.width - col_off)
        height = min(TILE_SIZE, src.height - row_off)
        window_label = Window(
            col_off=col_off,
            row_off=row_off,
            width=width,
            height=height,
        )
        label_tile = src.read(window=window_label)
        # If the tile is smaller than TILE_SIZE, pad it with no-data values
        if label_tile.shape[1] < TILE_SIZE or label_tile.shape[2] < TILE_SIZE:
            padded_label_tile = np.full(
                (label_tile.shape[0], TILE_SIZE, TILE_SIZE),
                NODATA_LABEL,
                dtype=label_tile.dtype,
            )
            padded_label_tile[:, :height, :width] = label_tile
            label_tile = padded_label_tile

        results["pixel_labels"] = label_tile.transpose(
            1, 2, 0
        )  # Convert entire array upfront  # HWC format

    # 3. Create validity mask
    valid_mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    for i in range(TILE_SIZE):
        for j in range(TILE_SIZE):
            if np.all(results["pixel_labels"][i, j]) != NODATA_LABEL:
                valid_mask[i, j] = process_pixel(results["pixel_labels"][i, j])
            else:
                valid_mask[i, j] = False  # Mark as invalid if it's a no-data value

    # Skip if valid pixels < 13 (10000m2)
    if np.sum(valid_mask) < 13:
        print(f"Skipping plot {plot_fid} - no valid pixels")
        return None
    else:
        results["valid_mask"] = valid_mask.astype(np.uint8)  # Save as binary (0 or 1)
        # 4. Process imagery
        for name, path in IMG_PATHS.items():
            with rasterio.open(path) as src:
                # Convert plot centroid to pixel coordinates
                px, py = src.index(centroid.x, centroid.y)
                # Calculate window bounds
                col_off = max(0, py - TILE_SIZE // 2)
                row_off = max(0, px - TILE_SIZE // 2)
                width = min(TILE_SIZE, src.width - col_off)
                height = min(TILE_SIZE, src.height - row_off)

                # Create the window
                window = Window(
                    col_off=col_off, row_off=row_off, width=width, height=height
                )

                # Read the data within the window
                tile = src.read(window=window, boundless=True, fill_value=np.nan)  # replace -inf with NaN

                # If the tile is smaller than TILE_SIZE, pad it with no-data values
                if tile.shape[1] < TILE_SIZE or tile.shape[2] < TILE_SIZE:
                    padded_tile = np.full(
                        (tile.shape[0], TILE_SIZE, TILE_SIZE),
                        np.nan,
                        dtype=tile.dtype,
                    )
                    padded_tile[:, :height, :width] = tile
                    tile = padded_tile

                # Convert to uint8 with dynamic scaling
                uint8_tile = np.full(tile.shape, NODATA_IMG, dtype=np.uint8)
                for band_idx in range(tile.shape[0]):
                    band_data = tile[band_idx]
                    # Create mask of valid pixels
                    valid = ~np.isnan(band_data)
                    if not valid.any():
                        continue

                    # Calculate dynamic range (1nd-99th percentile)
                    p_low, p_high = np.percentile(band_data[valid], [1, 99])
                    if p_high <= p_low:
                        p_low, p_high = band_data[valid].min(), band_data[valid].max()
                        p_high = p_low + 1  # Prevent division by zero

                    # Normalize band data 
                    normed = (band_data[valid] - p_low) / (p_high - p_low)

                    # Clip and scale to 0–254
                    scaled = np.clip(normed * 254, 0, 254).astype(np.uint8)
                    
                    # Apply to output
                    uint8_tile[band_idx][valid] = scaled

                # Store result in HWC format with uint8 type
                results[f"img_{name}"] = uint8_tile.transpose(1, 2, 0)

        # 5. Apply mask to labels
        results["pixel_labels"][~valid_mask] = NODATA_LABEL

        # 6. Add plot-level labels
        results["plot_label"] = get_plot_labels(plot["perc_specs"])

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
    plots_6661 = plots.copy()

    las_crs = CRS.from_epsg(6661)
    plots_6661 = plots_6661.to_crs(las_crs)  # Convert polygon to assumed LAS CRS

    # Process plots with valid point clouds
    final_results = Parallel(n_jobs=6)(
        delayed(process_plot)(
            plot, plot_6661, idx, LABEL_RASTER_PATH, LAS_FILES_DIR, MAX_POINTS
        )
        for (idx, plot), (_, plot_6661) in tqdm(
            zip(plots.iterrows(), plots_6661.iterrows()), total=len(plots)
        )
    )

    print(f"Processed {len([r for r in final_results if r])} valid plots")


# Run the pipeline
if __name__ == "__main__":
    main_workflow(
        plots_file="/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/plots/plot_test_prom10_rem100_Tilename_2958.gpkg"
    )

    """
    {
        's2_spring': (32, 32, bands),  # -1 for invalid pixels
        's2_summer': (32, 32, bands),
        'pixel_labels': (32, 32, classes),  # -1 for invalid pixels
        'valid_mask': (32, 32),        # Binary mask (1 = valid, 0 = invalid)
        'plot_label': (classes,),           # Original plot composition
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
