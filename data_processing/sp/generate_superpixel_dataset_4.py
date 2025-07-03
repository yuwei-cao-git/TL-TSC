import os
import numpy as np
import rasterio
import laspy
import geopandas as gpd
from tqdm import tqdm


def load_preprocessed_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    tile_images = data["tile_images"]  # List of images from all seasons
    label_array = data["label_array"]  # Shape: (num_classes, height, width)
    superpixel_mask = data["superpixel_mask"]  # Shape: (height, width)
    nodata_mask = data["nodata_mask"]  # Shape: (height, width)
    return tile_images, label_array, superpixel_mask, nodata_mask


def load_point_cloud(laz_file_path):
    # Read point cloud data from .laz file
    point_cloud = laspy.read(laz_file_path)
    # Extract x, y, z coordinates
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    classifications = point_cloud.classification

    # Filter out points with classification 0 (invalid points)
    valid_indices = classifications != 0
    valid_points = points[valid_indices]

    return valid_points


def get_polygon_labels(polygon_gdf):
    # Create a dictionary mapping POLYID to specs_perc
    polyid_to_label = {}
    for idx, row in polygon_gdf.iterrows():
        polyid = row["POLYID"]
        specs_perc = row["perc_specs"]  # Adjust the field name as needed
        specs_perc = specs_perc.replace("[", "")
        specs_perc = specs_perc.replace("]", "")
        specs_perc = specs_perc.split(",")
        polyid_to_label[polyid] = [
            float(i) for i in specs_perc
        ]  # convert items in label to float
        # polyid_to_label[polyid] = np.array(specs_perc, dtype=float)
    return polyid_to_label


def save_superpixel_data(
    polyid,
    superpixel_images,
    point_cloud,
    label,
    per_pixel_labels,
    nodata_mask,
    output_dir,
):
    output_file_path = os.path.join(output_dir, f"{polyid}.npz")
    np.savez_compressed(
        output_file_path,
        superpixel_images=superpixel_images,  # Shape: (num_seasons, num_channels, 128, 128)
        point_cloud=point_cloud,  # Shape: (7168, 3)
        label=label,  # Shape: (num_classes,)
        per_pixel_labels=per_pixel_labels,  # Shape: (num_classes, 128, 128)
        nodata_mask=nodata_mask,  # Shape: (128, 128)
    )
    print(f"Superpixel data saved to {output_file_path}")


def read_split_file(split_file_path):
    with open(split_file_path, "r") as f:
        tile_names = [line.strip() for line in f]
    return tile_names


def generate_combined_data_for_split(
    tile_npz_dir,
    point_cloud_dir,
    polygon_file_path,
    output_superpixel_dir,
    split_file_path,
):
    # Read the split file
    tile_names = read_split_file(split_file_path)

    # Load the polygon file with specs_perc per polygon
    polygon_gdf = gpd.read_file(polygon_file_path)
    polygon_gdf["POLYID"] = polygon_gdf["POLYID"].astype(int)
    polyid_to_label = get_polygon_labels(polygon_gdf)

    # Ensure output directory exists
    os.makedirs(output_superpixel_dir, exist_ok=True)

    for tile_name in tqdm(tile_names, desc="Processing Tiles"):
        tile_name = tile_name.split('.')[0]
        tile_file_path = os.path.join(tile_npz_dir, f"{tile_name}_combined.npz")

        # Check that all tile files exist
        if not os.path.exists(tile_file_path):
            print(f"Tile files for {tile_name} not found")
            continue

        # Load preprocessed tile data from all seasons
        tile_images, label_array, superpixel_mask, nodata_mask = load_preprocessed_data(
            tile_file_path
        )

        # Get unique superpixel IDs (excluding background)
        unique_polyids = np.unique(superpixel_mask)
        unique_polyids = unique_polyids[
            unique_polyids != 0
        ]  # Exclude background or no-data value (assuming 0)

        for polyid in unique_polyids:
            # Create a mask for the current superpixel
            superpixel_mask_binary = superpixel_mask == polyid

            # Handle NoData pixels and create combined mask
            # combined_mask is True for valid pixels (belongs to superpixel and not NoData)
            combined_mask = superpixel_mask_binary & (~nodata_mask)

            # Initialize padded images and masks
            num_channels = tile_images[0].shape[0]
            num_seasons = len(tile_images)
            padded_image_shape = (num_seasons, num_channels, 128, 128)
            padded_label_shape = (label_array.shape[0], 128, 128)
            padded_nodata_mask = np.ones(
                (128, 128), dtype=bool
            )  # Start with all True (NoData)

            # Update the NoData mask: set valid pixels to False (i.e., not NoData)
            padded_nodata_mask[combined_mask] = False

            # Initialize arrays for superpixel images
            superpixel_images = np.zeros(padded_image_shape, dtype=np.uint16)

            # Extract and pad images for each season
            for season_idx, season_image in enumerate(tile_images):
                # For each channel
                for channel_idx in range(num_channels):
                    # Initialize padded channel image
                    padded_channel_image = np.zeros(
                        (128, 128), dtype=season_image.dtype
                    )

                    # Assign pixel values for valid pixels
                    pixel_values = season_image[channel_idx][combined_mask]

                    # Assign scaled values
                    padded_channel_image[combined_mask] = pixel_values

                    # Assign to superpixel_images array
                    superpixel_images[season_idx, channel_idx, :, :] = (
                        padded_channel_image
                    )

            # Initialize per-pixel labels
            num_classes = label_array.shape[0]
            per_pixel_labels = np.zeros(padded_label_shape, dtype=label_array.dtype)

            # Assign per-pixel labels for valid pixels
            for class_idx in range(num_classes):
                # Initialize padded label image
                padded_label_image = np.zeros((128, 128), dtype=label_array.dtype)

                # Assign label values for valid pixels
                padded_label_image[combined_mask] = label_array[class_idx][
                    combined_mask
                ]

                # Assign to per_pixel_labels array
                per_pixel_labels[class_idx, :, :] = padded_label_image

            # Load point cloud data corresponding to the POLYID
            laz_file_path = os.path.join(point_cloud_dir, f"{polyid}.laz")
            if os.path.exists(laz_file_path):
                point_cloud = load_point_cloud(laz_file_path)
                if point_cloud.size == 0:
                    print(f"No valid points in point cloud for POLYID {polyid}")
                    continue
            else:
                print(f"Point cloud file for POLYID {polyid} not found.")
                continue  # Skip if point cloud file doesn't exist

            # Get the label (specs_perc) for the POLYID
            if polyid in polyid_to_label:
                superpixel_label = polyid_to_label[polyid]
            else:
                print(f"Label for POLYID {polyid} not found in polygon file.")
                continue  # Skip if label not found

            # Save the superpixel data
            save_superpixel_data(
                polyid,
                superpixel_images,
                point_cloud,
                superpixel_label,
                per_pixel_labels,
                padded_nodata_mask,
                output_superpixel_dir,
            )


if __name__ == "__main__":
    splits = ["train", "test", "val"]
    tile_npz_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_s2_compressed"
    for split in splits:
        point_cloud_dir = f"/mnt/g/ovf/superpxiel_plots/"  # Directory containing .laz files per POLYID
        polygon_file_path = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_plots/superpixel_plots/superpixel_plots_Tilename.gpkg"  # Path to polygon file with specs_perc per polygon
        output_superpixel_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_superpixel/{split}"

        split_file_path = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/dataset/{split}_tiles.txt"  # Path to the split file (train/test/val)

        # Generate combined data for the specified split
        generate_combined_data_for_split(
            tile_npz_dir,
            point_cloud_dir,
            polygon_file_path,
            output_superpixel_dir,
            split_file_path,
        )
