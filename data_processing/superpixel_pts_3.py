import laspy
import numpy as np
from shapely.geometry import Point
import os
from pathlib import Path
import geopandas as gpd
from resample_pts import farthest_point_sampling
import argparse
from multiprocessing import Pool, cpu_count


def resample_points_within_polygon(
    pts, max_pts
):
    if pts.shape[0] == 0:
        print("No points to sample from. Skipping.")
        return None, None

    if pts.shape[0] >= max_pts:
        use_idx = farthest_point_sampling(pts, max_pts)
        pts_sampled = pts[use_idx, :]
        classification = np.ones(max_pts, dtype=np.uint8)
    else:
        use_idx = np.random.choice(pts.shape[0], max_pts, replace=True)
        pts_sampled = pts[use_idx, :]
        classification = np.full(max_pts, 2, dtype=np.uint8)

    return pts_sampled, classification


def sample_points_within_polygon(
    las_file_path, polygon, max_pts, output_las_file_path, polyid, get_attributes=False
):
    try:
        extracted_points = []
        inFile = laspy.read(las_file_path)
        minx, miny, maxx, maxy = polygon.bounds

        points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()

        height_filtered = points[points[:, 2] > 2]
        mask = (
            (height_filtered[:, 0] >= minx)
            & (height_filtered[:, 0] <= maxx)
            & (height_filtered[:, 1] >= miny)
            & (height_filtered[:, 1] <= maxy)
        )
        candidate_points = height_filtered[mask]

        for point in candidate_points:
            if polygon.contains(Point(point[0], point[1])):
                extracted_points.append(point)

        extracted_points = np.array(extracted_points)

        output_las = laspy.create(
            point_format=inFile.header.point_format, file_version=inFile.header.version
        )

        extracted_points, classification = resample_points_within_polygon(
            extracted_points, max_pts
        )

        output_las.x = extracted_points[:, 0]
        output_las.y = extracted_points[:, 1]
        output_las.z = extracted_points[:, 2]

        if get_attributes:
            output_las.classification = classification

        output_las.write(output_las_file_path)
        print(f"Saved sampled points for {polyid} → {output_las_file_path}")

    except Exception as e:
        print(f"Error processing polygon {polyid}: {e}")


def process_polygon(row, las_dir, out_dir, max_pts):
    poly_id = row["POLYID"]
    tilename = row["Tilename"]
    polygon = row.geometry

    las_file_path = os.path.join(las_dir, f"{tilename}.laz")

    if not os.path.exists(las_file_path):
        print(f"LAS file missing: {tilename}.laz")
        return

    output_las_file_path = os.path.join(out_dir, f"{poly_id}.laz")

    if os.path.exists(output_las_file_path):
        print(f"Already exists: {output_las_file_path}")
        return

    sample_points_within_polygon(
        las_file_path,
        polygon,
        max_pts,
        output_las_file_path,
        poly_id,
        get_attributes=True,
    )


def sample_pts(
    polygons_file_path, las_files_directory, output_folder, max_pts
):

    las_files_directory = Path(las_files_directory)
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    gdf_polygons = gpd.read_file(Path(polygons_file_path))

    # Parallel processing
    num_cores = min(8, cpu_count())
    print(f"Using {num_cores} cores")

    args_list = [
        (row, las_files_directory, output_folder, max_pts)
        for _, row in gdf_polygons.iterrows()
    ]

    with Pool(processes=num_cores) as pool:
        pool.starmap(process_polygon, args_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Sample superpixel points from LAZ tiles"
    )
    parser.add_argument("--max_pts", type=int, default=7168)
    parser.add_argument("--dataset", type=str, default="wrf")

    args = parser.parse_args()

    dataset = args.dataset

    polygons_file_path = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/{dataset}/processed/{dataset}_plots/superpixel_plots_Tilename.gpkg"
    las_files_directory = f"/mnt/g/{dataset}/raw_laz"
    output_folder = f"/mnt/g/{dataset}/superpixel_plots"

    sample_pts(
        polygons_file_path, las_files_directory, output_folder, args.max_pts
    )

    # python superpixel_pts_3.py --dataset wrf --max_pts 7168
