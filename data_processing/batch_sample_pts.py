import laspy
import numpy as np
from shapely import Point
import os
from pathlib import Path
import geopandas as gpd
from resample_pts import farthest_point_sampling
import argparse
from multiprocessing import Pool, cpu_count

import os


def get_unique_filename(output_folder, polyid):
    base_path = os.path.join(output_folder, f"{polyid}.laz")

    if not os.path.exists(base_path):
        return base_path  # Return original filename if it doesn’t exist

    # If file exists, find next available index
    index = 1
    while True:
        new_path = os.path.join(output_folder, f"{polyid}_{index}.laz")
        if not os.path.exists(new_path):
            return new_path  # Return first available indexed filename
        index += 1


def resample_points_within_polygon(
    pts, max_pts, min_x, max_x, min_y, max_y, min_z, max_z
):
    # Number of points to sample
    num_points = max_pts

    # Randomly sample x, y, and z within the specified bounds
    if pts.shape[0] == 0:
        x = np.random.uniform(min_x, max_x, num_points)
        y = np.random.uniform(min_y, max_y, num_points)
        z = np.random.uniform(min_z, max_z, num_points)
        # Combine into an array of (x, y, z) points
        pts = np.column_stack((x, y, z))
        classification = np.zeros(num_points, dtype=np.uint8)
    elif pts.shape[0] >= max_pts:
        use_idx = farthest_point_sampling(pts, num_points)
        pts = pts[use_idx, :]
        classification = np.ones(num_points, dtype=np.uint8)
    elif pts.shape[0] <= max_pts // 2:
        x = np.random.uniform(min_x, max_x, num_points)
        y = np.random.uniform(min_y, max_y, num_points)
        z = np.random.uniform(min_z, max_z, num_points)
        # Combine into an array of (x, y, z) points
        pts = np.column_stack((x, y, z))
        classification = np.zeros(num_points, dtype=np.uint8)
    else:
        use_idx = np.random.choice(pts.shape[0], num_points, replace=True)
        pts = pts[use_idx, :]
        classification = np.full(num_points, 2, dtype=np.uint8)
    return pts, classification


def sample_points_within_polygon(
    las_file_path, polygon, max_pts, output_folder, polyid, get_attributes=False
):
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

    output_las_file_path = output_las_file_path = get_unique_filename(
        output_folder, polyid
    )
    output_las = laspy.create(
        point_format=inFile.header.point_format, file_version=inFile.header.version
    )

    extracted_points, classification = resample_points_within_polygon(
        extracted_points, max_pts, minx, maxx, miny, maxy, 2, 3
    )

    output_las.x = extracted_points[:, 0]
    output_las.y = extracted_points[:, 1]
    output_las.z = extracted_points[:, 2]

    if get_attributes:
        output_las.classification = classification

    output_las.write(output_las_file_path)
    print(f"Saved sampled points for {polyid} to {output_las_file_path}")


def process_polygon(polygon_row, las_files_directory, output_folder, max_pts):
    poly_id = polygon_row["POLYID"]
    tilename = polygon_row["Tilename"]
    polygon = polygon_row.geometry
    las_file_path = os.path.join(las_files_directory, f"{tilename}.laz")

    if os.path.exists(las_file_path):
        sample_points_within_polygon(
            las_file_path, polygon, max_pts, output_folder, poly_id, get_attributes=True
        )
    else:
        print(f"LAS file for {tilename} not found.")


def sample_pts(polygons_file_path, las_files_directory, output_folder, max_pts):
    las_files_directory = Path(las_files_directory)
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    gdf_polygons = gpd.read_file(Path(polygons_file_path))

    num_cores = min(32, cpu_count())
    print(f"Using {num_cores} cores for parallel processing.")

    with Pool(processes=num_cores) as pool:
        pool.starmap(
            process_polygon,
            [
                (polygon_row, las_files_directory, output_folder, max_pts)
                for _, polygon_row in gdf_polygons.iterrows()
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process some input files for tree species estimation."
    )

    parser.add_argument(
        "--polygons_file_path",
        type=str,
        required=True,
        help="Path to the polygons file (e.g., a .gpkg file).",
    )
    parser.add_argument(
        "--las_files_directory",
        type=str,
        required=True,
        help="Directory containing the raw .las files.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Directory where the output will be saved.",
    )
    parser.add_argument(
        "--max_pts", type=int, default=7168, help="Maximum number of points to sample."
    )

    args = parser.parse_args()

    sample_pts(
        args.polygons_file_path,
        args.las_files_directory,
        args.output_folder,
        args.max_pts,
    )
