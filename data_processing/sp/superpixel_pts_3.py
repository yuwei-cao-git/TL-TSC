import laspy
import numpy as np
from shapely.geometry import Point
import os
from pathlib import Path
import geopandas as gpd
from resample_pts import farthest_point_sampling
import argparse
from multiprocessing import Pool, cpu_count


def get_tilename(plots):
    ovf = gpd.read_file(
        r"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/data_processing/FORMGMT/LIO-2023-08-19/FOREST_MANAGEMENT_UNIT.shp",
        where="FMU_NAME='Ottawa Valley Forest'"
    ).to_crs(plots.crs)
    spl_tile_index = gpd.read_file(
        r"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/data_processing/FRI_Leaf_On_Tile_Index_SHP/FRI_Tile_Index.shp",
        columns=["Tilename"]
    )
    spl_tile_index_ovf = spl_tile_index.to_crs(plots.crs).clip(ovf)
    spl_tile_index_ovf.to_file(
        r"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/plots/spl_tile_index_ovf.shp"
    )
    # Perform the spatial join using 'within' operation
    plots_joined_att = plots.sjoin(spl_tile_index_ovf[['geometry', 'Tilename']], how='left', predicate='within')
    # Return DataFrame with duplicate rows removed. set keep=False to remove
    plots_joined_att.reset_index().drop_duplicates(subset=['POLYID'], keep=False)
    # remove NaN data
    plots_joined_att = plots_joined_att[plots_joined_att['Tilename'].notna()]
    if 'index' in plots_joined_att.columns:
        plots_joined_att = plots_joined_att.drop(columns=['index'])
    if 'index_right' in plots_joined_att.columns:
        plots_joined_att = plots_joined_att.drop(columns=['index_right'])
    if 'index_right0' in plots_joined_att.columns:
        plots_joined_att = plots_joined_att.drop(columns=['index_right0'])
    if 'index_righ' in plots_joined_att.columns:
        plots_joined_att = plots_joined_att.drop(columns=['index_righ'])
    return plots_joined_att


def resample_points_within_polygon(
    pts, max_pts, min_x, max_x, min_y, max_y, min_z, max_z
):
    num_points = max_pts

    if pts.shape[0] == 0:
        # No points available
        print("No points to sample from. Skipping.")
        return None, None
    elif pts.shape[0] >= max_pts:
        use_idx = farthest_point_sampling(pts, num_points)
        pts_sampled = pts[use_idx, :]
        classification = np.ones(num_points, dtype=np.uint8)
    else:
        # Randomly sample with replacement
        use_idx = np.random.choice(pts.shape[0], num_points, replace=True)
        pts_sampled = pts[use_idx, :]
        classification = np.full(num_points, 2, dtype=np.uint8)
    return pts_sampled, classification


def sample_points_within_polygon(
    las_file_path, polygon, max_pts, output_las_file_path, polyid, get_attributes=False
):
    try:
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

    except Exception as e:
        print(f"An error occurred while processing polygon {polyid}: {e}")


def process_polygon(polygon_row, las_files_directory, output_folder, max_pts):
    poly_id = polygon_row["POLYID"]
    tilename = polygon_row["Tilename"]
    polygon = polygon_row.geometry
    las_file_path = os.path.join(las_files_directory, f"{tilename}.laz")

    if os.path.exists(las_file_path):
        output_las_file_path = os.path.join(output_folder, f"{poly_id}.laz")
        if os.path.exists(output_las_file_path):
            print(f"Saved sampled points for {poly_id} to {output_las_file_path}")
        else:
            sample_points_within_polygon(
                las_file_path,
                polygon,
                max_pts,
                output_las_file_path,
                poly_id,
                get_attributes=True,
            )
    else:
        print(f"LAS file for {tilename} not found.")


def sample_pts(polygons_file_path, las_files_directory, output_folder, max_pts):
    las_files_directory = Path(las_files_directory)
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    gdf_polygons = gpd.read_file(Path(polygons_file_path))
    tilename_polygons = get_tilename(gdf_polygons)

    num_cores = min(8, cpu_count())
    print(f"Using {num_cores} cores for parallel processing.")

    args_list = [
        (row, las_files_directory, output_folder, max_pts)
        for _, row in tilename_polygons.iterrows()
    ]
    with Pool(processes=num_cores) as pool:
        pool.starmap(process_polygon, args_list)


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

    """
    python data_processing/superpixel_pts.py --polygons_file_path "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/plots/superpixel_plots/superpixel_plots.gpkg" --las_files_directory /mnt/g/ovf/raw_laz --output_folder /mnt/g/ovf/superpxiel_plots --max_pts 7168
    """
