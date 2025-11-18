import random
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def create_points(polygon, number, distance=25, max_attempts=1000):
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds

    for i in range(max_attempts):
        if len(points) < number:
            # Random point
            pt = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            pt = pt.buffer(11.28)

            # Conditions
            if polygon.contains(pt):
                if polygon.boundary.distance(pt.boundary) >= distance:
                    points.append(pt)
    return points


def random_plots(polygons, n, distance=50):
    random.seed(73)
    crs = polygons.crs
    all_points = []

    for _, row in polygons.iterrows():
        pts = create_points(row.geometry, n, distance)
        if len(pts) > 0:
            pts_gdf = gpd.GeoDataFrame({"geometry": pts})
            # Copy desired attributes
            for attribute in ["POLYID", "perc_specs"]:
                pts_gdf[attribute] = row[attribute]
            all_points.append(pts_gdf)

    if len(all_points) == 0:
        return None

    plots = gpd.GeoDataFrame(pd.concat(all_points, ignore_index=True), crs=crs)
    return plots


# --------------------------------------------------
# FUNCTION: assign Tilename
# --------------------------------------------------

def get_tilename(plots, dataset):
    """
    Assign SPL tile names to polygons by clipping tile index to FMU.
    """

    # FMU boundary
    if dataset == "wrf":
        FMU_name = "White River Forest"
    elif dataset == "rmf":
        FMU_name = "Romeo Mallate Forest"
    elif dataset == "ovf":
        FMU_name = "Ottawa Valley Forest"

    fmu = gpd.read_file(
        r"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/data_processing/FORMGMT/LIO-2023-08-19/FOREST_MANAGEMENT_UNIT.shp",
        where=f"FMU_NAME={FMU_name}",
    ).to_crs(plots.crs)

    # SPL tile index
    spl_tile_index = gpd.read_file(
        r"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/data_processing/FRI_Leaf_On_Tile_Index_SHP/FRI_Tile_Index.shp",
        columns=["Tilename"],
    )

    # Clip tile index to FMU
    spl_tile_index_fmu = spl_tile_index.to_crs(plots.crs).clip(fmu)

    # Save clipped tile index for debugging
    out_tile_index_path = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/{dataset}/processed/{dataset}_plots/spl_tile_index_{dataset}.shp"
    spl_tile_index_fmu.to_file(out_tile_index_path)

    # Spatial join
    plots_joined_att = plots.sjoin(
        spl_tile_index_fmu[["geometry", "Tilename"]], how="left", predicate="within"
    )

    # Remove bad records
    plots_joined_att = plots_joined_att[plots_joined_att["Tilename"].notna()]

    # Remove index columns from sjoin
    for col in ["index", "index_right", "index_right0", "index_righ"]:
        if col in plots_joined_att.columns:
            plots_joined_att = plots_joined_att.drop(columns=[col])

    return plots_joined_att


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    dataset = "wrf"

    polygons_gdf = gpd.read_file(
        f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/{dataset}/processed/{dataset}_fri/superpixel.shp"
    )

    # Generate random plots
    plots = random_plots(polygons_gdf, n=1, distance=25)
    print(plots.head())

    base_output_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/{dataset}/processed/{dataset}_plots"
    out_base = f"{base_output_dir}/superpixel_plots"

    # Save original random plots
    plots.to_file(f"{out_base}.gpkg")

    # ----------------------------------------
    # Add Tilename
    # ----------------------------------------

    plots_tilename = get_tilename(plots, dataset)

    # Save original CRS
    plots_tilename.to_file(f"{out_base}_Tilename.gpkg")

    # Save EPSG:2958
    plots_tilename_2958 = plots_tilename.to_crs("EPSG:2958")
    plots_tilename_2958.to_file(f"{out_base}_Tilename_2958.gpkg")

    print("Done! Saved:")
    print(f" - {out_base}_Tilename.gpkg")
    print(f" - {out_base}_Tilename_2958.gpkg")
