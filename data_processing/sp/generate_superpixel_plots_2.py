import random
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def create_points(polygon, number, distance=20, max_attempts=1000):
    # Empty list for points
    points = []

    # Get polygon bounds
    min_x, min_y, max_x, max_y = polygon.bounds

    for i in range(0, max_attempts):
        if len(points) < number:
            # Create random point
            point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            point = point.buffer(
                11.28
            )  # buffer point returns a circle polygonal result

            # Plot criteria
            if polygon.contains(point):  # polygon must contain point
                if (
                    polygon.boundary.distance(point.boundary) >= distance
                ):  # 100m from boundary
                    points.append(point)  # append points if criteria met
        else:
            pass
    return points


def random_plots(polygon, n, distance=50):
    random.seed(73)
    # Get crs
    crs = polygon.crs

    # Run random points
    all_points = []
    for i, row in polygons_gdf.iterrows():
        # print(f"{i} / {len(boundary)}")
        points = create_points(polygon.iloc[i].geometry, n, distance)
        if len(points) >= 1:
            points_gdf = gpd.GeoDataFrame({"geometry": points})
            for attribute in [
                "POLYID",
                "perc_specs",
            ]:  # List all attributes you want to include
                points_gdf[attribute] = row[attribute]
            all_points.append(points_gdf)

    # Create geodataframe of points
    if len(all_points) >= 1:
        plots = gpd.GeoDataFrame(pd.concat(all_points, ignore_index=True))
        plots.set_crs(crs=crs, inplace=True)  # set crs

        return plots

    else:
        return None


# Main code
if __name__ == "__main__":
    # Load the polygons from the superpixel.gpkg file
    polygons_gdf = gpd.read_file(
        "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/ovf_fri/superpixel.shp"
    )

    plots = random_plots(polygons_gdf, n=1, distance=20)
    print(plots.head())
    # Save the plots to a file if needed
    plots.to_file(
        "/mnt/d/Sync/research/tree_species_estimation/tree_dataset/ovf/processed/plots/superpixel_plots/superpixel_plots.gpkg"
    )
