import geopandas as gpd
import fiona
import rasterio
import tqdm
import numpy as np
import laspy
import os
from osgeo import gdal


def print_color(text, color):
    # ANSI escape sequences for different colors
    color_codes = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    if color in color_codes:
        color_code = color_codes[color]
        reset_code = color_codes["reset"]
        print(color_code + text + reset_code)
    else:
        print(text)


def pixel_centers(raster_path, shapefile_path):
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Get the transformation matrix
        transform = src.transform

        # Get the pixel size
        pixel_size_x = transform.a
        pixel_size_y = -transform.e  # Negative for y-coordinate flip

        # Calculate the half pixel offset
        half_pixel_x = pixel_size_x / 2
        half_pixel_y = pixel_size_y / 2

        # Get the CRS of the raster
        crs = src.crs
        # print("the shape of input raster: " + str(src.shape))
        # Create the shapefile with the same CRS as the raster
        schema = {
            "geometry": "Point",
            "properties": {"tile_name": "str", "tile_i": "int", "tile_j": "int"},
        }
        # Check if the shapefile exists
        if not os.path.exists(shapefile_path):
            # Create the shapefile first if it doesn't exist
            with fiona.open(
                shapefile_path, "w", "ESRI Shapefile", schema, crs=crs
            ) as dst:
                # Optionally, you can write an initial feature if needed
                pass
        with fiona.open(shapefile_path, "a", "ESRI Shapefile", schema, crs=crs) as dst:
            # Read the raster data
            raster_data = src.read(1)

            # Iterate over each pixel and write the center point to the shapefile if the pixel value is non-zero
            for i, j in list(np.ndindex(src.shape)):  # src.shape: (5807, 5196)
                pixel_value = raster_data[i, j]
                if pixel_value != (src.nodatavals[0] or -1 or np.nan):
                    # Calculate the center coordinates of the pixel
                    x = transform.c + (j + 0.5) * transform.a
                    y = transform.f + (i + 0.5) * transform.e

                    # Create the point geometry
                    point = {"type": "Point", "coordinates": (x, y)}

                    # Create the feature and write it to the shapefile
                    feature = {
                        "geometry": point,
                        "properties": {
                            "tile_name": os.path.basename(raster_path)[:-4],
                            "tile_i": i,
                            "tile_j": j,
                        },
                    }
                    dst.write(feature)
    # print("Center Points Created")


def buffer_points(input_shapefile, output_shapefile, buffer_distance=10):
    """
    Buffer points in a shapefile and save the result to a new shapefile.

    Args:
        input_shapefile (str): Path to the input points shapefile.
        output_shapefile (str): Path to the output shapefile where buffered data will be saved.
        buffer_distance (float): Buffer distance in the same units as the input shapefile (default is 10).

    Returns:
        None
    """
    # Read the points shapefile
    points_gdf = gpd.read_file(input_shapefile)

    # Create a buffer of 10 units around the points
    buffered_gdf = points_gdf.copy()
    buffered_gdf["geometry"] = points_gdf.buffer(buffer_distance)

    # Save the buffered polygons to a new shapefile
    buffered_gdf.to_file(output_shapefile)


# ref: https://pysal.org/tobler/_modules/tobler/area_weighted/area_join.html#area_join
import numpy as np
import pandas as pd
import warnings


def area_join(source_df, target_df, variables):
    """
    Join variables from source_df based on the largest intersection. In case of a tie it picks the first one.

    Parameters
    ----------
    source_df : geopandas.GeoDataFrame
        GeoDataFrame containing source values
    target_df : geopandas.GeoDataFrame
        GeoDataFrame containing source values
    variables : string or list-like
        column(s) in source_df dataframe for variable(s) to be joined

    Returns
    -------
    joined : geopandas.GeoDataFrame
         target_df GeoDataFrame with joined variables as additional columns

    """
    if not pd.api.types.is_list_like(variables):
        variables = [variables]

    for v in variables:
        if v in target_df.columns:
            raise ValueError(f"Column '{v}' already present in target_df.")

    target_df = target_df.copy()
    target_ix, source_ix = source_df.sindex.query(
        target_df.geometry, predicate="intersects"
    )
    areas = (
        target_df.geometry.values[target_ix]
        .intersection(source_df.geometry.values[source_ix])
        .area
    )

    main = []
    for i in range(len(target_df)):  # vectorise this loop?
        mask = target_ix == i
        if np.any(mask):
            main.append(source_ix[mask][np.argmax(areas[mask])])
        else:
            main.append(np.nan)

    main = np.array(main, dtype=float)
    mask = ~np.isnan(main)

    for v in variables:
        arr = np.empty(len(main), dtype=object)
        arr[mask] = source_df[v].values[main[mask].astype(int)]
        try:
            arr = arr.astype(source_df[v].dtype)
        except TypeError:
            warnings.warn(
                f"Cannot preserve dtype of '{v}'. Falling back to `dtype=object`.",
            )
        target_df[v] = arr
    # Create the new 'POLYID' field by concatenating 'tile_name', 'tile_i', and 'tile_j'
    target_df["POLYID"] = (
        target_df["tile_name"].astype(str)
        + "_"
        + target_df["tile_i"].astype(str)
        + "_"
        + target_df["tile_j"].astype(str)
    )

    return target_df


def read_las(pointcloudfile, get_attributes=False, useevery=1):
    """
    :param pointcloudfile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)
    :param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    """

    # Read file
    inFile = laspy.read(pointcloudfile)

    # Get coordinates (XYZ)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]

    # Return coordinates only
    if get_attributes == False:
        return coords

    # Return coordinates and attributes
    else:
        las_fields = [info.name for info in inFile.points.point_format.dimensions]
        attributes = {}
        # for las_field in las_fields[3:]:  # skip the X,Y,Z fields
        for las_field in las_fields:  # get all fields
            attributes[las_field] = inFile.points[las_field][::useevery]
        return (coords, attributes)


def write_las(outpoints, outfilepath, attribute_dict={}):
    """
    :param outpoints: 3D array of points to be written to output file
    :param outfilepath: specification of output file (format: las or laz)
    :param attribute_dict: dictionary of attributes (key: name of attribute; value: 1D array of attribute values in order of points in 'outpoints'); if not specified, dictionary is empty and nothing is added
    :return: None
    """
    import laspy

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.x_scale = 0.00025
    hdr.y_scale = 0.00025
    hdr.z_scale = 0.00025
    mean_extent = np.mean(outpoints, axis=0)
    hdr.x_offset = int(mean_extent[0])
    hdr.y_offset = int(mean_extent[1])
    hdr.z_offset = int(mean_extent[2])

    las = laspy.LasData(hdr)

    las.x = outpoints[:, 0]
    las.y = outpoints[:, 1]
    las.z = outpoints[:, 2]
    for key, vals in attribute_dict.items():
        try:
            las[key] = vals
        except:
            las.add_extra_dim(laspy.ExtraBytesParams(name=key, type=type(vals[0])))
            las[key] = vals

    las.write(outfilepath)


def normalize_point_cloud(xyz):
    # Center and scale spatial coordinates
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid
    max_distance = np.max(np.linalg.norm(xyz_centered, axis=1))
    xyz_normalized = xyz_centered / (max_distance + 1e-8)

    return xyz_normalized


def center_point_cloud(xyz):
    xyz_min = np.amin(xyz, axis=0, keepdims=True)
    xyz_max = np.amax(xyz, axis=0, keepdims=True)
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_center[0][-1] = xyz_min[0][-1]
    xyz = xyz - xyz_center
    return xyz
