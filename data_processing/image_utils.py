import math
import os
import shutil
import sys
import zipfile

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window

# from s2cloudless import S2PixelCloudDetector
from scipy.ndimage import zoom
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# from skimage import exposure
# from skimage.transform import AffineTransform, rotate, warp
from tqdm import tqdm
from utils import (
    unzip_directory,
    unzip_folder,
    print_color,
    delete_directory,
    find_subfolders,
    zip_and_remove_folder,
)


def set_nodata(folder_path, nodata_value):
    for filename in tqdm(os.listdir(folder_path), desc=f"{folder_path} set_nodata: "):
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith(".tif"):
            with rasterio.open(file_path, "r+") as src:
                src.nodata = nodata_value


def tile_raster(input_path, output_dir, tile_size, overlap_percent=0, nodata_value=0):
    with rasterio.open(input_path) as src:
        num_bands = src.count
        width, height = src.width, src.height
        nodata = src.nodata if src.nodata is not None else nodata_value

        # Adjust tile_size_overlap based on the overlap_percent
        tile_size_overlap = tile_size * overlap_percent
        tile_step = tile_size - tile_size_overlap
        if tile_step <= 0:
            raise ValueError("Tile step must be positive. Reduce the overlap_percent.")

        # Calculate the number of tiles in each dimension with overlap:
        num_tiles_x = math.ceil((width - tile_size_overlap) / tile_step)
        num_tiles_y = math.ceil((height - tile_size_overlap) / tile_step)

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Prepare a no data array to use for padding if necessary
        no_data_array = np.full(
            (num_bands, tile_size, tile_size), nodata, dtype=src.meta["dtype"]
        )

        # Loop through each tile
        for tile_y in tqdm(range(num_tiles_y), leave=True):
            for tile_x in range(num_tiles_x):
                # Calculate the pixel coordinates for the current tile
                x = int(tile_x * tile_step)
                y = int(tile_y * tile_step)
                tile_width = min(tile_size, width - x)
                tile_height = min(tile_size, height - y)

                # Create a window for the current tile
                window = Window(x, y, tile_width, tile_height)

                # Read the data for the current tile
                tile_data = src.read(window=window)

                # Check if any band in the tile has values
                if not tile_data.any():
                    continue

                # Pad the tile if its size is less than the tile_size
                if tile_width != tile_size or tile_height != tile_size:
                    padded_tile_data = no_data_array.copy()
                    padded_tile_data[:, :tile_height, :tile_width] = tile_data
                    tile_data = padded_tile_data

                # Create a new raster file for the current tile
                tile_path = os.path.join(output_dir, f"tile_{tile_x}_{tile_y}.tif")
                tile_profile = src.profile.copy()
                tile_profile.update(
                    width=tile_size,
                    height=tile_size,
                    transform=src.window_transform(window),
                )

                with rasterio.open(tile_path, "w", **tile_profile) as dst:
                    dst.write(tile_data)

        print("Raster Tiled")


def create_hls_folders(out_dir, file_path):
    subfolders = os.path.basename(file_path).split(".")[1:4]
    current_path = out_dir
    for subfolder in subfolders:
        current_path = os.path.join(current_path, subfolder)
        os.makedirs(current_path, exist_ok=True)

    new_file_path = os.path.join(current_path, os.path.basename(file_path))
    os.rename(file_path, new_file_path)


def composite_bands(input_files, output_file, band_names):
    # Open the first input file to get the metadata
    with rasterio.open(input_files[0]) as src:
        profile = src.profile

    # Update the metadata for the output file
    profile.update(count=len(input_files))

    # Create the output file
    with rasterio.open(output_file, "w", **profile) as dst:
        for i, (file, band_name) in enumerate(zip(input_files, band_names), start=1):
            with rasterio.open(file) as src:
                dst.write(src.read(1), i)
                dst.set_band_description(i, band_name)


def reproject_rasters(input_rasters, output_dir, target_crs):
    for raster_path in input_rasters:
        with rasterio.open(raster_path) as src:
            # Get the source CRS and transform
            src_crs = src.crs
            src_transform = src.transform

            # Calculate the transform to the target CRS
            dst_crs = target_crs
            width = src.width
            height = src.height
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, width, height, *src.bounds
            )

            # Update the destination transform to match the desired resolution
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs,
                dst_crs,
                width,
                height,
                *src.bounds,
                dst_width=src.width,
                dst_height=src.height,
            )

            # Create the output path using the input raster name
            output_path = f"{output_dir}/{os.path.basename(raster_path)}"

            # Reproject the raster
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                crs=dst_crs,
                transform=dst_transform,
                width=dst_width,
                height=dst_height,
                count=src.count,
                dtype=src.dtypes[0],
            ) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )


def clip_raster(raster_file, shapefile, output_file, invert=False, crop=True):
    # Load study area
    shapefile_data = gpd.read_file(shapefile)
    # Read the raster file using rasterio
    with rasterio.open(raster_file) as src:
        # Copy the metadata from the source raster
        profile = src.profile
        src_crs = src.crs

        # Clip the raster using the shapefile geometry
        # shapefile_data = shapefile_data.to_crs(src_crs)
        clipped_raster, transform = mask(
            src, shapefile_data.geometry, crop=crop, invert=invert
        )

    # Update the metadata with new dimensions and transform
    profile.update(
        {
            "driver": "GTiff",
            "height": clipped_raster.shape[1],
            "width": clipped_raster.shape[2],
            "transform": transform,
            "nodata": src.nodata,
        }
    )

    # Save the clipped raster to a new file
    with rasterio.open(output_file, "w", **profile) as dest:
        dest.write(clipped_raster)


def mosaic_rasters(raster_list, output_path, resampling=Resampling.bilinear):
    out_folder = os.path.dirname(output_path)
    os.makedirs(out_folder, exist_ok=True)
    temp_folder = os.path.join(out_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    # Reproject Files
    with rasterio.open(raster_list[0]) as src:
        crs = src.crs
        transform = src.transform
        band_descriptions = src.descriptions  # Get band descriptions
        nodata = src.nodata

    reproject_rasters(raster_list, temp_folder, crs)

    temp_raster_list = [
        os.path.join(temp_folder, file)
        for file in os.listdir(temp_folder)
        if file.endswith(".tif")
    ]

    # Open all input rasters
    src_files = []
    for raster_file in temp_raster_list:
        src = rasterio.open(raster_file)
        src_files.append(src)

    # Merge the rasters with defined interpolation
    mosaic, out_trans = merge(src_files, nodata=nodata, resampling=resampling)

    # Create the output raster file
    profile = src_files[0].profile
    profile.update(
        {"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans}
    )
    profile.update(decriptions=band_descriptions)

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(mosaic)

            # Save the mosaic to the output path
            with rasterio.open(output_path, "w", **profile) as dest:
                dest.write(dataset.read())

    # Close all the source rasters
    for src in src_files:
        src.close()

    delete_directory(temp_folder)


def merge_rasters(input_files, output_file, method="min"):
    out_folder = os.path.dirname(output_file)
    os.makedirs(out_folder, exist_ok=True)
    # Open the input files
    src_files = [rasterio.open(file) for file in input_files]

    band_descriptions = src_files[0].descriptions

    merged, out_trans = merge(src_files, method=method)

    # Create the output raster file
    profile = src_files[0].profile
    profile.update(
        {"height": merged.shape[1], "width": merged.shape[2], "transform": out_trans}
    )
    profile.update(descriptions=band_descriptions)

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(merged)

            # Save the mosaic to the output path
            with rasterio.open(output_file, "w", **profile) as dest:
                dest.write(dataset.read())


"""
def resample_raster(
    input_path, output_path, new_pixel_size=None, reference_raster_path=None
):
    with rasterio.open(input_path) as src:
        # Get the original pixel size and spatial extent
        orig_pixel_size = (src.res[0], src.res[1])
        orig_bounds = src.bounds

        # If a reference raster is provided, use its pixel size and spatial extent
        if reference_raster_path:
            with rasterio.open(reference_raster_path) as ref_src:
                new_pixel_size = (ref_src.res[0], ref_src.res[1])
                new_bounds = ref_src.bounds

        # If a new pixel size is not provided, return without resampling
        if not new_pixel_size:
            return

        # Calculate the resampling scale factors
        scale_x = orig_pixel_size[0] / new_pixel_size[0]
        scale_y = orig_pixel_size[1] / new_pixel_size[1]

        # Calculate the new dimensions
        new_width = int((orig_bounds[2] - orig_bounds[0]) / new_pixel_size[0])
        new_height = int((orig_bounds[3] - orig_bounds[1]) / new_pixel_size[1])

        # Update the new spatial extent based on the new dimensions
        new_bounds = (
            orig_bounds[0],
            orig_bounds[1],
            orig_bounds[0] + new_width * new_pixel_size[0],
            orig_bounds[1] + new_height * new_pixel_size[1],
        )

        # Resample the raster
        profile = src.profile
        data_type = src.dtypes
        profile.update(
            {
                "driver": "GTiff",
                "width": new_width,
                "height": new_height,
                "transform": rasterio.transform.from_bounds(
                    *new_bounds, new_width, new_height
                ),
            }
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=profile["transform"],
                    dst_crs=profile["crs"],
                    resampling=Resampling.bilinear,
                )
"""


def resample_raster(
    input_path, output_path, new_pixel_size=None, reference_raster_path=None
):
    with rasterio.open(input_path) as src:
        # Get the original pixel size and bounds
        orig_pixel_size = (src.res[0], src.res[1])
        orig_bounds = src.bounds
        print(orig_bounds)

        # If reference raster is provided, use its pixel size and extent
        if reference_raster_path:
            with rasterio.open(reference_raster_path) as ref_src:
                new_pixel_size = (ref_src.res[0], ref_src.res[1])
                new_bounds = ref_src.bounds
                dst_transform = ref_src.transform
                dst_crs = ref_src.crs
                new_width, new_height = ref_src.width, ref_src.height
        else:
            if not new_pixel_size:
                print("No new pixel size or reference raster provided.")
                return

            # Compute new raster dimensions
            new_width = int((orig_bounds[2] - orig_bounds[0]) / new_pixel_size[0])
            new_height = int((orig_bounds[3] - orig_bounds[1]) / new_pixel_size[1])
            dst_transform = rasterio.transform.from_bounds(
                *orig_bounds, new_width, new_height
            )
            dst_crs = src.crs  # Keep the same CRS

        # Update profile
        profile = src.profile
        profile.update(
            {
                "driver": "GTiff",
                "width": new_width,
                "height": new_height,
                "transform": dst_transform,
                "crs": dst_crs,
                "nodata": 0,  # Preserve NoData value
            }
        )

        # Resample the raster
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                data = src.read(i)
                dst_data = np.full((new_height, new_width), 0, dtype=data.dtype)

                reproject(
                    source=data,
                    destination=dst_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

                dst.write(dst_data, i)

    print("Resampling complete.")


def mask_raster(
    raster_path,
    scl_mask_values,
    scl_mask_path,
    aot_mask_path,
    cloud_mask_path,
    output_path,
):
    # Open the raster to be masked
    with rasterio.open(raster_path, "r+") as src:

        raster_data = src.read()
        band_descriptions = src.descriptions  # Get band descriptions

        # Open the scl mask raster
        with rasterio.open(scl_mask_path) as scl_mask_src:
            scl_mask_data = scl_mask_src.read(1)

            # Create a mask based on the values from the scl mask raster
            scl_mask = np.isin(scl_mask_data, scl_mask_values)

            # Open the AOT mask raster
            with rasterio.open(aot_mask_path) as aot_mask_src:
                aot_mask_data = aot_mask_src.read(1)

                # Create a mask based on values above 0.3 in the AOT mask raster
                aot_mask = aot_mask_data < 0.3

                # Open the cloud mask raster
                with rasterio.open(cloud_mask_path) as cloud_mask_src:
                    cloud_mask_data = cloud_mask_src.read(1)

                    # Create a mask based on the values equal to or above 1 in the Cloud mask raster
                    cloud_mask = cloud_mask_data >= 1

                    # Combine the masks
                    mask = np.logical_or(scl_mask, aot_mask, cloud_mask)

                    # Initialize an empty array to store masked bands
                    masked_data = np.copy(raster_data)

                    # Apply the mask to each band of the original raster
                    if src.meta["nodata"] is not None:
                        for band in range(src.count):
                            masked_data[band][mask] = src.meta["nodata"]
                    else:
                        src.nodata = 0
                        for band in range(src.count):
                            masked_data[band][mask] = src.meta["nodata"]

                    # Create metadata for the output raster
                    profile = src.profile
                    profile.update(
                        count=src.count,
                        descriptions=band_descriptions,
                        driver="GTiff",
                        dtype=rasterio.uint16,
                    )

                    # Write the masked raster to the output file
                    with rasterio.open(output_path, "w", **profile) as dst:
                        dst.write(masked_data)


def rasterize_shapefile(
    shapefile_path,
    output_raster_path,
    attribute_bands,
    pixel_size=10,
    reference_raster_path=None,
):
    """
    Rasterizes a shapefile and saves the output as a multi-band GeoTIFF raster,
    inheriting the properties of an existing reference raster if provided.

    Args:
        shapefile_path (str): Path to the input shapefile.
        output_raster_path (str): Path to save the output raster.
        pixel_size (float): Pixel size (resolution) for the output raster.
        attribute_bands (dict): Dictionary mapping attribute names to band indexes.
                               The key-value pair should be in the form: {attribute_name: band_index}.
        reference_raster_path (str, optional): Path to an existing raster to inherit its properties.
                                               Defaults to None.
    """

    if reference_raster_path is not None:
        with rasterio.open(reference_raster_path) as src:
            profile = src.profile.copy()
            profile.update(count=len(attribute_bands), dtype="float32", nodata=-1)
            pixel_size = src.res[0]  # Update pixel size to match the reference raster
            out_shape = (src.height, src.width)

    else:
        profile = {
            "driver": "GTiff",
            "height": 512,
            "width": 512,
            "count": len(attribute_bands),
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_origin(0, 0, pixel_size, pixel_size),
        }
        out_shape = (profile["height"], profile["width"])

    with rasterio.Env():
        with rasterio.open(output_raster_path, "w", **profile) as dst:
            shapes = []
            attributes = {}

            with fiona.open(shapefile_path, "r") as shp:
                for feature in tqdm(
                    shp, desc="Reading Shapefile Properties", leave=False, colour="red"
                ):
                    geom = feature["geometry"]
                    attr = feature["properties"]

                    shapes.append(geom)
                    for attribute, band_index in attribute_bands.items():
                        attributes.setdefault(attribute, []).append(
                            attr.get(attribute, -1)
                        )

            for attribute, band_index in tqdm(
                attribute_bands.items(),
                desc="Write Raster",
                leave=False,
                colour="green",
            ):
                attribute_values = attributes[attribute]
                attribute_values = [x * 0.01 for x in attribute_values]
                # attribute_values = [x for x in attribute_values]
                burned = np.zeros(
                    out_shape, dtype=np.float32
                )  # Create an empty array for rasterization

                # Iterate over each shape and its corresponding value
                geom_value = (
                    (geom, value) for geom, value in zip(shapes, attribute_values)
                )
                features.rasterize(
                    geom_value,
                    out_shape=out_shape,
                    transform=dst.transform,
                    all_touched=True,
                    fill=-1,
                    out=burned,
                    default_value=-1,
                )

                dst.write(burned, indexes=band_index)


def ntems_mask(raster_path, ntems_raster, ntems_values, output_path):
    # Open the raster to be masked
    with rasterio.open(raster_path, "r+") as src:
        raster_data = src.read()
        band_descriptions = src.descriptions  # Get band descriptions

        with rasterio.open(ntems_raster) as ntems_mask_src:
            ntems_mask_data = ntems_mask_src.read(1)

            rows_diff = raster_data.shape[1] - ntems_mask_data.shape[0]
            cols_diff = raster_data.shape[2] - ntems_mask_data.shape[1]

            pad_width = ((0, max(rows_diff, 0)), (0, max(cols_diff, 0)))

            ntems_mask_data_padded = np.pad(
                ntems_mask_data, pad_width, mode="constant", constant_values=0
            )
            ntems_mask_data_cropped = ntems_mask_data_padded[
                : raster_data.shape[1], : raster_data.shape[2]
            ]
            ntems_mask = np.isin(ntems_mask_data_cropped, ntems_values)

            masked_data = np.copy(raster_data)
            if src.meta["nodata"] is not None:
                for band in range(src.count):
                    masked_data[band][ntems_mask] = src.meta["nodata"]
            else:
                src.nodata = 0
                for band in range(src.count):
                    masked_data[band][ntems_mask] = src.meta["nodata"]

            profile = src.profile
            profile.update(
                count=src.count,
                height=ntems_mask_data_cropped.shape[0],
                width=ntems_mask_data_cropped.shape[1],
                descriptions=band_descriptions,
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(masked_data)


"""
This script demonstrates the use of GDAL to perform several common raster operations efficiently. The functions provided are optimized for use in an R environment and include:

    1. **Quick Clip**: Clips a raster to the extent of a shapefile.
    2. **Quick Mosaic**: Mosaics a set of rasters, taking the top value only.
    3. **Quick Translate**: Resamples a raster to a different CRS.
    4. **Quick Merge**: Merges two rasters into one file.
    ## Requirements
    - Python 3.x
    - GDAL library
    - osgeo 
    - glob
"""
### set libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, osr, ogr

gdal.UseExceptions()
from glob import glob
import time
import fiona
import rasterio


def quick_clip(rasters, shapefile, output_file, separate_bands=True):
    """
    Clips a raster to the specified bounding box.
       Parameters:
        - rasters (dir / file): Path to the input rasters
        - output_raster_path (str): Path to save the clipped raster file.
        - shapefile (str): Path to the shapefile to clip the raster with.

       Returns:
        - None
    """
    st = time.time()
    ## if rasters is a directory, get all the tifs in the directory
    if os.path.isdir(rasters):
        rasters = glob(os.path.join(rasters, "*.tif")) + glob(
            os.path.join(rasters, "*.tiff")
        )
    ## if rasters is still empty, return
    if len(rasters) == 0:
        print("No rasters found")
        return
    # if rasters is a list of files, do nothing
    if isinstance(rasters, list):
        pass
    ## else if rasters is a single file, make it a list
    elif os.path.isfile(rasters):
        rasters = [rasters]
    ## set up the output file names
    # if the output file has an extension, remove it
    if output_file.endswith(".*"):
        print("Output file should not have an extension")
        return
    output_file_vrt = output_file + ".vrt"
    output_file_tif = output_file + ".tif"
    ## set up vrt to virtual memory
    vrt_path = "/vsimem/cropped_vrt.vrt"
    ## set up the VRT options to place each in its own band
    vrt_options = gdal.BuildVRTOptions(separate=separate_bands)
    ## build a VRT from the rasters
    vrt = gdal.BuildVRT(vrt_path, rasters, options=vrt_options)
    ## save this to a file
    gdal.Translate(output_file_vrt, vrt)
    ## Open the cutline shapefile
    cutline_ds = gdal.OpenEx(shapefile, gdal.OF_VECTOR)
    ## Get the cutline layer
    cutline_layer = cutline_ds.GetLayer()
    ## Set the warp options with the croptoCutline feature
    warp_options = gdal.WarpOptions(
        format="GTiff", cutlineDSName=shapefile, cropToCutline=True
    )
    ## Warp the VRT file with the cutline
    gdal.Warp(output_file_tif, vrt, options=warp_options)
    print(f"Iter time: {time.time() - st}")
    print(f"Output file: {output_file_tif}")


def quick_mosaic(raster_paths, output_path):
    """
    Mosaics multiple rasters into a single raster.
    Parameters:
    - raster_paths (list of str): List of paths to the input raster files.
    - output_path (str): Path to save the mosaiced raster file.
    "
    Returns:
    - None
    """
    st = time.time()
    output_path_vrt = output_path + ".vrt"
    output_path_tif = output_path + ".tif"

    ## set up the VRT options to place each in is NOT in its own band (i.e. take the first)
    # Set up the VRT options to ignore NaN values
    # make vrt
    destName = output_path_vrt
    # kwargs = {'separate': True} ## I think this should be false?
    ds = gdal.BuildVRT(destName, raster_paths)
    # close and save ds
    ds = None
    # save vrt to tif with gdal translate
    kwargs = {"format": "GTiff"}
    fn = output_path_vrt
    dst_fn = output_path_tif
    ds = gdal.Translate(dst_fn, fn, **kwargs)
    ds = None
    ##
    print(f"Iter time: {time.time() - st}")
    print(f"Output file: {output_path_tif}")
    return None


def quick_translate(raster_path, output_path, target_crs):
    """
    Reprojects a raster to a specified coordinate reference system (CRS).
    "
    Parameters:
    - raster_path (str): Path to the input raster file.
    - output_path (str): Path to save the reprojected raster file.
    - target_crs (str): Target coordinate reference system (e.g., 'EPSG:4326').
    "
    Returns:
    - None
    """
    st = time.time()
    filename = raster_path
    input_raster = gdal.Open(filename)
    output_raster = output_path
    warp = gdal.Warp(output_raster, input_raster, dstSRS=target_crs)
    warp = None  # Closes the files
    print(f"Iter time: {time.time() - st}")
    print("Raster has been reprojected and saved to: ", output_path)


def reproject_align(infile, match, outfile):
    """Reproject a file to match the shape and projection of existing raster.

    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection
    outfile : (string) path to output file tif
    """
    # open input
    with rasterio.open(infile) as src:
        src_transform = src.transform

        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs

            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,  # input CRS
                dst_crs,  # output CRS
                match.width,  # input width
                match.height,  # input height
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
                "nodata": 0,
            }
        )
        print(
            "Coregistered to shape:", dst_height, dst_width, "\n Affine", dst_transform
        )
        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )


def clip_forest(input_raster, boundary, forest_fri, output_raster):
    out_temp = os.path.join(os.path.dirname(output_raster), "temp.tif")
    clip_raster(
        input_raster,
        boundary,
        out_temp,
    )
    print("clipped to the boundary")
    clip_raster(
        out_temp,
        forest_fri,
        output_raster,
        crop=False,
        invert=False,
    )
    print("clipped to the forest")
    os.remove(out_temp)
    print(f"raster clipped to {output_raster}")


import rasterio
import fiona


def rasterio_clip(shapefile_path, raster_path, output_path):
    with fiona.open(shapefile_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    for i, shape in enumerate(shapes):
        with rasterio.open(raster_path) as src:
            out_image, out_transform = rasterio.mask.mask(src, [shape], crop=True)
            out_meta = src.meta

        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        with rasterio.open(f"{output_path}_clipped_{i}.tif", "w", **out_meta) as dest:
            dest.write(out_image)
