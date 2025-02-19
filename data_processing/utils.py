"""
This module include helper functions for Coordinate System data manipulation 
and directory/file management
"""
import os
import math
from pyproj import Proj, transform, CRS
from rasterio.merge import merge
import zipfile
import shutil
from osgeo import gdal
import json
import os

def deg2num(lon_deg, lat_deg, zoom):
  """
  Transform a point in (lon, lat) to its corresponding tile given a specific zoom level.
  Note that it will the tile coordinate that contains this (lon, lat) point.
  """
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
  """
  Transform a tile at a specific zoom into a (lon, lat) value.
  """
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return lon_deg, lat_deg

def geodesic2spherical(x1, y1, inverse=False):
    """
    EPSG:4326 to EPSG:3857:
        x1: longitude
        y1: latitude
    EPSG:3857 to EPSG:4326:
        x1: x coordinate
        y1: y coordinate
    """
    if inverse:
        inProj = Proj(init='epsg:3857')
        outProj = Proj(init='epsg:4326')
    else:
        inProj = Proj(init='epsg:4326')
        outProj = Proj(init='epsg:3857')
    x2,y2 = transform(inProj, outProj, x1,y1)
    return x2, y2

def convert_crs(pts_dir, crs, out_dir):
    # supported / desirable extensions
    extensions = ['.laz'] # edit it to support more pc types
    # listing all files from the provided dir
    files = os.listdir(pts_dir)
    crs_pyproj = CRS.from_user_input(crs)
    for file in files:
        if not any(x in file for x in extensions): continue # if files with wrong extensions (rather silly check though it would work in simple cases)
        p = os.path.abspath(file) # get files abs path
        op = r"%s/%s" % (out_dir, file)
        jsonString = '{"pipeline": ["%s",{"type": "filters.reprojection","out_srs": "%s"},{"type":"writers.las","filename":"%s"}]}' % (p, crs_pyproj, op)  # json pipeline command
        # print(jsonString) 
        pipeline = gdal.Pipeline(jsonString) # create pdal pipeline
        pipeline.validate() # check if our JSON and options were good
        pipeline.execute() # run pdal pipeline execution    
        print("Processed: ", p) # print the result
        
def create_dir(folder):
  """
  Create dir if it does not exist.
  """
  if not os.path.exists(folder):
      os.makedirs(folder)

def unzip_directory(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(in_dir)
    for item in os.listdir(in_dir):
        if item.endswith(".zip"):
            file_name = os.path.abspath(item)
            zip_ref = zipfile.ZipFile(file_name)
            zip_ref.extractall(out_dir)
        zip_ref.close()

def unzip_folder(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        
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
        

def delete_directory(directory):
    try:
        shutil.rmtree(directory)
    except OSError as e:
        print_color(f"Error: {e.filename, e.strerror}", "red")

def move_files(file, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    filename = str(file).split("\\")[-1]
    outfile = os.path.join(out_folder, filename)
    shutil.move(file, outfile)   
        
def find_subfolders(directory):
    # Initialize an empty list to store the subfolder paths
    subfolders = []

    # Traverse the directory tree rooted at the given directory
    for root, dirs, files in os.walk(directory):
        # Check if there are no subdirectories in the current iteration
        if not dirs:
            # Append the current root directory (subfolder) to the subfolders list
            subfolders.append(root)

    # Return the list of subfolders
    return subfolders

def zip_and_remove_folder(input_folder):
    # Get the base folder name
    base_folder = os.path.basename(input_folder)

    # Create a zip file with the same name as the folder
    zip_file = input_folder + ".zip"

    try:
        # Create a ZipFile object
        with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
            # Iterate over all the files and folders in the input folder
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    # Get the absolute path of the file
                    file_path = os.path.join(root, file)
                    # Add the file to the zip file
                    zf.write(file_path, os.path.relpath(file_path, input_folder))

        # Remove the input folder
        shutil.rmtree(input_folder)

    except Exception as e:
        print_color(f"An error occurred: {e}", "red")
        
def load_tile_splits(folder_path):
    """Load train/val/test tile splits from a folder containing the text files."""
    train_file = os.path.join(folder_path, 'train_tiles.txt')
    val_file = os.path.join(folder_path, 'val_tiles.txt')
    test_file = os.path.join(folder_path, 'test_tiles.txt')

    with open(train_file, 'r') as f:
        train_tiles = f.read().splitlines()
    with open(val_file, 'r') as f:
        val_tiles = f.read().splitlines()
    with open(test_file, 'r') as f:
        test_tiles = f.read().splitlines()

    return train_tiles, val_tiles, test_tiles