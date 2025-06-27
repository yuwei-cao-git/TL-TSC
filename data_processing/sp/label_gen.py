import os
import numpy as np
from sklearn.model_selection import train_test_split
import rasterio

def calculate_species_proportions(raster_data):
    """
    Calculate the species proportions for each dataset split (train, val, test).
    
    Parameters:
    - raster_data: 3D numpy array of shape (num_tiles, 9, height, width), where each band represents
                   the percentage of a tree species.
    
    Returns:
    - proportions: A 2D array where each row is a tile, and each column is the proportion of a species in that tile.
    """
    # Calculate the mean proportion of each species across all pixels in each tile (mean along height and width axes)
    proportions = np.mean(raster_data, axis=(2, 3))
    
    return proportions

def check_balance(proportions, train_indices, val_indices, test_indices, target_split, tolerance=0.01):
    """
    Check if the species proportions are within the ±1% tolerance for the target split.
    
    Parameters:
    - proportions: A 2D array of species proportions (num_samples, num_species).
    - target_split: List with target proportions [train_size, val_size, test_size].
    - tolerance: The allowed deviation from the target split (default is 0.01 or ±1%).

    Returns:
    - True if the split is balanced, False otherwise.
    """
    total_samples = proportions.shape[0]
    species_sums = np.sum(proportions, axis=0)
    
    # Calculate the proportions of species in the splits
    train_proportions = np.sum(proportions[train_indices], axis=0) / species_sums
    val_proportions = np.sum(proportions[val_indices], axis=0) / species_sums
    test_proportions = np.sum(proportions[test_indices], axis=0) / species_sums
    
    # Check if all proportions are within the tolerance range
    return (
        np.all(np.abs(train_proportions - target_split[0]) <= tolerance) and
        np.all(np.abs(val_proportions - target_split[1]) <= tolerance) and
        np.all(np.abs(test_proportions - target_split[2]) <= tolerance)
    )

def iterative_split(raster_data, target_split=[0.7, 0.15, 0.15], max_iter=1000, tolerance=0.01):
    """
    Iteratively split the dataset until species proportions are balanced within the given tolerance.
    
    Parameters:
    - raster_data: 3D numpy array of shape (num_tiles, 9, height, width), where each band represents
                   the percentage of a tree species.
    - target_split: List with target split ratios [train_size, val_size, test_size].
    - max_iter: Maximum number of iterations to try for a balanced split.
    - tolerance: The allowed deviation from the target split (default is 0.01 or ±1%).

    Returns:
    - train_indices, val_indices, test_indices: Indices for the train, validation, and test sets.
    """
    proportions = calculate_species_proportions(raster_data)
    
    for i in range(max_iter):
        # Perform random splitting
        train_val_indices, test_indices = train_test_split(
            np.arange(proportions.shape[0]), test_size=target_split[2], random_state=i
        )
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=target_split[1] / (target_split[0] + target_split[1]), random_state=i
        )
        
        # Check if the split is balanced
        if check_balance(proportions, train_indices, val_indices, test_indices, target_split, tolerance):
            return train_indices, val_indices, test_indices

    raise ValueError("Could not find a balanced split within the given tolerance after {} iterations.".format(max_iter))

def get_tile_names_from_folder(folder_path, extension='.tif'):
    """
    Get all tile names from a specified folder with the given extension.
    
    Parameters:
    - folder_path: Path to the folder containing the tiles.
    - extension: File extension to filter (default is .tif).

    Returns:
    - tile_names: List of tile names found in the folder.
    """
    tile_names = [f for f in os.listdir(folder_path) if f.endswith(extension)]
    return sorted(tile_names)

def save_tile_names(tile_names, train_indices, val_indices, test_indices, output_dir):
    """
    Save the tile names for each dataset split (train, val, test) into .txt files.
    
    Parameters:
    - tile_names: List of tile names corresponding to the raster data.
    - train_indices, val_indices, test_indices: Indices for the train, validation, and test sets.
    - output_dir: Directory where the .txt files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_tiles = [tile_names[i] for i in train_indices]
    val_tiles = [tile_names[i] for i in val_indices]
    test_tiles = [tile_names[i] for i in test_indices]
    
    # Save the tile names to .txt files
    with open(os.path.join(output_dir, 'train_tiles.txt'), 'w') as f:
        f.write('\n'.join(train_tiles))
    
    with open(os.path.join(output_dir, 'val_tiles.txt'), 'w') as f:
        f.write('\n'.join(val_tiles))
    
    with open(os.path.join(output_dir, 'test_tiles.txt'), 'w') as f:
        f.write('\n'.join(test_tiles))


def load_raster_data_from_tiles(folder_path, tile_names):
    """
    Load raster data from the tiles in the folder.
    
    Parameters:
    - folder_path: Path to the folder containing the tiles.
    - tile_names: List of tile names to load.
    
    Returns:
    - raster_data: 4D numpy array of shape (num_tiles, 11, height, width).
    """
    raster_data = []
    
    for tile_name in tile_names:
        tile_path = os.path.join(folder_path, tile_name.split(" ")[0])
        
        # Open each tile using rasterio and read its bands
        with rasterio.open(tile_path) as src:
            tile_bands = src.read()  # This should be a 3D array (11 bands, height, width)
            raster_data.append(tile_bands)
    
    return np.stack(raster_data)

