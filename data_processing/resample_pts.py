# import glob
import os
import math
# from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm
from pts_utils import read_las, write_las


def farthest_point_sampling(coords, k):
    # Adapted from https://minibatchai.com/sampling/2021/08/07/FPS.html

    # Get points into numpy array
    points = np.array(coords)

    # Get points index values
    idx = np.arange(len(coords))

    # Initialize use_idx
    use_idx = np.zeros(k, dtype="int")

    # Initialize dists
    dists = np.ones_like(idx) * float("inf")

    # Select a point from its index
    selected = 0
    use_idx[0] = idx[selected]

    # Delete Selected
    idx = np.delete(idx, selected)

    # Iteratively select points for a maximum of k samples
    for i in range(1, k):
        # Find distance to last added point and all others
        last_added = use_idx[i - 1]  # get last added point
        dist_to_last_added_point = ((points[last_added] - points[idx]) ** 2).sum(-1)

        # Update dists
        dists[idx] = np.minimum(dist_to_last_added_point, dists[idx])

        # Select point with largest distance
        selected = np.argmax(dists[idx])
        use_idx[i] = idx[selected]

        # Update idx
        idx = np.delete(idx, selected)
    return use_idx


def get_distribution(z, z_bins):
    z_tot = len(z)
    z_bin_dists = []
    for i in range(0, len(z_bins)):
        if i == 0:  # if first bin
            z_idx = np.where(
                np.logical_and(z >= min(z_bins[i]), z <= min(z_bins[i + 1]))
            )  # get idx
        elif i == len(z_bins) - 1:  # if last bin
            z_idx = np.where(
                np.logical_and(z > min(z_bins[i]), z <= max(z_bins[i]))
            )  # get idx
        else:  # if between first and last bins
            z_idx = np.where(
                np.logical_and(z > min(z_bins[i]), z <= min(z_bins[i + 1]))
            )  # get idx

        z_bin_dist = len(z_idx[0]) / z_tot
        z_bin_dists.append(z_bin_dist)
    return z_bin_dists


def height_fps(coords, k, bins):
    z = coords[:, 2]  # get z values
    z_bins = np.array_split(
        np.array(range(math.floor(min(z)), math.ceil(max(z) + 1))), bins
    )  # bin height values
    z_dist = get_distribution(z, z_bins)  # get distribution
    n_z = [round(k * i) for i in z_dist]  # get number of points based on distribution

    idx_list = []
    for i in range(len(z_bins)):
        if i == 0:  # if first bin
            z_idx = np.where(
                np.logical_and(z >= min(z_bins[i]), z <= min(z_bins[i + 1]))
            )  # get idx
        elif i == len(z_bins) - 1:  # if last bin
            z_idx = np.where(
                np.logical_and(z > min(z_bins[i]), z <= max(z_bins[i]))
            )  # get idx
        else:  # if between first and last bins
            z_idx = np.where(
                np.logical_and(z > min(z_bins[i]), z <= min(z_bins[i + 1]))
            )  # get idx

        coords_idx = coords[z_idx[0], :]  # subset points
        if n_z[i] != 0: # check that n_z is not 0
            fps_idx = farthest_point_sampling(coords_idx, n_z[i])  # farthest point sampling
        else: # if n_z is zero skip fps step
            pass

        idx_list.append(fps_idx)

    use_idx = np.concatenate(idx_list, axis=0)

    return use_idx
    
    
def resample_point_clouds(root_dir, train_dataset_path, max_points_list, samp_meth, glob="*.laz", bins=None):
    # Create training set for each point density
    files = list(Path(root_dir).glob(glob))

    for max_points in tqdm(max_points_list, desc="Total: ", leave=False, colour="blue"):
        # Make folders
        if not os.path.exists(os.path.join(train_dataset_path, "trainingsets")):
            os.makedirs(os.path.join(train_dataset_path, "trainingsets"))
        if not os.path.exists(
            os.path.join(train_dataset_path, "trainingsets", samp_meth)
        ):
            os.makedirs(os.path.join(train_dataset_path, "trainingsets", samp_meth))
        if not os.path.exists(
            os.path.join(train_dataset_path, "trainingsets", samp_meth, str(max_points))
        ):
            os.makedirs(
                os.path.join(
                    train_dataset_path, "trainingsets", samp_meth, str(max_points)
                )
            )

        for file in tqdm(
            files, desc="Max Points: " + str(max_points), leave=False, colour="red"
        ):
            # Read las/laz file
            coords, attrs = read_las(file, get_attributes=True)
            filename = str(file).split("\\")[-1]

            # Resample number of points to max_points
            if coords.shape[0] >= max_points:
                if samp_meth == "random":
                    use_idx = np.random.choice(
                        coords.shape[0], max_points, replace=False
                    )
                if samp_meth == "fps":
                    use_idx = farthest_point_sampling(coords, max_points)
                if samp_meth == "height_fps":
                    use_idx = height_fps(coords, max_points, bins)
            else:
                use_idx = np.random.choice(coords.shape[0], max_points, replace=True)

            # Get subsetted point cloud
            coords = coords[use_idx, :]
            for key, vals in attrs.items():
                attrs[key] = vals[use_idx]

            # Write out files
            write_las(
                coords,
                os.path.join(
                    train_dataset_path,
                    "trainingsets",
                    samp_meth,
                    str(max_points),
                    filename,
                ),
                attrs,
            )
            
            
if __name__ == "__main__":
    root_dir = r"G:\ovf\dgcnn_spl\laz_plots\train\full"
    train_dataset_path = r"G:\ovf\dgcnn_spl\laz_plots\train"
    # max_points_list = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240]
    # max_points_list = [4096, 5120, 6144, 7168]
    max_points_list = [7168]
    samp_meth = "fps"
    resample_point_clouds(
        root_dir=root_dir,
        train_dataset_path=train_dataset_path,
        max_points_list=max_points_list,
        samp_meth=samp_meth,
    )
    '''
    samp_meth = "height_fps"
    resample_point_clouds(
        root_dir=train_dataset_path,
        max_points_list=max_points_list,
        samp_meth=samp_meth,
        bins=5
    )
    '''