#!/bin/bash

# Optional: path to Python executable
PYTHON_EXEC="python"

# Dataset argument
DATASET="wrf"  # Change this as needed (e.g., wrf, rmf, ovf)

# Loop over all Python scripts and execute them with --dataset
echo "Running generate_superpixels scripts with --dataset $DATASET ..."

echo "Generating superpixels"
$PYTHON_EXEC generate_superpixels_1.py --dataset "$DATASET"

echo "Generating plots in superpixels"
$PYTHON_EXEC generate_superpixel_plots_2.py --dataset "$DATASET"

echo "Sampling point cloud in pseudo plots"
$PYTHON_EXEC superpixel_pts_3.py --dataset "$DATASET"

echo "Combining image superpixels with point clouds"
$PYTHON_EXEC generate_superpixel_dataset_4.py --dataset "$DATASET"

# echo "Re-spliting the dataset (optional)"
# $PYTHON_EXEC superpixel_resplit_5.py --dataset "$DATASET"

echo "All scripts have been executed successfully."