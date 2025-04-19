import torch
import numpy as np
import os
import pandas as pd

import torch

def apply_mask(outputs, targets, mask, multi_class=True, keep_shp=False):
    """
    Applies the mask to outputs and targets to exclude invalid data points.

    Args:
        outputs: Model predictions.
                Images: (batch_size, num_classes, H, W)
                Point Clouds: (batch_size, num_points, num_classes) [Assuming class last]
                or (batch_size, num_classes, num_points) [Needs check]
        targets: Ground truth labels (same shape logic as outputs).
        mask: Boolean mask indicating invalid data points (True for invalid).
            Should correspond to spatial/point dimensions.
            Images: (batch_size, H, W)
            Point Clouds: (batch_size, num_points)
        multi_class (bool): Whether dealing with multiple classes.
        keep_shp (bool): If True, keeps original shape, setting invalid points to 255 (or other ignore value).
                        If False, removes invalid points and reshapes.

    Returns:
        valid_outputs: Masked outputs. Shape depends on keep_shp.
        valid_targets: Masked targets. Shape depends on keep_shp.
    """
    # Ensure mask is boolean
    mask = mask.bool()

    if not multi_class:
        # Simple case: Mask directly if no class dimension needs special handling
        expanded_mask = mask
        if keep_shp:
            outputs = outputs.clone()
            targets = targets.clone()
            outputs[expanded_mask] = 255 # Or another ignore value
            targets[expanded_mask] = 255 # Or appropriate ignore index
            return outputs, targets
        else:
            # Assuming mask applies element-wise or needs broadcasting correctly
            valid_outputs = outputs[~expanded_mask]
            valid_targets = targets[~expanded_mask]
            # May need reshaping depending on downstream use
            return valid_outputs, valid_targets

    # --- Multi-class handling ---
    num_classes = -1
    permute_dims = None # Initialize to None

    # Determine num_classes and expected mask shape based on tensor dimensions
    if outputs.ndim == 4: # Assume Image: (B, C, H, W)
        num_classes = outputs.size(1)
        expected_mask_shape = (outputs.size(0), outputs.size(2), outputs.size(3))
        class_dim = 1
        permute_dims = (0, 2, 3, 1) # Permute to (B, H, W, C) for masking
    elif outputs.ndim == 3: # Assume Point Cloud
        # Option 1: (B, N, C) - Class dimension is last
        if mask.shape == (outputs.size(0), outputs.size(1)):
            num_classes = outputs.size(2)
            expected_mask_shape = (outputs.size(0), outputs.size(1))
            class_dim = 2
        # Option 2: (B, C, N) - Class dimension is middle
        elif mask.shape == (outputs.size(0), outputs.size(2)):
            num_classes = outputs.size(1)
            expected_mask_shape = (outputs.size(0), outputs.size(2))
            class_dim = 1
            permute_dims = (0, 2, 1) # Permute to (B, N, C) for masking
        else:
            raise ValueError(f"Cannot determine point cloud format or mask shape mismatch. Outputs: {outputs.shape}, Mask: {mask.shape}")
    else:
        raise ValueError(f"Unsupported output dimensions: {outputs.ndim}")

    # Validate mask shape
    if mask.shape != expected_mask_shape:
        raise ValueError(f"Mask shape mismatch. Expected {expected_mask_shape}, got {mask.shape}")

    if keep_shp:
        # Expand mask across the class dimension
        expanded_mask = mask.unsqueeze(class_dim).expand_as(outputs)
        outputs = outputs.clone()
        targets = targets.clone()
        # Use an appropriate ignore value
        outputs[expanded_mask] = 255 # Or float('nan') / -1 etc.
        targets[expanded_mask] = 255 # Or appropriate ignore index
        return outputs, targets
    else:
        # --- Corrected logic for keep_shp=False ---
        if permute_dims:
            # Permute to put class dimension last: e.g., (B, H, W, C) or (B, N, C)
            outputs_permuted = outputs.permute(*permute_dims).contiguous()
            targets_permuted = targets.permute(*permute_dims).contiguous()
        else:
            # Already in desired format, e.g., (B, N, C)
            outputs_permuted = outputs
            targets_permuted = targets

        # Apply the *original* mask (e.g., (B, H, W) or (B, N))
        # This selects entire class vectors for valid locations
        # Result shape: (N_valid_locations, num_classes)
        valid_outputs = outputs_permuted[~mask]
        valid_targets = targets_permuted[~mask]

        return valid_outputs, valid_targets

def save_to_file(labels, outputs, classes, config):
    # Convert tensors to numpy arrays or lists as necessary
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    outputs = (
        outputs.cpu().numpy() if isinstance(outputs, torch.Tensor) else outputs
    )
    num_samples = labels.shape[0]
    data = {"SampleID": np.arange(num_samples)}

    # Add true and predicted values for each class
    for i, class_name in enumerate(classes):
        data[f"True_{class_name}"] = labels[:, i]
        data[f"Pred_{class_name}"] = outputs[:, i]

    df = pd.DataFrame(data)

    output_dir = os.path.join(
        config["save_dir"],
        config["log_name"],
        "outputs",
    )
    os.makedirs(output_dir, exist_ok=True)
    # Save DataFrame to a CSV file
    df.to_csv(
        os.path.join(output_dir, "test_outputs.csv"),
        mode="a",
    )