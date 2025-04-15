import torch
import numpy as np
import os
import pandas as pd

def apply_mask(outputs, targets, mask, multi_class=True):
    """
    Applies the mask to outputs and targets to exclude invalid data points.

    Args:
        outputs: Model predictions (batch_size, num_classes, H, W) for images or (batch_size, num_points, num_classes) for point clouds.
        targets: Ground truth labels (same shape as outputs).
        mask: Boolean mask indicating invalid data points (True for invalid).

    Returns:
        valid_outputs: Masked and reshaped outputs.
        valid_targets: Masked and reshaped targets.
    """
    # Expand the mask to match outputs and targets
    if multi_class:
        expanded_mask = mask.unsqueeze(1).expand_as(
            outputs
        )  # Shape: (batch_size, num_classes, H, W)
        num_classes = outputs.size(1)
    else:
        expanded_mask = mask

    # Apply mask to exclude invalid data points
    valid_outputs = outputs[~expanded_mask]
    valid_targets = targets[~expanded_mask]
    # Reshape to (-1, num_classes)
    if multi_class:
        valid_outputs = valid_outputs.view(-1, num_classes)
        valid_targets = valid_targets.view(-1, num_classes)

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