import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        weighted_squared_errors = squared_errors * self.weights
        loss = torch.mean(weighted_squared_errors)
        return loss


def calc_loss(y_true, y_pred, weights):
    weighted_mse = WeightedMSELoss(weights)
    loss = weighted_mse(y_pred, y_true)

    return loss


class NormWeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.eps = 1e-6  # Stability epsilon

    def forward(self, y_pred, y_true):
        # 1. Input validation
        if torch.isnan(y_pred).any() or torch.isnan(y_true).any():
            raise RuntimeError("NaN in loss inputs")

        # 2. Error calculation with clamping
        errors = y_pred - y_true
        clipped_errors = torch.clamp(errors, -1e3, 1e3)  # Prevent explosive gradients

        # 3. Safe squared calculation
        squared_errors = clipped_errors.pow(2)
        weighted_errors = squared_errors * self.weights

        # 4. Protected normalization
        valid_elements = (
            torch.numel(weighted_errors) - torch.isnan(weighted_errors).sum()
        )
        loss = weighted_errors.nansum() / (valid_elements + self.eps)

        return loss


def calc_nwmse_loss(valid_outputs, valid_targets, weights):
    weighted_mse = NormWeightedMSELoss(weights)
    loss = weighted_mse(valid_outputs, valid_targets)

    return loss


def apply_mask(outputs, targets, mask, multi_class=True, keep_shp=False):
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

    if keep_shp:
        # Set invalid outputs and targets to 255
        outputs = outputs.clone()
        targets = targets.clone()
        outputs[expanded_mask] = 255
        targets[expanded_mask] = 255
        return outputs, targets
    else:
        # Apply mask to exclude invalid data points
        valid_outputs = outputs[~expanded_mask]
        valid_targets = targets[~expanded_mask]
        # Reshape to (-1, num_classes)
        if multi_class:
            valid_outputs = valid_outputs.view(-1, num_classes)
            valid_targets = valid_targets.view(-1, num_classes)
        return valid_outputs, valid_targets
