import torch
import torch.nn as nn


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


class PinballLoss:
    def __init__(self, quantile=0.10, reduction="none"):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction

    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1 - self.quantile) * (abs(error)[bigger_index])

        if self.reduction == "sum":
            loss = loss.sum()
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


def calc_pinball_loss(y_true, y_pred):
    pinball_loss = PinballLoss(quantile=0.10, reduction="mean")
    loss = pinball_loss(y_pred, y_true)

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
