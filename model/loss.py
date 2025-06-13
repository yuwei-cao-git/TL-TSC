import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        loss = torch.mean(squared_errors)
        return loss

# MSE loss
def calc_mse_loss(valid_outputs, valid_targets):
    mse = MSELoss()
    loss = mse(valid_outputs, valid_targets)

    return loss
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        weighted_squared_errors = squared_errors * self.weights
        loss = torch.mean(weighted_squared_errors)
        return loss

def calc_wmse_loss(valid_outputs, valid_targets, weights):
    weighted_mse = WeightedMSELoss(weights)
    loss = weighted_mse(valid_outputs, valid_targets)

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

class AutomaticWeightedLoss(nn.Module):
    """
    Automatically weighted multi-task loss.

    Params:
        num: int
            The number of loss functions to combine.
        x: tuple
            A tuple containing multiple task losses.

    Examples:
        loss1 = 1
        loss2 = 2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        # Initialize parameters for weighting each loss, with gradients enabled
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *losses):
        """
        Forward pass to compute the combined loss.

        Args:
            *losses: Variable length argument list of individual loss values.

        Returns:
            torch.Tensor: The combined weighted loss.
        """
        loss_sum = 0
        for i, loss in enumerate(losses):
            # Compute the weighted loss component for each task
            weighted_loss = 0.5 / (self.params[i] ** 2) * loss
            # Add a regularization term to encourage the learning of useful weights
            regularization = torch.log(1 + self.params[i] ** 2)
            # Sum the weighted loss and the regularization term
            loss_sum += weighted_loss + regularization

        return loss_sum


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

def weighted_kl_divergence(y_true, y_pred, weights):
    loss = torch.sum(
        weights * y_true * torch.log((y_true + 1e-8) / (y_pred + 1e-8)), dim=1
    )
    return torch.mean(loss)

def calc_mae_loss(valid_outputs, valid_targets):
    loss = torch.sum(torch.abs(valid_outputs - valid_targets), dim=1)

    return torch.mean(loss)

# Rooted Weighted loss
def calc_rwmse_loss(valid_outputs, valid_targets, weights):
    weighted_mse = WeightedMSELoss(weights)
    loss = weighted_mse(valid_outputs, valid_targets)

    return torch.sqrt(loss)

def calc_masked_loss(loss_func_name, valid_outputs, valid_targets, weights):
    if loss_func_name == "wmse":
        return calc_wmse_loss(valid_outputs, valid_targets, weights)
    elif loss_func_name == "wrmse":
        return calc_rwmse_loss(valid_outputs, valid_targets, weights)
    elif loss_func_name == "mse":
        return calc_mse_loss(valid_outputs, valid_targets)
    elif loss_func_name == "wkl":
        return weighted_kl_divergence(valid_targets, valid_outputs, weights)
    elif loss_func_name == "mae":
        return calc_mae_loss(valid_outputs, valid_targets)
    elif loss_func_name == "pinball":
        return calc_pinball_loss(valid_outputs, valid_targets)