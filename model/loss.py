import torch
import torch.nn as nn
import torch.nn.functional as F

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


def calc_wmse_loss(y_true, y_pred, weights):
    weighted_mse = WeightedMSELoss(weights)
    loss = weighted_mse(y_pred, y_true)

    return loss


# Rooted Weighted loss
def calc_rwmse_loss(valid_outputs, valid_targets, weights):
    weighted_mse = WeightedMSELoss(weights)
    loss = weighted_mse(valid_outputs, valid_targets)

    return torch.sqrt(loss)


# kl loss
def weighted_kl_divergence(y_true, y_pred, weights):
    loss = torch.sum(
        weights * y_true * torch.log((y_true + 1e-8) / (y_pred + 1e-8)), dim=1
    )
    return torch.mean(loss)

# loss for leading species classification
def cal_leading_loss(y_true, y_pred, alpha_leading):
    correct = (y_pred.view(-1) == y_true.view(-1)).float()
    loss_pixel_leads = 1 - correct.mean()  # 1 - accuracy as pseudo-loss
    return loss_pixel_leads * alpha_leading

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
    
# MAE loss
def calc_mae_loss(valid_outputs, valid_targets):
    loss = torch.sum(torch.abs(valid_outputs - valid_targets), dim=1)

    return torch.mean(loss)

# margin loss
# mainfold loss


def calc_pinball_loss(y_true, y_pred):
    pinball_loss = PinballLoss(quantile=0.10, reduction="mean")
    loss = pinball_loss(y_pred, y_true)

    return loss


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