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

# MAE loss
def calc_mae_loss(valid_outputs, valid_targets):
    loss = torch.sum(torch.abs(valid_outputs - valid_targets), dim=1)

    return torch.mean(loss)

# margin loss
# mainfold loss

def smooth_l1_loss(
    input, target, beta: float = 1.0, reduction: str = "mean", size_average=False
):
    """
    Smooth L1 loss defined in the Fast R-CNN paper

    Returns:
        The loss with the reduction option applied.

    """
    if beta < 1e-5:
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)

    if reduction == "mean" or size_average:
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

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
            # regularization = torch.log(1 + self.params[i] ** 2)
            # Sum the weighted loss and the regularization term
            loss_sum += weighted_loss # + regularization

        return loss_sum


def apply_mask(outputs, targets, mask, multi_class=True, keep_shp=False):
    """
    Applies the mask to outputs and targets to exclude invalid data points.

    Args:
        outputs: Model predictions (batch_size, num_classes, H, W).
        targets: Ground truth labels (same shape as outputs).
        mask: Boolean mask indicating invalid data points (True for invalid).

    Returns:
        valid_outputs: Masked and reshaped outputs.
        valid_targets: Masked and reshaped targets.
    """
    # Expand the mask to match outputs and targets
    class_dim = 1
    if keep_shp:
        # Set invalid outputs and targets to 255
        if not multi_class:
            expanded_mask = mask
        else:
            expanded_mask = mask.unsqueeze(class_dim).expand_as(outputs)
        outputs = outputs.clone()
        targets = targets.clone()
        outputs[expanded_mask] = 255
        targets[expanded_mask] = 255
        return outputs, targets
    else:
        if multi_class:
            permute_dims = None # Initialize to None
            expected_mask_shape = (outputs.size(0), outputs.size(2), outputs.size(3))
            # Permute to (B, H, W, C) for masking
            permute_dims = (0, 2, 3, 1)
            # Validate mask shape
            if mask.shape != expected_mask_shape:
                raise ValueError(f"Mask shape mismatch. Expected {expected_mask_shape}, got {mask.shape}")
            
            # Permute to put class dimension last: (B, H, W, C)
            outputs_permuted = outputs.permute(*permute_dims).contiguous()
            targets_permuted = targets.permute(*permute_dims).contiguous()
            
            # Apply mask to exclude invalid data points
            valid_outputs = outputs_permuted[~mask]
            valid_targets = targets_permuted[~mask]

            return valid_outputs, valid_targets
        else:
            expanded_mask = mask
            # Assuming mask applies element-wise or needs broadcasting correctly
            valid_outputs = outputs[~expanded_mask]
            valid_targets = targets[~expanded_mask]
            return valid_outputs, valid_targets

def apply_mask_per_batch(preds, mask, multi_class=True):
    """
    Apply a mask to predictions and labels. Only valid pixels (mask == 1) are kept.
    Returns outputs grouped by batch for later aggregation.
    """
    if multi_class:
        B, C, H, W = preds.shape
        preds = preds.permute(0, 2, 3, 1)  # (B, H, W, C)
        mask = mask.squeeze(1)  # (B, H, W)

        masked_preds = []
        for b in range(B):
            valid = mask[b] > 0
            masked_preds.append(preds[b][valid])  # (N_valid, C)
        return masked_preds
    else:
        preds = preds[mask > 0]
        return preds


def weighted_kl_divergence(y_true, y_pred, weights=None):
    y_true = y_true.clamp(min=1e-8)
    y_pred = y_pred.clamp(min=1e-8)
    log_ratio = torch.log(y_true / y_pred)
    if weights is not None:
        log_ratio = weights * log_ratio
    loss = torch.sum(y_true * log_ratio, dim=1)
    return loss.mean()


class KLDivLoss(nn.Module):
    def __init__(self,
                temperature: float = 1.0,
                reduction: str = 'batchmean',
                loss_name: str = 'loss_kld'):
        super().__init__()
        assert reduction in ['none', 'batchmean', 'sum', 'mean']
        self.temperature = temperature
        self.reduction = reduction
        self._loss_name = loss_name

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: raw logits from the model (before softmax), shape (N, C)
            target: ground truth logits or already softmaxed probs, shape (N, C)
        """
        assert input.shape == target.shape

        # Apply temperature-scaled softmax
        log_probs = F.log_softmax(input / self.temperature, dim=1)
        probs_target = F.softmax(target / self.temperature, dim=1)

        # Apply KL divergence: log_probs vs. probs_target
        loss = F.kl_div(log_probs, probs_target, reduction=self.reduction, log_target=False)

        # Apply temperature scaling to loss
        loss = loss * (self.temperature ** 2)
        return loss


def calc_kl_loss(valid_outputs, valid_targets):
    klloss = KLDivLoss()
    loss = klloss(valid_outputs, valid_targets)
    return loss

def calc_mae_loss(valid_outputs, valid_targets):
    loss = torch.sum(torch.abs(valid_outputs - valid_targets), dim=1)

    return torch.mean(loss)

# Rooted Weighted loss
def calc_rwmse_loss(valid_outputs, valid_targets, weights):
    weighted_mse = WeightedMSELoss(weights)
    loss = weighted_mse(valid_outputs, valid_targets)

    return torch.sqrt(loss)

def get_class_grw_weight(class_weight, num_classes=9, exp_scale=0.2):
    """
    Caculate the Generalized Re-weight for Loss Computation
    """
    
    ratio = 1 / class_weight
        
    class_weight = 1 / (ratio**exp_scale)
    class_weight = class_weight / torch.sum(class_weight) * num_classes
    
    return class_weight

def calc_masked_loss(loss_func_name, valid_outputs, valid_targets, weights=None):
    # 1) Empty tensor check
    if valid_outputs.numel() == 0 or valid_targets.numel() == 0:
        print("\n[calc_masked_loss] Empty valid_outputs/valid_targets!")
        print("  shapes:", valid_outputs.shape, valid_targets.shape)
        # return a zero loss that still has grad
        return torch.zeros([], device=valid_outputs.device, requires_grad=True)

    # 2) Finite check
    if not torch.isfinite(valid_outputs).all():
        print("\n[calc_masked_loss] Non-finite values in valid_outputs")
        print("  stats:", valid_outputs.min().item(), valid_outputs.max().item())
        raise RuntimeError("NaN/Inf in valid_outputs")

    if not torch.isfinite(valid_targets).all():
        print("\n[calc_masked_loss] Non-finite values in valid_targets")
        print("  stats:", valid_targets.min().item(), valid_targets.max().item())
        raise RuntimeError("NaN/Inf in valid_targets")
    
    if loss_func_name in ["wmse", "ewmse"]:
        return calc_wmse_loss(valid_outputs, valid_targets, weights)
    elif loss_func_name == "wrmse":
        return calc_rwmse_loss(valid_outputs, valid_targets, weights)
    elif loss_func_name == "mse":
        return calc_mse_loss(valid_outputs, valid_targets)
    elif loss_func_name == "kl":
        return calc_kl_loss(valid_targets, valid_outputs)
    elif loss_func_name == "wkl":
        return weighted_kl_divergence(valid_targets, valid_outputs, weights)
    elif loss_func_name == "mae":
        return calc_mae_loss(valid_outputs, valid_targets)
    elif loss_func_name == "pinball":
        return calc_pinball_loss(valid_outputs, valid_targets)
    elif loss_func_name == "L1Smooth":
        return smooth_l1_loss(valid_outputs, valid_targets)
    elif loss_func_name == "bwkl":
        return plot_props_bias_aux_loss(valid_outputs, valid_targets, class_weights=weights, beta=2.0, eps=1e-8)

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: raw model output, shape (B, C)
        targets: 0/1 multi-label tensor, shape (B, C)
        """
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma

        loss = self.alpha * focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class SmoothClsLoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothClsLoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        # log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * pred).sum(dim=1).mean()
        return loss


class ClsLoss(nn.Module):
    def __init__(self):
        super(ClsLoss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss

def plot_props_bias_aux_loss(
    plot_props, target_props, class_weights=None, beta=2.0, eps=1e-8
):
    """
    Weighted KL(target_props || plot_props) with bias-aware weights and optional class weights.
    Upweights:
    - high-proportion classes that are underpredicted
    - low-proportion classes that are overpredicted
    """
    targ = target_props.clamp_min(eps)
    pred = plot_props.clamp_min(eps)
    kl_per_class = targ * (targ.log() - pred.log())  # [B,K]

    # --- Bias-aware weighting term
    s = (targ - pred) * (2 * targ - 1.0)
    w_bias = torch.exp(beta * s)
    w_bias = w_bias / w_bias.mean(dim=1, keepdim=True).clamp_min(1e-12)

    # --- Class weights
    if class_weights is not None:
        w_class = torch.as_tensor(
            class_weights, dtype=torch.float32, device=plot_props.device
        )
        w_class = w_class / w_class.mean().clamp_min(1e-12)
        w_class = w_class.unsqueeze(0).expand_as(kl_per_class)  # [B,K]
    else:
        w_class = torch.ones_like(kl_per_class)

    # --- Combine weights and compute mean loss
    weighted_kl = w_bias * w_class * kl_per_class
    return weighted_kl.sum(dim=1).mean()  # scalar
