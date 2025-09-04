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
    input, target, beta: float, reduction: str = "none", size_average=False
):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:

                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,

    where x = input - target.

    Smooth L1 loss is related to Huber loss, which is defined as:

                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise

    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:

     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.

    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.

    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.

    Returns:
        The loss with the reduction option applied.

    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
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
        

def weighted_kl_divergence(y_true, y_pred, weights):
    loss = torch.sum(
        weights * y_true * torch.log((y_true + 1e-8) / (y_pred + 1e-8)), dim=1
    )
    return torch.mean(loss)

class KLDivLoss(nn.Module):

    def __init__(self,
                temperature: float = 1.0,
                reduction: str = 'mean',
                loss_name: str = 'loss_kld'):
        """Kullback-Leibler divergence Loss.

        <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>

        Args:
            temperature (float, optional): Temperature param
            reduction  (str,  optional): The method to reduce the loss into a
            scalar. Default is "mean". Options are "none", "sum",
            and "mean"
        """

        assert isinstance(temperature, (float, int)), \
            'Expected temperature to be' \
            f'float or int, but got {temperature.__class__.__name__} instead'
        assert temperature != 0., 'Temperature must not be zero'

        assert reduction in ['mean', 'none', 'sum'], \
            'Reduction must be one of the options ("mean", ' \
            f'"sum", "none"), but got {reduction}'

        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self._loss_name = loss_name

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Forward function. Calculate KL divergence Loss.

        Args:
            input (Tensor): Logit tensor,
                the data type is float32 or float64.
                The shape is (N, C) where N is batchsize and C  is number of
                channels.
                If there more than 2 dimensions, shape is (N, C, D1, D2, ...
                Dk), k>= 1
            target (Tensor): Logit tensor,
                the data type is float32 or float64.
                input and target must be with the same shape.

        Returns:
            (Tensor): Reduced loss.
        """
        assert isinstance(input, torch.Tensor), 'Expected input to' \
            f'be Tensor, but got {input.__class__.__name__} instead'
        assert isinstance(target, torch.Tensor), 'Expected target to' \
            f'be Tensor, but got {target.__class__.__name__} instead'

        assert input.shape == target.shape, 'Input and target ' \
            'must have same shape,' \
            f'but got shapes {input.shape} and {target.shape}'

        input = F.softmax(input / self.temperature, dim=1)
        target = F.softmax(target / self.temperature, dim=1)

        loss = F.kl_div(input, target, reduction='none', log_target=False)
        loss = loss * self.temperature**2

        batch_size = input.shape[0]

        if self.reduction == 'sum':
            # Change view to calculate instance-wise sum
            loss = loss.view(batch_size, -1)
            return torch.sum(loss, dim=1)

        elif self.reduction == 'mean':
            # Change view to calculate instance-wise mean
            loss = loss.view(batch_size, -1)
            return torch.mean(loss, dim=1)

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

def get_class_grw_weight(class_weight, num_classes=9, exp_scale=0.3):
    """
    Caculate the Generalized Re-weight for Loss Computation
    """
    
    ratio = 1 / class_weight
        
    class_weight = 1 / (ratio**exp_scale)
    class_weight = class_weight / torch.sum(class_weight) * num_classes
    
    return class_weight

def calc_masked_loss(loss_func_name, valid_outputs, valid_targets, weights=None):
    if loss_func_name == "wmse":
        return calc_wmse_loss(valid_outputs, valid_targets, weights)
    elif loss_func_name == "wrmse":
        return calc_rwmse_loss(valid_outputs, valid_targets, weights)
    elif loss_func_name == "mse":
        return calc_mse_loss(valid_outputs, valid_targets)
    elif loss_func_name == "kl":
        return calc_kl_loss(valid_targets, valid_outputs, weights)
    elif loss_func_name == "wkl":
        return weighted_kl_divergence(valid_targets, valid_outputs, weights)
    elif loss_func_name == "mae":
        return calc_mae_loss(valid_outputs, valid_targets)
    elif loss_func_name == "pinball":
        return calc_pinball_loss(valid_outputs, valid_targets)
    elif loss_func_name == "L1Smooth":
        return smooth_l1_loss(valid_outputs, valid_targets)
    
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