
import torch
from torch import Tensor


def iou(y_pred: Tensor, y_target: Tensor) -> float:
    intersection = torch.sum(y_pred * y_target)
    union = torch.sum(torch.logical_or(y_pred, y_target))
    return intersection/union


def dice(y_pred: Tensor, y_target: Tensor, smooth=1) -> float:
    # smooth to prevent division by zero when no segmented pixels in image
    intersection = torch.sum(y_pred * y_target)
    return (2*intersection + smooth)/(torch.sum(y_pred) + torch.sum(y_target) + smooth)


def dice_loss(y_pred: Tensor, y_target: Tensor) -> float:
    return 1 - dice(y_pred, y_target)


def mclass_dice_loss(y_pred: Tensor, y_target: Tensor, epsilon=1e-6) -> float:
    """
    avoid skewed classes

    ref: https://arxiv.org/abs/1606.04797
    """
    axes = tuple(range(len(y_pred.shape)-3, len(y_pred.shape)))  # ignore batch/class axes
    numerator = 2.*torch.sum(y_pred*y_target, axes)
    denominator = torch.sum(torch.square(y_pred) + torch.square(y_target), axes)
    return 1 - torch.mean(numerator / (denominator + epsilon))


