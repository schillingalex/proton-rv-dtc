from typing import Optional
from typing import List
import torch
from torch import nn
from torch.nn.parameter import Parameter

from util.config import SingleTaskConfig, MultitaskConfig, MLConfig


def loss_from_string(loss_str) -> nn.Module:
    """
    Creates a loss object according to the given string.

    Currently supported: mse, mae, gauss_nll.

    Raises NotImplementedError for unsupported loss string.

    :param loss_str: The string to identify the desired loss.
    :return: The created loss object.
    """
    if loss_str == "mse":
        return nn.MSELoss()
    elif loss_str == "mae":
        return nn.L1Loss()
    elif loss_str == "gauss_nll":
        return nn.GaussianNLLLoss()

    raise NotImplementedError(f"Invalid value for loss: {loss_str}")


def loss_from_task_config(task_config: MLConfig) -> nn.Module:
    """
    Creates a loss object according to the given task config. Depending on the type of config, SingleTaskConfig or
    MultitaskConfig, a single loss or a compound loss is created.

    :param task_config: The SingleTaskConfig or MultitaskConfig to create a loss object from.
    :return: The loss object for the task config.
    """
    if isinstance(task_config, SingleTaskConfig):
        return loss_from_string(task_config.loss)
    elif isinstance(task_config, MultitaskConfig):
        losses = [loss_from_string(loss) for loss in task_config.task_losses]
        if task_config.loss == "homoscedastic":
            return HomoscedasticLoss(losses)
        elif task_config.loss == "weighted_sum":
            return MultitaskLoss(losses, task_config.task_weights)

    raise NotImplementedError(f"Invalid value for loss in task config: {task_config.loss}")


class MultitaskLoss(nn.Module):
    """
    Combines multiple losses through a weighted average with fixed weights.
    """

    def __init__(self, losses: List[nn.Module], weights: Optional[List[float]] = None):
        super(MultitaskLoss, self).__init__()
        # If no weights are given, give them equal weights
        if weights is None:
            weights = [1 / len(losses)] * len(losses)
        self._loss_weights = list(zip(losses, weights))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, var: torch.Tensor):
        l = torch.tensor(0., device=y_pred[0].device)
        for i, (loss, weight) in enumerate(self._loss_weights):
            l += weight * loss(y_pred[:, i], y_true[:, i], var[:, i])
        return l


class HomoscedasticLoss(nn.Module):
    """
    Combines multiple losses by learning the optimal weights based on task-dependent uncertainty.
    """

    def __init__(self, losses: List[nn.Module], device=None):
        super(HomoscedasticLoss, self).__init__()
        self.log_variance = Parameter(torch.zeros((len(losses),), requires_grad=True, device=device))
        self._losses = losses

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, var: torch.Tensor):
        l = 0
        for i, loss in enumerate(self._losses):
            precision = 1/len(self._losses) * torch.exp(-self.log_variance[i])
            l += precision * loss(y_pred[:, i], y_true[:, i], var[:, i]) + self.log_variance[i]
        return l

    @property
    def sigma(self):
        return torch.exp(self.log_variance)**0.5
