import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


class GradientAdamLRScheduler(ABC):
    """Noise to add to the logits before taking the softmax."""

    @abstractmethod
    def step(self, gradient_norm: float) -> None:
        pass


class GradientBasedAdamLRScheduler(GradientAdamLRScheduler):
    """Doesn't seem to work great. Just becomes instable (loss looks like sine wave)."""

    optimizer: torch.optim.Optimizer
    base_lr: float
    ceiling_lr: float | None

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, ceiling_lr: float | None = None):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.ceiling_lr = ceiling_lr

    def step(self, gradient_norm: float) -> None:
        new_lr = self.base_lr / gradient_norm
        if self.ceiling_lr is not None:
            new_lr = min(new_lr, self.ceiling_lr)

        self.optimizer.param_groups[0]["lr"] = new_lr


class GradientWatchingAdamLRScheduler(GradientAdamLRScheduler):
    """Watches gradient: if it has been too small for too long, double the LR.
    If the gradient exceeds the threshold, set it back to the original base LR."""

    optimizer: torch.optim.Optimizer
    base_lr: float
    gradient_ceiling: float = 1e-3
    patience: int = 40  # number of epochs to wait
    _current_epochs_waited: int = 0

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.patience = 30
        self._current_epochs_waited = 0
        self.gradient_ceiling = 1e-2

    def step(self, gradient_norm: float) -> None:
        if gradient_norm < self.gradient_ceiling:
            logger.debug("Gradient is too small, increasing LR. Current waited: %s", self._current_epochs_waited)
            self._current_epochs_waited += 1
            if self._current_epochs_waited > self.patience:
                self.optimizer.param_groups[0]["lr"] *= 2
                self._current_epochs_waited = 0
        else:
            logger.debug(f"Gradient norm is {gradient_norm}, not too small.")
            self.optimizer.param_groups[0]["lr"] = self.base_lr
            self._current_epochs_waited = 0
