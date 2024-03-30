from abc import ABC, abstractmethod

import torch


class ScheduledNoiseGenerator(ABC):
    """Noise to add to the logits before taking the softmax."""

    @abstractmethod
    def generate_noise(self, current_epoch: int, total_epochs: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class NoNoiseGenerator(ScheduledNoiseGenerator):
    shape: tuple[int]
    noise: torch.Tensor
    device: torch.device

    def __init__(self, shape: tuple[int], device: torch.device):
        self.shape = shape
        self.noise = torch.zeros(shape, device=device)
        self.device = device

    def generate_noise(self, current_epoch: int, total_epochs: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.noise, self.noise


class SPNoiseGenerator(ScheduledNoiseGenerator):
    """This noise is basically what subnetwork probing is doing."""

    shape: tuple[int]
    device: torch.device

    def __init__(self, shape: tuple[int], device: torch.device):
        self.shape = shape
        self.device = device

    def generate_noise(self, current_epoch: int, total_epochs: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.logit(torch.rand(self.shape, device=self.device)), torch.logit(
            torch.rand(self.shape, device=self.device)
        )


class ClampedSPNoiseGenerator(ScheduledNoiseGenerator):
    shape: tuple[int]
    device: torch.device
    max: float
    scaling: float

    def __init__(self, shape: tuple[int], device: torch.device, max: float, scaling: float = 1.0):
        self.shape = shape
        self.device = device
        self.max = max
        self.scaling = scaling

    def generate_noise(self, current_epoch: int, total_epochs: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._generate(), self._generate()

    def _generate(self) -> torch.Tensor:
        return torch.logit(torch.rand(self.shape, device=self.device)).clamp(-self.max, self.max) * self.scaling


class IntermittentNoiseGenerator(ScheduledNoiseGenerator):
    """This noise generator generates noise for a certain number of epochs (`noise_epoch_length`), then stops for
    a certain number of epochs (`nonoise_epoch_length`), repeating until the end."""

    base_generator: ScheduledNoiseGenerator
    no_noise_generator: NoNoiseGenerator
    shape: tuple[int]
    device: torch.device
    noise_epoch_length: int
    no_noise_epoch_length: int

    def __init__(
        self,
        base_generator: ScheduledNoiseGenerator,
        shape: tuple[int],
        device: torch.device,
        noise_epoch_length: int,
        no_noise_epoch_length: int,
    ):
        self.base_generator = base_generator
        self.no_noise_generator = NoNoiseGenerator(shape, device)
        self.shape = shape
        self.device = device
        self.noise_epoch_length = noise_epoch_length
        self.no_noise_epoch_length = no_noise_epoch_length

    def generate_noise(self, current_epoch: int, total_epochs: int) -> tuple[torch.Tensor, torch.Tensor]:
        if current_epoch % (self.noise_epoch_length + self.no_noise_epoch_length) < self.noise_epoch_length:
            return self.base_generator.generate_noise(current_epoch, total_epochs)
        else:
            return self.no_noise_generator.generate_noise(current_epoch, total_epochs)
