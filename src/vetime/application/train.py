"""Training use case shared by CLI and Hydra adapters."""

from __future__ import annotations

from collections.abc import Callable

from vetime.config import TrainingConfig
from vetime.interfaces.cli import namespace_from_training_config


class TrainUseCase:
    def __init__(self, runner: Callable[[TrainingConfig], object] | None = None):
        self.runner = runner or self._legacy_runner

    @staticmethod
    def _legacy_runner(config: TrainingConfig):
        """Execute the current training loop through a typed boundary during migration."""
        from train import train_univariate

        return train_univariate(namespace_from_training_config(config))

    def run(self, config: TrainingConfig):
        return self.runner(config)
