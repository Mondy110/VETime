"""Dependency-injected training loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrainingResult:
    last_epoch: int
    global_step: int
    train_loss: float
    validation_loss: float | None = None
    best_validation_loss: float | None = None


@dataclass
class TrainerDependencies:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: Iterable[Any]
    step_fn: Callable[[nn.Module, Any], Tensor]
    validation_loader: Iterable[Any] | None = None
    validation_fn: Callable[[nn.Module, Any], Tensor] | None = None
    scheduler: Any = None
    accelerator: Any = None
    callbacks: Sequence[Any] = field(default_factory=tuple)


class Trainer:
    def __init__(self, dependencies: TrainerDependencies, start_epoch: int = 0):
        self.dependencies = dependencies
        self.start_epoch = start_epoch
        self.global_step = 0
        self.best_validation_loss: float | None = None

    def _backward(self, loss: Tensor) -> None:
        if self.dependencies.accelerator is not None:
            self.dependencies.accelerator.backward(loss)
        else:
            loss.backward()

    def _validation_loss(self) -> float | None:
        if self.dependencies.validation_loader is None or self.dependencies.validation_fn is None:
            return None
        self.dependencies.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in self.dependencies.validation_loader:
                value = self.dependencies.validation_fn(self.dependencies.model, batch)
                losses.append(float(value.detach().cpu()))
        return sum(losses) / len(losses) if losses else None

    def fit(self, *, max_epochs: int | None = None, max_train_steps: int | None = None) -> TrainingResult:
        epochs = max_epochs if max_epochs is not None else 1
        last_epoch = self.start_epoch - 1
        last_train_loss = 0.0
        last_validation_loss = None
        for epoch_offset in range(epochs):
            epoch = self.start_epoch + epoch_offset
            last_epoch = epoch
            self.dependencies.model.train()
            losses: list[float] = []
            for batch in self.dependencies.train_loader:
                if max_train_steps is not None and self.global_step >= max_train_steps:
                    break
                self.dependencies.optimizer.zero_grad(set_to_none=True)
                loss = self.dependencies.step_fn(self.dependencies.model, batch)
                if not torch.is_tensor(loss) or loss.ndim != 0:
                    raise TypeError("training step must return a scalar torch.Tensor")
                self._backward(loss)
                self.dependencies.optimizer.step()
                if self.dependencies.scheduler is not None:
                    self.dependencies.scheduler.step()
                self.global_step += 1
                losses.append(float(loss.detach().cpu()))
                for callback in self.dependencies.callbacks:
                    on_step = getattr(callback, "on_step", None)
                    if on_step is not None:
                        on_step(epoch=epoch, global_step=self.global_step, loss=float(loss.detach().cpu()))
            last_train_loss = sum(losses) / len(losses) if losses else last_train_loss
            last_validation_loss = self._validation_loss()
            if last_validation_loss is not None:
                if self.best_validation_loss is None or last_validation_loss < self.best_validation_loss:
                    self.best_validation_loss = last_validation_loss
            for callback in self.dependencies.callbacks:
                on_epoch_end = getattr(callback, "on_epoch_end", None)
                if on_epoch_end is not None:
                    on_epoch_end(epoch=epoch, train_loss=last_train_loss, validation_loss=last_validation_loss)
            if max_train_steps is not None and self.global_step >= max_train_steps:
                break
        return TrainingResult(
            last_epoch=last_epoch,
            global_step=self.global_step,
            train_loss=last_train_loss,
            validation_loss=last_validation_loss,
            best_validation_loss=self.best_validation_loss,
        )
