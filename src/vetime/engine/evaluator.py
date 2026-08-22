"""Dependency-injected validation and metric evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EvaluationResult:
    loss: float | None
    metrics: dict[str, float]
    predictions: Any = None
    labels: Any = None


class Evaluator:
    def evaluate_validation(
        self,
        model: nn.Module,
        loader: Iterable[Any],
        step_fn: Callable[[nn.Module, Any], Tensor],
    ) -> EvaluationResult:
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                value = step_fn(model, batch)
                if not torch.is_tensor(value) or value.ndim != 0:
                    raise TypeError("validation step must return a scalar torch.Tensor")
                losses.append(float(value.detach().cpu()))
        mean_loss = sum(losses) / len(losses) if losses else None
        return EvaluationResult(loss=mean_loss, metrics={})

    def evaluate_tsb(
        self,
        model: nn.Module,
        loader: Iterable[Any],
        predict_fn: Callable[[nn.Module, Any], tuple[Any, Any]],
        metric_fn: Callable[[Any, Any], dict[str, float]],
    ) -> EvaluationResult:
        model.eval()
        predictions: list[Any] = []
        labels: list[Any] = []
        with torch.no_grad():
            for batch in loader:
                batch_predictions, batch_labels = predict_fn(model, batch)
                predictions.append(batch_predictions)
                labels.append(batch_labels)
        metrics = metric_fn(predictions, labels)
        return EvaluationResult(loss=None, metrics=metrics, predictions=predictions, labels=labels)
