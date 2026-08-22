"""Evaluation use case boundary."""

from __future__ import annotations

from collections.abc import Callable

from vetime.config import EvaluationConfig


class EvaluateUseCase:
    def __init__(self, runner: Callable[[EvaluationConfig], object]):
        self.runner = runner

    def run(self, config: EvaluationConfig):
        return self.runner(config)
