"""Training and evaluation engines."""

from .evaluator import EvaluationResult, Evaluator
from .trainer import Trainer, TrainerDependencies, TrainingResult

__all__ = ["EvaluationResult", "Evaluator", "Trainer", "TrainerDependencies", "TrainingResult"]
