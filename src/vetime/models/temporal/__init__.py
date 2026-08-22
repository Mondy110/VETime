"""Standalone temporal model and its encoder/head components."""

from .config import TemporalModelConfig
from .model import TemporalModel

__all__ = ["TemporalModel", "TemporalModelConfig"]
