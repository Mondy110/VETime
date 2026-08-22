"""Adapters from external configuration formats to vetime contracts."""

from .cli import training_config_from_namespace
from .hydra import training_config_from_mapping

__all__ = ["training_config_from_namespace", "training_config_from_mapping"]
