"""Application boundary for model construction."""

from vetime.config import TrainingConfig
from vetime.models.factory import build_vetime_model


def build_training_model(config: TrainingConfig, **kwargs):
    return build_vetime_model(config, **kwargs)
