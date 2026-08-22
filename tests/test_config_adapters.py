from argparse import Namespace

import pytest

from vetime.config import CheckpointPaths, DataConfig, ModelConfig, OptimizerConfig, TrainingConfig
from vetime.interfaces.cli import training_config_from_namespace
from vetime.interfaces.hydra import training_config_from_mapping


def test_cli_and_hydra_create_equivalent_training_configuration():
    cli = Namespace(
        seed=64,
        batch_size=32,
        ts_path="checkpoints/weight_ts/x.pth",
        vision_path="checkpoints/weight_v",
        vision_name="mae_visualize_base.pth",
    )
    hydra = {
        "seed": 64,
        "data": {"batch_size": 32},
        "paths": {"ts_path": "checkpoints/weight_ts/x.pth", "vision_path": "checkpoints/weight_v"},
        "model": {"vision_name": "mae_visualize_base.pth"},
    }

    assert training_config_from_namespace(cli) == training_config_from_mapping(hydra)


def test_config_rejects_non_positive_batch_size():
    paths = CheckpointPaths(
        temporal=None,
        vision_dir="checkpoints/weight_v",
        vision_name="mae_visualize_base.pth",
    )

    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainingConfig(
            seed=64,
            batch_size=0,
            paths=paths,
            model=ModelConfig(),
            training=OptimizerConfig(),
            data=DataConfig(),
        )
