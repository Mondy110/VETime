from argparse import Namespace

from vetime.application.train import TrainUseCase
from vetime.interfaces.cli import training_config_from_namespace
from vetime.interfaces.hydra import training_config_from_mapping


def test_cli_and_hydra_call_the_same_train_use_case():
    calls = []

    class RecordingRunner:
        def __call__(self, config):
            calls.append(config)
            return {"ok": True}

    cli_namespace = Namespace(
        seed=64,
        batch_size=2,
        ts_path=None,
        vision_path="checkpoints/weight_v",
        vision_name="mae_visualize_base.pth",
    )
    hydra_mapping = {
        "seed": 64,
        "data": {"batch_size": 2},
        "paths": {"ts_path": None, "vision_path": "checkpoints/weight_v"},
        "model": {"vision_name": "mae_visualize_base.pth"},
    }
    use_case = TrainUseCase(runner=RecordingRunner())

    use_case.run(training_config_from_namespace(cli_namespace))
    use_case.run(training_config_from_mapping(hydra_mapping))

    assert calls[0] == calls[1]
