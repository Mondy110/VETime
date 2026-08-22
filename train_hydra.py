"""Hydra entry point for the legacy QueryDecoder/CMRG experiment branch.

The existing ``train.py`` command-line interface remains available.  Use this
entry point for layered YAML configuration and Hydra command-line overrides.
"""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from vetime.application.train import TrainUseCase
from vetime.interfaces.hydra import training_config_from_mapping


@hydra.main(config_path="configs", config_name="univariate", version_base=None)
def run(cfg: DictConfig) -> None:
    """Compose config, adapt it, and run the established training implementation."""
    config = training_config_from_mapping(OmegaConf.to_container(cfg, resolve=True))
    result_path = Path(
        config.output_file_path.replace("result.json", f"{config.model.model_name.replace('/', '-')}_result.json")
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)

    results = TrainUseCase().run(config)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)


if __name__ == "__main__":
    run()
