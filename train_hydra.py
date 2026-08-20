"""Hydra entry point for the legacy QueryDecoder/CMRG experiment branch.

The existing ``train.py`` command-line interface remains available.  Use this
entry point for layered YAML configuration and Hydra command-line overrides.
"""

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from hydra_config import namespace_from_config


@hydra.main(config_path="configs", config_name="univariate", version_base=None)
def run(cfg: DictConfig) -> None:
    """Compose config, adapt it, and run the established training implementation."""
    from train import main

    args = namespace_from_config(OmegaConf.to_container(cfg, resolve=True))
    result_path = Path(
        args.output_file_path.replace("result.json", f"{args.model_name.replace('/', '-')}_result.json")
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)

    results = main(args)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)


if __name__ == "__main__":
    run()
