"""Validate the supplied VETime temporal and MAE checkpoints.

Run from the repository root with::

    python scripts/validate_checkpoints.py --full-model

The command only constructs models and reports compatibility; it never writes
to either checkpoint file and does not start a training job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temporal",
        default=str(ROOT / "checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth"),
    )
    parser.add_argument("--vision-dir", default=str(ROOT / "checkpoints/weight_v"))
    parser.add_argument("--vision-name", default="mae_visualize_base.pth")
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="also construct the composed VETime model with both checkpoints",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    temporal_path = Path(args.temporal)
    vision_path = Path(args.vision_dir) / args.vision_name
    if not temporal_path.is_file():
        raise FileNotFoundError(f"temporal checkpoint not found: {temporal_path}")
    if not vision_path.is_file():
        raise FileNotFoundError(f"vision checkpoint not found: {vision_path}")

    from vetime.config import CheckpointPaths, TrainingConfig
    from vetime.infrastructure.checkpointing.temporal_legacy import load_legacy_temporal_checkpoint
    from vetime.models.factory import build_vetime_model
    from vetime.models.temporal.config import TemporalModelConfig
    from vetime.models.temporal.model import TemporalModel
    from vetime.models.vision.mae import FrozenMAEEncoder

    temporal = TemporalModel(
        TemporalModelConfig(
            d_model=512,
            d_proj=256,
            patch_size=16,
            num_layers=8,
            num_heads=8,
            d_ff_dropout=0.05,
            use_rope=True,
            num_features=1,
        )
    )
    report = load_legacy_temporal_checkpoint(temporal, temporal_path)
    print(f"temporal: loaded={report.loaded_keys}, missing_required={len(report.missing_required_keys)}")

    vision = FrozenMAEEncoder.from_checkpoint(args.vision_name, args.vision_dir)
    print(
        f"vision: encoder={type(vision.encoder).__name__}, hidden_size={vision.hidden_size}, "
        f"patch_size={vision.patch_size}, frozen={all(not p.requires_grad for p in vision.parameters())}"
    )

    if args.full_model:
        config = TrainingConfig(
            seed=64,
            batch_size=1,
            paths=CheckpointPaths(
                temporal=str(temporal_path),
                vision_dir=str(args.vision_dir),
                vision_name=args.vision_name,
            ),
        )
        model = build_vetime_model(config)
        print(
            f"full_model: class={type(model).__name__}, temporal_dim={model.temporal.config.d_model}, "
            f"vision_dim={model.vision_encoder.hidden_size}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
