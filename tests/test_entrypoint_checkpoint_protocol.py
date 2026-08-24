"""Entrypoints must not retain a legacy full-VETime loading path."""

from pathlib import Path


def test_training_and_tsb_entrypoints_expose_v3_model_checkpoint_only():
    root = Path(__file__).resolve().parents[1]
    for entrypoint in (root / "train.py", root / "Test_TSB.py"):
        source = entrypoint.read_text(encoding="utf-8")
        assert "--model_checkpoint" in source
        assert "--vetime_path" not in source
        assert "from model.VETime import VETIME" not in source
