from pathlib import Path

import pytest
import torch
from torch import nn

from vetime.infrastructure.checkpointing.temporal_legacy import (
    CheckpointCompatibilityError,
    load_legacy_temporal_checkpoint,
    map_legacy_temporal_state_dict,
)


class TinyTemporalTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.Module()
        self.temporal.encoder = nn.Module()
        self.temporal.encoder.embedding = nn.Linear(2, 2)
        self.temporal.reconstruction_head = nn.Linear(2, 1)
        self.temporal.anomaly_head = nn.Linear(2, 2)


def legacy_state_dict():
    return {
        "module.ts_encoder.embedding.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "module.ts_encoder.embedding.bias": torch.tensor([1.0, 2.0]),
        "reconstruction_head.weight": torch.arange(2, dtype=torch.float32).reshape(1, 2),
        "reconstruction_head.bias": torch.tensor([3.0]),
        "anomaly_head.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "anomaly_head.bias": torch.tensor([4.0, 5.0]),
    }


def test_maps_all_required_legacy_prefixes_and_removes_module_prefix():
    mapped, report = map_legacy_temporal_state_dict(legacy_state_dict(), target_prefix="temporal.")

    assert set(mapped) == {
        "temporal.encoder.embedding.weight",
        "temporal.encoder.embedding.bias",
        "temporal.reconstruction_head.weight",
        "temporal.reconstruction_head.bias",
        "temporal.anomaly_head.weight",
        "temporal.anomaly_head.bias",
    }
    assert report.unconsumed_legacy_keys == ()


def test_loader_accepts_wrapped_and_naked_state_dicts(tmp_path: Path):
    for payload in (legacy_state_dict(), {"model_state_dict": legacy_state_dict()}):
        path = tmp_path / f"checkpoint-{len(list(tmp_path.iterdir()))}.pth"
        torch.save(payload, path)
        model = TinyTemporalTarget()

        report = load_legacy_temporal_checkpoint(model, path)

        assert not report.missing_required_keys
        assert not report.shape_conflicts
        assert report.loaded_keys == 6


def test_loader_rejects_shape_conflict(tmp_path: Path):
    payload = legacy_state_dict()
    payload["reconstruction_head.weight"] = torch.ones(2, 2)
    path = tmp_path / "bad-shape.pth"
    torch.save(payload, path)

    with pytest.raises(CheckpointCompatibilityError, match="shape"):
        load_legacy_temporal_checkpoint(TinyTemporalTarget(), path)


def test_lora_mapping_inserts_original_linear_for_supported_projection():
    mapped, _ = map_legacy_temporal_state_dict(
        {"ts_encoder.transformer.layers.0.q_proj.weight": torch.ones(2, 2)},
        target_prefix="temporal.",
        lora=True,
    )

    assert "temporal.encoder.transformer.layers.0.q_proj.original_linear.weight" in mapped
