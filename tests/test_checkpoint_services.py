from pathlib import Path

import pytest
import torch
from torch import nn

from vetime.infrastructure.checkpointing.model_checkpoint import (
    load_model_checkpoint,
    save_model_checkpoint,
)
from vetime.infrastructure.checkpointing.resume import (
    CheckpointCompatibilityError,
    ResumeState,
    load_resume_checkpoint,
    save_resume_checkpoint,
)


def test_new_model_checkpoint_contains_version_and_kind(tmp_path: Path):
    model = nn.Linear(2, 1)
    path = tmp_path / "model.pth"

    save_model_checkpoint(model, path, metadata={"run": "test"})

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 2
    assert payload["kind"] == "vetime_model"
    assert payload["metadata"]["run"] == "test"


def test_model_checkpoint_round_trip_restores_parameters(tmp_path: Path):
    source = nn.Linear(2, 1)
    path = tmp_path / "model.pth"
    save_model_checkpoint(source, path, metadata={})
    target = nn.Linear(2, 1)

    report = load_model_checkpoint(target, path)

    assert not report.missing_keys
    torch.testing.assert_close(target.weight, source.weight)
    torch.testing.assert_close(target.bias, source.bias)


def test_resume_loader_rejects_temporal_pretrain_checkpoint(tmp_path: Path):
    path = tmp_path / "temporal.pth"
    torch.save({"format_version": 2, "kind": "temporal_pretrain", "model_state_dict": {}}, path)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())

    with pytest.raises(CheckpointCompatibilityError, match="training_resume"):
        load_resume_checkpoint(path, model, optimizer, scheduler=None)


def test_resume_checkpoint_round_trip_restores_training_state(tmp_path: Path):
    source = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
    state = ResumeState(epoch=3, global_step=17, dataset_idx=2, current_dim=8)
    path = tmp_path / "resume.pth"

    save_resume_checkpoint(path, source, optimizer, scheduler=None, state=state, metadata={})

    target = nn.Linear(2, 1)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.01)
    restored = load_resume_checkpoint(path, target, target_optimizer, scheduler=None)

    assert restored == state
    torch.testing.assert_close(target.weight, source.weight)
