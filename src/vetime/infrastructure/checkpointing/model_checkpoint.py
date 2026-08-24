"""Versioned VETime model checkpoint persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .temporal_legacy import CheckpointCompatibilityError


@dataclass(frozen=True)
class ModelLoadReport:
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    metadata: Mapping[str, Any]


def save_model_checkpoint(model: nn.Module, path: str | Path, metadata: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 3,
            "kind": "vetime_model",
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata),
        },
        path,
    )


def load_model_checkpoint(model: nn.Module, path: str | Path) -> ModelLoadReport:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise CheckpointCompatibilityError("model checkpoint must be a versioned mapping")
    if payload.get("format_version") != 3:
        raise CheckpointCompatibilityError("expected checkpoint format_version=3")
    if payload.get("kind") != "vetime_model":
        raise CheckpointCompatibilityError(
            f"expected a vetime_model checkpoint, got {payload.get('kind')!r}"
        )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise CheckpointCompatibilityError("model checkpoint does not contain model_state_dict")
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, KeyError, ValueError) as error:
        raise CheckpointCompatibilityError(f"model checkpoint is incompatible: {error}") from error
    metadata = payload.get("metadata", {})
    return ModelLoadReport((), (), metadata)
