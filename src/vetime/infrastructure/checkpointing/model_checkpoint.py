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
            "format_version": 2,
            "kind": "vetime_model",
            "model_state_dict": model.state_dict(),
            "metadata": dict(metadata),
        },
        path,
    )


def load_model_checkpoint(model: nn.Module, path: str | Path) -> ModelLoadReport:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and payload.get("kind") == "temporal_pretrain":
        raise CheckpointCompatibilityError("expected a vetime_model checkpoint, got training pretrain")
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, Mapping) else None
    if not isinstance(state_dict, Mapping):
        raise CheckpointCompatibilityError("model checkpoint does not contain model_state_dict")
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except (RuntimeError, KeyError, ValueError) as error:
        raise CheckpointCompatibilityError(f"model checkpoint is incompatible: {error}") from error
    metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
    return ModelLoadReport(tuple(missing), tuple(unexpected), metadata)
