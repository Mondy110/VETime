"""Training progress checkpoint persistence and restoration."""

from __future__ import annotations

import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .temporal_legacy import CheckpointCompatibilityError


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    global_step: int
    dataset_idx: int = 0
    current_dim: int = 0
    prev_checkpoint_path: str | None = None
    best_val_loss: float | None = None
    patience_counter: int = 0


def save_resume_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer,
    scheduler,
    state: ResumeState,
    metadata: Mapping[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "kind": "training_resume",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "training_state": asdict(state),
        "metadata": dict(metadata),
        "random_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    if torch.cuda.is_available():
        payload["random_state"]["torch_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)


def _restore_optimizer(optimizer, state_dict) -> None:
    if optimizer is None or state_dict is None:
        return
    try:
        optimizer.load_state_dict(state_dict)
    except (RuntimeError, KeyError, ValueError) as error:
        warnings.warn(f"Optimizer restore skipped: {error}", RuntimeWarning, stacklevel=2)


def _restore_random_state(random_state: Mapping[str, Any]) -> None:
    if not random_state:
        return
    if "python" in random_state:
        random.setstate(random_state["python"])
    if "numpy" in random_state:
        np.random.set_state(random_state["numpy"])
    if "torch" in random_state:
        torch.set_rng_state(random_state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in random_state:
        torch.cuda.set_rng_state_all(random_state["torch_cuda"])


def load_resume_checkpoint(path: str | Path, model: nn.Module, optimizer, scheduler):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kind = payload.get("kind") if isinstance(payload, Mapping) else None
    if kind == "temporal_pretrain":
        raise CheckpointCompatibilityError("expected a training_resume checkpoint, got temporal_pretrain")
    if kind not in ("training_resume", None):
        raise CheckpointCompatibilityError(f"expected a training_resume checkpoint, got {kind!r}")
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise CheckpointCompatibilityError("resume checkpoint does not contain model_state_dict")
    try:
        model.load_state_dict(payload["model_state_dict"], strict=False)
    except (RuntimeError, KeyError, ValueError) as error:
        raise CheckpointCompatibilityError(f"resume model state is incompatible: {error}") from error
    _restore_optimizer(optimizer, payload.get("optimizer_state_dict"))
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    _restore_random_state(payload.get("random_state", {}))
    state = payload.get("training_state", {})
    return ResumeState(
        epoch=state.get("epoch", payload.get("epoch", 0)),
        global_step=state.get("global_step", payload.get("global_step", 0)),
        dataset_idx=state.get("dataset_idx", payload.get("dataset_idx", 0)),
        current_dim=state.get("current_dim", payload.get("current_dim", 0)),
        prev_checkpoint_path=state.get("prev_checkpoint_path", payload.get("prev_checkpoint_path")),
        best_val_loss=state.get("best_val_loss", payload.get("best_val_loss")),
        patience_counter=state.get("patience_counter", payload.get("patience_counter", 0)),
    )
