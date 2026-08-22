"""Strict compatibility loading for the original temporal pretraining model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


LEGACY_PREFIXES = {
    "ts_encoder.": "encoder.",
    "reconstruction_head.": "reconstruction_head.",
    "anomaly_head.": "anomaly_head.",
}
LORA_PROJECTION_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint cannot be safely applied to a target model."""


@dataclass(frozen=True)
class LoadReport:
    loaded_keys: int
    mapped_pairs: tuple[tuple[str, str], ...]
    unconsumed_legacy_keys: tuple[str, ...]
    missing_required_keys: tuple[str, ...]
    unexpected_target_keys: tuple[str, ...]
    shape_conflicts: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]


def extract_state_dict(payload: Mapping[str, Any]) -> Mapping[str, Tensor]:
    """Extract a naked or ``model_state_dict``-wrapped state dictionary."""
    candidate = payload.get("model_state_dict", payload)
    if not isinstance(candidate, Mapping):
        raise CheckpointCompatibilityError("checkpoint does not contain a state dictionary")
    if not all(isinstance(key, str) for key in candidate):
        raise CheckpointCompatibilityError("checkpoint state dictionary contains a non-string key")
    return candidate


def _normalise_key(key: str) -> str:
    return key.removeprefix("module.")


def _with_lora_original_linear(key: str) -> str:
    for projection_name in LORA_PROJECTION_NAMES:
        suffix = f"{projection_name}."
        marker = key.rfind(suffix)
        if marker >= 0:
            return f"{key[:marker]}{projection_name}.original_linear.{key[marker + len(suffix):]}"
    return key


def _target_prefix_from_state_dict(state_dict: Mapping[str, Tensor]) -> str:
    if any(key.startswith("temporal.") for key in state_dict):
        return "temporal."
    return ""


def _is_optional_temporal_key(key: str) -> bool:
    """CMRG gates and newly installed LoRA factors are not in old runs."""
    return (
        ".cmrg_" in key
        or key.endswith("cmrg_alpha")
        or key.endswith(".lora_A")
        or key.endswith(".lora_B")
    )


def map_legacy_temporal_state_dict(
    state_dict: Mapping[str, Tensor],
    *,
    target_prefix: str = "",
    lora: bool = False,
) -> tuple[dict[str, Tensor], LoadReport]:
    """Map legacy names to a standalone or composed temporal model namespace."""
    mapped: dict[str, Tensor] = {}
    pairs: list[tuple[str, str]] = []
    unconsumed: list[str] = []

    for original_key, value in state_dict.items():
        key = _normalise_key(original_key)
        source_prefix = next((prefix for prefix in LEGACY_PREFIXES if key.startswith(prefix)), None)
        if source_prefix is None:
            unconsumed.append(original_key)
            continue
        target_key = f"{target_prefix}{LEGACY_PREFIXES[source_prefix]}{key[len(source_prefix):]}"
        if lora and source_prefix == "ts_encoder.":
            target_key = _with_lora_original_linear(target_key)
        mapped[target_key] = value
        pairs.append((original_key, target_key))

    report = LoadReport(
        loaded_keys=0,
        mapped_pairs=tuple(pairs),
        unconsumed_legacy_keys=tuple(unconsumed),
        missing_required_keys=(),
        unexpected_target_keys=(),
        shape_conflicts=(),
    )
    return mapped, report


def load_legacy_temporal_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    lora: bool = False,
) -> LoadReport:
    """Load legacy temporal weights with strict required-key and shape checks."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = extract_state_dict(payload)
    target_state = model.state_dict()
    target_prefix = _target_prefix_from_state_dict(target_state)
    mapped, initial_report = map_legacy_temporal_state_dict(
        source_state,
        target_prefix=target_prefix,
        lora=lora,
    )

    required_prefix = (
        f"{target_prefix}encoder.",
        f"{target_prefix}reconstruction_head.",
        f"{target_prefix}anomaly_head.",
    )
    required_keys = {
        key
        for key in target_state
        if key.startswith(required_prefix) and not _is_optional_temporal_key(key)
    }
    missing = tuple(sorted(required_keys - mapped.keys()))
    unexpected_target = tuple(sorted(key for key in mapped if key not in target_state))
    shape_conflicts = tuple(
        sorted(
            (
                key,
                tuple(value.shape),
                tuple(target_state[key].shape),
            )
            for key, value in mapped.items()
            if key in target_state and tuple(value.shape) != tuple(target_state[key].shape)
        )
    )
    report = LoadReport(
        loaded_keys=len(mapped),
        mapped_pairs=initial_report.mapped_pairs,
        unconsumed_legacy_keys=initial_report.unconsumed_legacy_keys,
        missing_required_keys=missing,
        unexpected_target_keys=unexpected_target,
        shape_conflicts=shape_conflicts,
    )
    if report.unconsumed_legacy_keys:
        raise CheckpointCompatibilityError(
            f"unconsumed legacy checkpoint keys: {list(report.unconsumed_legacy_keys)}"
        )
    if report.missing_required_keys:
        raise CheckpointCompatibilityError(
            f"missing required temporal keys: {list(report.missing_required_keys)}"
        )
    if report.unexpected_target_keys:
        raise CheckpointCompatibilityError(
            f"mapped checkpoint keys are absent from target: {list(report.unexpected_target_keys)}"
        )
    if report.shape_conflicts:
        raise CheckpointCompatibilityError(f"checkpoint shape conflicts: {list(report.shape_conflicts)}")

    model.load_state_dict(mapped, strict=False)
    return report
