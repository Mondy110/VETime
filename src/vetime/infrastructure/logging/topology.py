"""Human-readable runtime topology reports for VETime training runs."""

from __future__ import annotations

import torch
from torch import nn


def _parameter_count(parameters) -> tuple[int, int]:
    values = tuple(parameters)
    return sum(parameter.numel() for parameter in values), sum(
        parameter.numel() for parameter in values if parameter.requires_grad
    )


def _group_counts(model: nn.Module) -> str:
    groups = {
        "vision": lambda name: name.startswith("vision_encoder."),
        "temporal": lambda name: name.startswith("temporal."),
        "fusion": lambda name: name.startswith(("mlp_i.", "I_att.", "fusion.", "mm_w.", "fusion_proj.")),
        "query_decoder": lambda name: name.startswith("query_decoder."),
        "cmrg": lambda name: name.startswith(("cmrg_distiller.", "cmrg_guider.")) or ".cmrg_" in name,
        "lora": lambda name: ".lora_A" in name or ".lora_B" in name,
    }
    named_parameters = tuple(model.named_parameters())
    return " | ".join(
        f"{group}={trainable:,}/{total:,}"
        for group, matches in groups.items()
        for total, trainable in (_parameter_count(
            parameter for name, parameter in named_parameters if matches(name)
        ),)
    )


def _device_description(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return f"cuda:{index} | {torch.cuda.get_device_name(index)}"


def format_runtime_topology(
    model: nn.Module,
    *,
    device: torch.device,
    initialization_source: str,
) -> str:
    """Describe the clean model assembly without changing training behavior."""
    temporal = model.temporal
    vision = model.vision_encoder
    temporal_config = temporal.config
    vision_frozen = all(not parameter.requires_grad for parameter in vision.parameters())
    lines = [
        "[INFO] ===== VETime Runtime Topology =====",
        f"[INFO] Device: {_device_description(device)}",
        "[INFO] Architecture: VETimeMultimodalModel (clean composition)",
        f"[INFO] Initialization: {initialization_source}",
        (
            "[INFO] Temporal: "
            f"{type(temporal).__name__} | layers={temporal_config.num_layers} | "
            f"d_model={temporal_config.d_model} | patch_size={temporal.patch_size} | "
            f"finetune={'LoRA' if temporal_config.use_lora else 'full/frozen'}"
        ),
        (
            "[INFO] Vision: "
            f"{type(vision).__name__} | hidden_size={vision.hidden_size} | "
            f"patch_size={vision.patch_size} | frozen={vision_frozen}"
        ),
        f"[INFO] CMRG: {'enabled' if model.cmrg_enabled else 'disabled'}",
        f"[INFO] QueryDecoder: {'enabled' if model.use_query_decoder else 'disabled'}",
        f"[INFO] Trainable parameters by group: {_group_counts(model)}",
        "[INFO] Legacy full VETime fallback: disabled",
    ]
    return "\n".join(lines)


def log_runtime_topology(model: nn.Module, *, device: torch.device, initialization_source: str) -> None:
    print(format_runtime_topology(model, device=device, initialization_source=initialization_source))
