"""Pure model assembly and trainability policies."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

from torch import nn

from vetime.config import TrainingConfig
from vetime.infrastructure.checkpointing.model_checkpoint import load_model_checkpoint
from vetime.infrastructure.checkpointing.temporal_legacy import load_legacy_temporal_checkpoint
from vetime.models.multimodal.model import VETimeMultimodalModel, VETimeOptions
from vetime.models.temporal.config import TemporalModelConfig
from vetime.models.temporal.model import TemporalModel
from vetime.models.vision.mae import FrozenMAEEncoder


def _temporal_config_for(config: TrainingConfig, temporal_config: TemporalModelConfig | None) -> TemporalModelConfig:
    base = temporal_config or TemporalModelConfig()
    return replace(base, use_lora=config.model.ts_finetune_type == "lora")


def build_vetime_model(
    config: TrainingConfig,
    *,
    temporal_config: TemporalModelConfig | None = None,
    temporal: TemporalModel | None = None,
    vision_encoder: nn.Module | None = None,
) -> VETimeMultimodalModel:
    """Build the composed model and apply initialization policy in one place."""
    if temporal is None:
        temporal = TemporalModel(_temporal_config_for(config, temporal_config))
    if vision_encoder is None:
        vision_encoder = FrozenMAEEncoder.from_checkpoint(
            config.paths.vision_name,
            config.paths.vision_dir,
            max_length=5000,
            use_vectorized_fold=config.model.use_vectorized_fold,
        )
    options = VETimeOptions(
        vision_dim=vision_encoder.hidden_size,
        temporal_dim=temporal.config.d_model,
        max_length=getattr(vision_encoder, "MAX_L", 5000),
        model_name=config.model.model_name,
        cmrg_enabled=config.model.cmrg_enabled,
        cmrg_num_relation_tokens=config.model.cmrg_num_relation_tokens,
        cmrg_guide_dim=config.model.cmrg_guide_dim,
        cmrg_num_heads=config.model.cmrg_num_heads,
        cmrg_metric_init=config.model.cmrg_metric_init,
        cmrg_gate_init=config.model.cmrg_gate_init,
        cmrg_injection_mode=config.model.cmrg_injection_mode,
        cmrg_factorized=config.model.cmrg_factorized,
        use_query_decoder=config.model.use_query_decoder,
        use_gradient_checkpointing=config.model.use_gradient_checkpointing,
    )
    model = VETimeMultimodalModel(temporal=temporal, vision_encoder=vision_encoder, options=options)
    if config.paths.temporal:
        load_legacy_temporal_checkpoint(
            model,
            Path(config.paths.temporal),
            lora=config.model.ts_finetune_type == "lora",
        )
    elif config.paths.model_checkpoint:
        load_model_checkpoint(model, config.paths.model_checkpoint)
    apply_temporal_finetune_policy(model, config.model.ts_finetune_type)
    return model


def apply_temporal_finetune_policy(model: nn.Module, mode: str) -> None:
    """Apply full, LoRA, or frozen temporal trainability policy."""
    if mode not in {"freeze", "lora", "full"}:
        raise ValueError(f"unsupported temporal fine-tuning mode: {mode}")
    if mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    for name, parameter in model.named_parameters():
        if mode == "freeze" and name.startswith("temporal.encoder."):
            parameter.requires_grad = ".cmrg_" in name or name.endswith("cmrg_alpha")
        elif mode == "lora" and "original_linear" in name:
            parameter.requires_grad = False


def freeze_for_cls_warmup(model: nn.Module) -> dict[str, bool]:
    """Freeze all parameters except classification heads for a warmup phase."""
    previous = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".anomaly_head." in name or name.startswith("anomaly_head.")
    return previous


def restore_requires_grad(model: nn.Module, previous: Mapping[str, bool]) -> None:
    for name, parameter in model.named_parameters():
        if name in previous:
            parameter.requires_grad = previous[name]
