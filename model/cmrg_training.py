"""Training utilities for CMRG that do not depend on the training runtime."""

from typing import Optional

from model.CMRG import CMRGContext


def configure_freeze_mode(model):
    """Freeze temporal backbone weights while retaining CMRG training."""
    frozen_backbone_parts = ("transformer_encoder", "embedding_layer", "rope_embedder")
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        is_cmrg_parameter = name.startswith("cmrg_") or ".cmrg_" in name
        if is_cmrg_parameter:
            param.requires_grad = True
            trainable_count += 1
        elif any(part in name for part in frozen_backbone_parts):
            param.requires_grad = False
            frozen_count += 1
    print("[INFO] TS Encoder 开启选择性冻结：")
    print(f"  已冻结核心骨干: {frozen_count} 个张量")
    print(f"  保持可训练 CMRG 张量: {trainable_count} 个")


def collect_cmrg_monitoring(model, cmrg_context: Optional[CMRGContext]):
    """Return per-layer CMRG gates and factorized correction strengths."""
    if not getattr(model, "cmrg_enabled", False) or cmrg_context is None:
        return {}

    layers = model.ts_encoder.ts_encoder.transformer_encoder.layers
    metrics = {}
    for layer_idx, layer in enumerate(layers):
        alpha = layer.cmrg_alpha.detach()
        metrics[f"cmrg/alpha_{layer_idx}"] = alpha.item()
        metrics[f"cmrg/rho_{layer_idx}"] = cmrg_context.frobenius_strength(alpha).item()
    return metrics
