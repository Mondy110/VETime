"""Training utilities for CMRG that do not depend on the training runtime."""

from typing import Optional
import warnings

from model.CMRG import CMRGContext


def add_cmrg_injection_mode_argument(parser):
    """Register the supported CMRG layer-injection ablations."""
    parser.add_argument(
        "--cmrg_injection_mode",
        choices=["all_layers", "last_layer"],
        default="all_layers",
        help="CMRG attention injection mode",
    )


def load_model_state_compat(model, state_dict, description="model"):
    """Load legacy weights non-strictly and report compatibility gaps."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] {description} loaded (strict=False)")
    if missing:
        print(f"  Missing parameters ({len(missing)}): {list(missing)}")
    if unexpected:
        print(f"  Unexpected parameters ({len(unexpected)}): {list(unexpected)}")
    return missing, unexpected


def restore_optimizer_state_compat(optimizer, state_dict):
    """Restore optimizer state when compatible, otherwise continue safely."""
    if state_dict is None:
        warnings.warn(
            "Optimizer restore skipped: checkpoint has no optimizer state",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    try:
        optimizer.load_state_dict(state_dict)
    except (ValueError, KeyError, RuntimeError) as error:
        warnings.warn(f"Optimizer restore skipped: {error}", RuntimeWarning, stacklevel=2)
        return False
    return True


def configure_freeze_mode(model):
    """Freeze temporal backbone weights while retaining CMRG training."""
    frozen_backbone_parts = ("transformer_encoder", "embedding_layer", "rope_embedder")
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        is_cmrg_parameter = name.startswith("cmrg_") or ".cmrg_" in name
        if is_cmrg_parameter:
            if param.requires_grad:
                trainable_count += 1
        elif any(part in name for part in frozen_backbone_parts):
            param.requires_grad = False
            frozen_count += 1
    print("[INFO] TS Encoder 开启选择性冻结：")
    print(f"  已冻结核心骨干: {frozen_count} 个张量")
    print(f"  保持可训练 CMRG 张量: {trainable_count} 个")


def collect_cmrg_monitoring(model, cmrg_context: Optional[CMRGContext]):
    """Return all layer gates and active-layer factorized correction strengths."""
    if not getattr(model, "cmrg_enabled", False) or cmrg_context is None:
        return {}

    layers = model.ts_encoder.ts_encoder.transformer_encoder.layers
    metrics = {}
    for layer_idx, layer in enumerate(layers):
        alpha = layer.cmrg_alpha.detach()
        metrics[f"cmrg/alpha_{layer_idx}"] = alpha.item()

        # Inactive layers (e.g. last-layer injection) still expose their gate,
        # but do not receive CMRG context and therefore have no meaningful rho.
        if not getattr(layer, "cmrg_active", True):
            continue

        numerator = cmrg_context.frobenius_strength(alpha)
        qk_frobenius = getattr(layer.self_attn, "cmrg_qk_frobenius", None)
        if qk_frobenius is None:
            if alpha.item() == 0.0:
                metrics[f"cmrg/rho_{layer_idx}"] = 0.0
            continue
        metrics[f"cmrg/rho_{layer_idx}"] = (
            numerator / (qk_frobenius.to(numerator) + 1e-12)
        ).item()
    return metrics
