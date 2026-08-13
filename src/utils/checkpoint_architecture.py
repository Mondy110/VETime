"""Checkpoint helpers that keep VETime's decoder architecture reproducible."""

from collections.abc import Mapping
from typing import Any, Dict


def checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return model weights from either a legacy or metadata-wrapped checkpoint."""
    state_dict = checkpoint.get("model_state_dict")
    return state_dict if isinstance(state_dict, Mapping) else checkpoint


def checkpoint_model_config(checkpoint: Mapping[str, Any]) -> Dict[str, bool]:
    """Read decoder architecture metadata, falling back to legacy key inference."""
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, Mapping) and "use_query_decoder" in model_config:
        return {"use_query_decoder": bool(model_config["use_query_decoder"])}

    state_dict = checkpoint_state_dict(checkpoint)
    return {
        "use_query_decoder": any(
            key.startswith("query_decoder.") for key in state_dict
        )
    }


def make_model_checkpoint(state_dict: Mapping[str, Any], *, use_query_decoder: bool) -> Dict[str, Any]:
    """Wrap model weights with the architecture flag required for evaluation."""
    return {
        "model_state_dict": state_dict,
        "model_config": {"use_query_decoder": use_query_decoder},
    }
