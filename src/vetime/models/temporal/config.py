"""Configuration for the standalone temporal model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TemporalModelConfig:
    d_model: int = 512
    d_proj: int = 256
    patch_size: int = 16
    num_layers: int = 8
    num_heads: int = 8
    d_ff_dropout: float = 0.1
    max_total_tokens: int = 8192
    num_query_tokens: int = 1
    use_rope: bool = True
    activation: str = "gelu"
    num_features: int = 1
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    cmrg_injection_mode: str = "all_layers"

    @classmethod
    def from_legacy(cls, config) -> "TemporalModelConfig":
        names = cls.__dataclass_fields__
        return cls(**{name: getattr(config, name) for name in names if hasattr(config, name)})
