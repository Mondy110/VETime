from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class TimeSeriesConfig:
    """Configuration for time series encoder.

    Attributes:
        d_model: Dimension of model hidden states.
        d_proj: Dimension of projection layer.
        patch_size: Size of time series patches.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        d_ff_dropout: Dropout rate for feed-forward networks.
        use_rope: Whether to use Rotary Position Embedding.
        activation: Activation function name.
        num_features: Number of input features.
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning (as per paper).
        lora_r: LoRA rank (default: 8, as per paper).
        lora_alpha: LoRA alpha (default: 16, as per paper).
    """
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
    num_features: int = 8
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16


default_config_t = TimeSeriesConfig()
