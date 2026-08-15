import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Tuple, Optional
from model.TS_encoder.encoding_utils import CustomTransformerEncoder, RotaryEmbedding

from dataclasses import dataclass, field
from typing import Dict, Optional

from model.CMRG import CMRGContext


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
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning.
        lora_r: LoRA rank (default: 8, as per paper).
        lora_alpha: LoRA alpha (default: 16, as per paper).
    """
    d_model: int = 512
    d_proj: int = 256
    patch_size: int = 14
    num_query_tokens: int = 1
    num_layers: int = 8
    num_heads: int = 8
    d_ff_dropout: float = 0.1
    use_rope: bool = True
    activation: str = "gelu"
    num_features: int = 1
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    cmrg_enabled: bool = False
    cmrg_num_relation_tokens: int = 16
    cmrg_guide_dim: int = 512
    cmrg_num_heads: int = 8
    cmrg_metric_init: str = "identity"
    cmrg_gate_init: float = 0.0
    cmrg_injection_mode: str = "all_layers"
    cmrg_factorized: bool = True


@dataclass
class PreparedTimeSeriesInputs:
    """Patch-level temporal encoder inputs and metadata."""

    embedded_patches: torch.Tensor
    freqs: Optional[torch.Tensor]
    feature_id: torch.Tensor
    full_mask: torch.Tensor
    batch_size: int
    seq_len: int
    num_features: int
    num_patches: int


class TimeSeriesEncoder(nn.Module):
    """
    Time Series Encoder with PatchTST-like patching, RoPE, and LoRA support.

    Args:
        d_model (int): Model dimension
        d_proj (int): Projection dimension
        patch_size (int): Size of each patch
        num_layers (int): Number of encoder layers
        num_heads (int): Number of attention heads
        d_ff_dropout (float): Dropout rate
        max_total_tokens (int): Maximum sequence length
        use_rope (bool): Use RoPE if True
        num_features (int): Number of features in the time series
        activation (str): "relu" or "gelu"
        use_lora (bool): Use LoRA for parameter-efficient fine-tuning (as per paper)
        lora_r (int): LoRA rank (default: 8, as per paper)
        lora_alpha (int): LoRA alpha (default: 16, as per paper)

    Inputs:
        time_series (Tensor): Shape (batch_size, seq_len, num_features)
        mask (Tensor): Shape (batch_size, seq_len)

    Outputs:
        local_embeddings (Tensor): Shape (batch_size, seq_len, num_features, d_proj)
    """
    def __init__(self, d_model=512, d_proj=256, patch_size=14, num_layers=8, num_heads=8,
                 d_ff_dropout=0.1, max_total_tokens=8192, use_rope=True, num_features=1,
                 activation="gelu", use_lora=True, lora_r=8, lora_alpha=16, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_proj = d_proj
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff_dropout = d_ff_dropout
        self.max_total_tokens = max_total_tokens
        self.use_rope = use_rope
        self.num_features = num_features
        self.activation = activation
        self.use_lora = use_lora

        # Patch embedding layer
        self.embedding_layer = nn.Linear(patch_size, d_model)

        if use_rope:
            # Initialize RoPE and custom encoder with LoRA support
            self.rope_embedder = RotaryEmbedding(d_model)
            self.transformer_encoder = CustomTransformerEncoder(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=d_ff_dropout,
                activation=activation,
                num_layers=num_layers,
                num_features=num_features,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha
            )
        else:
            # Standard encoder without RoPE
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=d_ff_dropout,
                batch_first=True,
                activation=activation
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection layers
        self.projection_layer = nn.Linear(d_model, patch_size * d_proj)
        self._init_parameters()

        # Gradient checkpointing flag
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Use Xavier uniform for GELU activation
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def prepare_inputs(self, time_series, mask):
        """Create patch embeddings and metadata for temporal encoding."""
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)
        device = time_series.device
        B, seq_len, num_features = time_series.size()
        assert num_features == self.num_features, f"Number of features mismatch with data: {num_features} vs param: {self.num_features}"
        assert mask.size() == (B, seq_len), f"Mask shape mismatch: {mask.size()} vs {(B, seq_len)}"

        # Pad sequence to be divisible by patch_size
        padded_length = math.ceil(seq_len / self.patch_size) * self.patch_size
        if padded_length > seq_len:
            pad_amount = padded_length - seq_len
            time_series = F.pad(time_series, (0, 0, 0, pad_amount), value=0)
            mask = F.pad(mask, (0, pad_amount), value=0)

        # Convert to patches
        num_patches = padded_length // self.patch_size
        total_length = num_patches * num_features
        patches = time_series.view(B, num_patches, self.patch_size, num_features)
        patches = patches.permute(0, 3, 1, 2).contiguous()  # (B, num_features, num_patches, patch_size)
        patches = patches.view(B, num_features * num_patches, self.patch_size)  # (B, L, patch_size)
        # Create feature IDs for patches
        feature_id = torch.arange(num_features, device=device).repeat_interleave(
            num_patches)  # (num_features * num_patches = L,)
        feature_id = feature_id.unsqueeze(0).expand(B, -1)  # (B, L)

        # Embed patches
        embedded_patches = self.embedding_layer(patches)  # (B, L, d_model)

        # Create patch-level mask
        mask = mask.view(B, num_patches, self.patch_size)
        patch_mask = mask.sum(dim=-1) > 0  # (B, num_patches)
        full_mask = patch_mask.unsqueeze(1).expand(-1, num_features, -1)  # (B, num_features, num_patches)
        full_mask = full_mask.reshape(B, num_features * num_patches)  # (B, L)

        # Generate RoPE frequencies if applicable
        if self.use_rope:
            freqs = self.rope_embedder(total_length).to(device)
        else:
            freqs = None

        return PreparedTimeSeriesInputs(
            embedded_patches=embedded_patches,
            freqs=freqs,
            feature_id=feature_id,
            full_mask=full_mask,
            batch_size=B,
            seq_len=seq_len,
            num_features=num_features,
            num_patches=num_patches,
        )

    def _run_transformer(self, prepared, cmrg_context=None):
        if prepared.num_features > 1:
            return self.transformer_encoder(
                prepared.embedded_patches,
                freqs=prepared.freqs,
                src_id=prepared.feature_id,
                attn_mask=prepared.full_mask,
                cmrg_context=cmrg_context,
            )
        return self.transformer_encoder(
            prepared.embedded_patches,
            freqs=prepared.freqs,
            attn_mask=prepared.full_mask,
            cmrg_context=cmrg_context,
        )

    def encode_prepared(self, prepared, cmrg_context: Optional[CMRGContext] = None):
        """Encode prepared patches, optionally with cross-modal guidance."""
        if self._gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(
                lambda embedded_patches: self._run_transformer(
                    PreparedTimeSeriesInputs(
                        embedded_patches=embedded_patches,
                        freqs=prepared.freqs,
                        feature_id=prepared.feature_id,
                        full_mask=prepared.full_mask,
                        batch_size=prepared.batch_size,
                        seq_len=prepared.seq_len,
                        num_features=prepared.num_features,
                        num_patches=prepared.num_patches,
                    ),
                    cmrg_context,
                ),
                prepared.embedded_patches,
                use_reentrant=False,
            )
        return self._run_transformer(prepared, cmrg_context)

    def project_local_embeddings(self, patch_embeddings, prepared):
        """Project encoded patches back to local time-step embeddings."""
        patch_proj = self.projection_layer(patch_embeddings)
        local_embeddings = patch_proj.view(
            prepared.batch_size,
            prepared.num_features,
            prepared.num_patches,
            self.patch_size,
            self.d_proj,
        )
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4)
        return local_embeddings.view(
            prepared.batch_size, -1, prepared.num_features, self.d_proj
        )[:, :prepared.seq_len, :, :]

    def forward(self, time_series, mask, cmrg_context: Optional[CMRGContext] = None):
        """Forward pass to generate local embeddings."""
        prepared = self.prepare_inputs(time_series, mask)
        patch_embeddings = self.encode_prepared(prepared, cmrg_context)

        # Extract and project local embeddings
        local_embeddings = self.project_local_embeddings(patch_embeddings, prepared)
        return patch_embeddings, local_embeddings, prepared.full_mask
