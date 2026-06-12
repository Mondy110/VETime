"""
    Scripts for encoding time series into TS-MLLM Format.
"""
from einops import rearrange
import numpy as np
from typing import *
from jaxtyping import Float, Int
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Tuple, Optional


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) for linear layers.

    As per paper: "we apply Low-Rank Adaptation (LoRA) to fine-tune the linear
    projection matrices within both the Attention mechanism and the FFN modules.
    Specifically, we set the LoRA rank r = 8 and the scaling hyperparameter α = 16,
    resulting in a scaling factor of α/r = 2."

    Args:
        original_linear: The original linear layer (frozen)
        r: LoRA rank (default: 8)
        alpha: LoRA scaling factor (default: 16)
    """
    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # = 2 as per paper

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # LoRA matrices: A (down-projection) and B (up-projection)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze the original linear layer
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output (frozen weights) + LoRA adaptation
        return self.original_linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora_to_module(module: nn.Module, r: int = 8, alpha: int = 16) -> nn.Module:
    """
    Apply LoRA to all linear layers in a module.

    Args:
        module: The module to apply LoRA to
        r: LoRA rank
        alpha: LoRA scaling factor

    Returns:
        The module with LoRA applied
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Replace Linear with LoRALinear
            lora_linear = LoRALinear(child, r=r, alpha=alpha)
            setattr(module, name, lora_linear)
        else:
            # Recursively apply to child modules
            apply_lora_to_module(child, r=r, alpha=alpha)
    return module


class RMSNorm(nn.Module):
    """Root Mean Square Normalization layer."""
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.to(torch.float32).pow(2).mean(dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)

class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding for injecting positional information."""
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs  # Shape: (seq_len, dim // 2)


class BinaryAttentionBias(nn.Module):
    """Binary Variate Attention for time series data."""

    def __init__(self,
                 num_heads: Int):
        super().__init__()
        self.num_heads = num_heads
        self.emd = nn.Embedding(2, num_heads)

    def forward(self,
                query_id: Int[torch.Tensor, "batch_size q_len"],
                kv_id: Int[torch.Tensor, "batch_size kv_len"],
                ) -> Float[torch.Tensor, "batch_size num_heads q_len kv_len"]:
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        ind = ind.unsqueeze(1)  # (batch_size, 1, q_len, kv_len)
        weight = rearrange(self.emd.weight, "two num_heads -> two num_heads 1 1")  # (2, num_heads, 1, 1)
        bias = ~ind * weight[:1] + ind * weight[1:]  # (batch_size, num_heads, q_len, kv_len)
        return bias


class LlamaMLP(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, use_lora=False, lora_r=8, lora_alpha=16):
        super().__init__()
        self.hidden_size = d_model
        self.intermediate_size = dim_feedforward
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = F.gelu

        # Apply LoRA if specified (as per paper)
        if use_lora:
            self.gate_proj = LoRALinear(self.gate_proj, r=lora_r, alpha=lora_alpha)
            self.up_proj = LoRALinear(self.up_proj, r=lora_r, alpha=lora_alpha)
            self.down_proj = LoRALinear(self.down_proj, r=lora_r, alpha=lora_alpha)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class CustomTransformerEncoder(nn.Module):
    """Stack of Transformer Encoder Layers."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, num_layers, num_features,
                 use_lora=False, lora_r=8, lora_alpha=16):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                num_features=num_features,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha
            ) for _ in range(num_layers)
        ])
    def forward(self, src, freqs, src_id=None, attn_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, freqs, src_id, attn_mask=attn_mask)
        return output

class TransformerEncoderLayerWithRoPE(nn.Module):
    """Transformer Encoder Layer with RoPE and RMSNorm."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", num_features=1,
                 use_lora=False, lora_r=8, lora_alpha=16):
        super().__init__()
        self.self_attn = MultiheadAttentionWithRoPE(d_model, nhead, num_features,
                                                    use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)
        self.dropout = nn.Dropout(dropout)
        self.input_norm = RMSNorm(d_model)
        self.output_norm = RMSNorm(d_model)
        self.mlp = LlamaMLP(d_model, dim_feedforward,
                           use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src, freqs, src_id=None, attn_mask=None):
        residual = src
        src = self.input_norm(src)
        src = self.self_attn(src, src, src, freqs, src_id, src_id, attn_mask=attn_mask)
        src = src + residual
        residual = src
        src = self.output_norm(src)
        src = self.mlp(src)
        src = residual + self.dropout2(src)
        return src

class MultiheadAttentionWithRoPE(nn.Module):
    """Multi-head Attention with Rotary Positional Encoding (RoPE), non-causal by default."""
    "========== NOtice that this applies BinaryAttentionBias ==========="

    def __init__(self, embed_dim, num_heads, num_features, use_lora=False, lora_r=8, lora_alpha=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V, and output
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Apply LoRA to attention projections if specified (as per paper)
        if use_lora:
            self.q_proj = LoRALinear(self.q_proj, r=lora_r, alpha=lora_alpha)
            self.k_proj = LoRALinear(self.k_proj, r=lora_r, alpha=lora_alpha)
            self.v_proj = LoRALinear(self.v_proj, r=lora_r, alpha=lora_alpha)
            self.out_proj = LoRALinear(self.out_proj, r=lora_r, alpha=lora_alpha)

        # Binary attention bias for time series
        if num_features > 1:
            self.binary_attention_bias = BinaryAttentionBias(num_heads)

    def apply_rope(self, x, freqs):
        """Apply Rotary Positional Encoding to the input tensor."""
        B, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"
        assert freqs.shape == (seq_len, embed_dim // 2), "freqs shape mismatch"

        # Reshape for rotation: split embed_dim into pairs
        x_ = x.view(B, seq_len, embed_dim // 2, 2)
        cos = freqs.cos().unsqueeze(0)  # (1, seq_len, embed_dim // 2, 1)
        sin = freqs.sin().unsqueeze(0)  # (1, seq_len, embed_dim // 2, 1)

        # Apply rotation to each pair
        x_rot = torch.stack(
            [
                x_[..., 0] * cos - x_[..., 1] * sin,
                x_[..., 0] * sin + x_[..., 1] * cos,
            ],
            dim=-1
        )
        return x_rot.view(B, seq_len, embed_dim)

    def forward(self, query, key, value, freqs, query_id=None, kv_id=None, attn_mask=None):
        """
        Forward pass for multi-head attention with RoPE.

        Args:
            query (Tensor): Shape (B, T, C)
            key (Tensor): Shape (B, T, C)
            value (Tensor): Shape (B, T, C)
            freqs (Tensor): RoPE frequencies, shape (T, embed_dim // 2)
            query_id (Tensor, optional): Shape (B, q_len), feature IDs for query
            kv_id (Tensor, optional): Shape (B, kv_len), feature IDs for key/value
            attn_mask (Tensor, optional): Shape (B, T), True for valid positions, False for padding.

        Returns:
            Tensor: Attention output, shape (B, T, C)
        """
        B, T, C = query.shape
        assert key.shape == (B, T, C) and value.shape == (B, T, C), "query, key, value shapes must match"

        # Project inputs to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Apply RoPE to Q and K
        Q_rot = self.apply_rope(Q, freqs)
        K_rot = self.apply_rope(K, freqs)

        # Reshape for multi-head attention
        Q_rot = Q_rot.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        K_rot = K_rot.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Prepare attention mask for padding
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        else:
            attn_mask = None

        if query_id is not None and kv_id is not None:
            # Add binary attention bias
            attn_bias = self.binary_attention_bias(query_id, kv_id)  # (B, num_heads, q_len, kv_len)
            scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1)) / math.sqrt(
                self.head_dim)  # (B, num_heads, q_len, kv_len)
            scores += attn_bias
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, q_len, kv_len)
            y = torch.matmul(attn_weights, V)  # (B, num_heads, q_len, hs)

        else:
            # Compute scaled dot-product attention (non-causal) without binary bias
            # for param in self.binary_attention_bias.parameters():
            #     param.requires_grad = False
            # 注意：为了可复现性，需要确保 Flash Attention 被禁用
            # 可以在 train.py 的 set_seed() 中设置 torch.backends.cuda.enable_flash_sdp(False)
            y = F.scaled_dot_product_attention(
                Q_rot, K_rot, V,
                attn_mask=attn_mask,
                is_causal=False  # Non-causal attention for encoder
            )  # (B, nh, T, hs)

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y

