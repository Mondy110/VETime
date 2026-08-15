"""Cross-modal relational guidance components."""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


class RelationDistiller(nn.Module):
    """Distill a fixed set of learnable relation tokens from visual patches."""

    def __init__(self, vision_dim, guide_dim, num_relation_tokens, num_heads):
        super().__init__()
        self.visual_projection = nn.Linear(vision_dim, guide_dim)
        self.relation_queries = nn.Parameter(torch.randn(1, num_relation_tokens, guide_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            guide_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.norm = nn.LayerNorm(guide_dim)

    def forward(self, visual_tokens):
        keys = self.visual_projection(visual_tokens)
        queries = self.relation_queries.expand(visual_tokens.size(0), -1, -1)
        distilled, _ = self.cross_attention(queries, keys, keys, need_weights=False)
        return self.norm(distilled + queries)


class CrossModalRelationGuider(nn.Module):
    """Produce unnormalized temporal-to-relation logits and metric factors."""

    def __init__(self, temporal_dim, guide_dim, num_heads, num_relation_tokens):
        super().__init__()
        if guide_dim % num_heads:
            raise ValueError("guide_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.guide_dim = guide_dim
        self.head_dim = guide_dim // num_heads
        self.temporal_proj = nn.Linear(temporal_dim, guide_dim, bias=False)
        self.relation_proj = nn.Linear(guide_dim, guide_dim, bias=False)
        self.relation_metric = nn.Parameter(torch.eye(num_relation_tokens))

    def forward(self, temporal_tokens, relation_tokens, temporal_valid_mask=None):
        batch, num_temporal, _ = temporal_tokens.shape
        num_relation = relation_tokens.size(1)
        q = self.temporal_proj(F.normalize(temporal_tokens, dim=-1))
        k = self.relation_proj(F.normalize(relation_tokens, dim=-1))
        q = q.view(batch, num_temporal, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, num_relation, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if temporal_valid_mask is not None:
            logits = logits * temporal_valid_mask[:, None, :, None].to(logits.dtype)
        factor = torch.matmul(logits, self.relation_metric)
        return logits, factor


@dataclass
class CMRGContext:
    relation_logits: torch.Tensor
    relation_factor: torch.Tensor
    temporal_valid_mask: torch.Tensor

    def correction(self):
        return torch.matmul(self.relation_factor, self.relation_logits.transpose(-2, -1))

    def frobenius_strength(self, alpha=1.0):
        """Return ``||alpha * (M Rᵀ)||_F`` using factor Gram products."""
        left = self.relation_factor
        right = self.relation_logits
        left_gram = torch.matmul(left.transpose(-2, -1), left)
        right_gram = torch.matmul(right.transpose(-2, -1), right)
        squared = (left_gram * right_gram).sum(dim=(-2, -1))
        return torch.as_tensor(alpha, device=left.device, dtype=left.dtype).abs() * squared.clamp_min(0).sum().sqrt()
