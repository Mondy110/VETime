"""
Random masking utilities for self-supervised pretraining.

This module provides patch-level random masking functions used by the
collate pipeline to generate masked time series inputs for denoising
reconstruction tasks.
"""
from typing import Tuple
import torch


def create_random_mask(
    time_series: torch.Tensor,
    attention_mask: torch.Tensor,
    patch_size: int = 14,
    mask_ratio: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create random mask for time series patches in self-supervised learning.

    Vectorized implementation: generates per-sample random scores for all patches,
    then selects the top-K scoring patches to mask. This avoids the slow Python
    for-loop over samples and patches.

    Args:
        time_series: Input time series tensor of shape (B, L, C).
        attention_mask: Boolean tensor of shape (B, L) indicating valid positions.
        patch_size: Size of each patch for masking. Default: 14.
        mask_ratio: Ratio of patches to mask within valid regions. Default: 0.3.

    Returns:
        A tuple containing:
            masked_time_series: Time series with masked positions replaced by
                                small Gaussian noise. Same shape as input (B, L, C).
            mask: Boolean tensor of shape (B, L) indicating masked positions.
    """
    B, L, C = time_series.shape
    num_patches = (L + patch_size - 1) // patch_size

    # 每个 patch 的有效标记：只要 patch 内有任一有效位置就视为有效 patch
    # patch_mask: (B, num_patches)
    patch_mask = attention_mask[:, :num_patches * patch_size].reshape(B, num_patches, patch_size).any(dim=2)

    # 每个样本要 mask 的 patch 数
    num_valid = patch_mask.sum(dim=1)  # (B,)
    num_to_mask = (num_valid.float() * mask_ratio).clamp(min=1).long()
    num_to_mask = num_to_mask.clamp(max=num_valid)

    # 对无效 patch 赋极低分数，确保不会被选中
    scores = torch.rand(B, num_patches)
    scores[~patch_mask] = -1.0

    # 选 top-K 分数的 patch 作为 masked
    K = num_to_mask.max().item()
    if K > 0:
        _, topk_indices = scores.topk(K, dim=1)  # (B, K)

        # 构建patch级别的mask，再展开到token级别
        patch_level_mask = torch.zeros(B, num_patches, dtype=torch.bool)
        for i in range(B):
            patch_level_mask[i, topk_indices[i, :num_to_mask[i]]] = True

        # 展开 patch mask -> token mask
        mask = patch_level_mask.unsqueeze(2).expand(-1, -1, patch_size).reshape(B, num_patches * patch_size)
        mask = mask[:, :L]  # 截断到原始长度
    else:
        mask = torch.zeros(B, L, dtype=torch.bool)

    mask = mask & attention_mask

    masked_time_series = time_series.clone()
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, C)
    masked_time_series[mask_expanded] = torch.randn_like(masked_time_series[mask_expanded]) * 0.1

    return masked_time_series, mask
