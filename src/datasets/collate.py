"""
Collation utilities for batching anomaly detection samples.

This module provides the collate function and supporting classes for
assembling variable-length time series samples into padded batches,
with random masking for self-supervised learning and dynamic batch
sizing to maximize GPU utilization.
"""
from typing import Tuple, List, Dict, Any, Optional, Union
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

from src.datasets.masking import create_random_mask


def collate_fn(
    batch: List[Tuple],
    patch_size: int
) -> Dict[str, Union[torch.Tensor, List, Tuple]]:
    """
    Collate function for batching anomaly detection samples with dual-branch images.

    This function processes a batch of samples from AnomalyDataset and:
    1. Concatenates all time series and computes global mean/std for normalization
    2. Pads all sequences to the same length (multiple of patch_size)
    3. Generates attention masks for valid sequence positions
    4. Applies random masking for self-supervised learning
    5. Pads VETime images to match the target width
    6. Stacks ViCO images (already fixed 224x224 size)

    Args:
        batch: List of samples from AnomalyDataset.__getitem__. Each sample is
               a tuple of (time_series, normal_time_series, img_vetime, img_vico,
               labels, attribute, period, padding_value).
        patch_size: Size of patches for masking and padding alignment.

    Returns:
        A dictionary containing:
            - 'time_series': Padded time series tensor (B, L_max, C)
            - 'normal_time_series': Padded normal reference tensor (B, L_max, C)
            - 'mask_time_series': Time series with random patches masked (B, L_max, C)
            - 'image': Padded VETime time-domain image tensor (B, 3, H, W_max)
            - 'image_vico': ViCO frequency-domain image tensor (B, 3, 224, 224)
            - 'mask': Boolean mask indicating masked positions (B, L_max)
            - 'labels': Padded label tensor (B, L_max) with -1 for padding
            - 'attention_mask': Boolean mask for valid positions (B, L_max)
            - 'period': Tuple of periods for each sample in batch
            - 'padding_value': Tensor of padding values (B, 3, C, 1)

    Note:
        - Time series are normalized using batch-wide statistics
        - Labels are padded with -1 (ignored in loss computation)
        - Random masking applies mask_ratio=0.3 to valid sequence regions only
        - VETime images are padded with adaptive padding values
        - ViCO images are already 224x224, just stacked
    """
    time_series_list, normal_time_series_list, img_vetime_list, img_vico_list, labels_list, attribute_list, period, padding_value = zip(*batch)

    if time_series_list[0].ndim == 1:
        time_series_tensors = [ts.unsqueeze(-1) for ts in time_series_list]
        normal_time_series_tensors = [nts.unsqueeze(-1) for nts in normal_time_series_list]
    else:
        time_series_tensors = [ts for ts in time_series_list]
        normal_time_series_tensors = [nts for nts in normal_time_series_list]

    concatenated = torch.cat(time_series_tensors, dim=0)
    mean = concatenated.mean(dim=0, keepdim=True)
    std = concatenated.std(dim=0, keepdim=True) + 1e-4
    time_series_tensors = [(ts - mean) / std for ts in time_series_tensors]
    normal_time_series_tensors = [(nts - mean) / std for nts in normal_time_series_tensors]

    labels = [label for label in labels_list]
    lengths = [t.size(0) for t in labels]
    max_len = max(lengths)
    max_idx = lengths.index(max_len)
    target_length = ((max_len + patch_size - 1) // patch_size) * patch_size

    def padding_to_target_length(list0, value):
        original_tensor = list0[max_idx]
        pad_shape = [0, 0] * original_tensor.dim()
        pad_shape[-1] = target_length - max_len
        padded_tensor = torch.nn.functional.pad(original_tensor, pad=pad_shape, mode='constant', value=value)
        list0[max_idx] = padded_tensor
        return torch.nn.utils.rnn.pad_sequence(list0, batch_first=True, padding_value=value)

    padded_time_series = padding_to_target_length(time_series_tensors, 0.0)
    normal_time_series_tensors = padding_to_target_length(normal_time_series_tensors, 0.0)
    padded_labels = padding_to_target_length(labels, -1)

    # VETime 分支: 需要自适应 padding
    image_inputs_vetime = image_right_padding(img_vetime_list, target_length, padding_value)

    # ViCO 分支: 已经是固定 224x224，直接 stack
    image_inputs_vico = torch.stack(img_vico_list)  # (B, 3, 224, 224)

    sequence_lengths = [ts.size(0) for ts in time_series_tensors]
    B, max_seq_len, num_features = padded_time_series.shape
    attention_mask = torch.ones(B, max_seq_len, dtype=torch.bool)

    for i, length in enumerate(sequence_lengths):
        attention_mask[i, length:] = False

    mask_time_series, mask = create_random_mask(padded_time_series, attention_mask, patch_size)
    normal_time_series_tensors, mask = create_random_mask(normal_time_series_tensors, attention_mask, patch_size)

    return {
        'time_series': padded_time_series,
        'normal_time_series': normal_time_series_tensors,
        'mask_time_series': mask_time_series,
        'image': image_inputs_vetime,  # VETime 时域图像 (保持旧 key 名便于兼容)
        'image_vico': image_inputs_vico,  # ViCO 频域图像
        'mask': mask,
        'labels': padded_labels,
        'attention_mask': attention_mask,
        'period': period,
        'padding_value': padding_value,
    }


def image_right_padding(
    imgs: List[torch.Tensor],
    max_width: int,
    p_values: torch.Tensor
) -> torch.Tensor:
    """
    Pad images on the right side to match target width.

    This function extends images that are shorter than max_width by padding
    on the right side. The padding uses the provided padding values to maintain
    consistency with the time series padding strategy.

    Args:
        imgs: List of image tensors, each of shape (3, H, W_i) where W_i may
              vary across samples.
        max_width: Target width for all images. Images with W < max_width will
                   be padded; images with W >= max_width remain unchanged.
        p_values: Tensor of padding values with shape (B, 3, C, 1) or compatible.
                  Each sample's padding value is used to fill its padded region.

    Returns:
        torch.Tensor: Stacked tensor of padded images with shape (B, 3, H, max_width).
                      All images have the same width after processing.

    Note:
        - Padding is applied only on the right side (width dimension)
        - Padding values are transposed to match the image channel format
    """
    padded_images = []
    for i in range(len(imgs)):
        img = imgs[i]
        C, H_size, W = img.shape
        p_value = p_values[i]
        if max_width > W:
            right_padding = max_width - img.shape[2]
            padding = (0, right_padding, 0, 0)
            padded_img = F.pad(img.unsqueeze(0), padding, mode='constant', value=0).squeeze(0)
            padded_img[:, :, W:] = p_value.T[:, :, None]
        else:
            padded_img = img
        padded_images.append(padded_img)
    return torch.stack(padded_images)


class DynamicLengthBatchSampler(Sampler):
    """
    按序列长度动态调整 batch_size 的采样器。

    核心思路：保持每 batch 的总 token 数（B * L_max）大致恒定，
    短样本时自动增大 batch_size 充分利用 GPU，长样本时保持原始 batch_size。

    同时通过 padding_ratio 约束同一 batch 内的 padding 浪费：
    当新样本长度超过当前 batch 最短样本的 padding_ratio 倍时，强制切分，
    避免 1K 样本和 68K 样本同 batch 导致大量 padding 浪费。

    数据集需已按长度排序（AnomalyDataset 默认行为）。

    Args:
        lengths: 每个样本的序列长度列表（已排序）。
        max_tokens_per_batch: 每 batch 允许的最大 token 数
                              (B * L_max <= max_tokens_per_batch)。
        min_batch_size: 最小 batch_size，防止梯度噪声过大。默认 32。
        max_batch_size: 最大 batch_size，防止极短样本时 batch 过大。默认 256。
        padding_ratio: 同一 batch 内允许的最大/最小长度比。
                       超过此比例时强制切 batch，减少 padding 浪费。默认 4.0。
        drop_last: 是否丢弃最后不满一个 batch 的数据。默认 True。
        effective_batch_size: 目标有效 batch_size，用于计算梯度累积步数。
                              设为 0 则不启用（accumulation_steps 固定为 1）。默认 0。
        shuffle_each_epoch: 是否在每个 epoch 内打乱同长度区间的样本顺序。
                            保持长度排序的宏观顺序不变。默认 False。
        seed: 用于 shuffle 的随机种子。默认 42。
    """

    def __init__(
        self,
        lengths: List[int],
        max_tokens_per_batch: int,
        min_batch_size: int = 32,
        max_batch_size: int = 256,
        padding_ratio: float = 4.0,
        drop_last: bool = True,
        effective_batch_size: int = 0,
        shuffle_each_epoch: bool = False,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.max_tokens = max_tokens_per_batch
        self.min_bs = min_batch_size
        self.max_bs = max_batch_size
        self.padding_ratio = padding_ratio
        self.drop_last = drop_last
        self.effective_bs = effective_batch_size
        self.shuffle = shuffle_each_epoch
        self.seed = seed
        self.epoch = 0

        # 预计算所有 batch
        self._batches = self._compute_batches(self.lengths)

    def _compute_batches(self, lengths: List[int]) -> List[List[int]]:
        """
        根据长度列表预计算所有 batch 的索引划分。

        双重约束：
        1. B * L_max <= max_tokens（显存约束）
        2. L_max / L_min <= padding_ratio（padding 浪费约束）

        遍历已排序的长度列表，逐步累加样本到当前 batch，
        当任一约束被打破时切分出当前 batch。
        """
        batches = []
        current_batch = []
        current_max_len = 0
        current_min_len = float('inf')

        for idx, length in enumerate(lengths):
            new_max_len = max(current_max_len, length)
            new_min_len = min(current_min_len, length)
            new_bs = len(current_batch) + 1

            should_split = False
            if new_bs > 1:
                # 约束1: token 预算
                if new_bs * new_max_len > self.max_tokens:
                    should_split = True
                # 约束2: padding 浪费（L_max / L_min 不超过 padding_ratio）
                elif new_max_len / new_min_len > self.padding_ratio:
                    should_split = True
                # 约束3: batch_size 上限
                elif new_bs > self.max_bs:
                    should_split = True

            if should_split:
                if len(current_batch) >= self.min_bs or not self.drop_last:
                    batches.append(current_batch)
                current_batch = [idx]
                current_max_len = length
                current_min_len = length
            else:
                current_batch.append(idx)
                current_max_len = new_max_len
                current_min_len = new_min_len

        # 处理最后一个 batch
        if current_batch:
            if self.drop_last and len(current_batch) < self.min_bs:
                pass  # 丢弃
            else:
                batches.append(current_batch)

        return batches

    def get_accumulation_steps(self) -> int:
        """
        返回推荐的梯度累积步数，用于保证有效 batch_size 一致。

        计算方式：effective_batch_size / median(actual_batch_size)
        向上取整到最近的 2 的幂次，使累积更均匀。
        """
        if self.effective_bs <= 0:
            return 1
        median_bs = int(np.median([len(b) for b in self._batches]))
        if median_bs <= 0:
            return 1
        steps = max(1, self.effective_bs // median_bs)
        return steps

    def get_batch_info(self) -> str:
        """返回 batch 统计信息字符串，用于日志输出。"""
        batch_sizes = [len(b) for b in self._batches]
        max_lens = [max(self.lengths[i] for i in b) for b in self._batches]
        return (
            f"DynamicBatchSampler: {len(self._batches)} batches, "
            f"bs range [{min(batch_sizes)}, {max(batch_sizes)}], "
            f"len range [{min(max_lens)}, {max(max_lens)}], "
            f"median bs={int(np.median(batch_sizes))}, "
            f"accumulation_steps={self.get_accumulation_steps()}"
        )

    def __iter__(self):
        batches = list(self._batches)

        if self.shuffle:
            # 在保持长度排序的前提下，打乱相邻同长度样本的顺序
            rng = random.Random(self.seed + self.epoch)
            # 将索引按长度分组，组内打乱
            i = 0
            while i < len(batches):
                # 找到长度相近的连续 batch 区间
                j = i + 1
                while j < len(batches):
                    len_i = max(self.lengths[idx] for idx in batches[i])
                    len_j = max(self.lengths[idx] for idx in batches[j])
                    if len_j > len_i * 1.5:  # 长度差异超过 50% 视为不同区间
                        break
                    j += 1
                # 打乱 [i, j) 区间内的 batch 顺序
                segment = batches[i:j]
                rng.shuffle(segment)
                batches[i:j] = segment
                i = j

        self.epoch += 1

        for batch in batches:
            yield batch

    def __len__(self):
        return len(self._batches)
