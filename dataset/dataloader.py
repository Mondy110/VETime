"""
Data loading utilities for VETime anomaly detection.

This module provides dataset classes and collate functions for loading and
preprocessing time series anomaly detection data. It supports:
- Loading preprocessed datasets from pickle files
- Converting time series to image representations on-the-fly
- Padding and batching sequences of variable lengths
- Random masking for self-supervised pretraining
"""
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, Sampler
import random

from dataset.pre_image import ts2image_1d


class AnomalyDataset(Dataset):
    """
    PyTorch Dataset for time series anomaly detection.

    This dataset class loads preprocessed time series data from pickle files
    and optionally generates image representations on-the-fly. It supports
    train/test split and filters out very short sequences.

    The dataset expects pickle files containing a list of dictionaries, where
    each dictionary represents a sample with keys:
        - 'time_series': numpy array of shape (L, C)
        - 'normal_time_series': numpy array for normal reference
        - 'labels': numpy array of anomaly labels (0=normal, 1=anomaly)
        - 'attribute': metadata dictionary

    Args:
        dataset_dir: Path to the pickle file containing the dataset.
        patch_size: Size of patches for image generation. Used to determine
                    target image width.
        gen_image: If True, generate image representations for all samples
                   during initialization. Default: True.
        split: Data split to use. 'train' uses all data, 'test' uses the
               last (1 - train_ratio) portion. Default: 'train'.
        train_ratio: Ratio of data to use for training when split='test'.
                     Only used when split='test'. Default: 0.95.
        seed: Random seed for shuffling indices. Default: 42.
        name: Optional name identifier for the dataset. Default: None.

    Attributes:
        data: List of sample dictionaries after filtering and splitting.
        image_type: Type of image representation ('RGB').
        image_h: Height of each channel tile in generated images.

    """

    def __init__(
        self,
        dataset_dir: str,
        patch_size: int,
        gen_image: bool = True,
        split: str = 'train',
        train_ratio: float = 0.95,
        seed: int = 42,
        name: Optional[str] = None
    ):
        """
        Initialize the AnomalyDataset.

        Args:
            dataset_dir: Path to the pickle file containing the dataset.
            patch_size: Size of patches for image generation.
            gen_image: If True, generate image representations during init.
            split: Data split ('train' or 'test').
            train_ratio: Ratio of data for training split.
            seed: Random seed for reproducibility.
            name: Optional dataset name identifier.
        """
        file_path = dataset_dir
        self.image_h = patch_size
        self.gen_image = gen_image
        self.patch_size = patch_size
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        random.seed(seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        num_train = int(len(dataset) * train_ratio)
        if split == 'train':
            selected_indices = indices[:num_train]  # 修复：使用前 train_ratio 的数据
        elif split == 'test':
            selected_indices = indices[num_train:]  # 后 (1-train_ratio) 作为验证集
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.data = [dataset[i] for i in selected_indices]
        self.data = [x for x in self.data if len(x['time_series']) > 100]
        self.data.sort(key=lambda x: len(x['time_series']))

        self.image_type = 'RGB'
        self.name = name
        if self.gen_image:
            self.generate_image(self.data)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def generate_image(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate image representations for time series samples.

        This function converts each time series in the data list to an image
        using ts2image_1d. The image width is determined by the sequence length
        and patch_size, rounded up to the nearest multiple of patch_size.

        Args:
            data: List of sample dictionaries. Each dictionary should contain
                  at least 'time_series' key. The image, period, and padding
                  value will be added in-place.

        Returns:
            List[Dict[str, Any]]: The same data list with added keys:
                - 'image': Generated image array of shape (3, C*h_size, width)
                - 'period': Detected period (integer)
                - 'padding_value': Padding values for the image

        Note:
            This function modifies the input data list in-place and also
            returns it for convenience.
        """
        # 串行处理（更稳定，避免多进程内存问题）
        for idx, data0 in enumerate(data):
            target_length = ((len(data0['time_series']) + self.patch_size - 1) // self.patch_size) * self.patch_size
            img, period, padding_value = ts2image_1d(data0['time_series'], target_length, self.patch_size)
            data[idx]['image'] = img
            data[idx]['period'] = period
            data[idx]['padding_value'] = padding_value
        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict, int, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing:
                - time_series: Time series data as float32 tensor (L, C)
                - normal_time_series: Normal reference time series (L, C)
                - image: Image representation as float32 tensor (3, H, W)
                - labels: Anomaly labels as long tensor (L,)
                - attribute: Metadata dictionary
                - period: Detected period (int)
                - padding_value: Padding values as float32 tensor (3, C, 1)
        """
        sample = self.data[idx]
        img_tensor = torch.tensor(sample['image'], dtype=torch.float32)
        time_series = torch.tensor(sample['time_series'], dtype=torch.float32)
        normal_time_series = torch.tensor(sample['normal_time_series'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.long)
        attribute = sample['attribute']
        period = sample['period']
        padding_value = torch.tensor(sample['padding_value'], dtype=torch.float32)
        return time_series, normal_time_series, img_tensor, labels, attribute, period, padding_value


def collate_fn(
    batch: List[Tuple],
    patch_size: int
) -> Dict[str, Union[torch.Tensor, List, Tuple]]:
    """
    Collate function for batching anomaly detection samples.

    This function processes a batch of samples from AnomalyDataset and:
    1. Concatenates all time series and computes global mean/std for normalization
    2. Pads all sequences to the same length (multiple of patch_size)
    3. Generates attention masks for valid sequence positions
    4. Applies random masking for self-supervised learning
    5. Pads images to match the target width

    Args:
        batch: List of samples from AnomalyDataset.__getitem__. Each sample is
               a tuple of (time_series, normal_time_series, img_tensor, labels,
               attribute, period, padding_value).
        patch_size: Size of patches for masking and padding alignment.

    Returns:
        A dictionary containing:
            - 'time_series': Padded time series tensor (B, L_max, C)
            - 'normal_time_series': Padded normal reference tensor (B, L_max, C)
            - 'mask_time_series': Time series with random patches masked (B, L_max, C)
            - 'image': Padded image tensor (B, 3, H, W_max)
            - 'mask': Boolean mask indicating masked positions (B, L_max)
            - 'labels': Padded label tensor (B, L_max) with -1 for padding
            - 'attention_mask': Boolean mask for valid positions (B, L_max)
            - 'period': Tuple of periods for each sample in batch
            - 'padding_value': Tensor of padding values (B, 3, C, 1)

    Note:
        - Time series are normalized using batch-wide statistics
        - Labels are padded with -1 (ignored in loss computation)
        - Random masking applies mask_ratio=0.3 to valid sequence regions only
    """
    time_series_list, normal_time_series_list, img_tensor, labels_list, attribute_list, period, padding_value = zip(*batch)
    
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

    image_inputs = image_right_padding(img_tensor, target_length, padding_value)
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
        'image': image_inputs,
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
