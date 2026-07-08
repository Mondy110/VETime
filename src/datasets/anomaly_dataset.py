"""
Anomaly detection dataset class for VETime.

This module provides the AnomalyDataset class that loads preprocessed time
series data from pickle files and optionally generates dual-branch image
representations on-the-fly.
"""
from typing import Tuple, List, Dict, Any, Optional
import pickle
import random

import torch
from torch.utils.data import Dataset

from src.datasets.pre_image import ts2image_1d, vico_render_timeseries


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
        Generate dual-branch image representations for time series samples.

        This function converts each time series in the data list to two images:
        1. VETime image: Time-domain rendering using ts2image_1d
        2. ViCO image: Frequency-domain rendering using vico_render_timeseries

        The image width is determined by the sequence length and patch_size,
        rounded up to the nearest multiple of patch_size.

        Args:
            data: List of sample dictionaries. Each dictionary should contain
                  at least 'time_series' key. The images, period, and padding
                  value will be added in-place.

        Returns:
            List[Dict[str, Any]]: The same data list with added keys:
                - 'image_vetime': VETime time-domain image (3, C*h_size, width)
                - 'image_vico': ViCO frequency-domain image (3, 224, 224)
                - 'period': Detected period (integer)
                - 'padding_value': Padding values for VETime image

        Note:
            This function modifies the input data list in-place and also
            returns it for convenience.
        """
        # 串行处理（更稳定，避免多进程内存问题）
        for idx, data0 in enumerate(data):
            target_length = ((len(data0['time_series']) + self.patch_size - 1) // self.patch_size) * self.patch_size

            # === VETime 时域渲染 (现有) ===
            img_vetime, period, padding_value = ts2image_1d(
                data0['time_series'], target_length, self.patch_size
            )

            # === ViCO 频域渲染 (新增) ===
            img_vico = vico_render_timeseries(
                data0['time_series'], period, img_size=224
            )

            # 存储两分支图像
            data[idx]['image_vetime'] = img_vetime  # [3, C*h_size, width]
            data[idx]['image_vico'] = img_vico      # [3, 224, 224]
            data[idx]['period'] = period
            data[idx]['padding_value'] = padding_value

            # 删除旧的 'image' 键（避免混淆）
            if 'image' in data[idx]:
                del data[idx]['image']

        return data

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict, int, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing:
                - time_series: Time series data as float32 tensor (L, C)
                - normal_time_series: Normal reference time series (L, C)
                - image_vetime: VETime time-domain image as float32 tensor (3, H, W)
                - image_vico: ViCO frequency-domain image as float32 tensor (3, 224, 224)
                - labels: Anomaly labels as long tensor (L,)
                - attribute: Metadata dictionary
                - period: Detected period (int)
                - padding_value: Padding values as float32 tensor (3, C, 1)
        """
        sample = self.data[idx]
        # 双分支图像
        img_vetime_tensor = torch.tensor(sample['image_vetime'], dtype=torch.float32)
        img_vico_tensor = torch.tensor(sample['image_vico'], dtype=torch.float32)
        time_series = torch.tensor(sample['time_series'], dtype=torch.float32)
        normal_time_series = torch.tensor(sample['normal_time_series'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.long)
        attribute = sample['attribute']
        period = sample['period']
        padding_value = torch.tensor(sample['padding_value'], dtype=torch.float32)
        return time_series, normal_time_series, img_vetime_tensor, img_vico_tensor, labels, attribute, period, padding_value
