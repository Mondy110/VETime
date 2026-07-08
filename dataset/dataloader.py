"""
Data loading utilities for VETime anomaly detection.

DEPRECATED: This module now re-exports from src.datasets for backward
compatibility. New code should import directly from src.datasets:

    from src.datasets import AnomalyDataset, collate_fn, DynamicLengthBatchSampler, create_random_mask

This module provides dataset classes and collate functions for loading and
preprocessing time series anomaly detection data. It supports:
- Loading preprocessed datasets from pickle files
- Converting time series to image representations on-the-fly
- Padding and batching sequences of variable lengths
- Random masking for self-supervised pretraining
"""

# Backward compatibility: re-export from new location
from src.datasets.anomaly_dataset import AnomalyDataset
from src.datasets.collate import collate_fn, DynamicLengthBatchSampler, image_right_padding
from src.datasets.masking import create_random_mask
from src.datasets.pre_image import ts2image_1d, ts2image_Test, vico_render_timeseries

__all__ = [
    'AnomalyDataset',
    'collate_fn',
    'DynamicLengthBatchSampler',
    'image_right_padding',
    'create_random_mask',
    'ts2image_1d',
    'ts2image_Test',
    'vico_render_timeseries',
]
