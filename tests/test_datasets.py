"""
Tests for the src.datasets module.

Verifies that the migrated dataset pipeline works correctly:
- create_random_mask produces correct shapes and respects attention masks
- collate_fn returns all expected keys
"""
import torch
import numpy as np


def test_create_random_mask_shape():
    """Verify create_random_mask output shapes and that padding positions are not masked."""
    from src.datasets.masking import create_random_mask
    B, L, C = 2, 128, 1
    ts = torch.randn(B, L, C)
    att_mask = torch.ones(B, L, dtype=torch.long)
    att_mask[0, 100:] = 0
    masked_ts, mask = create_random_mask(ts, att_mask, patch_size=16, mask_ratio=0.3)
    assert masked_ts.shape == ts.shape
    assert mask.shape == (B, L)
    assert mask[0, 100:].sum() == 0  # padding positions should not be masked


def test_collate_fn_output_keys():
    """Verify collate_fn returns all expected dictionary keys."""
    from src.datasets.collate import collate_fn
    batch = []
    for _ in range(2):
        C = 1  # univariate
        ts = torch.randn(200, C, dtype=torch.float32)
        normal_ts = ts.clone()
        # ts2image_1d produces image shape (3, C*h_size, W) with h_size=1 -> (3, 1, W)
        img_vetime = torch.randint(0, 255, (3, C, 200), dtype=torch.float32)
        labels = torch.zeros(200, dtype=torch.long)
        attribute = {'key': 'value'}
        period = 10
        # ts2image_1d produces pad_values shape (C, 3) as uint8, cast to float32
        padding_value = torch.zeros(C, 3, dtype=torch.float32)
        batch.append((ts, normal_ts, img_vetime, labels, attribute, period, padding_value))
    result = collate_fn(batch, patch_size=16)
    expected_keys = {'time_series', 'time_series_raw', 'normal_time_series', 'mask_time_series',
                     'image', 'mask', 'labels', 'attention_mask',
                     'period', 'padding_value'}
    assert expected_keys.issubset(set(result.keys()))


def test_backward_compat_imports():
    """Verify that old import paths still work via backward-compat re-exports."""
    from dataset.dataloader import AnomalyDataset, collate_fn, DynamicLengthBatchSampler, create_random_mask
    assert AnomalyDataset is not None
    assert collate_fn is not None
    assert DynamicLengthBatchSampler is not None
    assert create_random_mask is not None


def test_src_datasets_init_reexports():
    """Verify that src.datasets.__init__ re-exports work."""
    from src.datasets import AnomalyDataset, collate_fn, DynamicLengthBatchSampler, create_random_mask
    assert AnomalyDataset is not None
    assert collate_fn is not None
    assert DynamicLengthBatchSampler is not None
    assert create_random_mask is not None
