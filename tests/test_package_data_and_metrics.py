import torch
import numpy as np

from dataset.dataloader import collate_fn as legacy_collate_fn
from evaluation.metrics import fast_get_metrics as legacy_fast_get_metrics
from loss.loss import load_balance_loss as legacy_load_balance_loss
from postprocess_runtime import resolve_postprocess_workers as legacy_workers
from training_logging import DeferredLossMetrics as LegacyDeferredLossMetrics

from vetime.data.collate import collate_fn
from vetime.losses.contrastive import load_balance_loss
from vetime.metrics.tsb import fast_get_metrics
from vetime.infrastructure.logging.training import DeferredLossMetrics
from vetime.infrastructure.logging.runtime import resolve_postprocess_workers


def test_packaged_collate_matches_legacy_collate():
    sample = (
        torch.ones(2, 1),
        torch.zeros(3, 1, 2),
        torch.zeros(2, dtype=torch.long),
        {},
        2,
        torch.zeros(1, 3),
    )
    packaged = collate_fn([sample], patch_size=2)
    legacy = legacy_collate_fn([sample], patch_size=2)
    assert packaged.keys() == legacy.keys()
    for key in ("time_series", "image", "labels", "attention_mask"):
        torch.testing.assert_close(packaged[key], legacy[key])


def test_packaged_loss_and_metrics_match_legacy_implementations():
    probs = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
    assert load_balance_loss(probs) == legacy_load_balance_loss(probs)
    scores = np.asarray([0.1, 0.8, 0.2, 0.9])
    labels = np.asarray([0, 1, 0, 1])
    assert fast_get_metrics(scores, labels) == legacy_fast_get_metrics(scores, labels)


def test_packaged_logging_and_postprocess_helpers_preserve_legacy_identity():
    assert issubclass(DeferredLossMetrics, LegacyDeferredLossMetrics)
    assert resolve_postprocess_workers(None, cpu_count=24) == legacy_workers(None, cpu_count=24)
