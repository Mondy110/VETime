import torch

from training_logging import DeferredLossMetrics


def test_deferred_loss_metrics_averages_only_completed_update_window():
    """Micro-batch metrics remain tensors until an optimizer-update snapshot."""
    metrics = DeferredLossMetrics()

    metrics.add(total=torch.tensor(2.0), bce=torch.tensor(1.0), mse=torch.tensor(0.5), cl=torch.tensor(0.3), balance=torch.tensor(0.2))
    metrics.add(total=torch.tensor(4.0), bce=torch.tensor(3.0), mse=torch.tensor(1.5), cl=torch.tensor(0.7), balance=torch.tensor(0.4))

    assert metrics.consume_update_average() == {
        "total": 3.0,
        "bce": 2.0,
        "mse": 1.0,
        "cl": 0.5,
        "balance": 0.3,
    }
    assert metrics.consume_update_average() is None
    assert metrics.epoch_average() == {
        "total": 3.0,
        "bce": 2.0,
        "mse": 1.0,
        "cl": 0.5,
        "balance": 0.3,
    }
