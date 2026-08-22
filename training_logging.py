"""Small, reusable boundary for scalar training metrics."""

from typing import Callable, Mapping, Any


class DeferredLossMetrics:
    """Accumulate detached scalar tensors and materialize them only at log boundaries."""

    _NAMES = ("total", "bce", "mse", "cl", "balance")

    def __init__(self) -> None:
        self._epoch_sums = None
        self._epoch_count = 0
        self._window_sums = None
        self._window_count = 0

    def add(self, *, total: Any, bce: Any, mse: Any, cl: Any, balance: Any) -> None:
        values = {
            "total": total.detach(),
            "bce": bce.detach(),
            "mse": mse.detach(),
            "cl": cl.detach(),
            "balance": balance.detach(),
        }
        if self._epoch_sums is None:
            self._epoch_sums = {name: values[name].clone() for name in self._NAMES}
            self._window_sums = {name: values[name].clone() for name in self._NAMES}
        else:
            for name in self._NAMES:
                self._epoch_sums[name] += values[name]
            if self._window_sums is None:
                self._window_sums = {name: values[name].clone() for name in self._NAMES}
            else:
                for name in self._NAMES:
                    self._window_sums[name] += values[name]
        self._epoch_count += 1
        self._window_count += 1

    def consume_update_average(self):
        """Return one CPU scalar snapshot per optimizer update, then reset its window."""
        if self._window_count == 0:
            return None
        averages = {
            name: round((self._window_sums[name] / self._window_count).item(), 7)
            for name in self._NAMES
        }
        self._window_sums = None
        self._window_count = 0
        return averages

    def epoch_average(self):
        """Materialize the full-epoch mean only after the epoch has finished."""
        if self._epoch_count == 0:
            return {name: 0.0 for name in self._NAMES}
        return {
            name: round((self._epoch_sums[name] / self._epoch_count).item(), 7)
            for name in self._NAMES
        }


def log_batch_metrics(
    log_fn: Callable[[Mapping[str, Any], int], None],
    *,
    global_step: int,
    batch_loss: float,
    batch_loss_bce: float,
    batch_loss_mse: float,
    batch_loss_cl: float,
    batch_loss_e: float,
    learning_rate: float,
) -> None:
    """Write batch losses and learning rate after the first optimizer update."""
    if global_step <= 0:
        return

    log_fn(
        {
            "Loss/Total": batch_loss,
            "Loss/BCE_Anomaly": batch_loss_bce,
            "Loss/MSE_Recon": batch_loss_mse,
            "Loss/CL_Contrastive": batch_loss_cl,
            "Loss/Balance": batch_loss_e,
            "Train/LR": learning_rate,
        },
        step=global_step,
    )
