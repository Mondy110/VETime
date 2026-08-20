"""Small, reusable boundary for scalar training metrics."""

from typing import Callable, Mapping, Any


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
