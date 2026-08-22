"""Task heads used by temporal pretraining."""

from torch import nn


def build_reconstruction_head(d_proj: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_proj, d_proj * 4),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(d_proj * 4, d_proj * 4),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(d_proj * 4, 1),
    )


def build_anomaly_head(d_proj: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_proj, d_proj // 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(d_proj // 2, 2),
    )
