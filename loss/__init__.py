"""
DEPRECATED: This module now re-exports from src.losses for backward compatibility.
New code should import directly from src.losses:
    from src.losses import win_Contrastive_Loss, load_balance_loss

This module provides loss functions for VETime anomaly detection.
"""

# Backward compatibility: re-export from new location
from src.losses.contrastive import win_Contrastive_Loss, win_ContrastiveLoss_init
from src.losses.balance import load_balance_loss

__all__ = [
    'win_Contrastive_Loss',
    'win_ContrastiveLoss_init',
    'load_balance_loss',
]
