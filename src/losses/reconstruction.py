import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_reconstruction_loss(local_embeddings, original_time_series, mask, labels,
                                  reconstruction_head=None):
    """Weighted reconstruction loss, excluding anomaly points when labels available.

    Standalone function extracted from TS_Model.weighted_reconstruction_loss.

    Args:
        local_embeddings: [B, seq_len, d_proj] or [B, seq_len, num_features, d_proj]
        original_time_series: [B, seq_len, 1] or [B, seq_len, C]
        mask: [B, seq_len], bool or float (1 = masked)
        labels: [B, seq_len] or None
        reconstruction_head: nn.Module to apply to local_embeddings.
            If None, local_embeddings is assumed to be already reconstructed.

    Returns:
        reconstruction_loss: scalar loss
        reconstructed: [B, seq_len, C] reconstructed time series
    """
    batch_size, seq_len, num_f = original_time_series.shape
    device = original_time_series.device

    # Ensure mask is boolean
    if not mask.dtype == torch.bool:
        mask = mask > 0.5  # threshold if float mask

    # Apply reconstruction head if provided
    if reconstruction_head is not None:
        reconstructed = reconstruction_head(local_embeddings).squeeze(-1)  # [B, seq_len, C]
    else:
        # Assume local_embeddings is already reconstructed
        if local_embeddings.size(-1) == num_f:
            reconstructed = local_embeddings
        else:
            reconstructed = local_embeddings.squeeze(-1)

    effective_mask = mask.clone()  # [B, L]
    if labels is not None:
        labels = labels.bool()  # [B, L]
        effective_mask = effective_mask & (~labels)

    flat_mask = effective_mask.view(-1)

    reconstruction_loss = F.mse_loss(
        reconstructed.reshape(-1, num_f)[flat_mask],
        original_time_series.reshape(-1, num_f)[flat_mask],
    )

    return reconstruction_loss, reconstructed


def masked_reconstruction_loss(local_embeddings, original_time_series, mask,
                                reconstruction_head=None):
    """Reconstruction loss on masked positions only.

    Standalone function extracted from TS_Model.masked_reconstruction_loss.

    Args:
        local_embeddings: [B, seq_len, d_proj] or [B, seq_len, num_features, d_proj]
        original_time_series: [B, seq_len, 1] or [B, seq_len, C]
        mask: [B, seq_len], bool or float (1 = masked)
        reconstruction_head: nn.Module to apply to local_embeddings.
            If None, local_embeddings is assumed to be already reconstructed.

    Returns:
        reconstruction_loss: scalar loss
        error: absolute error per position [B, seq_len, C]
    """
    batch_size, seq_len, num_features = original_time_series.shape

    mask = mask.bool()

    # Apply reconstruction head if provided
    if reconstruction_head is not None:
        reconstructed = reconstruction_head(local_embeddings).squeeze(-1)  # [B, seq_len, C]
    else:
        if local_embeddings.size(-1) == num_features:
            reconstructed = local_embeddings
        else:
            reconstructed = local_embeddings.squeeze(-1)

    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_features)

    reconstruction_loss = F.mse_loss(
        reconstructed[mask_expanded],
        original_time_series[mask_expanded],
    )
    error = (reconstructed - original_time_series).abs()
    return reconstruction_loss, error
