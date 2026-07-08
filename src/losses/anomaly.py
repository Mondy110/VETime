import torch
import torch.nn as nn
import torch.nn.functional as F


def anomaly_detection_loss(local_embeddings, labels, anomaly_head=None,
                           focal_gamma=2.0, w_anomaly=1.2, w_normal=0.8):
    """Masked Focal Loss for anomaly detection.

    Standalone function extracted from TS_Model.anomaly_detection_loss.

    When anomaly_head is provided, applies the projection head to local_embeddings
    (matching the model's method behavior). When None, expects local_embeddings
    to already be logits of shape [B, seq_len, 2].

    - Per-timestep focal loss on valid (non-padding) positions only
    - Decoupled + normalised class weights to keep loss scale stable

    Args:
        local_embeddings: [B, seq_len, num_features, d_proj] embeddings from model,
                          or [B, seq_len, 2] logits if anomaly_head is None
        labels: [B, seq_len] integer labels (-1=padding, 0=normal, 1=anomaly)
        anomaly_head: nn.Module for projecting embeddings to 2-class logits (optional)
        focal_gamma: Focal loss gamma parameter (default: 2.0)
        w_anomaly: Weight for anomaly class (default: 1.2)
        w_normal: Weight for normal class (default: 0.8)

    Returns:
        anomaly_loss: scalar loss tensor
        logits: [B, seq_len, 2] classification logits
    """
    if anomaly_head is not None:
        # Apply anomaly_head projection + average across num_features
        # (matches TS_Model.anomaly_detection_loss behavior)
        logits = anomaly_head(local_embeddings)  # [B, seq_len, num_features, 2]
        logits = logits.mean(dim=-2)              # [B, seq_len, 2]
    else:
        # Expect pre-computed logits
        if local_embeddings.dim() != 3 or local_embeddings.size(-1) != 2:
            raise ValueError(
                f"When anomaly_head is None, local_embeddings must be logits "
                f"of shape [B, seq_len, 2], got {local_embeddings.shape}"
            )
        logits = local_embeddings

    # Mask out padding timesteps (labels == -1)
    attention_mask = (labels != -1)  # [B, seq_len]

    if attention_mask.sum() > 0:
        valid_logits = logits[attention_mask]        # [N_valid, 2]
        valid_labels = labels[attention_mask].long()  # [N_valid]

        # Per-token focal loss
        bce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** focal_gamma * bce_loss

        # Decoupled per-class mean + normalised weights
        is_anomaly = (valid_labels == 1)
        is_normal = (valid_labels == 0)

        loss_anom = focal_loss[is_anomaly]
        loss_norm = focal_loss[is_normal]

        mean_anom = loss_anom.mean() if loss_anom.numel() > 0 else torch.tensor(0.0, device=logits.device)
        mean_norm = loss_norm.mean() if loss_norm.numel() > 0 else torch.tensor(0.0, device=logits.device)

        if loss_anom.numel() > 0 and loss_norm.numel() > 0:
            anomaly_loss = (mean_anom * w_anomaly + mean_norm * w_normal) / (w_anomaly + w_normal)
        elif loss_anom.numel() > 0:
            anomaly_loss = mean_anom
        else:
            anomaly_loss = mean_norm
    else:
        anomaly_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

    return anomaly_loss, logits
