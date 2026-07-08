import torch
import torch.nn as nn
import torch.nn.functional as F


def anomaly_detection_loss(local_embeddings, labels, focal_gamma=2.0, w_anomaly=1.2, w_normal=0.8):
    """Masked Focal Loss for anomaly detection.

    Standalone function extracted from TS_Model.anomaly_detection_loss.
    Uses the anomaly_head passed separately or constructed externally.

    - Per-timestep focal loss on valid (non-padding) positions only
    - Decoupled + normalised class weights to keep loss scale stable

    Args:
        local_embeddings: [B, seq_len, num_features, d_proj] embeddings from model
        labels: [B, seq_len] integer labels (-1=padding, 0=normal, 1=anomaly)
        focal_gamma: Focal loss gamma parameter (default: 2.0)
        w_anomaly: Weight for anomaly class (default: 1.2)
        w_normal: Weight for normal class (default: 0.8)

    Returns:
        anomaly_loss: scalar loss tensor
        logits: [B, seq_len, 2] classification logits
    """
    # This function expects local_embeddings and labels
    # The anomaly_head projection is applied inside the model's method;
    # here we take the logits directly if already projected, or compute them.
    # For standalone use, local_embeddings should already be the output of anomaly_head.
    # But since the model's method applies anomaly_head internally,
    # this standalone version accepts pre-computed logits.

    # If local_embeddings is 4D (from model output), it needs to go through anomaly_head
    # which is model-specific. So this standalone function takes logits directly
    # as the first argument when used externally.
    # For maximum compatibility with the model's method signature,
    # we check if the input looks like logits (3D with last dim = 2).
    if local_embeddings.dim() == 3 and local_embeddings.size(-1) == 2:
        # Already logits
        logits = local_embeddings
    else:
        # Need anomaly_head projection - but this standalone function doesn't have it.
        # The model's method should be used instead.
        raise ValueError(
            "Standalone anomaly_detection_loss expects pre-computed logits "
            "(shape [B, seq_len, 2]). Use model.anomaly_detection_loss() for "
            "raw embeddings, or apply anomaly_head first."
        )

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
