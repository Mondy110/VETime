from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.TS_encoder.ts_encoder import TimeSeriesEncoder

class TS_Model(nn.Module):
    """Model for time series pretraining with masked reconstruction and anomaly detection."""
    
    def __init__(self, config_t, **kwargs):
        super().__init__()
        self.config = config_t
        ts_config = self.config
        self.ts_encoder = TimeSeriesEncoder(
            d_model=ts_config.d_model,
            d_proj=ts_config.d_proj,
            patch_size=ts_config.patch_size,
            num_layers=ts_config.num_layers,
            num_heads=ts_config.num_heads,
            d_ff_dropout=ts_config.d_ff_dropout,
            use_rope=ts_config.use_rope,
            num_features=ts_config.num_features,
            activation=ts_config.activation
        )

        self.d_proj=ts_config.d_proj
        self.patch_size=ts_config.patch_size

        self.token_hidden_size=512
        self.MAX_L=5000

        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.d_proj, self.d_proj* 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_proj* 4, self.d_proj * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_proj * 4, 1)
        )
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(self.d_proj, self.d_proj // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_proj // 2, 2)  # binary classification
        )

    def forward(self,time_series: torch.Tensor, mask: Optional[torch.Tensor] = None):
        
        patch_embeddings,local_embeddings,full_mask=self.ts_encoder(time_series,mask)
        return patch_embeddings,local_embeddings,full_mask
    
    def masked_reconstruction_loss(self, 
                                   local_embeddings: torch.Tensor,
                                   original_time_series: torch.Tensor,
                                   mask: torch.Tensor) -> torch.Tensor:
        """Compute masked reconstruction loss."""
        batch_size, seq_len,num_features = original_time_series.shape
        patch_size = self.patch_size
        
        mask = mask.bool()
        reconstructed = self.reconstruction_head(local_embeddings).squeeze(-1)  # [B, seq_len, 1]
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_features)
        
        reconstruction_loss = F.mse_loss(
                reconstructed[mask_expanded],
                original_time_series[mask_expanded],
            )
        error = (reconstructed - original_time_series).abs()
        return reconstruction_loss,error
    
    def weighted_reconstruction_loss(
        self,
        local_embeddings: torch.Tensor,
        original_time_series: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Compute weighted reconstruction loss with optional mask weakening,
        and return the self-similarity matrix of reconstructed features.

        Args:
            local_embeddings: [B, seq_len, d_proj]
            original_time_series: [B, seq_len, 1] or [B, seq_len, C]
            mask: [B, seq_len], bool or float (1 = masked)
            mask_weight: weight for masked positions (e.g., 0.5 to weaken)

        Returns:
            total_loss: scalar loss
            error: absolute error per position [B, seq_len, C]
            sim_matrix: self-similarity matrix of reconstructed embeddings [B, seq_len, seq_len]
        """
        batch_size, seq_len, num_f = original_time_series.shape
        device = original_time_series.device
        # Ensure mask is boolean
        if not mask.dtype == torch.bool:
            mask = mask > 0.5  # threshold if float mask

        # Reconstruct time series from embeddings
        reconstructed = self.reconstruction_head(local_embeddings).squeeze(-1)  # [B, seq_len, C]
        

        effective_mask = mask.clone()  # [B, L]
        if labels is not None:
            labels = labels.bool()  # [B, L]
            effective_mask = effective_mask & (~labels)

        flat_mask = effective_mask.view(-1)

       
        reconstruction_loss = F.mse_loss(
            reconstructed.reshape(-1, num_f)[flat_mask],
            original_time_series.reshape(-1, num_f)[flat_mask],
        )

        emb_norm = F.normalize(local_embeddings.flatten(2), p=2, dim=-1)  # [B, seq_len, d_proj]
        # sim_matrix = torch.bmm(emb_norm, emb_norm.transpose(1, 2))  # [B, seq_len, seq_len]

        return reconstruction_loss, reconstructed
    
    def anomaly_detection_loss(self,
                               local_embeddings: torch.Tensor,
                               labels: torch.Tensor) -> torch.Tensor:
        """Masked Focal Loss for extreme class-imbalanced anomaly detection.

        - Per-timestep focal loss on valid (non-padding) positions only
        - Decoupled + normalised class weights to keep loss scale stable
        """
        # Project & average across num_features (same as original)
        logits = self.anomaly_head(local_embeddings)  # [B, seq_len, num_features, 2]
        logits = logits.mean(dim=-2)                  # [B, seq_len, 2]

        # Mask out padding timesteps (labels == -1)
        attention_mask = (labels != -1)  # [B, seq_len]

        if attention_mask.sum() > 0:
            valid_logits = logits[attention_mask]        # [N_valid, 2]
            valid_labels = labels[attention_mask].long()  # [N_valid]

            # Per-token focal loss
            bce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='none')
            pt = torch.exp(-bce_loss)
            gamma = 2.0
            focal_loss = (1 - pt) ** gamma * bce_loss

            # Decoupled per-class mean + normalised weights
            is_anomaly = (valid_labels == 1)
            is_normal = (valid_labels == 0)

            loss_anom = focal_loss[is_anomaly]
            loss_norm = focal_loss[is_normal]

            mean_anom = loss_anom.mean() if loss_anom.numel() > 0 else torch.tensor(0.0, device=logits.device)
            mean_norm = loss_norm.mean() if loss_norm.numel() > 0 else torch.tensor(0.0, device=logits.device)

            w_anom, w_norm = 1.2, 0.8
            if loss_anom.numel() > 0 and loss_norm.numel() > 0:
                anomaly_loss = (mean_anom * w_anom + mean_norm * w_norm) / (w_anom + w_norm)
            elif loss_anom.numel() > 0:
                anomaly_loss = mean_anom
            else:
                anomaly_loss = mean_norm
        else:
            anomaly_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return anomaly_loss, logits