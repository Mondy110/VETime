"""Standalone temporal encoder with reconstruction and anomaly heads."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import TemporalModelConfig
from .encoder import TimeSeriesEncoder
from .heads import build_anomaly_head, build_reconstruction_head


class TemporalModel(nn.Module):
    """Temporal pretraining model with canonical ``encoder.*`` state keys."""

    def __init__(self, config: TemporalModelConfig):
        super().__init__()
        config = TemporalModelConfig.from_legacy(config)
        self.config = config
        self.encoder = TimeSeriesEncoder(
            d_model=config.d_model,
            d_proj=config.d_proj,
            patch_size=config.patch_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff_dropout=config.d_ff_dropout,
            max_total_tokens=config.max_total_tokens,
            use_rope=config.use_rope,
            num_features=config.num_features,
            activation=config.activation,
            use_lora=config.use_lora,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            cmrg_injection_mode=config.cmrg_injection_mode,
        )
        self.d_proj = config.d_proj
        self.patch_size = config.patch_size
        self.MAX_L = 5000
        self.reconstruction_head = build_reconstruction_head(config.d_proj)
        self.anomaly_head = build_anomaly_head(config.d_proj)

    def forward(self, time_series: Tensor, mask: Optional[Tensor] = None):
        if mask is None:
            if time_series.dim() == 2:
                sequence_length = time_series.shape[1]
            else:
                sequence_length = time_series.shape[1]
            mask = torch.ones(
                time_series.shape[0], sequence_length, dtype=torch.bool, device=time_series.device
            )
        return self.encoder(time_series, mask)

    def legacy_state_dict(self) -> dict[str, Tensor]:
        """Return keys in the original pretraining checkpoint namespace."""
        return {
            f"ts_encoder.{key.removeprefix('encoder.')}"
            if key.startswith("encoder.")
            else key: value
            for key, value in self.state_dict().items()
        }

    def masked_reconstruction_loss(
        self,
        local_embeddings: Tensor,
        original_time_series: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mask = mask.bool()
        reconstructed = self.reconstruction_head(local_embeddings).squeeze(-1)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, original_time_series.shape[-1])
        reconstruction_loss = F.mse_loss(
            reconstructed[mask_expanded], original_time_series[mask_expanded]
        )
        error = (reconstructed - original_time_series).abs()
        return reconstruction_loss, error

    def weighted_reconstruction_loss(
        self,
        local_embeddings: Tensor,
        original_time_series: Tensor,
        mask: Tensor,
        labels: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if mask.dtype is not torch.bool:
            mask = mask > 0.5
        reconstructed = self.reconstruction_head(local_embeddings).squeeze(-1)
        effective_mask = mask.clone()
        if labels is not None:
            effective_mask = effective_mask & (~labels.bool())
        flat_mask = effective_mask.reshape(-1)
        num_features = original_time_series.shape[-1]
        reconstruction_loss = F.mse_loss(
            reconstructed.reshape(-1, num_features)[flat_mask],
            original_time_series.reshape(-1, num_features)[flat_mask],
        )
        return reconstruction_loss, reconstructed

    def anomaly_detection_loss(self, local_embeddings: Tensor, labels: Tensor):
        logits = self.anomaly_head(local_embeddings).mean(dim=-2)
        attention_mask = labels != -1
        if attention_mask.sum() > 0:
            valid_logits = logits[attention_mask]
            valid_labels = labels[attention_mask].long()
            bce_loss = F.cross_entropy(valid_logits, valid_labels, reduction="none")
            pt = torch.exp(-bce_loss)
            focal_loss = (1 - pt) ** 2 * bce_loss
            is_anomaly = valid_labels == 1
            is_normal = valid_labels == 0
            mean_anomaly = (
                focal_loss[is_anomaly].mean()
                if is_anomaly.any()
                else torch.tensor(0.0, device=logits.device)
            )
            mean_normal = (
                focal_loss[is_normal].mean()
                if is_normal.any()
                else torch.tensor(0.0, device=logits.device)
            )
            if is_anomaly.any() and is_normal.any():
                loss = (mean_anomaly * 1.2 + mean_normal * 0.8) / 2.0
            elif is_anomaly.any():
                loss = mean_anomaly
            else:
                loss = mean_normal
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        return loss, logits

    def split_data(self, images, time_series, att_mask, labels):
        """Split long sequences on patch boundaries for the TSB evaluator."""
        _, sequence_length, _ = time_series.shape
        if sequence_length != labels.shape[1]:
            raise ValueError("Data and labels must have the same length")
        if sequence_length % self.patch_size != 0:
            raise ValueError("sequence length must be divisible by patch_size")
        max_patches = self.MAX_L // self.patch_size
        if max_patches <= 0:
            raise ValueError("MAX_L must be >= patch_size")
        patch_count = sequence_length // self.patch_size
        split_count = max(1, (patch_count + max_patches - 1) // max_patches)
        base, remainder = divmod(patch_count, split_count)
        chunks = []
        start = 0
        for index in range(split_count):
            length = (base + (index < remainder)) * self.patch_size
            end = start + length
            chunks.append(
                (
                    images[:, :, :, start:end],
                    time_series[:, start:end, :],
                    att_mask[:, start:end],
                    labels[:, start:end],
                )
            )
            start = end
        return chunks
