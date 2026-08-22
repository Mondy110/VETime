"""Composed VETime model with an explicit temporal model dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from loss.loss import win_Contrastive_Loss
from model.CMRG import CMRGContext, CrossModalRelationGuider, RelationDistiller
from model.VTS_module import M_moe, QueryDecoder, V_Attention, VTS_Alignment

from vetime.models.temporal.model import TemporalModel


@dataclass
class VETimeOptions:
    vision_dim: int
    temporal_dim: int
    max_length: int = 5000
    model_name: str | None = None
    cmrg_enabled: bool = False
    cmrg_num_relation_tokens: int = 16
    cmrg_guide_dim: int | None = None
    cmrg_num_heads: int = 8
    cmrg_metric_init: str = "identity"
    cmrg_gate_init: float = 0.0
    cmrg_injection_mode: str = "all_layers"
    cmrg_factorized: bool = True
    use_query_decoder: bool = True
    query_decoder_num_heads: int = 8
    use_gradient_checkpointing: bool = False


class VETimeMultimodalModel(nn.Module):
    """Multimodal VETime composition; temporal weights live under ``temporal``."""

    def __init__(self, temporal: TemporalModel, vision_encoder: nn.Module, options: VETimeOptions):
        super().__init__()
        self.temporal = temporal
        self.vision_encoder = vision_encoder
        self.name = options.model_name
        self.MAX_L = options.max_length
        self.patch_size = temporal.patch_size
        self.d_proj = temporal.d_proj
        self.use_gradient_checkpointing = options.use_gradient_checkpointing
        self.cmrg_enabled = options.cmrg_enabled
        self.cmrg_distiller: nn.Module | None = None
        self.cmrg_guider: nn.Module | None = None

        vision_dim = options.vision_dim
        temporal_dim = options.temporal_dim
        self.mlp_i = nn.Sequential(
            nn.Linear(vision_dim, temporal_dim * 2),
            nn.GELU(),
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.LayerNorm(temporal_dim),
        )
        self.pos_emb_v = nn.Parameter(torch.zeros(1, self.MAX_L, vision_dim))
        nn.init.normal_(self.pos_emb_v, std=0.02)
        self.I_att = V_Attention(temporal_dim)
        self.fusion = VTS_Alignment(vision_dim, temporal_dim)
        self.mm_w = M_moe(temporal_dim)
        self.cl_loss = win_Contrastive_Loss(temporal_dim, temperature=0.1)
        self.use_query_decoder = options.use_query_decoder
        self.query_decoder = None
        self.fusion_proj = None
        if self.use_query_decoder:
            self.query_decoder = QueryDecoder(
                d_model=temporal_dim,
                num_heads=options.query_decoder_num_heads,
                dropout=0.1,
            )
            self.fusion_proj = nn.Sequential(
                nn.Linear(temporal_dim * 2, temporal_dim),
                nn.LayerNorm(temporal_dim),
            )

        if self.cmrg_enabled:
            self._initialize_cmrg(options)
        if self.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

    @property
    def vit_encoder(self):
        """Compatibility view used by the existing data preparation loop."""
        return self.vision_encoder

    @property
    def ts_encoder(self):
        """Compatibility view; the canonical module remains ``temporal``."""
        return self.temporal

    def weighted_reconstruction_loss(self, *args, **kwargs):
        return self.temporal.weighted_reconstruction_loss(*args, **kwargs)

    def anomaly_detection_loss(self, *args, **kwargs):
        return self.temporal.anomaly_detection_loss(*args, **kwargs)

    def split_data(self, *args, **kwargs):
        return self.temporal.split_data(*args, **kwargs)

    def _initialize_cmrg(self, options: VETimeOptions) -> None:
        guide_dim = options.cmrg_guide_dim or options.temporal_dim
        if not self.temporal.config.use_rope:
            raise ValueError("CMRG requires the custom RoPE temporal attention implementation")
        if guide_dim != options.temporal_dim:
            raise ValueError("CMRG guide dimension must match temporal dimension")
        if options.cmrg_num_heads != self.temporal.config.num_heads:
            raise ValueError("CMRG guide head count must match temporal attention heads")
        if options.cmrg_metric_init != "identity":
            raise ValueError("CMRG currently supports only identity metric initialization")
        if options.cmrg_injection_mode not in ("all_layers", "last_layer"):
            raise ValueError("CMRG injection mode must be all_layers or last_layer")
        if not options.cmrg_factorized:
            raise ValueError("CMRG requires factorized relation context")

        self.cmrg_distiller = RelationDistiller(
            options.vision_dim,
            guide_dim,
            options.cmrg_num_relation_tokens,
            options.cmrg_num_heads,
        )
        self.cmrg_guider = CrossModalRelationGuider(
            options.temporal_dim,
            guide_dim,
            options.cmrg_num_heads,
            options.cmrg_num_relation_tokens,
        )
        encoder = self.temporal.encoder
        encoder.transformer_encoder.set_cmrg_injection_mode(options.cmrg_injection_mode)
        with torch.no_grad():
            for layer in encoder.transformer_encoder.layers:
                layer.cmrg_alpha.fill_(options.cmrg_gate_init if layer.cmrg_active else 0.0)

    def _encode_temporal_with_cmrg(self, hidden_states, time_series, att_mask):
        encoder = self.temporal.encoder
        prepared = encoder.prepare_inputs(time_series, att_mask)
        image_features, _ = self.vision_encoder(hidden_states)
        relation_tokens = self.cmrg_distiller(image_features)
        relation_logits, relation_factor = self.cmrg_guider(
            prepared.embedded_patches,
            relation_tokens,
            prepared.full_mask,
        )
        context = CMRGContext(relation_logits, relation_factor, prepared.full_mask)
        patch_embeddings = encoder.encode_prepared(prepared, context)
        local_embeddings = encoder.project_local_embeddings(patch_embeddings, prepared)
        return patch_embeddings, local_embeddings, prepared.full_mask, image_features

    def forward(
        self,
        hidden_states: Tensor,
        time_series: Tensor,
        att_mask: Optional[Tensor] = None,
        init_img_size=None,
        labels=None,
    ):
        if att_mask is None:
            att_mask = torch.ones(
                time_series.shape[0], time_series.shape[1], dtype=torch.bool, device=time_series.device
            )
        return self._forward_impl(hidden_states, time_series, att_mask, init_img_size, labels)

    def _forward_impl(self, hidden_states, time_series, att_mask, init_img_size, labels):
        if self.cmrg_enabled:
            ts_embeddings, _, patch_mask, image_features = self._encode_temporal_with_cmrg(
                hidden_states, time_series, att_mask
            )
        else:
            ts_embeddings, _, patch_mask = self.temporal(time_series, att_mask)
            image_features, _ = self.vision_encoder(hidden_states)

        batch_size, seq_len, num_features = time_series.shape
        patch_num = patch_mask.size(1) // num_features
        temporal_pos = self.pos_emb_v[:, :patch_num, :]
        multivariate_pos = temporal_pos.repeat(1, num_features, 1)
        image_embeddings = self.vision_encoder.unfold_image(image_features, init_img_size)
        image_embeddings = self.mlp_i(image_embeddings + multivariate_pos)
        image_embeddings0 = self.I_att(image_embeddings, patch_mask)
        image_embeddings, temporal_embeddings = self.fusion(
            image_embeddings0, ts_embeddings, patch_mask
        )
        contrastive_loss = self.compute_cl(image_embeddings, temporal_embeddings, labels, num_features)

        if self.use_query_decoder and self.query_decoder is not None:
            mixed = torch.cat([temporal_embeddings, image_embeddings], dim=-1)
            fusion_embeddings = self.fusion_proj(mixed)
            reconstruction_embeddings, classification_embeddings = self.query_decoder(
                ts_embeddings, image_embeddings, fusion_embeddings, patch_mask
            )
            local_anomaly = self._project_local(classification_embeddings, batch_size, patch_num, num_features, seq_len)
            local_reconstruction = self._project_local(reconstruction_embeddings, batch_size, patch_num, num_features, seq_len)
            return local_anomaly, None, contrastive_loss, local_reconstruction

        mixed = torch.cat([temporal_embeddings, image_embeddings], dim=-1)
        anomaly_features, anomaly_weights = self.mm_w(
            mixed, ts_embeddings, image_embeddings0, mixed, task_id=1
        )
        reconstruction_features, reconstruction_weights = self.mm_w(
            mixed, ts_embeddings, image_embeddings0, mixed, task_id=0
        )
        local_anomaly = self._project_local(anomaly_features, batch_size, patch_num, num_features, seq_len)
        local_reconstruction = self._project_local(
            reconstruction_features, batch_size, patch_num, num_features, seq_len
        )
        return (
            local_anomaly,
            {0: reconstruction_weights, 1: anomaly_weights},
            contrastive_loss,
            local_reconstruction,
        )

    def _project_local(self, features, batch_size, patch_num, num_features, seq_len):
        projected = self.temporal.encoder.projection_layer(features)
        local = projected.view(batch_size, num_features, patch_num, self.patch_size, self.d_proj)
        local = local.permute(0, 2, 3, 1, 4).contiguous()
        return local.view(batch_size, -1, num_features, self.d_proj)[:, :seq_len]

    def compute_cl(self, image_embeddings, temporal_embeddings, labels, num_features=1):
        if not self.training:
            return 0.0
        return self.cl_loss(image_embeddings, temporal_embeddings, labels, num_features)

    def _enable_gradient_checkpointing(self):
        self.temporal.encoder.gradient_checkpointing_enable()

    def disable_gradient_checkpointing(self):
        self.temporal.encoder.gradient_checkpointing_disable()
