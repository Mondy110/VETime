from dataclasses import replace

import torch
from torch import nn

from vetime.config import CheckpointPaths, ModelConfig, TrainingConfig
from vetime.models.factory import (
    apply_temporal_finetune_policy,
    build_vetime_model,
)
from vetime.models.temporal.config import TemporalModelConfig
from vetime.models.temporal.model import TemporalModel


class TinyVisionEncoder(nn.Module):
    hidden_size = 8
    MAX_L = 8
    patch_size = 16

    def forward(self, hidden_states):
        return hidden_states, None

    def unfold_image(self, image_features, init_img_size=None):
        return image_features


def build_config(**model_overrides):
    paths = CheckpointPaths(None, "checkpoints/weight_v", "mae_visualize_base.pth")
    model = replace(ModelConfig(), **model_overrides)
    return TrainingConfig(seed=64, batch_size=2, paths=paths, model=model)


def tiny_temporal_config():
    return TemporalModelConfig(
        d_model=8,
        d_proj=2,
        patch_size=2,
        num_layers=1,
        num_heads=2,
        d_ff_dropout=0.0,
        num_features=1,
        use_lora=False,
    )


def test_freeze_policy_keeps_cmrg_parameters_trainable():
    config = build_config(cmrg_enabled=True, cmrg_guide_dim=8, cmrg_num_heads=2)
    model = build_vetime_model(
        config,
        temporal_config=tiny_temporal_config(),
        vision_encoder=TinyVisionEncoder(),
    )

    apply_temporal_finetune_policy(model, "freeze")

    assert not model.temporal.encoder.embedding_layer.weight.requires_grad
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(("cmrg_distiller.", "cmrg_guider.")) or ".cmrg_alpha" in name
    )


def test_factory_installs_lora_before_loading_temporal_weights():
    config = build_config(ts_finetune_type="lora")
    model = build_vetime_model(
        config,
        temporal_config=tiny_temporal_config(),
        vision_encoder=TinyVisionEncoder(),
    )

    names = dict(model.named_parameters())
    assert any("original_linear" in name for name in names)
    assert any("lora_A" in name for name in names)
    assert all(
        not parameter.requires_grad
        for name, parameter in names.items()
        if "original_linear" in name
    )


def test_factory_accepts_injected_temporal_dependency():
    config = build_config(ts_finetune_type="freeze")
    temporal = TemporalModel(tiny_temporal_config())
    model = build_vetime_model(
        config,
        temporal=temporal,
        vision_encoder=TinyVisionEncoder(),
    )

    assert model.temporal is temporal
