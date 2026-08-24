from dataclasses import replace

import torch
from torch import nn

from vetime.models.multimodal.model import VETimeMultimodalModel, VETimeOptions
from vetime.models.temporal.config import TemporalModelConfig
from vetime.models.temporal.model import TemporalModel
from vetime.models.vision.mae import FrozenMAEEncoder


class TinyVisionEncoder(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.MAX_L = 8
        self.patch_size = 16
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.forward_calls = 0
        self.unfold_calls = 0

    def forward(self, hidden_states):
        self.forward_calls += 1
        return self.projection(hidden_states), None

    def unfold_image(self, image_features, init_img_size=None):
        self.unfold_calls += 1
        return image_features


def temporal_config():
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


def model_options(**overrides):
    options = VETimeOptions(
        vision_dim=8,
        temporal_dim=8,
        max_length=8,
        cmrg_enabled=False,
        use_query_decoder=False,
    )
    return replace(options, **overrides)


def test_multimodal_model_composes_instead_of_inherits():
    model = VETimeMultimodalModel(
        temporal=TemporalModel(temporal_config()),
        vision_encoder=TinyVisionEncoder(),
        options=model_options(),
    )

    assert isinstance(model.temporal, TemporalModel)
    assert not isinstance(model, TemporalModel)
    assert any(key.startswith("temporal.encoder.") for key in model.state_dict())
    assert not any(key.startswith("ts_encoder.") for key in model.state_dict())


def test_composed_model_keeps_four_value_forward_contract():
    vision = TinyVisionEncoder()
    model = VETimeMultimodalModel(
        temporal=TemporalModel(temporal_config()),
        vision_encoder=vision,
        options=model_options(),
    )
    model.eval()

    returns = model(
        torch.randn(1, 2, 8),
        torch.randn(1, 4, 1),
        torch.ones(1, 4, dtype=torch.bool),
    )

    assert len(returns) == 4
    assert returns[0].shape == (1, 4, 1, 2)
    assert returns[3].shape == (1, 4, 1, 2)
    assert vision.forward_calls == 1
    assert vision.unfold_calls == 1


def test_composed_model_exposes_training_protocol_without_duplicate_modules():
    model = VETimeMultimodalModel(
        temporal=TemporalModel(temporal_config()),
        vision_encoder=TinyVisionEncoder(),
        options=model_options(),
    )

    assert model.vit_encoder is model.vision_encoder
    assert model.ts_encoder is model.temporal
    assert callable(model.anomaly_detection_loss)
    assert callable(model.weighted_reconstruction_loss)
    assert callable(model.split_data)


def test_frozen_vision_adapter_freezes_wrapped_encoder():
    wrapped = FrozenMAEEncoder(TinyVisionEncoder())
    assert not any(parameter.requires_grad for parameter in wrapped.parameters())


def test_frozen_vision_adapter_forwards_checkpoint_directory(monkeypatch, tmp_path):
    calls = {}

    class FakeVModel(TinyVisionEncoder):
        def __init__(self, **kwargs):
            calls.update(kwargs)
            super().__init__()

    import vetime.models.vision.legacy_mae.V_encoder as legacy_vision

    monkeypatch.setattr(legacy_vision, "V_model", FakeVModel, raising=False)
    wrapped = FrozenMAEEncoder.from_checkpoint(
        "mae_visualize_base.pth",
        tmp_path,
        max_length=32,
        use_vectorized_fold=True,
    )

    assert isinstance(wrapped, FrozenMAEEncoder)
    assert calls["vision_dir"] == str(tmp_path)
    assert calls["MAX_L"] == 32
    assert calls["use_vectorized_fold"] is True


def test_composed_model_supports_cmrg_and_query_decoder_modes():
    for options in (
        model_options(cmrg_enabled=True, cmrg_guide_dim=8, cmrg_num_heads=2),
        model_options(use_query_decoder=True, query_decoder_num_heads=2),
        model_options(use_gradient_checkpointing=True),
    ):
        model = VETimeMultimodalModel(
            temporal=TemporalModel(temporal_config()),
            vision_encoder=TinyVisionEncoder(),
            options=options,
        )
        model.eval()
        returns = model(
            torch.randn(1, 2, 8),
            torch.randn(1, 4, 1),
            torch.ones(1, 4, dtype=torch.bool),
        )
        assert returns[0].shape == (1, 4, 1, 2)
        assert returns[3].shape == (1, 4, 1, 2)
