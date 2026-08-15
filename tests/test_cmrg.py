import math

import torch

from model.CMRG import (
    CMRGContext,
    CrossModalRelationGuider,
    RelationDistiller,
)
from model.TS_encoder.encoding_utils import (
    CustomTransformerEncoder,
    MultiheadAttentionWithRoPE,
    RotaryEmbedding,
)
from model.TS_encoder.ts_encoder import TimeSeriesEncoder
from model.TS_encoder.ts_model import TS_Model
from model.VETime import VETIME


def test_relation_distiller_returns_relation_tokens_without_dropout():
    distiller = RelationDistiller(vision_dim=768, guide_dim=512, num_relation_tokens=16, num_heads=8)
    visual_tokens = torch.randn(2, 11, 768)

    relation_tokens = distiller(visual_tokens)

    assert relation_tokens.shape == (2, 16, 512)
    assert not any(isinstance(module, torch.nn.Dropout) for module in distiller.modules())


def test_guider_returns_unnormalized_relation_factors():
    guider = CrossModalRelationGuider(8, 8, 2, 3)
    temporal = torch.randn(2, 5, 8)
    relation = torch.randn(2, 3, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)

    logits, factor = guider(temporal, relation, valid)

    assert logits.shape == (2, 2, 5, 3)
    assert factor.shape == (2, 2, 5, 3)
    assert not torch.allclose(logits.sum(-1), torch.ones(2, 2, 5))


def test_guider_identity_metric_matches_direct_factorization():
    torch.manual_seed(4)
    guider = CrossModalRelationGuider(8, 8, 2, 3)
    temporal = torch.randn(2, 5, 8)
    relation = torch.randn(2, 3, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)

    logits, factor = guider(temporal, relation, valid)

    torch.testing.assert_close(factor, logits @ guider.relation_metric)
    torch.testing.assert_close(guider.relation_metric, torch.eye(3))


def test_guider_zeros_padded_temporal_rows():
    guider = CrossModalRelationGuider(8, 8, 2, 3)
    temporal = torch.randn(1, 4, 8)
    relation = torch.randn(1, 3, 8)
    valid = torch.tensor([[True, False, True, False]])

    logits, factor = guider(temporal, relation, valid)

    assert torch.count_nonzero(logits[:, :, ~valid[0], :]) == 0
    assert torch.count_nonzero(factor[:, :, ~valid[0], :]) == 0


def test_guider_uses_scaled_dot_products_not_softmax_probabilities():
    guider = CrossModalRelationGuider(8, 8, 2, 3)
    temporal = torch.randn(1, 5, 8)
    relation = torch.randn(1, 3, 8)
    valid = torch.ones(1, 5, dtype=torch.bool)

    logits, _ = guider(temporal, relation, valid)
    q = guider.temporal_proj(torch.nn.functional.normalize(temporal, dim=-1))
    k = guider.relation_proj(torch.nn.functional.normalize(relation, dim=-1))
    q = q.view(1, 5, 2, 4).transpose(1, 2)
    k = k.view(1, 3, 2, 4).transpose(1, 2)
    expected = q @ k.transpose(-2, -1) / math.sqrt(4)

    torch.testing.assert_close(logits, expected)
    assert torch.any(logits < 0) or torch.any(logits > 1)


def test_cmrg_context_exposes_factorized_correction_and_strength():
    logits = torch.randn(2, 2, 4, 3)
    factor = torch.randn(2, 2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, False, True, True]])
    context = CMRGContext(logits, factor, mask)

    correction = context.correction()
    expected = factor @ logits.transpose(-2, -1)
    torch.testing.assert_close(correction, expected)

    alpha = torch.tensor(0.25)
    strength = context.frobenius_strength(alpha)
    expected_strength = (alpha * correction).square().sum().sqrt()
    torch.testing.assert_close(strength, expected_strength)


def _attention_inputs():
    torch.manual_seed(12)
    attention = MultiheadAttentionWithRoPE(embed_dim=4, num_heads=2, num_features=2)
    attention.eval()
    tokens = torch.randn(1, 3, 4)
    freqs = RotaryEmbedding(4)(3)
    feature_ids = torch.tensor([[0, 0, 1]])
    return attention, tokens, freqs, feature_ids


def _cmrg_context():
    logits = torch.tensor(
        [[[[1.0, -1.0], [0.5, 2.0], [-1.0, 0.25]], [[-0.5, 1.5], [1.0, 0.0], [0.75, -0.25]]]]
    )
    factor = torch.tensor(
        [[[[0.5, 1.0], [1.0, -0.5], [0.25, 1.5]], [[1.0, 0.25], [-0.75, 0.5], [0.5, 1.0]]]]
    )
    return CMRGContext(logits, factor, torch.ones(1, 3, dtype=torch.bool))


def test_rope_attention_zero_cmrg_gate_preserves_output_exactly():
    attention, tokens, freqs, feature_ids = _attention_inputs()
    context = _cmrg_context()

    without_context = attention(tokens, tokens, tokens, freqs, feature_ids, feature_ids)
    with_zero_gate = attention(
        tokens,
        tokens,
        tokens,
        freqs,
        feature_ids,
        feature_ids,
        cmrg_context=context,
        cmrg_alpha=torch.zeros(()),
    )

    assert torch.equal(without_context, with_zero_gate)


def test_rope_attention_nonzero_cmrg_gate_changes_explicit_score_output():
    attention, tokens, freqs, feature_ids = _attention_inputs()
    context = _cmrg_context()
    without_context = attention(tokens, tokens, tokens, freqs, feature_ids, feature_ids)
    with_context = attention(
        tokens,
        tokens,
        tokens,
        freqs,
        feature_ids,
        feature_ids,
        cmrg_context=context,
        cmrg_alpha=torch.ones(()),
    )

    assert not torch.allclose(without_context, with_context)


def test_each_rope_transformer_layer_owns_a_zero_initialized_cmrg_gate():
    encoder = CustomTransformerEncoder(
        d_model=4,
        nhead=2,
        dim_feedforward=8,
        dropout=0.0,
        activation="gelu",
        num_layers=2,
        num_features=2,
        use_lora=False,
    )

    assert all(torch.equal(layer.cmrg_alpha.detach(), torch.zeros(())) for layer in encoder.layers)


def test_cmrg_context_correction_matches_its_factorization():
    context = _cmrg_context()

    torch.testing.assert_close(
        context.correction(), context.relation_factor @ context.relation_logits.transpose(-2, -1)
    )


def test_time_series_encoder_forward_keeps_return_tuple_compatible():
    torch.manual_seed(7)
    encoder = TimeSeriesEncoder(
        d_model=4,
        d_proj=3,
        patch_size=2,
        num_layers=1,
        num_heads=2,
        d_ff_dropout=0.0,
        num_features=2,
        use_lora=False,
    )
    encoder.eval()
    time_series = torch.randn(2, 5, 2)
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])

    patch_embeddings, local_embeddings, full_mask = encoder(time_series, mask)

    assert patch_embeddings.shape == (2, 6, 4)
    assert local_embeddings.shape == (2, 5, 2, 3)
    assert full_mask.shape == (2, 6)


class _CountingVisionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 8
        self.MAX_L = 8
        self.forward_calls = 0
        self.unfold_calls = 0

    def forward(self, hidden_states):
        self.forward_calls += 1
        return hidden_states, None

    def unfold_image(self, image_features, init_img_size):
        self.unfold_calls += 1
        return image_features


class _RecordingTimeSeriesEncoder(TimeSeriesEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.received_cmrg_context = None

    def encode_prepared(self, prepared, cmrg_context=None):
        self.received_cmrg_context = cmrg_context
        return super().encode_prepared(prepared, cmrg_context)


def _tiny_vetime_config(cmrg_enabled):
    from model.TS_encoder.ts_encoder import TimeSeriesConfig

    return TimeSeriesConfig(
        d_model=8,
        d_proj=2,
        patch_size=2,
        num_layers=1,
        num_heads=2,
        d_ff_dropout=0.0,
        num_features=1,
        use_lora=False,
        cmrg_enabled=cmrg_enabled,
        cmrg_guide_dim=8,
        cmrg_num_heads=2,
    )


def _make_tiny_vetime(cmrg_enabled=True, use_gradient_checkpointing=False):
    config = _tiny_vetime_config(cmrg_enabled)
    ts_model = TS_Model(config)
    ts_model.ts_encoder = _RecordingTimeSeriesEncoder(
        d_model=config.d_model,
        d_proj=config.d_proj,
        patch_size=config.patch_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff_dropout=config.d_ff_dropout,
        num_features=config.num_features,
        use_lora=config.use_lora,
    )
    vision = _CountingVisionEncoder()
    return VETIME(
        config,
        vision,
        config,
        ts_model,
        use_query_decoder=False,
        use_gradient_checkpointing=use_gradient_checkpointing,
    ), vision


def test_vetime_cmrg_uses_raw_mae_tokens_without_replacing_fusion_path():
    model, vision = _make_tiny_vetime(cmrg_enabled=True)
    model.eval()
    hidden_states = torch.randn(1, 2, 8)
    time_series = torch.randn(1, 4, 1)
    mask = torch.ones(1, 4, dtype=torch.bool)

    returns = model(hidden_states, time_series, mask, init_img_size=None)

    context = model.ts_encoder.ts_encoder.received_cmrg_context
    assert vision.forward_calls == 1
    assert vision.unfold_calls == 1
    assert context.relation_logits.shape == (1, 2, 2, 16)
    assert len(returns) == 4
    assert returns[0].shape == (1, 4, 1, 2)
    assert returns[3].shape == (1, 4, 1, 2)


def test_vetime_cmrg_checkpointing_preserves_vision_and_return_contract():
    model, vision = _make_tiny_vetime(
        cmrg_enabled=True,
        use_gradient_checkpointing=True,
    )
    model.train()
    hidden_states = torch.randn(1, 2, 8)
    time_series = torch.randn(1, 4, 1)
    mask = torch.ones(1, 4, dtype=torch.bool)
    labels = torch.zeros(1, 4, dtype=torch.long)

    returns = model(hidden_states, time_series, mask, init_img_size=None, labels=labels)

    context = model.ts_encoder.ts_encoder.received_cmrg_context
    assert vision.forward_calls == 1
    assert vision.unfold_calls == 1
    assert context.relation_logits.shape == (1, 2, 2, 16)
    assert len(returns) == 4
    assert returns[0].shape == (1, 4, 1, 2)
    assert returns[3].shape == (1, 4, 1, 2)
