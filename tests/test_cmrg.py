import math

import torch

from model.CMRG import (
    CMRGContext,
    CrossModalRelationGuider,
    RelationDistiller,
)


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
