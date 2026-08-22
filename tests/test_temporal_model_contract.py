from pathlib import Path

import pytest
import torch

from vetime.infrastructure.checkpointing.temporal_legacy import load_legacy_temporal_checkpoint
from vetime.models.temporal.config import TemporalModelConfig
from vetime.models.temporal.model import TemporalModel
from model.TS_encoder.ts_model import TS_Model


@pytest.fixture()
def tiny_temporal_config():
    return TemporalModelConfig(
        d_model=32,
        d_proj=8,
        patch_size=4,
        num_layers=1,
        num_heads=4,
        d_ff_dropout=0.0,
        use_rope=True,
        activation="gelu",
        num_features=1,
        use_lora=False,
    )


def test_temporal_model_keeps_pretraining_forward_contract(tiny_temporal_config):
    model = TemporalModel(tiny_temporal_config)
    patch_embeddings, local_embeddings, full_mask = model(
        torch.randn(2, 8, 1),
        torch.ones(2, 8, dtype=torch.bool),
    )

    assert patch_embeddings.shape[0] == 2
    assert local_embeddings.shape == (2, 8, 1, 8)
    # The legacy encoder returns a patch-level mask; 8 samples with patch_size=4
    # produce two temporal patches for the single feature.
    assert full_mask.shape == (2, 2)


def test_temporal_model_has_single_canonical_module_path(tiny_temporal_config):
    model = TemporalModel(tiny_temporal_config)
    assert all(
        key.startswith(("encoder.", "reconstruction_head.", "anomaly_head."))
        for key in model.state_dict()
    )


def test_legacy_ts_model_alias_accepts_original_checkpoint_namespace(tiny_temporal_config):
    source_model = TemporalModel(tiny_temporal_config)
    source = {
        (f"ts_encoder.{key.removeprefix('encoder.')}" if key.startswith("encoder.") else key): value
        for key, value in source_model.state_dict().items()
    }

    target = TS_Model(tiny_temporal_config)
    missing, unexpected = target.load_state_dict(source, strict=False)

    assert not [key for key in missing if not key.endswith("cmrg_alpha")]
    assert not unexpected


@pytest.mark.skipif(
    not Path("checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth").exists(),
    reason="real temporal checkpoint is absent",
)
def test_real_pretrain_checkpoint_loads_all_temporal_parameters():
    config = TemporalModelConfig(
        d_model=512,
        d_proj=256,
        patch_size=16,
        num_layers=8,
        num_heads=8,
        d_ff_dropout=0.05,
        use_rope=True,
        activation="gelu",
        num_features=1,
        use_lora=False,
    )
    model = TemporalModel(config)
    report = load_legacy_temporal_checkpoint(
        model,
        "checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth",
    )

    assert report.loaded_keys == 111
    assert not report.missing_required_keys
    assert not report.shape_conflicts
