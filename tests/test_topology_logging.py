import torch
from torch import nn

from vetime.models.multimodal.model import VETimeMultimodalModel, VETimeOptions
from vetime.models.temporal.config import TemporalModelConfig
from vetime.models.temporal.model import TemporalModel


class TinyVision(nn.Module):
    hidden_size = 8
    MAX_L = 8
    patch_size = 2

    def forward(self, inputs):
        return inputs, None

    def unfold_image(self, features, init_img_size=None):
        return features


def test_topology_declares_clean_model_and_enabled_modules():
    from vetime.infrastructure.logging.topology import format_runtime_topology

    temporal = TemporalModel(
        TemporalModelConfig(
            d_model=8,
            d_proj=2,
            patch_size=2,
            num_layers=1,
            num_heads=2,
            num_features=1,
        )
    )
    model = VETimeMultimodalModel(
        temporal=temporal,
        vision_encoder=TinyVision(),
        options=VETimeOptions(
            vision_dim=8,
            temporal_dim=8,
            max_length=8,
            cmrg_enabled=True,
            cmrg_guide_dim=8,
            cmrg_num_heads=2,
            use_query_decoder=True,
            query_decoder_num_heads=2,
        ),
    )

    text = format_runtime_topology(
        model,
        device=torch.device("cuda:0"),
        initialization_source="temporal_pretrain",
    )

    assert "VETimeMultimodalModel (clean composition)" in text
    assert "CMRG: enabled" in text
    assert "QueryDecoder: enabled" in text
    assert "Initialization: temporal_pretrain" in text
