"""Adapter for the frozen MAE/ViT vision encoder used by VETime."""

from __future__ import annotations

from pathlib import Path

from torch import nn


class FrozenMAEEncoder(nn.Module):
    """Expose the vision features and fold API while enforcing frozen weights."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = encoder.hidden_size
        self.MAX_L = encoder.MAX_L
        self.patch_size = encoder.patch_size
        self._freeze()

    @classmethod
    def from_checkpoint(
        cls,
        vision_name: str,
        vision_dir: str | Path = "checkpoints/weight_v",
        *,
        max_length: int = 5000,
        use_vectorized_fold: bool = False,
    ) -> "FrozenMAEEncoder":
        from model.Vision_encoder.V_encoder import V_model

        encoder = V_model(
            vision_name=vision_name,
            vision_dir=str(vision_dir),
            MAX_L=max_length,
            unpatch=True,
            finetune_type="none",
            use_vectorized_fold=use_vectorized_fold,
        )
        return cls(encoder)

    def _freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def forward(self, hidden_states):
        return self.encoder(hidden_states)

    def unfold_image(self, image_features, init_img_size=None):
        return self.encoder.unfold_image(image_features, init_img_size)
