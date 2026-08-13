"""Tests for the strict multi-scale STFT time-frequency renderer."""

import torch


def test_factory_creates_multiscale_stft_renderer():
    """A configured time-frequency branch must resolve through the renderer factory."""
    from src.datasets.renderers import MultiScaleSTFTRenderer, create_renderer

    assert isinstance(create_renderer("stft_multiscale"), MultiScaleSTFTRenderer)


def test_multiscale_stft_renderer_returns_three_bounded_spectrograms():
    """Removing a spectral view or emitting invalid image values breaks this contract."""
    from src.datasets.renderers import create_renderer

    time = torch.arange(512, dtype=torch.float32)
    series = torch.stack(
        [torch.sin(2 * torch.pi * time / 16), torch.sin(2 * torch.pi * time / 64)],
        dim=-1,
    ).unsqueeze(0)

    image = create_renderer("stft_multiscale").render_batch(series, img_size=64)

    assert image.shape == (1, 3, 64, 64)
    assert image.dtype == torch.float32
    assert image.device == series.device
    assert torch.isfinite(image).all()
    assert image.min() >= 0
    assert image.max() <= 255
    assert not torch.allclose(image[:, 0], image[:, 1])


def test_multiscale_stft_renderer_ignores_masked_padding():
    """Using masked tail values in STFT would leak padding artifacts into the image."""
    from src.datasets.renderers import create_renderer

    valid = torch.sin(torch.arange(512, dtype=torch.float32) / 8).view(1, 512, 1)
    padded = torch.cat([valid, torch.full((1, 128, 1), 1_000_000.0)], dim=1)
    mask = torch.cat([torch.ones(1, 512, dtype=torch.bool), torch.zeros(1, 128, dtype=torch.bool)], dim=1)
    renderer = create_renderer("stft_multiscale")

    reference = renderer.render_batch(valid, img_size=64)
    masked = renderer.render_batch(padded, att_mask=mask, img_size=64)

    assert torch.allclose(masked, reference)
