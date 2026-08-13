"""Strict multi-scale STFT renderer for the time-frequency visual branch."""

from typing import Optional, Sequence

import numpy as np
import torch
from scipy.ndimage import zoom

from src.datasets.pre_image import _normalise_01_np, _stft_spectrogram_np

from . import RendererRegistry
from .base import BaseRenderer


@RendererRegistry.register("stft_multiscale")
class MultiScaleSTFTRenderer(BaseRenderer):
    """Render a raw series as three log-magnitude STFT spectrograms.

    RGB channels correspond to short, medium, and long analysis windows.  The
    source sequence is never resampled: only completed 2-D spectrograms are
    resized for the vision encoder.
    """

    windows: Sequence[int] = (32, 64, 128)

    def render_batch(
        self,
        time_series: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        img_size: int = 224,
    ) -> torch.Tensor:
        """Return [B, 3, img_size, img_size] float32 RGB time-frequency images."""
        if time_series.ndim != 3:
            raise ValueError(
                "time_series must have shape [batch, time, features], "
                f"received {tuple(time_series.shape)}"
            )
        if att_mask is not None and att_mask.shape != time_series.shape[:2]:
            raise ValueError(
                "att_mask must have shape [batch, time], "
                f"received {tuple(att_mask.shape)}"
            )

        series_np = time_series.detach().cpu().float().numpy()
        mask_np = None if att_mask is None else att_mask.detach().cpu().bool().numpy()
        images = []
        for batch_index, series in enumerate(series_np):
            valid_series = series if mask_np is None else series[mask_np[batch_index]]
            if valid_series.shape[0] < self.windows[-1]:
                raise ValueError(
                    f"STFT renderer requires at least {self.windows[-1]} valid time steps; "
                    f"received {valid_series.shape[0]}"
                )
            images.append(self._render_series(valid_series, img_size))

        return torch.from_numpy(np.stack(images)).to(
            device=time_series.device,
            dtype=torch.float32,
        )

    def _render_series(self, series: np.ndarray, img_size: int) -> np.ndarray:
        centered = series - series.mean(axis=0, keepdims=True)
        scale = centered.std(axis=0, keepdims=True) + 1e-8
        normalised = centered / scale
        channels = []
        for window in self.windows:
            spectrogram = _stft_spectrogram_np(
                normalised,
                win_len=window,
                hop_len=window // 4,
                n_fft=window,
            )
            resized = zoom(
                spectrogram,
                (img_size / spectrogram.shape[0], img_size / spectrogram.shape[1]),
                order=1,
            )
            channels.append(_normalise_01_np(resized))

        return (np.stack(channels, axis=0) * 255.0).clip(0, 255).astype(np.float32)
