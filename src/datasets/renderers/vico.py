# src/datasets/renderers/vico.py
"""ViCO 频域渲染器：STFT + 热力图 + 梯度图三视图。"""

import torch
from typing import Optional
from .base import BaseRenderer
from . import RendererRegistry


@RendererRegistry.register('vico')
class ViCORenderer(BaseRenderer):
    """ViCO 频域渲染器。

    使用 STFT 频谱图 + 周期折叠热力图 + 梯度图组成三通道 RGB 图像。
    周期通过 FFT 自动检测（find_period_fft）。
    """

    def render_batch(
        self,
        time_series: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        img_size: int = 224,
    ) -> torch.Tensor:
        """渲染 ViCO 频域图像批次。

        内部调用 src.datasets.pre_image.render_vico_batch。

        Args:
            time_series: [B, T, F] 原始时序数据（未归一化）
            att_mask: [B, T] 注意力掩码, True=有效, False=padding
            img_size: 输出图像尺寸

        Returns:
            [B, 3, img_size, img_size] float32 tensor,值在 [0, 255] 范围
        """
        from src.datasets.pre_image import render_vico_batch
        return render_vico_batch(time_series, att_mask, img_size)
