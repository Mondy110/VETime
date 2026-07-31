# src/datasets/renderers/base.py
"""渲染器抽象基类。"""

from abc import ABC, abstractmethod
import torch
from typing import Optional


class BaseRenderer(ABC):
    """频域图像渲染器基类。

    所有渲染器必须实现 render_batch 方法,将时序数据转换为图像。
    """

    @abstractmethod
    def render_batch(
        self,
        time_series: torch.Tensor,    # [B, T, F] 原始时序
        att_mask: Optional[torch.Tensor] = None,  # [B, T]
        img_size: int = 224,
    ) -> torch.Tensor:
        """渲染频域图像批次。

        Args:
            time_series: [B, T, F] 原始时序数据（未归一化）
            att_mask: [B, T] 注意力掩码, True=有效, False=padding
            img_size: 输出图像尺寸

        Returns:
            [B, 3, img_size, img_size] float32 tensor,值在 [0, 255] 范围
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"
