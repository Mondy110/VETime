# tests/datasets/renderers/test_vico_renderer.py
"""ViCO 渲染器单元测试。"""

import pytest
import torch
import numpy as np


class TestRendererRegistry:
    """注册表测试。"""

    def test_register_and_get(self):
        """测试注册和获取。"""
        from src.datasets.renderers import RendererRegistry, BaseRenderer

        @RendererRegistry.register('test_renderer')
        class TestRenderer(BaseRenderer):
            def render_batch(self, ts, att_mask=None, img_size=224):
                return ts

        assert 'test_renderer' in RendererRegistry.list_available()
        cls = RendererRegistry.get('test_renderer')
        assert cls is TestRenderer

    def test_duplicate_register_raises(self):
        """测试重复注册抛出异常。"""
        from src.datasets.renderers import RendererRegistry, BaseRenderer

        with pytest.raises(ValueError, match="already registered"):
            @RendererRegistry.register('vico')  # 已存在
            class Another(BaseRenderer):
                def render_batch(self, ts, att_mask=None, img_size=224):
                    return ts

    def test_unknown_renderer_raises(self):
        """测试获取未注册渲染器抛出异常。"""
        from src.datasets.renderers import RendererRegistry

        with pytest.raises(ValueError, match="Unknown renderer"):
            RendererRegistry.get('nonexistent')


class TestViCORenderer:
    """ViCO 渲染器测试。"""

    def test_create_renderer_factory(self):
        """测试工厂函数创建。"""
        from src.datasets.renderers import create_renderer, ViCORenderer

        renderer = create_renderer('vico')
        assert isinstance(renderer, ViCORenderer)

    def test_render_batch_output_shape(self):
        """测试输出形状正确。"""
        from src.datasets.renderers import create_renderer

        renderer = create_renderer('vico')

        # 创建测试数据 [B=2, T=100, F=1]
        ts = torch.randn(2, 100, 1)

        output = renderer.render_batch(ts, img_size=224)

        assert output.shape == (2, 3, 224, 224)
        assert output.dtype == torch.float32

    def test_render_batch_with_att_mask(self):
        """测试带注意力掩码的渲染。"""
        from src.datasets.renderers import create_renderer

        renderer = create_renderer('vico')

        ts = torch.randn(2, 100, 1)
        att_mask = torch.ones(2, 100, dtype=torch.bool)
        att_mask[1, 50:] = False  # 第二个样本 padding

        output = renderer.render_batch(ts, att_mask=att_mask, img_size=224)

        assert output.shape == (2, 3, 224, 224)
