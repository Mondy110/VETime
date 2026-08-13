# tests/integration/test_renderer_integration.py
"""渲染器集成测试：验证 Trainer 正确使用渲染器。"""

import pytest
import torch
from unittest.mock import MagicMock, patch


def test_trainer_uses_configured_renderer():
    """测试 Trainer 使用配置指定的渲染器。"""
    from omegaconf import OmegaConf
    from src.engines.trainer import Trainer

    # 创建 mock 对象
    cfg = OmegaConf.create({
        'model': {
            'model_name': 'test',
            'vision_branch': {'vico_renderer': 'vico'}
        },
        'training': {'total_epochs': 1, 'stage1_epochs': 0, 'early_stopping': {'patience': 1}},
        'loss': {},
        'data': {'dynamic_batch': False, 'effective_batch_size': 32},
    })

    model = MagicMock()
    model.MAX_L = 5000
    train_loader = MagicMock()
    val_loader = MagicMock()
    accelerator = MagicMock()
    accelerator.device = 'cpu'
    data_setting = {'img_size': 224}
    patch_size = 16

    trainer = Trainer(
        cfg, model, train_loader, val_loader, accelerator,
        data_setting, patch_size
    )

    # 验证渲染器已初始化
    from src.datasets.renderers import ViCORenderer
    assert isinstance(trainer.vico_renderer, ViCORenderer)


def test_trainer_default_renderer_without_config():
    """测试无配置时默认使用 vico 渲染器。"""
    from omegaconf import OmegaConf
    from src.engines.trainer import Trainer

    cfg = OmegaConf.create({
        'model': {'model_name': 'test'},
        'training': {'total_epochs': 1, 'stage1_epochs': 0, 'early_stopping': {'patience': 1}},
        'loss': {},
        'data': {'dynamic_batch': False, 'effective_batch_size': 32},
    })

    model = MagicMock()
    model.MAX_L = 5000
    train_loader = MagicMock()
    val_loader = MagicMock()
    accelerator = MagicMock()
    accelerator.device = 'cpu'
    data_setting = {'img_size': 224}
    patch_size = 16

    trainer = Trainer(
        cfg, model, train_loader, val_loader, accelerator,
        data_setting, patch_size
    )

    from src.datasets.renderers import ViCORenderer
    assert isinstance(trainer.vico_renderer, ViCORenderer)


def test_trainer_uses_multiscale_stft_renderer_when_configured():
    """A time-frequency configuration must not silently fall back to ViCO."""
    from omegaconf import OmegaConf
    from src.engines.trainer import Trainer
    from src.datasets.renderers import MultiScaleSTFTRenderer

    cfg = OmegaConf.create({
        'model': {
            'model_name': 'test',
            'vision_branch': {'vico_renderer': 'stft_multiscale'},
        },
        'training': {'total_epochs': 1, 'stage1_epochs': 0, 'early_stopping': {'patience': 1}},
        'loss': {},
        'data': {'dynamic_batch': False, 'effective_batch_size': 32},
    })
    model = MagicMock()
    model.MAX_L = 5000
    accelerator = MagicMock()
    accelerator.device = 'cpu'

    trainer = Trainer(
        cfg, model, MagicMock(), MagicMock(), accelerator,
        {'img_size': 224}, 16,
    )

    assert isinstance(trainer.vico_renderer, MultiScaleSTFTRenderer)
