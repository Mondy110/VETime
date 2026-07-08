"""Trainer 集成测试（需要 GPU 和预训练权重，标记为集成测试）。"""

import pytest


def test_trainer_init():
    """Trainer 可以被实例化（需要完整依赖，标记为集成测试）。"""
    pytest.skip("需要 GPU 和预训练权重，标记为集成测试")


def test_hooks_freeze_restore():
    """freeze_for_cls_warmup / restore_requires_grad 模块可导入。"""
    from src.engines.hooks import freeze_for_cls_warmup, restore_requires_grad
    assert callable(freeze_for_cls_warmup)
    assert callable(restore_requires_grad)


def test_trainer_importable():
    """Trainer 类可从 src.engines 导入。"""
    from src.engines.trainer import Trainer
    assert Trainer is not None
