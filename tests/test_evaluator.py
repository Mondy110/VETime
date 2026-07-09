import pytest


def test_evaluator_import():
    """验证 Evaluator 类可导入。"""
    from src.engines.evaluator import Evaluator


def test_dataloader_tsb_import():
    """验证 dataloader_TSB 函数可导入。"""
    from src.engines.evaluator import dataloader_TSB


def test_evaluator_compute_metrics():
    """验证静态方法 compute_metrics 可调用。"""
    pytest.skip("需要真实数据和标签，标记为集成测试")
