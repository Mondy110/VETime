import os
import pytest
import torch
import numpy as np
import random
import logging
import tempfile

def test_seed_everything_reproducible():
    from src.utils.seed import seed_everything
    seed_everything(42)
    a = random.random()
    b = np.random.randn()
    c = torch.randn(3)
    seed_everything(42)
    d = random.random()
    e = np.random.randn()
    f = torch.randn(3)
    assert a == d, "Python random not reproducible"
    assert np.allclose(b, e), "NumPy random not reproducible"
    assert torch.allclose(c, f), "PyTorch random not reproducible"

def test_seed_worker():
    from src.utils.seed import seed_worker
    seed_worker(0)
    seed_worker(3)

def test_get_logger_returns_logger():
    from src.utils.logger import get_logger
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"

def test_get_logger_has_handler():
    from src.utils.logger import get_logger
    logger = get_logger("test_handler")
    assert len(logger.handlers) > 0

def test_save_load_checkpoint_roundtrip():
    from src.utils.checkpoint import save_checkpoint, load_checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_ckpt.pt")
        data = {"epoch": 5, "loss": 0.123, "tags": ["a", "b"]}
        save_checkpoint(data, path)
        loaded = load_checkpoint(path)
        assert loaded["epoch"] == 5
        assert abs(loaded["loss"] - 0.123) < 1e-6
        assert loaded["tags"] == ["a", "b"]

def test_load_checkpoint_missing_file():
    from src.utils.checkpoint import load_checkpoint
    with pytest.raises(FileNotFoundError):
        load_checkpoint("/nonexistent/path/checkpoint.pt")

def test_save_checkpoint_no_directory():
    from src.utils.checkpoint import save_checkpoint, load_checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")  # 纯文件名，无子目录
        data = {"epoch": 1}
        save_checkpoint(data, path)
        loaded = load_checkpoint(path)
        assert loaded["epoch"] == 1
