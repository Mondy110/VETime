import os
import torch
from typing import Any, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_checkpoint(state: Dict[str, Any], path: str):
    """保存 checkpoint 到磁盘。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logger.info(f"Checkpoint 已保存: {path}")

def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """从磁盘加载 checkpoint。"""
    assert os.path.isfile(path), f"Checkpoint 不存在: {path}"
    state = torch.load(path, map_location=map_location, weights_only=False)
    logger.info(f"Checkpoint 已加载: {path}")
    return state
