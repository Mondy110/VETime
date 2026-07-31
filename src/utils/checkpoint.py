import os
import random
import torch
import numpy as np
from typing import Any, Dict, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_checkpoint(state: Dict[str, Any], path: str):
    """保存 checkpoint 到磁盘。"""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    torch.save(state, path)
    logger.info(f"Checkpoint 已保存: {path}")

def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """从磁盘加载 checkpoint。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint 不存在: {path}")
    state = torch.load(path, map_location=map_location, weights_only=False)
    logger.info(f"Checkpoint 已加载: {path}")
    return state


def save_full_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    dataset_idx: int,
    current_dim: int,
    prev_checkpoint_path: Optional[str],
    best_val_loss: float,
    patience_counter: int,
    save_path: str,
    accelerator
):
    """
    保存完整的训练状态checkpoint

    Args:
        model: 模型实例
        optimizer: 优化器实例
        epoch: 当前epoch（已完成）
        global_step: 全局步数
        dataset_idx: 当前数据集索引
        current_dim: 当前维度
        prev_checkpoint_path: 上一维度的checkpoint路径
        best_val_loss: 最佳验证损失
        patience_counter: 早停计数器
        save_path: 保存路径
        accelerator: Accelerator实例
    """
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    checkpoint = {
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'dataset_idx': dataset_idx,
        'current_dim': current_dim,
        'prev_checkpoint_path': prev_checkpoint_path,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'random_state': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }

    if accelerator.is_main_process:
        torch.save(checkpoint, save_path)
        logger.info(f"完整Checkpoint已保存: {save_path}")
