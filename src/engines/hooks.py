"""训练钩子函数：分类预热冻结 / 解冻。"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def freeze_for_cls_warmup(model, accelerator):
    """冻结除异常分类相关参数外的所有参数，用于分类头平稳预热。

    可训练参数:
      - anomaly_head.*            (分类 MLP)
      - mm_w.task_proj.1.T.*      (分类时序专家投影)
      - mm_w.task_proj.1.I.*      (分类图像专家投影)
      - mm_w.task_proj.1.M.*      (分类混合专家投影)
      - mm_w.Router.*             (任务路由，task_embedding 按任务隔离)

    其余全部冻结（视觉编码器、时序编码器、LoRA、重构头、重构专家、共享 mlp_m、fusion 等）。
    返回 saved_requires_grad 字典，用于后续恢复。
    """
    unwrapped = accelerator.unwrap_model(model)
    classification_patterns = ['anomaly_head', 'mm_w.task_proj.1.', 'mm_w.Router.']
    saved_requires_grad = {}
    frozen_count = 0
    trainable_count = 0
    for name, param in unwrapped.named_parameters():
        saved_requires_grad[name] = param.requires_grad
        if not any(pattern in name for pattern in classification_patterns):
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
        else:
            trainable_count += 1
    logger.info(
        f"[Cls Warmup] 冻结 {frozen_count} 个参数组，"
        f"保留 {trainable_count} 个分类相关参数组可训练"
    )
    return saved_requires_grad


def restore_requires_grad(model, accelerator, saved_requires_grad):
    """恢复参数的 requires_grad 状态（分类预热结束后调用）。"""
    unwrapped = accelerator.unwrap_model(model)
    for name, param in unwrapped.named_parameters():
        if name in saved_requires_grad:
            param.requires_grad = saved_requires_grad[name]
