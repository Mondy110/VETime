# train_ad_qwen_vl.py
"""
VETime Training Script

As per paper (B.4 Implementation Details):
- Vision Encoder: Frozen MAE (no fine-tuning)
- Time-Series Encoder: LoRA fine-tuning (r=8, α=16)
- Learning Rate: 5e-4
- Weight Decay: 1e-5
- Optimizer: AdamW
- Epochs: 25 (with early stopping patience=4)
- Batch Size: 32
"""
import argparse
import gc
import json
import random
import numpy as np
import yaml
from typing import Dict, Any, Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate import Accelerator
from accelerate.logging import get_logger
from Test_TSB import PASS_LIST, TSB_test
from evaluation.metrics import fast_get_metrics
import logging
from tqdm.auto import tqdm
import os
from datetime import datetime
from Test_TSB import EarlyStopping
from functools import partial
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
torch.cuda.empty_cache()


def set_seed(seed: int):
    """设置所有随机种子以保证可复现性（PyTorch 2.4+ 支持确定性 Flash Attention）"""
    import os
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # CUDA 确定性所需

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保CUDA操作确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ✅ PyTorch 2.1+ 支持确定性 Flash Attention
    # 这会启用所有操作的确定性模式，包括 Flash Attention
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """为每个DataLoader worker设置不同的种子"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


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
    print(f"[Cls Warmup] 冻结 {frozen_count} 个参数组，保留 {trainable_count} 个分类相关参数组可训练")
    return saved_requires_grad


def restore_requires_grad(model, accelerator, saved_requires_grad):
    """恢复参数的 requires_grad 状态（分类预热结束后调用）。"""
    unwrapped = accelerator.unwrap_model(model)
    for name, param in unwrapped.named_parameters():
        if name in saved_requires_grad:
            param.requires_grad = saved_requires_grad[name]



def train_univariate_hydra(cfg):
    """新入口：基于 src/ 的训练流程，使用 Hydra 配置。

    与 train_univariate 完全独立的入口函数，使用 src.engines.trainer.Trainer
    执行两阶段课程训练循环。原始 train_univariate 不受影响。

    Args:
        cfg: OmegaConf DictConfig，由 Hydra 装饰器注入或手动构建。
    """
    from src.utils.seed import seed_everything
    from src.utils.logger import get_logger as get_src_logger
    from src.engines.trainer import Trainer
    from src.models.vision_encoder.v_encoder import V_model
    from src.models.ts_encoder.config import TimeSeriesConfig
    from src.models.ts_encoder.ts_model import TS_Model
    from src.models.vetime import VETIME
    from src.datasets.anomaly_dataset import AnomalyDataset
    from src.datasets.collate import collate_fn, DynamicLengthBatchSampler
    from omegaconf import OmegaConf

    log = get_src_logger(__name__)
    seed_everything(cfg.seed)

    # ---- Accelerator ----
    gradient_accumulation_steps = max(1, cfg.data.effective_batch_size // cfg.data.batch_size)
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir="./output/logs",
    )
    log.info(f"Using {accelerator.num_processes} "
             f"{'GPUs' if accelerator.num_processes > 1 else 'CPU'}")

    # ---- Vision Encoder ----
    log.info(f"加载 Vision Encoder: {cfg.model.vision_name}")
    vision_model = V_model(
        cfg.model.vision_name,
        MAX_L=cfg.data.max_seq_length,
        unpatch=True,
        finetune_type='none',
        use_vectorized_fold=cfg.model.use_vectorized_fold,
    )
    config_v = vision_model.config
    patch_size = config_v['patch_size'] if isinstance(config_v, dict) else config_v.patch_size
    log.info(f"Vision Encoder 加载完成: patch_size={patch_size}, hidden_size={vision_model.hidden_size}")

    # ---- TS Encoder ----
    config_t = TimeSeriesConfig(
        **OmegaConf.to_container(OmegaConf.load("configs/model/vetime.yaml"), resolve=True)
    )
    if cfg.model.ts_finetune_type == 'lora':
        config_t.use_lora = True
        log.info(f"TS Encoder 微调类型: LoRA (r={config_t.lora_r}, alpha={config_t.lora_alpha})")
    else:
        config_t.use_lora = False
        log.info("TS Encoder 微调类型: 完全冻结")

    ts_model = TS_Model(config_t)
    if cfg.paths.ts_path:
        log.info(f"加载 TS Encoder 权重: {cfg.paths.ts_path}")
        state_ts_dict = torch.load(cfg.paths.ts_path, map_location='cpu')['model_state_dict']

        if cfg.model.ts_finetune_type == 'lora':
            new_state_dict = {}
            for key, value in state_ts_dict.items():
                if any(x in key for x in ['q_proj.weight', 'k_proj.weight', 'v_proj.weight',
                                           'out_proj.weight', 'gate_proj.weight', 'gate_proj.bias',
                                           'up_proj.weight', 'up_proj.bias', 'down_proj.weight',
                                           'down_proj.bias']):
                    parts = key.rsplit('.', 1)
                    new_key = f"{parts[0]}.original_linear.{parts[1]}"
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            ts_model.load_state_dict(new_state_dict, strict=False)
        else:
            ts_model.load_state_dict(state_ts_dict, strict=False)
        log.info("TS Encoder 权重加载完成")

    # Freeze 模式：选择性冻结
    if cfg.model.ts_finetune_type == 'freeze':
        for name, param in ts_model.named_parameters():
            if any(key in name for key in ['transformer_encoder', 'embedding_layer', 'rope_embedder']):
                param.requires_grad = False

    # ---- VETIME Model ----
    # 解码器模式：use_query_decoder=True 时使用 Query-based 解码器替代 MoE
    use_query_decoder = getattr(cfg.model, 'use_query_decoder', False)
    model = VETIME(
        config_v, vision_model, config_t, ts_model, cfg.model.model_name,
        use_query_decoder=use_query_decoder,
        use_gradient_checkpointing=cfg.model.use_gradient_checkpointing
    )
    if use_query_decoder:
        log.info("使用 Query-based 解码器模式（单阶段训练，无需课程学习）")
    else:
        log.info("使用 MoE 解码器模式（二阶段课程训练）")
    if hasattr(cfg.paths, 'vetime_path') and cfg.paths.vetime_path:
        log.info(f"加载 VETime 完整权重: {cfg.paths.vetime_path}")
        state_dict = torch.load(cfg.paths.vetime_path, map_location='cpu')
        model.load_state_dict(state_dict)
        log.info("VETime 权重加载完成")

    del vision_model, ts_model

    # ---- 参数统计 ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,} "
             f"({100*trainable_params/total_params:.2f}%)")

    # ---- DataSetting ----
    data_setting = {"img_size": 224, "T_sqrt": False}
    collatefn = partial(collate_fn, patch_size=patch_size)
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    # ---- DataLoader ----
    val_loader = None
    val_mode = getattr(cfg.data, 'val_mode', 'tsb')
    dynamic_batch = getattr(cfg.data, 'dynamic_batch', False)

    if val_mode == 'split':
        train_ratio = 1.0 - cfg.data.val_ratio
        train_dataset = AnomalyDataset(cfg.paths.dataset_path, patch_size=patch_size,
                                       split="train", train_ratio=train_ratio, seed=cfg.seed)
        val_dataset = AnomalyDataset(cfg.paths.dataset_path, patch_size=patch_size,
                                     split="test", train_ratio=train_ratio, seed=cfg.seed)
        log.info(f"验证模式: split, 训练集 {len(train_dataset)} 样本, "
                 f"验证集 {len(val_dataset)} 样本 (val_ratio={cfg.data.val_ratio})")

        if dynamic_batch:
            train_lengths = [len(train_dataset.data[i]['time_series']) for i in range(len(train_dataset))]
            val_lengths = [len(val_dataset.data[i]['time_series']) for i in range(len(val_dataset))]
            max_tokens = cfg.data.batch_size * max(train_lengths)

            train_sampler = DynamicLengthBatchSampler(
                train_lengths, max_tokens_per_batch=max_tokens,
                min_batch_size=cfg.data.batch_size,
                max_batch_size=getattr(cfg.data, 'max_batch_size', 256),
                padding_ratio=getattr(cfg.data, 'padding_ratio', 1.5),
                drop_last=True, effective_batch_size=cfg.data.batch_size,
                shuffle_each_epoch=getattr(cfg.data, 'shuffle_bucket', False), seed=cfg.seed,
            )
            val_sampler = DynamicLengthBatchSampler(
                val_lengths, max_tokens_per_batch=max_tokens,
                min_batch_size=cfg.data.batch_size,
                max_batch_size=getattr(cfg.data, 'max_batch_size', 256),
                padding_ratio=getattr(cfg.data, 'padding_ratio', 1.5),
                drop_last=False, effective_batch_size=0,
            )
            log.info(f"动态 Batch Size: {train_sampler.get_batch_info()}")
            accelerator.gradient_accumulation_steps = 1

            from src.utils.seed import seed_worker
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                      collate_fn=collatefn, num_workers=cfg.data.num_workers,
                                      pin_memory=True, persistent_workers=True,
                                      worker_init_fn=seed_worker)
            val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                                    collate_fn=collatefn, num_workers=cfg.data.num_workers,
                                    pin_memory=True, persistent_workers=True,
                                    worker_init_fn=seed_worker)
        else:
            from src.utils.seed import seed_worker
            train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size,
                                      collate_fn=collatefn, shuffle=False, num_workers=cfg.data.num_workers,
                                      pin_memory=True, drop_last=True, persistent_workers=True,
                                      worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size,
                                    collate_fn=collatefn, shuffle=False, num_workers=cfg.data.num_workers,
                                    pin_memory=True, drop_last=False, persistent_workers=True,
                                    worker_init_fn=seed_worker, generator=g)
    else:
        train_dataset = AnomalyDataset(cfg.paths.dataset_path, patch_size=patch_size, split="train")
        log.info(f"验证模式: tsb, 训练集 {len(train_dataset)} 样本")

        if dynamic_batch:
            train_lengths = [len(train_dataset.data[i]['time_series']) for i in range(len(train_dataset))]
            max_tokens = cfg.data.batch_size * max(train_lengths)

            train_sampler = DynamicLengthBatchSampler(
                train_lengths, max_tokens_per_batch=max_tokens,
                min_batch_size=cfg.data.batch_size,
                max_batch_size=getattr(cfg.data, 'max_batch_size', 256),
                padding_ratio=getattr(cfg.data, 'padding_ratio', 1.5),
                drop_last=True, effective_batch_size=cfg.data.batch_size,
                shuffle_each_epoch=getattr(cfg.data, 'shuffle_bucket', False), seed=cfg.seed,
            )
            log.info(f"动态 Batch Size: {train_sampler.get_batch_info()}")

            accumulation_steps = train_sampler.get_accumulation_steps()
            accelerator.gradient_accumulation_steps = accumulation_steps

            from src.utils.seed import seed_worker
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                      collate_fn=collatefn, num_workers=cfg.data.num_workers,
                                      pin_memory=True, persistent_workers=True,
                                      worker_init_fn=seed_worker)
        else:
            from src.utils.seed import seed_worker
            train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size,
                                      collate_fn=collatefn, shuffle=False, num_workers=cfg.data.num_workers,
                                      pin_memory=True, drop_last=True, persistent_workers=True,
                                      worker_init_fn=seed_worker, generator=g)

    # ---- Trainer ----
    trainer = Trainer(cfg, model, train_loader, val_loader, accelerator,
                      data_setting, patch_size)
    return trainer.run()



if __name__ == "__main__":
    # Use Hydra for configuration
    from omegaconf import OmegaConf
    import hydra

    @hydra.main(config_path="configs", config_name="base", version_base=None)
    def main(cfg):
        train_univariate_hydra(cfg)

    main()
