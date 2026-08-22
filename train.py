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
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate import Accelerator
from accelerate.logging import get_logger
from evaluation.metrics import fast_get_metrics
from model.Vision_encoder.V_encoder import V_model
from loss.loss import load_balance_loss
from model.TS_encoder.ts_model import TS_Model
from model.TS_encoder.config import default_config_t
from dataset.dataloader import AnomalyDataset, collate_fn, DynamicLengthBatchSampler
import logging
from tqdm.auto import tqdm
import os
from datetime import datetime
from model.VETime import VETIME
from model.CMRG import CMRGContext
from model.cmrg_training import (
    add_cmrg_injection_mode_argument,
    collect_cmrg_monitoring,
    configure_freeze_mode,
    load_model_state_compat,
    restore_optimizer_state_compat,
)
from functools import partial
from training_logging import DeferredLossMetrics, log_batch_metrics

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



def apply_cmrg_config(args):
    """Apply CMRG CLI controls to the shared temporal configuration."""
    for option in (
        "cmrg_enabled",
        "cmrg_num_relation_tokens",
        "cmrg_guide_dim",
        "cmrg_num_heads",
        "cmrg_metric_init",
        "cmrg_gate_init",
        "cmrg_injection_mode",
        "cmrg_factorized",
        "cmrg_log_interval",
    ):
        setattr(default_config_t, option, getattr(args, option))


def enable_cmrg_monitoring(model):
    """Capture the latest detached factorized context for interval logging."""
    if not getattr(model, "cmrg_enabled", False):
        return None

    def capture_context(_, inputs, output):
        temporal_valid_mask = inputs[2]
        model._cmrg_monitoring_context = CMRGContext(
            output[0].detach(), output[1].detach(), temporal_valid_mask.detach()
        )

    return model.cmrg_guider.register_forward_hook(capture_context)



def freeze_for_cls_warmup(model, accelerator):
    """冻结除异常分类相关参数外的所有参数，用于分类头平稳预热。

    可训练参数:
      - anomaly_head.*            (分类 MLP)
      - mm_w.task_proj.1.T.*      (分类时序专家投影) - M_moe 模式
      - mm_w.task_proj.1.I.*      (分类图像专家投影) - M_moe 模式
      - mm_w.task_proj.1.M.*      (分类混合专家投影) - M_moe 模式
      - mm_w.Router.*             (任务路由，task_embedding 按任务隔离) - M_moe 模式
      - query_decoder.*           (QueryDecoder 全部参数) - QueryDecoder 模式
      - fusion_proj.*             (融合投影层) - QueryDecoder 模式

    其余全部冻结（视觉编码器、时序编码器、LoRA、重构头、重构专家、共享 mlp_m、fusion 等）。
    返回 saved_requires_grad 字典，用于后续恢复。
    """
    unwrapped = accelerator.unwrap_model(model)

    # 根据模式选择可训练参数模式
    if hasattr(model, 'use_query_decoder') and model.use_query_decoder:
        # QueryDecoder 模式：训练分类相关 + query_decoder
        classification_patterns = ['anomaly_head', 'query_decoder.', 'fusion_proj.']
    else:
        # M_moe 模式：训练分类相关专家投影
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


def train_univariate(args):
    # Import the benchmark adapter only when an actual training run starts.
    # Keeping it out of module import time makes ``train.py --help`` and
    # configuration inspection usable in environments where the optional
    # tsb-ad benchmark package is not installed.
    from Test_TSB import PASS_LIST, TSB_test, EarlyStopping

    """
    单变量训练：完全保留原有训练逻辑

    此函数与原 main() 完全相同，确保单变量训练路径不受任何影响。
    """
    # 设置随机种子（必须在任何随机操作之前）
    set_seed(args.seed)
    print(f"[INFO] 随机种子已设置: {args.seed}")

    # 为 TSB_test 兼容性添加缺失的属性
    if not hasattr(args, 'save_dir'):
        args.save_dir = './output'
    if not hasattr(args, 'target_dir'):
        args.target_dir = os.path.join(args.save_dir, args.model_name)
        os.makedirs(args.target_dir, exist_ok=True)
    if not hasattr(args, 'dataset_dir'):
        args.dataset_dir = args.dataset_test_dir
    if not hasattr(args, 'file_list') or isinstance(args.file_list, str):
        if hasattr(args, 'file_list') and args.file_list.endswith('.csv'):
            df = pd.read_csv(args.file_list)
            args.file_list = df['filename'].tolist() if 'filename' in df.columns else df.iloc[:, 0].tolist()
        else:
            args.file_list = sorted(os.listdir(args.dataset_dir))

    # 计算梯度累积步数：确保每次更新的样本数 = effective_batch_size
    gradient_accumulation_steps = max(1, args.effective_batch_size // args.batch_size)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir="./output/logs"
    )

    logger.info(f"Using {accelerator.num_processes} {'GPUs' if accelerator.num_processes > 1 else 'CPU'}")
    print(f"[INFO] 梯度累积: {gradient_accumulation_steps} 步 (batch_size={args.batch_size} × 累积步数 = {args.batch_size * gradient_accumulation_steps} 样本/更新)")

    # ========== Vision Encoder (Frozen MAE, as per paper) ==========
    print(f"[INFO] 正在加载 Vision Encoder (MAE) 权重: checkpoints/weight_v/{args.vision_name}")
    # finetune_type='none' means fully frozen (as per paper: "the encoder of the frozen MAE")
    vision_model = V_model(args.vision_name, MAX_L=5000, unpatch=True, finetune_type='none',
                           use_vectorized_fold=args.use_vectorized_fold)
    print(f"[INFO] Vision Encoder 权重加载完成！Patch Size: {vision_model.patch_size}, Hidden Size: {vision_model.hidden_size}")
    print(f"[INFO] Vision Encoder 状态: 完全冻结 (as per paper)")
    if args.use_vectorized_fold:
        print(f"[INFO] fold_image 使用向量化版本 (约 150 倍加速)")

    config_v = vision_model.config
    if 'mae' in args.vision_name:
        patch_size = config_v['patch_size']
    else:
        patch_size = config_v.patch_size

    # ========== Time-Series Encoder (LoRA fine-tuning or Freeze, controlled by --ts_finetune_type) ==========
    # 根据 ts_finetune_type 设置 use_lora 配置
    if args.ts_finetune_type == 'lora':
        default_config_t.use_lora = True
        print(f"[INFO] TS Encoder 微调类型: LoRA (r={default_config_t.lora_r}, α={default_config_t.lora_alpha})")
    else:  # freeze
        default_config_t.use_lora = False
        print(f"[INFO] TS Encoder 微调类型: 完全冻结")

    apply_cmrg_config(args)

    ts_model = TS_Model(default_config_t)
    if args.ts_path is not None:
        print(f"[INFO] 正在加载 TS Encoder 权重: {args.ts_path}")
        state_ts_dict = torch.load(args.ts_path, map_location='cpu', weights_only=False)['model_state_dict']

        if args.ts_finetune_type == 'lora':
            # LoRA 模式：需要将预训练权重映射到 LoRALinear 的 original_linear 中
            # 预训练权重的 key: ts_encoder.xxx.weight
            # LoRA 模型的 key: ts_encoder.xxx.original_linear.weight
            new_state_dict = {}
            for key, value in state_ts_dict.items():
                # 检查是否是需要映射的线性层权重
                if any(x in key for x in ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'out_proj.weight',
                                            'gate_proj.weight', 'gate_proj.bias', 'up_proj.weight', 'up_proj.bias',
                                            'down_proj.weight', 'down_proj.bias']):
                    # 插入 .original_linear 到 key 中
                    parts = key.rsplit('.', 1)
                    new_key = f"{parts[0]}.original_linear.{parts[1]}"
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            # 使用 strict=False 因为 LoRA 参数 (lora_A, lora_B) 不在预训练权重中
            missing, unexpected = ts_model.load_state_dict(new_state_dict, strict=False)
            print(f"[INFO] TS Encoder 权重加载完成！")
            if missing:
                print(f"[INFO]   缺失的参数 (LoRA 参数，将随机初始化): {len([m for m in missing if 'lora' in m])} 个")
        else:  # freeze 模式
            # Freeze 模式：直接加载权重，不修改键名
            ts_model.load_state_dict(state_ts_dict, strict=False)
            print(f"[INFO] TS Encoder 权重加载完成！")
    else:
        print(f"[WARNING] 未指定 --ts_path，TS Encoder 使用随机初始化！")

    # ========== Create VETime Model ==========
    model = VETIME(config_v, vision_model, default_config_t, ts_model, args.model_name)
    if args.vetime_path is not None:
        print(f"[INFO] 正在加载 VETime 完整权重: {args.vetime_path}")
        state_dict = torch.load(args.vetime_path, map_location='cpu', weights_only=False)
        load_model_state_compat(model, state_dict, "legacy VETime weights")
        print(f"[INFO] VETime 权重加载完成（用于继续训练）")
    else:
        print(f"[INFO] 未指定 --vetime_path，VETime 融合模块从头训练")

    if args.ts_finetune_type == 'freeze':
        configure_freeze_mode(model)
    enable_cmrg_monitoring(model)

    del vision_model, ts_model

    # Print trainable parameters statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] 模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # ========== Dataset and DataLoader ==========
    collatefn = partial(collate_fn, patch_size=patch_size)

    # 为DataLoader设置随机种子生成器
    g = torch.Generator()
    g.manual_seed(args.seed)

    val_loader = None
    if args.val_mode == 'split':
        # 模式1: 从训练集中划分验证集（默认10%）
        train_ratio = 1.0 - args.val_ratio
        train_dataset = AnomalyDataset(args.dataset_path, patch_size=patch_size, split="train",
                                       train_ratio=train_ratio, seed=args.seed)
        val_dataset = AnomalyDataset(args.dataset_path, patch_size=patch_size, split="test",
                                     train_ratio=train_ratio, seed=args.seed)
        print(f"[INFO] 验证模式: split (从训练集划分)")
        print(f"[INFO] 数据集划分: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本 "
              f"(val_ratio={args.val_ratio})")

        # 动态 batch size：按序列长度动态调整，短样本增大 batch 充分利用 GPU
        train_lengths = [len(train_dataset.data[i]['time_series']) for i in range(len(train_dataset))]
        val_lengths = [len(val_dataset.data[i]['time_series']) for i in range(len(val_dataset))]
        # max_tokens = batch_size * max_length：保证最长样本 batch_size 不低于原始值
        max_tokens = args.batch_size * max(train_lengths) if args.dynamic_batch else None

        if args.dynamic_batch and max_tokens:
            train_sampler = DynamicLengthBatchSampler(
                train_lengths, max_tokens_per_batch=max_tokens,
                min_batch_size=args.batch_size,
                max_batch_size=args.max_batch_size,
                padding_ratio=args.padding_ratio,
                drop_last=True, effective_batch_size=args.batch_size,
                shuffle_each_epoch=args.shuffle_bucket, seed=args.seed,
            )
            val_sampler = DynamicLengthBatchSampler(
                val_lengths, max_tokens_per_batch=max_tokens,
                min_batch_size=args.batch_size,
                max_batch_size=args.max_batch_size,
                padding_ratio=args.padding_ratio,
                drop_last=False, effective_batch_size=0,
            )
            print(f"[INFO] 动态 Batch Size: {train_sampler.get_batch_info()}")

            # 动态 batch 模式：由训练循环按样本数手动控制累积，accelerator 设为 1
            accelerator.gradient_accumulation_steps = 1
            print(f"[INFO] 梯度累积: 按样本数触发，每 {args.effective_batch_size} 个样本反向传播一次")

            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                      collate_fn=collatefn, num_workers=args.num_workers,
                                      pin_memory=True, persistent_workers=True,
                                      worker_init_fn=seed_worker)
            val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                                    collate_fn=collatefn, num_workers=args.num_workers,
                                    pin_memory=True, persistent_workers=True,
                                    worker_init_fn=seed_worker)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      collate_fn=collatefn, shuffle=False, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True, persistent_workers=True,
                                      worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    collate_fn=collatefn, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True, drop_last=False, persistent_workers=True,
                                    worker_init_fn=seed_worker, generator=g)
    else:
        # 模式2: 使用真实数据集(TSB_test)作为验证（原有逻辑）
        train_dataset = AnomalyDataset(args.dataset_path, patch_size=patch_size, split="train")
        print(f"[INFO] 验证模式: tsb (真实数据集验证)")
        print(f"[INFO] 训练集: {len(train_dataset)} 样本")

        # 动态 batch size
        train_lengths = [len(train_dataset.data[i]['time_series']) for i in range(len(train_dataset))]
        max_tokens = args.batch_size * max(train_lengths) if args.dynamic_batch else None

        if args.dynamic_batch and max_tokens:
            train_sampler = DynamicLengthBatchSampler(
                train_lengths, max_tokens_per_batch=max_tokens,
                min_batch_size=args.batch_size,
                max_batch_size=args.max_batch_size,
                padding_ratio=args.padding_ratio,
                drop_last=True, effective_batch_size=args.batch_size,
                shuffle_each_epoch=args.shuffle_bucket, seed=args.seed,
            )
            print(f"[INFO] 动态 Batch Size: {train_sampler.get_batch_info()}")

            accumulation_steps = train_sampler.get_accumulation_steps()
            accelerator.gradient_accumulation_steps = accumulation_steps
            print(f"[INFO] 梯度累积步数: {accumulation_steps} (有效 batch size ≈ {accumulation_steps * int(np.median([len(b) for b in train_sampler._batches]))})")

            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                      collate_fn=collatefn, num_workers=args.num_workers,
                                      pin_memory=True, persistent_workers=True,
                                      worker_init_fn=seed_worker)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      collate_fn=collatefn, shuffle=False, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True, persistent_workers=True,
                                      worker_init_fn=seed_worker, generator=g)

    # ========== Optimizer (as per paper: AdamW, lr=5e-4, weight_decay=1e-5) ==========
    trainable_params_list = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params_list,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    print(f"\n[INFO] 优化器配置 (as per paper):")
    print(f"  Optimizer: AdamW")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Weight Decay: {args.weight_decay}")

    if val_loader is not None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )

    # 创建学习率调度器 (Warmup + CosineAnnealingLR, batch级别衰减)
    if args.dynamic_batch:
        steps_per_epoch = len(train_dataset) // args.effective_batch_size
    else:
        steps_per_epoch = len(train_loader) // accelerator.gradient_accumulation_steps
    warmup_steps = steps_per_epoch  # 1 epoch warmup
    total_steps = (args.early_stop_patience + 2) * steps_per_epoch  # 基于早停耐心值估算实际训练步数
    cosine_steps = total_steps - warmup_steps
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    print(f"  Scheduler: Warmup({warmup_steps} steps) + Cosine(cosine_steps={cosine_steps}, eta_min=1e-6), total={total_steps} steps ({args.early_stop_patience + 2} epochs)")

    # 初始化 TensorBoard tracker（每次训练使用独立的实验名称）
    run_name = f"vetime_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    accelerator.init_trackers(run_name)
    print(f"[INFO] TensorBoard 实验名称: {run_name}")

    model.train()
    global_step = 0
    accumulated_samples = 0  # 动态 batch 模式下按样本数累积
    epochs = args.num_epochs
    output = []
    device = accelerator.device
    data_setting = args.data_setting
    img_size = data_setting['img_size']
    name_save = f'./output/{args.model_name}__{img_size}_best.pth'

    early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True, path=name_save)
    output_path0 = f'./output/score/uni/{args.model_name}_train'
    os.makedirs(output_path0, exist_ok=True)

    # ========== 处理resume恢复 ==========
    start_epoch = 0
    best_val_loss_resume = None  # 用于恢复早停中的最佳验证损失
    checkpoint_dir = f'./output/checkpoints/{args.model_name}'
    if args.resume:
        resume_path = args.resume
        if not os.path.exists(resume_path):
            print(f"[ERROR] Resume checkpoint 不存在: {resume_path}")
        else:
            print(f"[INFO] 正在加载resume checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)

            if 'model_state_dict' in checkpoint:
                # 完整checkpoint格式
                unwrapped_model = accelerator.unwrap_model(model)
                missing, unexpected = unwrapped_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"[INFO] 模型权重已恢复")
                if missing:
                    print(f"  缺失的参数: {len(missing)} 个")
                if unexpected:
                    print(f"  未预期的参数: {len(unexpected)} 个")

                if restore_optimizer_state_compat(
                    optimizer, checkpoint.get('optimizer_state_dict')
                ):
                    print(f"[INFO] Optimizer状态已恢复")

                start_epoch = checkpoint['epoch'] + 1
                global_step = checkpoint['global_step']
                best_val_loss_resume = checkpoint.get('best_val_loss')
                if best_val_loss_resume is not None:
                    early_stopping.val_loss_min = best_val_loss_resume
                    print(f"[INFO] 早停最佳验证损失已恢复: {best_val_loss_resume:.4f}")

                # 恢复调度器状态
                scheduler_state = checkpoint.get('scheduler_state_dict')
                if scheduler_state is not None:
                    scheduler.load_state_dict(scheduler_state)
                    print(f"[INFO] 学习率调度器状态已恢复，当前LR={scheduler.get_last_lr()[0]:.2e}")
                else:
                    print(f"[WARNING] 旧checkpoint无调度器状态，手动推进{global_step}步（可能存在微小误差）")
                    for _ in range(global_step):
                        scheduler.step()

                # 恢复随机状态
                random_state = checkpoint.get('random_state', {})
                if 'python' in random_state:
                    random.setstate(random_state['python'])
                if 'numpy' in random_state:
                    np.random.set_state(random_state['numpy'])
                if 'torch' in random_state:
                    torch.set_rng_state(random_state['torch'])
                if 'cuda' in random_state and random_state['cuda'] is not None:
                    torch.cuda.set_rng_state_all(random_state['cuda'])
                print(f"[INFO] 随机状态已恢复")

                print(f"[INFO] 从epoch {start_epoch} 继续训练，global_step={global_step}")
            else:
                # 旧格式：仅模型权重
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.load_state_dict(checkpoint, strict=False)
                print(f"[INFO] 旧格式checkpoint，仅恢复了模型权重，从epoch 0开始训练")

    for epoch in range(start_epoch, epochs):
        # ========== 两阶段训练范式（Two-Stage Training）==========
        # 阶段1 (epoch < stage1_epochs): 纯重建预训练，切断异常分类损失，防止"梯度海啸"淹没分类头
        # 阶段2 (epoch >= stage1_epochs): 正常多任务联合训练
        # QueryDecoder 模式下：'joint' 模式跳过两阶段，'staged' 模式使用两阶段
        use_two_stage = True
        if hasattr(model, 'use_query_decoder') and model.use_query_decoder:
            if args.query_decoder_training_mode == 'joint':
                use_two_stage = False
                print(f"[QueryDecoder-Joint] Epoch {epoch+1}/{epochs}: 多任务同时训练 (两任务通过独立 Query 自然解耦)")

        is_stage_1 = use_two_stage and (epoch < args.stage1_epochs)
        if use_two_stage:
            if is_stage_1:
                print(f"[Stage 1] Epoch {epoch+1}/{epochs}: 纯重建预训练 (异常分类损失已切断，门控负载均衡仅保留重建分支)")
            else:
                print(f"[Stage 2] Epoch {epoch+1}/{epochs}: 多任务联合训练 (异常分类损失恢复，门控负载均衡恢复双分支)")

        # ========== 分类预热（Classification Warmup）==========
        # Stage 2 首个 epoch：前 cls_warmup_ratio 比例的 batch 仅训练分类相关参数
        # 让分类网络先平稳初始化，避免突然引入分类梯度造成巨大震荡
        # QueryDecoder 模式：两个任务通过独立 Query 解耦，不需要分类预热
        use_cls_warmup = (not is_stage_1) and (epoch == args.stage1_epochs) and (args.cls_warmup_ratio > 0)
        if hasattr(model, 'use_query_decoder') and model.use_query_decoder:
            use_cls_warmup = False  # QueryDecoder 模式跳过分类预热
        is_cls_warmup_epoch = use_cls_warmup
        cls_warmup_active = False
        saved_requires_grad = None
        cls_warmup_batches = 0
        if is_cls_warmup_epoch:
            cls_warmup_batches = max(1, int(len(train_loader) * args.cls_warmup_ratio))
            saved_requires_grad = freeze_for_cls_warmup(model, accelerator)
            cls_warmup_active = True
            print(f"[Cls Warmup] Epoch {epoch+1}: 前 {cls_warmup_batches}/{len(train_loader)} 个 batch 仅训练分类网络 (ratio={args.cls_warmup_ratio})")
            print(f"  可训练: anomaly_head, mm_w.task_proj.1.{{T,I,M}}, mm_w.Router")
            print(f"  已冻结: 视觉编码器、时序编码器(含LoRA)、重构头、重构专家、共享mlp_m、fusion 等")

        model.train()
        loss_metrics = DeferredLossMetrics()
        all_probs, all_preds, all_labels = [], [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}[Train]", disable=not accelerator.is_local_main_process)
        for batch_idx, batch in enumerate(progress_bar):

            # ========== 分类预热解冻检查 ==========
            if cls_warmup_active and batch_idx == cls_warmup_batches:
                restore_requires_grad(model, accelerator, saved_requires_grad)
                optimizer.zero_grad()
                cls_warmup_active = False
                progress_bar.set_description(f"Epoch {epoch+1}[Train]")
                print(f"\n[Cls Warmup] 分类预热完成 (batch {batch_idx}/{len(train_loader)})，已解冻所有参数，恢复多任务联合训练")
            labels = batch["labels"]
            images = batch["image"]  # (B, C, H, W)
            time_series, att_mask = batch['time_series'], batch['attention_mask']
            # Legacy masked-input feature; disabled in dataloader because the
            # current Query/CMRG forward path does not consume it.
            # mask = batch['mask']
            period = batch['period']
            p_value = batch['padding_value']

            # 【新增】定义重构损失的缩放系数，大象的体重缩水 20 倍
            alpha_recon = 0.05  

            if labels.shape[1] > model.MAX_L:
                data_splits = model.split_data(images, time_series, att_mask, labels)
                loss1 = 0
                loss2 = 0
                batch_loss_bce = torch.zeros((), device=device)
                batch_loss_mse = torch.zeros((), device=device)
                batch_loss_cl = torch.zeros((), device=device)
                batch_loss_e = torch.zeros((), device=device)
                logits_list = []
                for data_part in data_splits:
                    img_part, ts_part, att_mask_part, label_part = data_part
                    images_folded, init_img_size = model.vit_encoder.fold_image(img_part, period, p_value, **data_setting)

                    local_embeddings1, m_w, loss_cl, local_embeddings2 = model(images_folded, ts_part, att_mask_part, init_img_size, label_part)

                    loss01, logit = model.anomaly_detection_loss(local_embeddings1, label_part)
                    
                    # 这是我们要缩放的纯重构损失
                    loss02, rec = model.weighted_reconstruction_loss(local_embeddings2, ts_part, att_mask_part, label_part)

                    # ========== 两阶段训练范式：门控负载均衡损失 ==========
                    # 阶段1: 仅保留重建任务分支 m_w[0]，切断异常路由 m_w[1]，防止异常路由器为平均分配产生无意义梯度
                    # 阶段2: 恢复双分支计算
                    # QueryDecoder 模式下 m_w 为 None，无需负载均衡损失
                    if m_w is not None:
                        if is_stage_1:
                            batch_loss_e_part = 0.01 * load_balance_loss(m_w[0])
                        else:
                            batch_loss_e_part = 0.01 * 0.5 * (load_balance_loss(m_w[0]) + load_balance_loss(m_w[1]))
                    else:
                        batch_loss_e_part = torch.zeros((), device=device)

                    # ========== 两阶段训练范式：异常分类损失 ==========
                    # 阶段1: 强制切断，loss01 归零（脱离异常分类头的梯度）
                    # 阶段2: 恢复正常的异常分类损失
                    if is_stage_1:
                        loss01 = torch.tensor(0.0, device=device)

                    # 记录未缩放的原始数值用于 log 打印 (方便你和之前的实验对比)
                    batch_loss_bce += loss01.detach()
                    batch_loss_mse += loss02.detach()
                    batch_loss_cl += (0.1 * loss_cl).detach()
                    batch_loss_e += batch_loss_e_part.detach()

                    # 【核心修改】只对 loss02 乘 alpha_recon，其他 Loss 保持原样！
                    loss2 = loss2 + (alpha_recon * loss02) + 0.1 * loss_cl + batch_loss_e_part
                    loss1 = loss1 + loss01
                    logits_list.append(logit)

                logits = torch.cat(logits_list, dim=1)

                num_splits = len(data_splits)
                if num_splits > 0:
                    loss1 = loss1 / num_splits
                    loss2 = loss2 / num_splits
                    batch_loss_bce /= num_splits
                    batch_loss_mse /= num_splits
                    batch_loss_cl /= num_splits
                    batch_loss_e /= num_splits

            else:
                images_folded, init_img_size = model.vit_encoder.fold_image(images, period, p_value, **data_setting)

                local_embeddings1, m_w, loss_cl, local_embeddings2 = model(images_folded, time_series, att_mask, init_img_size, labels)

                loss1, logits = model.anomaly_detection_loss(local_embeddings1, labels)

                # 为了代码清晰，原有的 loss2 重命名为 loss_recon，代表纯重构损失
                loss_recon, rec = model.weighted_reconstruction_loss(local_embeddings2, time_series, att_mask, labels)

                # ========== 两阶段训练范式：门控负载均衡损失 ==========
                # 阶段1: 仅保留重建任务分支 m_w[0]，切断异常路由 m_w[1]，防止异常路由器为平均分配产生无意义梯度
                # 阶段2: 恢复双分支计算 (保留 Tensor 格式，为了后续反向传播)
                # QueryDecoder 模式下 m_w 为 None，无需负载均衡损失
                if m_w is not None:
                    if is_stage_1:
                        loss_e_tensor = 0.01 * load_balance_loss(m_w[0])
                    else:
                        loss_e_tensor = 0.01 * 0.5 * (load_balance_loss(m_w[0]) + load_balance_loss(m_w[1]))
                else:
                    loss_e_tensor = torch.tensor(0.0, device=time_series.device)

                # ========== 两阶段训练范式：异常分类损失 ==========
                # 阶段1: 强制切断，loss1 归零（脱离异常分类头的梯度）
                # 阶段2: 恢复正常的异常分类损失
                if is_stage_1:
                    loss1 = torch.tensor(0.0, device=device)
                
                # 【核心修改】只对 loss_recon 乘以缩放系数
                loss2 = (alpha_recon * loss_recon) + loss_e_tensor + 0.1 * loss_cl

                # 提取纯数值用于打 log
                batch_loss_bce = loss1.detach()
                batch_loss_mse = loss_recon.detach()  # 记录原始未缩放的重构损失
                batch_loss_cl = (0.1 * loss_cl).detach()
                batch_loss_e = loss_e_tensor.detach()

            # 最终反向传播：稳定的 BCE 分类 + 经过合理缩放后的 loss2 (包含降权的重构 + 对比 + 负载均衡)
            accelerator.backward(loss1 + loss2)

            # 梯度累积：按样本数触发反向传播
            current_bs = labels.shape[0]
            did_optimizer_step = False
            if args.dynamic_batch:
                accumulated_samples += current_bs
                if accumulated_samples >= args.effective_batch_size:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    accumulated_samples = 0
                    did_optimizer_step = True
            else:
                global_step += 1
                if global_step % accelerator.gradient_accumulation_steps == 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    did_optimizer_step = True

            loss_metrics.add(
                total=(loss1 + loss2).detach(),
                bce=batch_loss_bce,
                mse=batch_loss_mse,
                cl=batch_loss_cl,
                balance=batch_loss_e,
            )
            if did_optimizer_step:
                update_metrics = loss_metrics.consume_update_average()
                progress_bar.set_postfix({"Tot": f"{update_metrics['total']:.3f}", "BCE": f"{update_metrics['bce']:.3f}", "MSE": f"{update_metrics['mse']:.3f}", "CL": f"{update_metrics['cl']:.3f}", "Bal": f"{update_metrics['balance']:.4f}"})
                log_batch_metrics(
                    accelerator.log,
                    global_step=global_step,
                    batch_loss=update_metrics["total"],
                    batch_loss_bce=update_metrics["bce"],
                    batch_loss_mse=update_metrics["mse"],
                    batch_loss_cl=update_metrics["cl"],
                    batch_loss_e=update_metrics["balance"],
                    learning_rate=optimizer.param_groups[0]['lr'],
                )
            if cls_warmup_active and batch_idx < cls_warmup_batches:
                progress_bar.set_description(f"Epoch {epoch+1}[Train|CLS_Warmup]")

            if (args.cmrg_enabled and global_step > 0
                    and global_step % args.cmrg_log_interval == 0):
                cmrg_metrics = collect_cmrg_monitoring(
                    accelerator.unwrap_model(model),
                    getattr(accelerator.unwrap_model(model), "_cmrg_monitoring_context", None),
                )
                if cmrg_metrics:
                    accelerator.log(cmrg_metrics, step=global_step)

            probs = torch.softmax(logits, dim=-1)[:, :, 1]
            preds = (probs > 0.5).float()

            probs, preds, labels = accelerator.gather_for_metrics((probs, preds, labels))
            if global_step % 10 == 0:
                for i in range(probs.shape[0]):
                    all_probs.append(probs[i].detach().cpu().numpy().reshape(-1))
                    all_preds.append(preds[i].detach().cpu().numpy().reshape(-1))
                    all_labels.append(labels[i].detach().cpu().numpy().reshape(-1).astype(int))

            # 清理变量
            del images_folded, logits, loss1, probs, preds, labels, loss2
            # Legacy cleanup when ``mask = batch['mask']`` is restored:
            # del local_embeddings1, local_embeddings2, m_w, loss_cl, rec, mask, period, p_value
            del local_embeddings1, local_embeddings2, m_w, loss_cl, rec, period, p_value
            del images, time_series, att_mask, init_img_size

        # epoch 结束时 flush 剩余梯度
        if args.dynamic_batch and accumulated_samples > 0:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            accumulated_samples = 0

        # 分类预热安全保障：若预热因 batch 数不足未能中途解冻，在此强制恢复
        if cls_warmup_active:
            restore_requires_grad(model, accelerator, saved_requires_grad)
            optimizer.zero_grad()
            cls_warmup_active = False
            print(f"[Cls Warmup] Epoch {epoch+1} 结束，强制解冻所有参数（预热覆盖整个 epoch）")

        if len(all_probs) > 0:
            # 将收集到的小部分数据拼接起来
            all_probs_arr = np.concatenate(all_probs)
            all_preds_arr = np.concatenate(all_preds)
            all_labels_arr = np.concatenate(all_labels)

            if np.any(np.isnan(all_probs_arr)):
                print("⚠️ Warning: all_probs contains NaN values!")

            # 拿着这部分“抽样”的数据去算精确指标
            train_metrics = fast_get_metrics(all_probs_arr, all_labels_arr)

            # 👇【把打印代码加回来，让你能在屏幕上看到效果】
            for k, v in train_metrics.items():
                print(f"  Train {k}: {v:.4f}")

            # 👇【超级重要】：算完指标后，这些大数组就没用了，立刻手动删掉并回收内存！
            del all_probs_arr, all_preds_arr, all_labels_arr
            gc.collect()
        else:
            # 防御性代码：如果因为某些原因没采样到数据，给个空字典防止后面报错
            train_metrics = {}

        epoch_metrics = loss_metrics.epoch_average()
        avg_train_loss = epoch_metrics["total"]
        avg_loss_bce = epoch_metrics["bce"]
        avg_loss_mse = epoch_metrics["mse"]
        avg_loss_cl = epoch_metrics["cl"]
        avg_loss_e = epoch_metrics["balance"]

        accelerator.log({
            "epoch_train_loss": avg_train_loss,
            "epoch_loss_bce": avg_loss_bce,
            "epoch_loss_mse": avg_loss_mse,
            "epoch_loss_cl": avg_loss_cl,
            "epoch_loss_e": avg_loss_e,
        }, step=epoch)

        print(f"\n[Epoch {epoch + 1}/{epochs}] Training Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f} (BCE: {avg_loss_bce:.4f}, MSE: {avg_loss_mse:.4f}, CL: {avg_loss_cl:.4f}, Bal: {avg_loss_e:.4f})")

        # epoch结束后清理大数组（保留 train_metrics 供后续使用）
        del all_probs, all_preds, all_labels
        gc.collect()

        # ========== 验证阶段 ==========
        if args.val_mode == 'split':
            # split 模式: 每个epoch用划分的验证集评估，用于早停
            avg_val_loss = evaluate_univariate(model, val_loader, accelerator, data_setting)
            accelerator.log({"epoch_val_loss": avg_val_loss}, step=epoch)
            print(f"  Avg Val Loss (split): {avg_val_loss:.4f}")

            # 早停判断
            early_stopping(avg_val_loss, model)
            if avg_val_loss <= early_stopping.val_loss_min:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                best_model_path = f'./output/{args.model_name}__{img_size}_best.pth'
                if accelerator.is_main_process:
                    torch.save(unwrapped_model.state_dict(), best_model_path)
                    print(f"  Best model saved: {best_model_path} (val_loss={avg_val_loss:.4f})")
            if early_stopping.early_stop:
                print("Early stopping triggered (based on validation split).")
                break

            # split 模式下仍定期用 TSB_test 观测（不用于早停）
            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                model.eval()
                avg_tsb_val_loss = TSB_test(model, args, args.data_setting, device,
                                            dataset_setting=PASS_LIST, verbose=False,
                                            postprocess_workers=args.tsb_postprocess_workers,
                                            cpu_threads_per_worker=args.tsb_worker_cpu_threads)
                gc.collect()
                torch.cuda.empty_cache()
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                timestamp = datetime.now().strftime("%m%d-%H")
                name_save = f'./output/{args.model_name}__{img_size}_{avg_tsb_val_loss:.4f}_{timestamp}.pth'
                torch.save(unwrapped_model.state_dict(), name_save)
                logger.info(f"Model saved at epoch {epoch+1} with TSB_val_loss={avg_tsb_val_loss:.4f}")

                epoch_log = {
                    "epoch": epoch + 1,
                    "train_loss": round(avg_train_loss, 6),
                    "train_metrics": {k: round(v, 6) for k, v in train_metrics.items()},
                    "val_loss_split": round(avg_val_loss, 6),
                    "val_loss_tsb": round(avg_tsb_val_loss, 6) if avg_tsb_val_loss is not None else None,
                }
                output.append(epoch_log)
                model.train()
                gc.collect()
                torch.cuda.empty_cache()
            else:
                epoch_log = {
                    "epoch": epoch + 1,
                    "train_loss": round(avg_train_loss, 6),
                    "train_metrics": {k: round(v, 6) for k, v in train_metrics.items()},
                    "val_loss_split": round(avg_val_loss, 6),
                }
                output.append(epoch_log)

        else:
            # tsb 模式: 用 TSB_test 作为验证集（原有逻辑），用于早停
            if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
                model.eval()
                avg_val_loss = TSB_test(model, args, args.data_setting, device,
                                        dataset_setting=PASS_LIST, verbose=False,
                                        postprocess_workers=args.tsb_postprocess_workers,
                                        cpu_threads_per_worker=args.tsb_worker_cpu_threads)
                gc.collect()
                torch.cuda.empty_cache()
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                timestamp = datetime.now().strftime("%m%d-%H")
                name_save = f'./output/{args.model_name}__{img_size}_{avg_val_loss:.4f}_{timestamp}.pth'
                torch.save(unwrapped_model.state_dict(), name_save)
                logger.info(f"Model saved at epoch {epoch+1} with val_loss={avg_val_loss:.4f}")

                epoch_log = {
                    "epoch": epoch + 1,
                    "train_loss": round(avg_train_loss, 6),
                    "train_metrics": {k: round(v, 6) for k, v in train_metrics.items()},
                    "val_loss_tsb": round(avg_val_loss, 6) if avg_val_loss is not None else None,
                }
                output.append(epoch_log)

                early_stopping(avg_val_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered (based on TSB validation).")
                    break

                model.train()
                gc.collect()
                torch.cuda.empty_cache()
            else:
                epoch_log = {
                    "epoch": epoch + 1,
                    "train_loss": round(avg_train_loss, 6),
                    "train_metrics": {k: round(v, 6) for k, v in train_metrics.items()},
                }
                output.append(epoch_log)

        # ========== 保存完整checkpoint（每个epoch）==========
        accelerator.wait_for_everyone()
        os.makedirs(checkpoint_dir, exist_ok=True)
        epoch_checkpoint_path = os.path.join(
            checkpoint_dir,
            f'univariate_epoch{epoch}_full.pth'
        )
        save_full_checkpoint(
            model, optimizer, scheduler, epoch, global_step,
            early_stopping.val_loss_min,  # best_val_loss
            0,  # patience_counter (由 EarlyStopping 对象管理)
            epoch_checkpoint_path, accelerator
        )

        # 最后清理 train_metrics
        del train_metrics
        gc.collect()

    # 加载最佳模型（split 模式下由早停保存）
    if args.val_mode == 'split':
        best_model_path = f'./output/{args.model_name}__{img_size}_best.pth'
        if os.path.exists(best_model_path):
            print(f"\n[INFO] 加载早停保存的最佳模型: {best_model_path}")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(torch.load(best_model_path, map_location='cpu', weights_only=False))

    loss_all = TSB_test(
        model, args, args.data_setting, device, dataset_setting=PASS_LIST, verbose=False,
        postprocess_workers=args.tsb_postprocess_workers,
        cpu_threads_per_worker=args.tsb_worker_cpu_threads,
    )
    print(f"Final TSB validation loss: {loss_all}")
    accelerator.end_training()
    logger.info("Training completed!")

    return output



def save_full_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
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
        print(f"[INFO] 完整Checkpoint已保存: {save_path}")




def evaluate_univariate(model, val_loader, accelerator, data_setting):
    """
    单变量训练验证函数（基于训练集划分的验证集）

    Args:
        model: VETIME 模型
        val_loader: 验证集 DataLoader
        accelerator: Accelerator 实例
        data_setting: 数据配置

    Returns:
        avg_val_loss: 平均验证损失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"]
            images = batch["image"]
            time_series, att_mask = batch['time_series'], batch['attention_mask']
            period = batch['period']
            p_value = batch['padding_value']

            if labels.shape[1] > model.MAX_L:
                data_splits = model.split_data(images, time_series, att_mask, labels)
                loss1_total = 0
                loss2_total = 0

                for data_part in data_splits:
                    img_part, ts_part, att_mask_part, label_part = data_part
                    images_folded, init_img_size = model.vit_encoder.fold_image(
                        img_part, period, p_value, **data_setting)

                    local_embeddings1, m_w, loss_cl, local_embeddings2 = model(
                        images_folded, ts_part, att_mask_part, init_img_size, label_part)

                    loss01, _ = model.anomaly_detection_loss(local_embeddings1, label_part)
                    loss02, _ = model.weighted_reconstruction_loss(
                        local_embeddings2, ts_part, att_mask_part, label_part)

                    lb_loss = 0.01 * 0.5 * (load_balance_loss(m_w[0]) + load_balance_loss(m_w[1])) if m_w is not None else 0.0
                    loss2_total = loss2_total + loss02 + 0.1 * loss_cl + lb_loss
                    loss1_total = loss1_total + loss01

                batch_loss = loss1_total.item() + loss2_total.item()
            else:
                images_folded, init_img_size = model.vit_encoder.fold_image(
                    images, period, p_value, **data_setting)

                local_embeddings1, m_w, loss_cl, local_embeddings2 = model(
                    images_folded, time_series, att_mask, init_img_size, labels)

                loss1, _ = model.anomaly_detection_loss(local_embeddings1, labels)
                loss2, _ = model.weighted_reconstruction_loss(
                    local_embeddings2, time_series, att_mask, labels)
                lb_loss = 0.01 * 0.5 * (load_balance_loss(m_w[0]) + load_balance_loss(m_w[1])) if m_w is not None else 0.0
                loss2 = loss2 + lb_loss + 0.1 * loss_cl

                batch_loss = loss1.item() + loss2.item()

            total_loss += batch_loss
            num_batches += 1

            del images_folded, local_embeddings1, local_embeddings2
            del images, time_series, att_mask, labels

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    model.train()
    return avg_loss



def main(args):
    """Run the univariate training workflow."""
    from vetime.application.train import TrainUseCase
    from vetime.interfaces.cli import training_config_from_namespace

    return TrainUseCase().run(training_config_from_namespace(args))

if __name__ == "__main__":
    # Default settings as per paper (B.4 Implementation Details)
    DATA_INIT_SETTING = {
        "img_size": 224,
        "T_sqrt": False,
    }

    parser = argparse.ArgumentParser(description='VETime Training (as per paper)')
    parser.add_argument('--dataset_path', default='./dataset', type=str, help='Path to the training data')
    parser.add_argument('--dataset_test_dir', type=str, default='./dataset/TSB-AD/Datasets/TSB-AD-U')
    parser.add_argument('--file_list', type=str, default='./dataset/TSB-AD/Datasets/File_List/TSB-AD-U.csv')
    parser.add_argument('--model_name', default='VETime', type=str, help='Name of the model')
    parser.add_argument('--seed', type=int, default=64, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (paper: 32)')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of data loader workers')
    parser.add_argument('--tsb_postprocess_workers', type=int, default=4,
                        help='CPU processes for TSB metric post-processing (independent of DataLoader workers)')
    parser.add_argument('--tsb_worker_cpu_threads', type=int, default=1,
                        help='PyTorch CPU threads allowed inside each TSB post-process worker')
    parser.add_argument('--effective_batch_size', type=int, default=256,
                        help='梯度累积的目标有效 batch size，每累积这么多样本就反向传播一次 (默认: 128)')
    parser.add_argument('--dynamic_batch', action='store_true', default=False,
                        help='启用动态 batch size，短样本时自动增大 batch 以充分利用 GPU')
    parser.add_argument('--max_batch_size', type=int, default=256,
                        help='动态 batch 模式下，短样本时的 batch_size 上限 (默认: 256)')
    parser.add_argument('--padding_ratio', type=float, default=1.5,
                        help='动态 batch 模式下，同一 batch 内最大/最小长度比阈值，'
                             '超过则强制切 batch 以减少 padding 浪费 (默认: 4.0)')
    parser.add_argument('--shuffle_bucket', action='store_true', default=False,
                        help='动态 batch 模式下，在每个 epoch 内打乱同长度区间的 batch 顺序（保持宏观排序不变）')
    parser.add_argument('--use_vectorized_fold', action='store_true', default=False,
                        help='使用向量化版本的 fold_image，约 150 倍加速，固定 T_sqrt=True')
    parser.add_argument('--num_epochs', type=int, default=25, help='Epochs number (paper: 25)')
    parser.add_argument('--stage1_epochs', type=int, default=1,
                        help='纯重建预训练的 epoch 数量（两阶段训练：前 stage1_epochs 个 epoch 仅训练重建任务，切断异常分类损失）')
    parser.add_argument('--cls_warmup_ratio', type=float, default=0.5,
                        help='分类预热比例：Stage 2 首个 epoch 中，前 cls_warmup_ratio 比例的 batch 仅训练分类相关参数，其余冻结。设为 0 跳过预热')
    parser.add_argument('--query_decoder_training_mode', type=str, default='joint', choices=['joint', 'staged'],
                        help="QueryDecoder 训练模式: 'joint'=同时训练两任务(推荐), 'staged'=分阶段训练(先重构后分类)")
    parser.add_argument('--early_stop_patience', type=int, default=4, help='Early stopping patience (paper: 4)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of training data used for validation split')
    parser.add_argument('--val_mode', type=str, default='tsb', choices=['tsb', 'split'],
                        help="Validation mode: 'tsb' uses real test data (original), 'split' uses train split for early stopping")
    parser.add_argument('--output_file_path', default='./output/result.json', type=str, help='Path to the output file')
    parser.add_argument('--keep_idx_path', type=str, required=False, help='Path to the keep idx file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for evaluation')
    parser.add_argument('--data_setting', type=str, default=DATA_INIT_SETTING, help='Data settings')
    parser.add_argument('--vision_path', type=str, default='./checkpoints/weight_v', help='vision_weight directory')
    parser.add_argument('--ts_path', type=str, default=None, help='TS Encoder pre-trained weight path')
    parser.add_argument('--vetime_path', type=str, default=None, help='VETime full model weight path')
    parser.add_argument('--vision_name', type=str, default='mae_visualize_base.pth', help='vision_weight filename')
    # Optimizer parameters (as per paper)
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate (paper: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (paper: 1e-5)')
    parser.add_argument('--ts_finetune_type', type=str, default='lora', choices=['lora', 'freeze'],
                        help="TS Encoder fine-tuning type: 'lora' or 'freeze'")
    parser.add_argument('--cmrg_enabled', action='store_true', default=False,
                        help='Enable cross-modal relational guidance')
    parser.add_argument('--cmrg_num_relation_tokens', type=int, default=16,
                        help='Number of distilled CMRG relation tokens')
    parser.add_argument('--cmrg_guide_dim', type=int, default=512,
                        help='CMRG guide dimension (must match temporal d_model)')
    parser.add_argument('--cmrg_num_heads', type=int, default=8,
                        help='CMRG guide head count (must match temporal num_heads)')
    parser.add_argument('--cmrg_metric_init', choices=['identity'], default='identity',
                        help='CMRG relation metric initialization')
    parser.add_argument('--cmrg_gate_init', type=float, default=0.0,
                        help='Initial per-layer CMRG gate')
    add_cmrg_injection_mode_argument(parser)
    parser.add_argument('--cmrg_factorized', action='store_true', default=True,
                        help='Keep the CMRG relation context factorized')
    parser.add_argument('--no_cmrg_factorized', action='store_false', dest='cmrg_factorized',
                        help='Disable factorized CMRG context (unsupported by the model)')
    parser.add_argument('--cmrg_log_interval', type=int, default=100,
                        help='Global-step interval for CMRG monitoring')
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint继续训练的路径（完整状态恢复）')

    args = parser.parse_args()
    output_file_path = args.output_file_path.replace('result.json', f'{args.model_name.replace("/", "-")}_result.json')

    results = main(args)

    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)
