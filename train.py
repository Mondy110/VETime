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
from typing import Dict, Any
import pandas as pd
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from Test_TSB import PASS_LIST, TSB_test
from evaluation.metrics import fast_get_metrics
from model.Vision_encoder.V_encoder import V_model
from loss.loss import load_balance_loss
from model.TS_encoder.ts_model import TS_Model
from model.TS_encoder.config import default_config_t
from dataset.dataloader import AnomalyDataset, collate_fn
import logging
from tqdm.auto import tqdm
import os
from datetime import datetime
from model.VETime import VETIME
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


def create_model(args, num_features: int, vision_model, config_v):
    """
    创建 VETime 模型（复用已存在的 vision_model）

    根据 num_features 自动构建对应维度的组件：
    - num_features=1: 不创建 BinaryAttentionBias
    - num_features>1: 创建 BinaryAttentionBias

    Args:
        args: 命令行参数
        num_features: 当前数据集的特征维度
        vision_model: 已实例化的视觉编码器（复用，不重建）
        config_v: 视觉模型配置

    Returns:
        model: VETIME 模型
        ts_model: 时间序列编码器
    """
    # 更新 TS Encoder 配置
    default_config_t.num_features = num_features

    # 仅实例化 TS Encoder（根据 num_features 自动构建 BinaryAttentionBias）
    ts_model = TS_Model(default_config_t)

    # 构建 VETime 完整模型
    model = VETIME(config_v, vision_model, default_config_t, ts_model, args.model_name)

    return model, ts_model


def train_univariate(args):
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

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=4,
        log_with="tensorboard",
        project_dir="./output/logs"
    )

    logger.info(f"Using {accelerator.num_processes} {'GPUs' if accelerator.num_processes > 1 else 'CPU'}")

    # ========== Vision Encoder (Frozen MAE, as per paper) ==========
    print(f"[INFO] 正在加载 Vision Encoder (MAE) 权重: checkpoints/weight_v/{args.vision_name}")
    # finetune_type='none' means fully frozen (as per paper: "the encoder of the frozen MAE")
    vision_model = V_model(args.vision_name, MAX_L=5000, unpatch=True, finetune_type='none')
    print(f"[INFO] Vision Encoder 权重加载完成！Patch Size: {vision_model.patch_size}, Hidden Size: {vision_model.hidden_size}")
    print(f"[INFO] Vision Encoder 状态: 完全冻结 (as per paper)")

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

    ts_model = TS_Model(default_config_t)
    if args.ts_path is not None:
        print(f"[INFO] 正在加载 TS Encoder 权重: {args.ts_path}")
        state_ts_dict = torch.load(args.ts_path, map_location='cpu')['model_state_dict']

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

    # Freeze 模式：选择性冻结（仅冻结核心骨干，保留任务头可训练）
    if args.ts_finetune_type == 'freeze':
        frozen_layers = []
        trainable_layers = []
        for name, param in ts_model.named_parameters():
            if any(key in name for key in ['transformer_encoder', 'embedding_layer', 'rope_embedder']):
                param.requires_grad = False
                frozen_layers.append(name)
            else:
                param.requires_grad = True
                trainable_layers.append(name)
        print(f"[INFO] TS Encoder 开启选择性冻结：")
        print(f"  已冻结核心骨干: {len(frozen_layers)} 个张量")
        print(f"  保持可训练 (projection/anomaly/reconstruction): {len(trainable_layers)} 个张量")

    # ========== Create VETime Model ==========
    model = VETIME(config_v, vision_model, default_config_t, ts_model, args.model_name)
    if args.vetime_path is not None:
        print(f"[INFO] 正在加载 VETime 完整权重: {args.vetime_path}")
        state_dict = torch.load(args.vetime_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"[INFO] VETime 权重加载完成（用于继续训练）")
    else:
        print(f"[INFO] 未指定 --vetime_path，VETime 融合模块从头训练")

    del vision_model, ts_model

    # Print trainable parameters statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] 模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    # ========== Dataset and DataLoader ==========
    collatefn = partial(collate_fn, patch_size=patch_size)
    train_dataset = AnomalyDataset(args.dataset_path, patch_size=patch_size, split="train")

    # 为DataLoader设置随机种子生成器
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=collatefn, shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, persistent_workers=False,
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

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    model.train()
    global_step = 0
    epochs = args.num_epochs
    output = []
    device = accelerator.device
    data_setting = args.data_setting
    img_size = data_setting['img_size']
    name_save = f'./output/{args.model_name}__{img_size}_best.pth'

    early_stopping = EarlyStopping(patience=args.early_stop_patience, verbose=True, path=name_save)
    output_path0 = f'./output/score/uni/{args.model_name}_train'
    os.makedirs(output_path0, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_probs, all_preds, all_labels = [], [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}[Train]", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            labels = batch["labels"]
            images = batch["image"]  # (B, C, H, W)
            time_series, att_mask = batch['time_series'], batch['attention_mask']
            mask = batch['mask']
            period = batch['period']
            p_value = batch['padding_value']

            if labels.shape[1] > model.MAX_L:
                data_splits = model.split_data(images, time_series, att_mask, labels)
                loss1 = 0
                loss2 = 0
                logits_list = []
                for data_part in data_splits:
                    img_part, ts_part, att_mask_part, label_part = data_part
                    images_folded, init_img_size = model.vit_encoder.fold_image(img_part, period, p_value, **data_setting)

                    local_embeddings1, m_w, loss_cl, local_embeddings2 = model(images_folded, ts_part, att_mask_part, init_img_size, label_part)

                    loss01, logit = model.anomaly_detection_loss(local_embeddings1, label_part)

                    loss02, rec = model.weighted_reconstruction_loss(local_embeddings2, ts_part, att_mask_part, label_part)
                    loss2 = loss2 + loss02
                    loss2 = loss2 + 0.1 * loss_cl + 0.2 * load_balance_loss(m_w)
                    loss1 = loss1 + loss01
                    logits_list.append(logit)

                logits = torch.cat(logits_list, dim=1)

            else:
                images_folded, init_img_size = model.vit_encoder.fold_image(images, period, p_value, **data_setting)

                local_embeddings1, m_w, loss_cl, local_embeddings2 = model(images_folded, time_series, att_mask, init_img_size, labels)

                loss1, logits = model.anomaly_detection_loss(local_embeddings1, labels)

                loss2, rec = model.weighted_reconstruction_loss(local_embeddings2, time_series, att_mask, labels)

                loss2 = loss2 + 0.2 * load_balance_loss(m_w) + 0.1 * loss_cl

            accelerator.backward(loss1 + loss2)

            global_step += 1
            if global_step % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss1.item() + loss2.item()
            total_loss += batch_loss
            progress_bar.set_postfix({"loss": batch_loss})

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
            del local_embeddings1, local_embeddings2, m_w, loss_cl, rec, mask, period, p_value
            del images, time_series, att_mask, init_img_size
            torch.cuda.empty_cache()

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

        avg_train_loss = total_loss / len(train_loader)
        accelerator.log({"epoch_train_loss": avg_train_loss}, step=epoch)
        print(f"\n[Epoch {epoch + 1}/{epochs}] 🟩 Training Summary:")
        print(f"  Avg Loss: {avg_train_loss:.4f}")

        # epoch结束后清理大数组（保留 train_metrics 供后续使用）
        del all_probs, all_preds, all_labels
        gc.collect()

        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            model.eval()
            avg_val_loss = TSB_test(model, args, args.data_setting, device, dataset_setting=PASS_LIST, verbose=False)
            # 验证后清理内存
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
                "val_loss": round(avg_val_loss, 6) if avg_val_loss is not None else None,
            }
            output.append(epoch_log)

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            # 验证后清理
            model.train()
            gc.collect()
            torch.cuda.empty_cache()

        # 最后清理 train_metrics
        del train_metrics
        gc.collect()

    loss_all = TSB_test(model, args, args.data_setting, device, dataset_setting=PASS_LIST, verbose=False)
    print(f"Final validation loss: {loss_all}")
    accelerator.end_training()
    logger.info("Training completed!")

    return output


def evaluate_multivariate(model, val_loader, accelerator, device, data_setting, vision_model):
    """
    多变量训练验证函数（参考 TimeRCD）

    Returns:
        avg_val_loss: 平均验证损失
    """
    model.eval()
    vision_model.eval()
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

                    loss2_total = loss2_total + loss02 + 0.1 * loss_cl + 0.2 * load_balance_loss(m_w)
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
                loss2 = loss2 + 0.2 * load_balance_loss(m_w) + 0.1 * loss_cl

                batch_loss = loss1.item() + loss2.item()

            total_loss += batch_loss
            num_batches += 1

            del images_folded, local_embeddings1, local_embeddings2
            del images, time_series, att_mask, labels
            torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    model.train()
    return avg_loss


def train_multivariate(args, config: Dict[str, Any]):
    """
    多变量顺序训练（参考 TimeRCD 范式）

    核心策略：
    1. 销毁重建：每次切换维度时全新实例化模型
    2. 知识延续：strict=False 加载 checkpoint
    3. 参数纳管：重建 Optimizer 确保所有参数可训练
    4. 显存安全：Accelerator 循环外初始化，Vision Encoder 单次实例化
    5. 动态梯度累积：目标有效 batch size = 128
    6. 早停机制：基于验证集 loss（95%/5% 划分）

    Args:
        args: 命令行参数
        config: YAML 配置字典
    """
    mv_config = config['multivariate']
    datasets = mv_config['datasets']
    dim_batch_size_map = mv_config['dim_batch_size_map']
    checkpoint_dir = mv_config['checkpoint_dir']
    enforce_dim_order = mv_config.get('enforce_dim_order', False)

    # 目标有效 batch size（用于动态梯度累积）
    TARGET_EFFECTIVE_BATCH_SIZE = 128

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 设置随机种子
    set_seed(args.seed)
    print(f"[INFO] 随机种子已设置: {args.seed}")

    # ========== [关键] Accelerator 全局初始化（只执行一次）==========
    # 注意：gradient_accumulation_steps 会在每个维度动态设置
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,  # 初始值，会在每个维度动态更新
        log_with="tensorboard",
        project_dir="./output/logs"
    )
    device = accelerator.device

    logger.info(f"Using {accelerator.num_processes} {'GPUs' if accelerator.num_processes > 1 else 'CPU'}")

    # ========== [关键] Vision Encoder 只实例化一次 ==========
    print(f"[INFO] 正在加载 Vision Encoder (MAE) 权重: checkpoints/weight_v/{args.vision_name}")
    vision_model = V_model(
        args.vision_name,
        MAX_L=3000,
        unpatch=True,
        finetune_type='none'
    )
    # 移至正确的设备并开启 eval 模式
    vision_model = vision_model.to(device)
    vision_model.eval()

    # 【防御性代码】显式冻结所有参数，防止意外梯度计算
    for param in vision_model.parameters():
        param.requires_grad = False

    config_v = vision_model.config
    if 'mae' in args.vision_name:
        patch_size = config_v['patch_size']
    else:
        patch_size = config_v.patch_size
    args.patch_size = patch_size

    print(f"[INFO] Vision Encoder 权重加载完成！Patch Size: {patch_size}")
    print(f"[INFO] Vision Encoder 状态: 完全冻结 (作为特征对齐锚点)")

    # 检查维度顺序（可选）
    if enforce_dim_order:
        dims = [ds['dim'] for ds in datasets]
        if dims != sorted(dims):
            raise ValueError(f"数据集维度顺序非递增: {dims}，请调整 datasets 列表顺序或设置 enforce_dim_order: false")

    # 设置 LoRA 配置
    if args.ts_finetune_type == 'lora':
        default_config_t.use_lora = True
        print(f"[INFO] TS Encoder 微调类型: LoRA (r={default_config_t.lora_r}, α={default_config_t.lora_alpha})")
    else:
        default_config_t.use_lora = False
        print(f"[INFO] TS Encoder 微调类型: 完全冻结")

    # 全局 checkpoint 路径（用于维度间传递）
    prev_checkpoint_path = None

    # 数据设置
    data_setting = args.data_setting

    # ========== 外层维度循环 ==========
    for dataset_idx, dataset_info in enumerate(datasets):
        dataset_path = dataset_info['path']
        current_dim = dataset_info['dim']
        batch_size = dim_batch_size_map.get(str(current_dim), 1)

        # ========== 动态计算梯度累积步数 ==========
        accumulation_steps = max(1, TARGET_EFFECTIVE_BATCH_SIZE // batch_size)
        accelerator.gradient_accumulation_steps = accumulation_steps

        print(f"\n{'='*60}")
        print(f"[多变量训练] 数据集 {dataset_idx+1}/{len(datasets)}: {dataset_path}")
        print(f"  维度: {current_dim}, Batch Size: {batch_size}")
        print(f"  梯度累积步数: {accumulation_steps} (有效 Batch Size: {batch_size * accumulation_steps})")
        if prev_checkpoint_path is not None:
            print(f"  继承上一维度权重: {prev_checkpoint_path}")
        print(f"{'='*60}\n")

        # ========== 1. 刷新配置 ==========
        default_config_t.num_features = current_dim
        args.batch_size = batch_size

        # ========== 2. 全新实例化模型（复用 vision_model）==========
        model, ts_model = create_model(args, current_dim, vision_model, config_v)

        # ========== 3. 知识延续（strict=False 是灵魂）==========
        if prev_checkpoint_path is not None:
            print(f"[INFO] 加载上一维度 checkpoint: {prev_checkpoint_path}")
            state_dict = torch.load(prev_checkpoint_path, map_location='cpu')

            # 【防御性代码】剥离 DDP 的 'module.' 前缀，适配不同保存格式
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] 权重加载完成 (strict=False)")
            if missing:
                print(f"  缺失的参数（新维度组件，使用初始化值）: {len(missing)} 个")
            if unexpected:
                print(f"  未预期的参数: {len(unexpected)} 个")

        # 打印参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[INFO] 模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

        # ========== 4. 重建 Optimizer ==========
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params_list,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        print(f"\n[INFO] 优化器配置: AdamW, lr={args.learning_rate}, weight_decay={args.weight_decay}")

        # ========== 5. 创建 DataLoader（95%/5% 划分训练/验证集）==========
        train_dataset = AnomalyDataset(dataset_path, patch_size=patch_size, split="train", train_ratio=1.0)
        val_dataset = AnomalyDataset(dataset_path, patch_size=patch_size, split="test", train_ratio=0)

        print(f"[INFO] 数据集划分: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")

        g = torch.Generator()
        g.manual_seed(args.seed)

        collatefn = partial(collate_fn, patch_size=patch_size)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collatefn,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=g
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collatefn,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=g
        )

        # ========== 6. prepare 动态组件 ==========
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader)

        # ========== 7. 设置训练模式（注意：vision_model 需保持 eval）==========
        model.train()
        vision_model.eval()

        # ========== 8. 早停机制初始化（基于验证集 loss）==========
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = args.early_stop_patience

        # ========== 9. 内层 Epoch 训练 ==========
        global_step = 0
        epochs = args.num_epochs
        img_size = data_setting['img_size']

        for epoch in range(epochs):
            model.train()
            vision_model.eval()  # 确保每个 epoch 开始时 vision_model 保持 eval

            total_loss = 0
            all_probs, all_preds, all_labels = [], [], []

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}[Train]",
                               disable=not accelerator.is_local_main_process)

            for batch in progress_bar:
                labels = batch["labels"]
                images = batch["image"]
                time_series, att_mask = batch['time_series'], batch['attention_mask']
                mask = batch['mask']
                period = batch['period']
                p_value = batch['padding_value']

                if labels.shape[1] > model.MAX_L:
                    data_splits = model.split_data(images, time_series, att_mask, labels)
                    loss1 = 0
                    loss2 = 0
                    logits_list = []

                    for data_part in data_splits:
                        img_part, ts_part, att_mask_part, label_part = data_part
                        images_folded, init_img_size = model.vit_encoder.fold_image(
                            img_part, period, p_value, **data_setting)

                        local_embeddings1, m_w, loss_cl, local_embeddings2 = model(
                            images_folded, ts_part, att_mask_part, init_img_size, label_part)

                        loss01, logit = model.anomaly_detection_loss(local_embeddings1, label_part)
                        loss02, rec = model.weighted_reconstruction_loss(
                            local_embeddings2, ts_part, att_mask_part, label_part)

                        loss2 = loss2 + loss02 + 0.1 * loss_cl + 0.2 * load_balance_loss(m_w)
                        loss1 = loss1 + loss01
                        logits_list.append(logit)

                    logits = torch.cat(logits_list, dim=1)
                else:
                    images_folded, init_img_size = model.vit_encoder.fold_image(
                        images, period, p_value, **data_setting)

                    local_embeddings1, m_w, loss_cl, local_embeddings2 = model(
                        images_folded, time_series, att_mask, init_img_size, labels)

                    loss1, logits = model.anomaly_detection_loss(local_embeddings1, labels)
                    loss2, rec = model.weighted_reconstruction_loss(
                        local_embeddings2, time_series, att_mask, labels)
                    loss2 = loss2 + 0.2 * load_balance_loss(m_w) + 0.1 * loss_cl

                accelerator.backward(loss1 + loss2)

                global_step += 1
                if global_step % accelerator.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                batch_loss = loss1.item() + loss2.item()
                total_loss += batch_loss
                progress_bar.set_postfix({"loss": batch_loss})

                probs = torch.softmax(logits, dim=-1)[:, :, 1]
                preds = (probs > 0.5).float()
                probs, preds, labels = accelerator.gather_for_metrics((probs, preds, labels))

                if global_step % 10 == 0:
                    for i in range(probs.shape[0]):
                        all_probs.append(probs[i].detach().cpu().numpy().reshape(-1))
                        all_preds.append(preds[i].detach().cpu().numpy().reshape(-1))
                        all_labels.append(labels[i].detach().cpu().numpy().reshape(-1).astype(int))

                del images_folded, logits, loss1, probs, preds, labels, loss2
                del local_embeddings1, local_embeddings2, m_w, loss_cl, rec, mask, period, p_value
                del images, time_series, att_mask, init_img_size
                torch.cuda.empty_cache()

            # 计算 epoch 训练指标
            if len(all_probs) > 0:
                all_probs_arr = np.concatenate(all_probs)
                all_preds_arr = np.concatenate(all_preds)
                all_labels_arr = np.concatenate(all_labels)
                train_metrics = fast_get_metrics(all_probs_arr, all_labels_arr)
                for k, v in train_metrics.items():
                    print(f"  Train {k}: {v:.4f}")
                del all_probs_arr, all_preds_arr, all_labels_arr
                gc.collect()
            else:
                train_metrics = {}

            avg_train_loss = total_loss / len(train_loader)
            accelerator.log({"epoch_train_loss": avg_train_loss}, step=epoch)
            print(f"\n[Epoch {epoch + 1}/{epochs}] Training Summary:")
            print(f"  Avg Train Loss: {avg_train_loss:.4f}")

            del all_probs, all_preds, all_labels
            gc.collect()

            # ========== 验证阶段（每个 epoch 都验证）==========
            avg_val_loss = evaluate_multivariate(model, val_loader, accelerator, device, data_setting, vision_model)
            print(f"  Avg Val Loss: {avg_val_loss:.4f}")

            accelerator.log({"epoch_val_loss": avg_val_loss}, step=epoch)

            # ========== 早停判断（基于验证 loss）==========
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"  ✓ New best validation loss: {best_val_loss:.4f}")

                # 保存最佳模型
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                best_model_path = os.path.join(checkpoint_dir, f'vetime_dim{current_dim}_best.pth')
                if accelerator.is_main_process:
                    torch.save(unwrapped_model.state_dict(), best_model_path)
                    print(f"  ✓ Best model saved: {best_model_path}")
            else:
                patience_counter += 1
                print(f"  ✗ Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")

            # ========== 早停触发检查 ==========
            if patience_counter >= early_stopping_patience:
                print(f"\n[早停] 维度 {current_dim} 训练提前终止于 epoch {epoch + 1}")
                print(f"  最佳验证损失: {best_val_loss:.4f}")
                break

            gc.collect()
            torch.cuda.empty_cache()

            del train_metrics
            gc.collect()

        # ========== 10. 保存维度最终 Checkpoint ==========
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint_path = os.path.join(checkpoint_dir, f'vetime_dim{current_dim}_final.pth')

        if accelerator.is_main_process:
            torch.save(unwrapped_model.state_dict(), checkpoint_path)
            print(f"[INFO] 维度 {current_dim} 最终 Checkpoint 已保存: {checkpoint_path}")

        prev_checkpoint_path = checkpoint_path

        # ========== 11. 彻底清理显存 ==========
        del model, optimizer, train_loader, val_loader, ts_model, train_dataset, val_dataset
        accelerator.free_memory()
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*60}")
    print("[多变量训练] 所有数据集训练完成！")
    print(f"{'='*60}")

    accelerator.end_training()
    return {"status": "completed", "datasets": len(datasets)}


def main(args):
    """
    主入口：根据配置选择训练模式

    - univariate: 单变量训练（原有逻辑）
    - multivariate: 多变量顺序训练
    """
    # 加载配置文件
    if args.config is not None:
        config = load_config(args.config)
        training_mode = config.get('training_mode', 'univariate')
        print(f"[INFO] 加载配置文件: {args.config}")
        print(f"[INFO] 训练模式: {training_mode}")
    else:
        config = None
        training_mode = 'univariate'

    if training_mode == 'univariate':
        return train_univariate(args)
    elif training_mode == 'multivariate':
        if config is None:
            raise ValueError("多变量训练模式需要指定 --config 参数")
        return train_multivariate(args, config)
    else:
        raise ValueError(f"未知的训练模式: {training_mode}")


if __name__ == "__main__":
    # Default settings as per paper (B.4 Implementation Details)
    DATA_INIT_SETTING = {
        "img_size": 224,
        "T_sqrt": False,
    }

    parser = argparse.ArgumentParser(description='VETime Training (as per paper)')
    parser.add_argument('--config', type=str, default=None,
                        help='训练配置文件路径（YAML格式），指定后将忽略部分命令行参数')
    parser.add_argument('--dataset_path', default='./dataset', type=str, help='Path to the training data')
    parser.add_argument('--dataset_test_dir', type=str, default='./dataset/TSB-AD/Datasets/TSB-AD-U')
    parser.add_argument('--file_list', type=str, default='./dataset/TSB-AD/Datasets/File_List/TSB-AD-U.csv')
    parser.add_argument('--model_name', default='VETime', type=str, help='Name of the model')
    parser.add_argument('--seed', type=int, default=64, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (paper: 32)')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of data loader workers')
    parser.add_argument('--num_epochs', type=int, default=25, help='Epochs number (paper: 25)')
    parser.add_argument('--early_stop_patience', type=int, default=4, help='Early stopping patience (paper: 4)')
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
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint继续训练的路径（完整状态恢复）')
    parser.add_argument('--pretrain_from', type=str, default=None,
                        help='预训练权重路径（仅模型权重），用于启动多变量训练')

    args = parser.parse_args()
    output_file_path = args.output_file_path.replace('result.json', f'{args.model_name.replace("/", "-")}_result.json')

    results = main(args)

    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)
