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


def main(args):
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

    args = parser.parse_args()
    output_file_path = args.output_file_path.replace('result.json', f'{args.model_name.replace("/", "-")}_result.json')

    results = main(args)

    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)
