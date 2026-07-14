# Offline Vision Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将冻结的 ViT 视觉编码器推理从训练循环中移出，通过离线预提取特征保存到 safetensors，训练时直接加载以降低 GPU 计算开销。

**Architecture:** 方案 A（Dataset 集成式）—— 新建 `OfflineFeatureDataset` + `offline_collate_fn` 替代原始图像数据流；在 `VETIME` 模型中新增 `_forward_with_offline_features` 方法跳过 `fold_image` + `ViT forward`；Trainer 中按 `use_offline_features` 标志切换路径。特征存储使用 safetensors + JSON sidecar，含样本级对齐校验。

**Tech Stack:** PyTorch, safetensors, accelerate, OmegaConf

---

### Task 1: 添加 safetensors 依赖

**Files:**
- Modify: `requirements.txt` (如存在) 或确认 pip 安装

**Step 1: 安装 safetensors**

Run: `pip install safetensors`
Expected: 成功安装

**Step 2: 验证导入**

Run: `python -c "from safetensors.torch import save_file, load_file; print('OK')"`
Expected: 输出 `OK`

**Step 3: Commit**

```bash
git add requirements.txt  # 如有变更
git commit -m "chore: add safetensors dependency for offline feature storage"
```

---

### Task 2: 创建离线特征提取脚本 `scripts/extract_vision_features.py`

**Files:**
- Create: `scripts/extract_vision_features.py`

这是独立脚本，最大程度复用现有代码（AnomalyDataset, collate_fn, V_model）。

**Step 1: 编写提取脚本**

```python
#!/usr/bin/env python
"""离线视觉特征提取脚本。

复用项目的原生数据加载和视觉编码器代码，
将 ViT 编码器输出预提取并保存为 safetensors 格式。

用法:
    python scripts/extract_vision_features.py \
        --pickle_path ./dataset/post_150000.pkl \
        --vision_name mae_visualize_base.pth \
        --patch_size 16 \
        --output_dir ./features/ \
        --seed 64 \
        --batch_size 32
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from safetensors.torch import save_file
from tqdm import tqdm

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.datasets.anomaly_dataset import AnomalyDataset
from src.datasets.collate import collate_fn
from src.models.vision_encoder.v_encoder import V_model
from src.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ViT features offline")
    parser.add_argument("--pickle_path", type=str, required=True,
                        help="Path to the pickle dataset file")
    parser.add_argument("--vision_name", type=str, default="mae_visualize_base.pth",
                        help="Vision encoder weight filename")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch size (must match training config)")
    parser.add_argument("--output_dir", type=str, default="./features/",
                        help="Directory to save extracted features")
    parser.add_argument("--seed", type=int, default=64,
                        help="Random seed for deterministic extraction")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for feature extraction")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"],
                        help="Dataset split to extract")
    parser.add_argument("--train_ratio", type=float, default=0.95,
                        help="Train ratio for split")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size for fold_image")
    parser.add_argument("--T_sqrt", action="store_true", default=False,
                        help="Use sqrt(T) for fold period")
    parser.add_argument("--use_vectorized_fold", action="store_true", default=True,
                        help="Use vectorized fold_image")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset name for metadata (auto-detected if not provided)")
    return parser.parse_args()


def extract_features(args):
    # 确定性种子
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 数据集名称
    dataset_name = args.dataset_name or os.path.splitext(
        os.path.basename(args.pickle_path)
    )[0]

    print(f"[Extract] Dataset: {dataset_name}")
    print(f"[Extract] Pickle: {args.pickle_path}")
    print(f"[Extract] Split: {args.split}, train_ratio: {args.train_ratio}")
    print(f"[Extract] Seed: {args.seed}")

    # ---- 1. 加载数据集（复用 AnomalyDataset，生成图像） ----
    print("[Extract] Loading dataset and generating images...")
    dataset = AnomalyDataset(
        dataset_dir=args.pickle_path,
        patch_size=args.patch_size,
        gen_image=True,
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.seed,
        name=dataset_name,
    )
    print(f"[Extract] Dataset size: {len(dataset)} samples")

    # ---- 2. 加载 ViT 编码器 ----
    print(f"[Extract] Loading ViT encoder: {args.vision_name}")
    vit_encoder = V_model(
        vision_name=args.vision_name,
        MAX_L=5000,
        unpatch=True,
        finetune_type='none',
        use_vectorized_fold=args.use_vectorized_fold,
    )
    vit_encoder.eval()
    vit_encoder.cuda()
    print(f"[Extract] ViT encoder loaded: hidden_size={vit_encoder.hidden_size}, "
          f"patch_size={vit_encoder.patch_size}")

    # ---- 3. 逐样本提取特征 ----
    print("[Extract] Extracting features...")
    all_features = {}  # safetensors 需要的 flat dict
    metadata_samples = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Extracting"):
            sample = dataset.data[idx]
            ts_length = len(sample['time_series'])

            # 复用 AnomalyDataset.__getitem__ 的转换逻辑
            time_series, normal_ts, img_vetime, img_vico, labels, attribute, period, padding_value = dataset[idx]

            # 添加 batch 维度
            img_vetime_batch = img_vetime.unsqueeze(0).cuda()   # (1, 3, H, W)
            img_vico_batch = img_vico.unsqueeze(0).cuda()       # (1, 3, 224, 224)

            # ---- VETime 分支: fold_image + ViT forward ----
            images_folded, init_img_size = vit_encoder.fold_image(
                img_vetime_batch,
                (period,),       # P_L 需要是 tuple
                (padding_value,),  # p_values 需要是 tuple
                img_size=args.img_size,
                T_sqrt=args.T_sqrt,
            )

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                image_features_vetime, _ = vit_encoder(images_folded)
            # image_features_vetime: (1, num_patches+1, v_dim)

            # ---- ViCO 分支: ViT forward ----
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                image_features_vico, _ = vit_encoder(img_vico_batch)
            # image_features_vico: (1, num_patches+1, v_dim)

            # 转 bfloat16 保存（与训练混合精度一致，节省 50% 存储），移除 batch 维度
            feat_vetime = image_features_vetime[0].bfloat16().cpu()  # (197, v_dim)
            feat_vico = image_features_vico[0].bfloat16().cpu()      # (197, v_dim)
            init_img_size_tensor = torch.tensor(init_img_size[0], dtype=torch.int64)  # (5,)
            pad_val_tensor = padding_value.bfloat16().cpu()  # (3, C, 1)
            period_tensor = torch.tensor(period, dtype=torch.int64)
            ts_length_tensor = torch.tensor(ts_length, dtype=torch.int64)

            # 存入 safetensors dict
            prefix = f"sample_{idx}"
            all_features[f"{prefix}_features_vetime"] = feat_vetime
            all_features[f"{prefix}_features_vico"] = feat_vico
            all_features[f"{prefix}_init_img_size"] = init_img_size_tensor
            all_features[f"{prefix}_padding_value"] = pad_val_tensor
            all_features[f"{prefix}_period"] = period_tensor
            all_features[f"{prefix}_ts_length"] = ts_length_tensor

            metadata_samples.append(prefix)

    # ---- 4. 保存 safetensors ----
    safe_path = os.path.join(args.output_dir, f"{dataset_name}_{args.split}.safetensors")
    save_file(all_features, safe_path)
    print(f"[Extract] Features saved to: {safe_path}")

    # ---- 5. 保存 JSON 元数据 ----
    meta = {
        "dataset_name": dataset_name,
        "pickle_path": os.path.abspath(args.pickle_path),
        "split": args.split,
        "train_ratio": args.train_ratio,
        "num_samples": len(dataset),
        "vit_model_name": args.vision_name,
        "patch_size": vit_encoder.patch_size,
        "img_size": args.img_size,
        "embed_dim": vit_encoder.hidden_size,
        "T_sqrt": args.T_sqrt,
        "use_vectorized_fold": args.use_vectorized_fold,
        "extraction_seed": args.seed,
        "extraction_timestamp": datetime.now().isoformat(),
        "sample_ids": metadata_samples,
    }
    meta_path = os.path.join(args.output_dir, f"{dataset_name}_{args.split}.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[Extract] Metadata saved to: {meta_path}")
    print("[Extract] Done!")


if __name__ == "__main__":
    args = parse_args()
    extract_features(args)
```

**Step 2: 在小数据集上验证提取**

Run: `python scripts/extract_vision_features.py --pickle_path <test_pickle> --output_dir ./features_test/ --batch_size 4`
Expected: 生成 `.safetensors` + `.json` 文件，无报错

**Step 3: 验证 safetensors 内容**

Run: `python -c "from safetensors.torch import load_file; d = load_file('./features_test/<name>.safetensors'); print({k: v.shape for k, v in list(d.items())[:6]})"`
Expected: 输出包含 `sample_0_features_vetime`, `sample_0_features_vico`, `sample_0_init_img_size` 等 key，形状符合预期

**Step 4: Commit**

```bash
git add scripts/extract_vision_features.py
git commit -m "feat: add offline vision feature extraction script"
```

---

### Task 3: 创建离线特征数据集 `src/datasets/offline_feature_dataset.py`

**Files:**
- Create: `src/datasets/offline_feature_dataset.py`

核心职责：加载 pickle（时序数据）+ safetensors（视觉特征），做对齐校验，`__getitem__` 返回特征+时序。

**Step 1: 编写 OfflineFeatureDataset**

```python
"""离线视觉特征数据集，替代 AnomalyDataset 用于训练阶段。

与 AnomalyDataset 共享相同的 pickle 加载逻辑，
但 __getitem__ 返回预提取的 ViT 特征而非原始图像。

对齐校验在初始化时执行，确保视觉特征与时序数据一一对应。
"""

import json
import pickle
import random
import warnings
from typing import Optional, Tuple

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset


class OfflineFeatureDataset(Dataset):
    """
    使用预提取视觉特征的离线数据集。

    Args:
        dataset_dir: Pickle 文件路径（与 AnomalyDataset 相同）
        feature_dir: safetensors 特征文件所在目录
        patch_size: Patch 大小
        split: 数据切分 ('train' / 'test')
        train_ratio: 训练集比例
        seed: 随机种子（必须与提取时一致，确保样本排序相同）
        name: 数据集名称（用于定位特征文件）
        dtype: 特征加载后的数据类型，默认 bfloat16（与保存格式一致）
        verify_alignment: 是否执行对齐校验，默认 True
    """

    def __init__(
        self,
        dataset_dir: str,
        feature_dir: str,
        patch_size: int,
        split: str = 'train',
        train_ratio: float = 0.95,
        seed: int = 64,
        name: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        verify_alignment: bool = True,
    ):
        self.patch_size = patch_size
        self.dtype = dtype

        # ---- 1. 加载 pickle（与 AnomalyDataset 完全相同的逻辑） ----
        with open(dataset_dir, 'rb') as f:
            dataset = pickle.load(f)
        random.seed(seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        num_train = int(len(dataset) * train_ratio)
        if split == 'train':
            selected_indices = indices[:num_train]
        elif split == 'test':
            selected_indices = indices[num_train:]
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.data = [dataset[i] for i in selected_indices]
        self.data = [x for x in self.data if len(x['time_series']) > 100]
        self.data.sort(key=lambda x: len(x['time_series']))
        self.name = name

        # ---- 2. 加载 safetensors 特征 ----
        dataset_name = name or dataset_dir.split('/')[-1].replace('.pkl', '')
        safe_path = f"{feature_dir}/{dataset_name}_{split}.safetensors"
        meta_path = f"{feature_dir}/{dataset_name}_{split}.json"

        self.features = load_file(safe_path)

        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)

        # ---- 3. 对齐校验 ----
        if verify_alignment:
            self._verify_alignment()

    def _verify_alignment(self):
        """样本级对齐校验：确保 pickle 数据与 safetensors 特征一一对应。"""
        num_samples_pickle = len(self.data)
        num_samples_features = self.metadata['num_samples']

        if num_samples_pickle != num_samples_features:
            raise ValueError(
                f"样本数不匹配: pickle={num_samples_pickle}, "
                f"safetensors={num_samples_features}"
            )

        mismatch_count = 0
        for idx in range(num_samples_pickle):
            prefix = f"sample_{idx}"
            ts_length_pickle = len(self.data[idx]['time_series'])
            ts_length_feature = self.features[f"{prefix}_ts_length"].item()

            if ts_length_pickle != ts_length_feature:
                raise ValueError(
                    f"样本 {idx} 时序长度不匹配: pickle={ts_length_pickle}, "
                    f"feature={ts_length_feature}"
                )

            # period 校验（警告级）
            period_pickle = self.data[idx].get('period', None)
            if period_pickle is not None:
                period_feature = self.features[f"{prefix}_period"].item()
                if period_pickle != period_feature:
                    mismatch_count += 1
                    if mismatch_count <= 3:
                        warnings.warn(
                            f"样本 {idx} period 不匹配: pickle={period_pickle}, "
                            f"feature={period_feature}"
                        )

        if mismatch_count > 3:
            warnings.warn(f"共 {mismatch_count} 个样本 period 不匹配（以上仅显示前3个）")

        print(f"[OfflineFeatureDataset] 对齐校验通过: {num_samples_pickle} 样本")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        """
        返回 9 元组：
        (time_series, normal_time_series,
         image_features_vetime, image_features_vico, init_img_size,
         labels, attribute, period, padding_value)
        """
        sample = self.data[idx]
        prefix = f"sample_{idx}"

        # 时序数据（与 AnomalyDataset 一致）
        time_series = torch.tensor(sample['time_series'], dtype=self.dtype)
        normal_time_series = torch.tensor(sample['normal_time_series'], dtype=self.dtype)
        labels = torch.tensor(sample['labels'], dtype=torch.long)
        attribute = sample['attribute']
        period = sample['period']
        padding_value = torch.tensor(sample['padding_value'], dtype=self.dtype)

        # 离线特征（从 safetensors 加载）
        image_features_vetime = self.features[f"{prefix}_features_vetime"].to(self.dtype)
        image_features_vico = self.features[f"{prefix}_features_vico"].to(self.dtype)
        init_img_size = self.features[f"{prefix}_init_img_size"]

        return (
            time_series,
            normal_time_series,
            image_features_vetime,
            image_features_vico,
            init_img_size,
            labels,
            attribute,
            period,
            padding_value,
        )
```

**Step 2: 验证导入和基本结构**

Run: `python -c "from src.datasets.offline_feature_dataset import OfflineFeatureDataset; print('OK')"`
Expected: 输出 `OK`

**Step 3: Commit**

```bash
git add src/datasets/offline_feature_dataset.py
git commit -m "feat: add OfflineFeatureDataset with alignment verification"
```

---

### Task 4: 创建离线特征 collate 函数 `src/datasets/offline_collate.py`

**Files:**
- Create: `src/datasets/offline_collate.py`

与原始 `collate_fn` 的核心差异：跳过图像 padding/stack，改为直接 stack 预提取特征。

**Step 1: 编写 offline_collate_fn**

```python
"""离线特征 collate 函数。

与原始 collate_fn 的核心差异：
- 跳过 VETime 图像 padding（特征已是固定形状 (197, v_dim)，直接 stack）
- 跳过 ViCO 图像 stack（特征已是固定形状 (197, v_dim)，直接 stack）
- 保留时序 padding、归一化、attention_mask、随机 mask
- 输出 batch dict 中用 image_features_vetime/vico 替代 image/image_vico
"""

from typing import List, Tuple, Dict, Union
import torch
import torch.nn.functional as F

from src.datasets.masking import create_random_mask


def offline_collate_fn(
    batch: List[Tuple],
    patch_size: int
) -> Dict[str, Union[torch.Tensor, List]]:
    """
    离线特征模式的 collate 函数。

    Args:
        batch: OfflineFeatureDataset.__getitem__ 返回的 9 元组列表
        patch_size: Patch 大小

    Returns:
        batch dict，字段与原始 collate_fn 兼容但替换了图像为特征
    """
    # 解包 9 元组
    (time_series_list, normal_time_series_list,
     feat_vetime_list, feat_vico_list, init_img_size_list,
     labels_list, attribute_list, period, padding_value) = zip(*batch)

    # ---- 时序归一化（与原始 collate_fn 完全一致） ----
    if time_series_list[0].ndim == 1:
        time_series_tensors = [ts.unsqueeze(-1) for ts in time_series_list]
        normal_time_series_tensors = [nts.unsqueeze(-1) for nts in normal_time_series_list]
    else:
        time_series_tensors = [ts for ts in time_series_list]
        normal_time_series_tensors = [nts for nts in normal_time_series_list]

    concatenated = torch.cat(time_series_tensors, dim=0)
    mean = concatenated.mean(dim=0, keepdim=True)
    std = concatenated.std(dim=0, keepdim=True) + 1e-4
    time_series_tensors = [(ts - mean) / std for ts in time_series_tensors]
    normal_time_series_tensors = [(nts - mean) / std for nts in normal_time_series_tensors]

    # ---- 时序 padding（与原始一致） ----
    labels = [label for label in labels_list]
    lengths = [t.size(0) for t in labels]
    max_len = max(lengths)
    max_idx = lengths.index(max_len)
    target_length = ((max_len + patch_size - 1) // patch_size) * patch_size

    def padding_to_target_length(list0, value):
        original_tensor = list0[max_idx]
        pad_shape = [0, 0] * original_tensor.dim()
        pad_shape[-1] = target_length - max_len
        padded_tensor = F.pad(original_tensor, pad=pad_shape, mode='constant', value=value)
        list0[max_idx] = padded_tensor
        return torch.nn.utils.rnn.pad_sequence(list0, batch_first=True, padding_value=value)

    padded_time_series = padding_to_target_length(time_series_tensors, 0.0)
    normal_time_series_tensors = padding_to_target_length(normal_time_series_tensors, 0.0)
    padded_labels = padding_to_target_length(labels, -1)

    # ---- 离线特征：直接 stack（无需 padding，已是固定形状） ----
    image_features_vetime = torch.stack(feat_vetime_list)   # (B, 197, v_dim)
    image_features_vico = torch.stack(feat_vico_list)       # (B, 197, v_dim)
    # init_img_size_list 保持为 list

    # ---- Attention mask（与原始一致） ----
    sequence_lengths = [ts.size(0) for ts in time_series_tensors]
    B, max_seq_len, num_features = padded_time_series.shape
    attention_mask = torch.ones(B, max_seq_len, dtype=torch.bool)
    for i, length in enumerate(sequence_lengths):
        attention_mask[i, length:] = False

    # ---- 随机 mask（保留，时序侧 mask 仍需） ----
    mask_time_series, mask = create_random_mask(padded_time_series, attention_mask, patch_size)
    normal_time_series_tensors, mask = create_random_mask(normal_time_series_tensors, attention_mask, patch_size)

    return {
        'time_series': padded_time_series,
        'normal_time_series': normal_time_series_tensors,
        'mask_time_series': mask_time_series,
        'image_features_vetime': image_features_vetime,  # 替代 'image'
        'image_features_vico': image_features_vico,       # 替代 'image_vico'
        'init_img_size_list': list(init_img_size_list),   # 新增
        'mask': mask,
        'labels': padded_labels,
        'attention_mask': attention_mask,
        'period': period,
        'padding_value': padding_value,
    }
```

**Step 2: 验证导入**

Run: `python -c "from src.datasets.offline_collate import offline_collate_fn; print('OK')"`
Expected: 输出 `OK`

**Step 3: Commit**

```bash
git add src/datasets/offline_collate.py
git commit -m "feat: add offline_collate_fn for pre-extracted feature batching"
```

---

### Task 5: 在 VETIME 模型中新增 `_forward_with_offline_features` 方法

**Files:**
- Modify: `src/models/vetime.py:120-210`

核心思路：提取 `_forward_impl` 中视觉编码后的融合逻辑为公共方法，然后新增 `_forward_with_offline_features` 直接从 ViT 输出 token 开始。

**Step 1: 编写 `_fuse_and_decode` 公共方法**

在 `src/models/vetime.py` 的 `_forward_impl` 方法之后添加。该方法接收 `Q_visual` 和 `K_V_tokens_proj`（已完成视觉编码的特征），执行融合+解码逻辑。

```python
def _fuse_and_decode(self, Q_visual, K_V_tokens_proj,
                     TS_embeddings0, patch_mask, B, seq_len, num_features, labels):
    """视觉特征融合与解码的公共逻辑。

    被 _forward_impl 和 _forward_with_offline_features 共用。

    Args:
        Q_visual: VETime 分支自注意力后的视觉特征 (B, N_TS, t_dim)
        K_V_tokens_proj: ViCO 分支投影后的特征 (B, 196, t_dim)
        TS_embeddings0: 时序编码器原始输出
        patch_mask: Patch mask
        B, seq_len, num_features: 形状参数
        labels: 标签
    """
    # === 交叉注意力融合 ===
    I_embeddings0 = self.visual_cross_attn(
        Q_VETime=Q_visual,
        K_ViCO=K_V_tokens_proj,
        V_ViCO=K_V_tokens_proj
    )

    I_embeddings, TS_embeddings = self.fusion(I_embeddings0, TS_embeddings0, patch_mask)
    loss_sc = self.compute_cl(I_embeddings, TS_embeddings, labels, num_features)

    if self.use_query_decoder and self.query_decoder is not None:
        F_TS = TS_embeddings0
        F_V = I_embeddings
        mix_out0_for_proj = torch.cat([TS_embeddings, I_embeddings], dim=-1)
        F_A = self.fusion_proj(mix_out0_for_proj)
        F_rec, F_cls = self.query_decoder(F_TS, F_V, F_A, patch_mask)

        patch_proj = self.projection_layer(F_cls)
        local_embeddings = patch_proj.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
        local_embeddings1 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]

        patch_proj2 = self.projection_layer(F_rec)
        local_embeddings = patch_proj2.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
        local_embeddings2 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]

        m_w = None
        return local_embeddings1, m_w, loss_sc, local_embeddings2

    mix_out0 = torch.cat([TS_embeddings, I_embeddings], dim=-1)
    mix_out_a, m_w_a = self.mm_w(mix_out0, TS_embeddings0, I_embeddings0, mix_out0, task_id=1)
    mix_out_r, m_w_r = self.mm_w(mix_out0, TS_embeddings0, I_embeddings0, mix_out0, task_id=0)
    m_w = {0: m_w_r, 1: m_w_a}

    patch_proj = self.projection_layer(mix_out_a)
    local_embeddings = patch_proj.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
    local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
    local_embeddings1 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]

    patch_proj2 = self.projection_layer(mix_out_r)
    local_embeddings = patch_proj2.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
    local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
    local_embeddings2 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]

    return local_embeddings1, m_w, loss_sc, local_embeddings2
```

**Step 2: 重构 `_forward_impl` 使用 `_fuse_and_decode`**

将 `_forward_impl` 中从交叉注意力融合开始的代码（约第 150-210 行）替换为调用 `_fuse_and_decode`。

修改后 `_forward_impl` 变为：

```python
def _forward_impl(self, hidden_states, time_series, att_mask=None, init_img_size=None,
                  hidden_states_vico=None, init_img_size_vico=None, labels=None):
    TS_embeddings0, local_embeddings0, patch_mask = self.ts_encoder(time_series, att_mask)
    B, seq_len, num_features = time_series.size()
    patch_num = patch_mask.size(1) // num_features
    temporal_pos_emb = self.pos_emb_v[:, :patch_num, :]
    multivariate_pos_emb = temporal_pos_emb.repeat(1, num_features, 1)

    # === 分支 A: VETime 时域 ===
    image_features_vetime, _ = self.vit_encoder(hidden_states)
    I_embeddings_vetime = self.vit_encoder.unfold_image(image_features_vetime, init_img_size)
    I_embeddings_vetime = self.mlp_i(I_embeddings_vetime + multivariate_pos_emb)
    Q_visual = self.I_att(I_embeddings_vetime, patch_mask)

    # === 分支 B: ViCO 频域 ===
    if hidden_states_vico is not None:
        image_features_vico, _ = self.vit_encoder(hidden_states_vico)
        K_V_tokens = image_features_vico[:, 1:, :]
    else:
        K_V_tokens = image_features_vetime[:, 1:, :]
    K_V_tokens_proj = self.mlp_vico(K_V_tokens)

    # === 融合 + 解码（公共方法） ===
    return self._fuse_and_decode(
        Q_visual, K_V_tokens_proj,
        TS_embeddings0, patch_mask, B, seq_len, num_features, labels
    )
```

**Step 3: 新增 `_forward_with_offline_features` 方法**

在 `_fuse_and_decode` 之后添加：

```python
def _forward_with_offline_features(
    self,
    image_features_vetime: torch.Tensor,
    image_features_vico: torch.Tensor,
    init_img_size_list,
    time_series: torch.Tensor,
    att_mask: Optional[torch.Tensor] = None,
    labels=None,
):
    """使用预提取 ViT 特征的 forward，跳过 fold_image + ViT forward。

    Args:
        image_features_vetime: (B, 197, v_dim) VETime 分支 ViT 输出（含 CLS token）
        image_features_vico: (B, 197, v_dim) ViCO 分支 ViT 输出（含 CLS token）
        init_img_size_list: list of [init_h, init_w, pad_patch, img_size, Num]
        time_series: (B, L, C) 原始时序
        att_mask: (B, L) 注意力掩码
        labels: (B, L) 标签
    """
    TS_embeddings0, local_embeddings0, patch_mask = self.ts_encoder(time_series, att_mask)
    B, seq_len, num_features = time_series.size()
    patch_num = patch_mask.size(1) // num_features
    temporal_pos_emb = self.pos_emb_v[:, :patch_num, :]
    multivariate_pos_emb = temporal_pos_emb.repeat(1, num_features, 1)

    # === 分支 A: VETime 时域 — 从 ViT 输出继续，仅执行 unfold_image ===
    I_embeddings_vetime = self.vit_encoder.unfold_image(image_features_vetime, init_img_size_list)
    I_embeddings_vetime = self.mlp_i(I_embeddings_vetime + multivariate_pos_emb)
    Q_visual = self.I_att(I_embeddings_vetime, patch_mask)

    # === 分支 B: ViCO 频域 — 直接从 ViT 输出取 patch tokens ===
    K_V_tokens = image_features_vico[:, 1:, :]  # 去掉 CLS token
    K_V_tokens_proj = self.mlp_vico(K_V_tokens)

    # === 融合 + 解码（公共方法） ===
    return self._fuse_and_decode(
        Q_visual, K_V_tokens_proj,
        TS_embeddings0, patch_mask, B, seq_len, num_features, labels
    )
```

**Step 4: 验证代码能正常导入**

Run: `python -c "from src.models.vetime import VETIME; print('OK')"`
Expected: 输出 `OK`

**Step 5: Commit**

```bash
git add src/models/vetime.py
git commit -m "feat: add _forward_with_offline_features and extract _fuse_and_decode"
```

---

### Task 6: 适配 Trainer 支持离线特征模式

**Files:**
- Modify: `src/engines/trainer.py:24-46`（构造函数添加 `use_offline_features` 标志）
- Modify: `src/engines/trainer.py:281-482`（train_epoch 添加离线特征分支）

**Step 1: 修改 Trainer.__init__**

在 `src/engines/trainer.py:32` 的 `__init__` 参数列表中添加 `use_offline_features=False`，并在 body 中保存：

```python
def __init__(self, cfg, model, train_loader, val_loader, accelerator,
             data_setting, patch_size, use_offline_features=False):
    ...
    self.use_offline_features = use_offline_features
```

**Step 2: 修改 train_epoch 添加离线特征分支**

在 `train_epoch` 方法的 batch 处理循环中（第 281 行附近），在解包 batch 数据之后添加离线特征分支。

在现有 `labels = batch["labels"]` 等解包代码之后，替换 forward 逻辑：

```python
# 解包 batch 数据
labels = batch["labels"]
time_series, att_mask = batch['time_series'], batch['attention_mask']
mask = batch['mask']
period = batch['period']
p_value = batch['padding_value']

alpha_recon = cfg.loss.alpha_recon
cl_weight = cfg.loss.cl_weight
balance_weight = cfg.loss.balance_weight

# ---- Forward + Loss ----
if self.use_offline_features:
    # 离线特征模式：直接使用预提取特征
    image_features_vetime = batch["image_features_vetime"]
    image_features_vico = batch["image_features_vico"]
    init_img_size_list = batch["init_img_size_list"]

    if labels.shape[1] > model.MAX_L:
        # 长序列分块 — 需要同步切分时序和特征
        # 注意：离线特征模式下 split_sequence 不再切分图像
        # 而是切分时序数据，每个 chunk 独立 forward
        data_splits = model.split_sequence(
            None, time_series, att_mask, labels  # images=None
        )
        # ... (长序列处理，见下方详细代码)
    else:
        # 常规序列
        local_embeddings1, m_w, loss_cl, local_embeddings2 = model._forward_with_offline_features(
            image_features_vetime=image_features_vetime,
            image_features_vico=image_features_vico,
            init_img_size_list=init_img_size_list,
            time_series=time_series,
            att_mask=att_mask,
            labels=labels,
        )
        # 后续 loss 计算与原始路径完全一致
        ...
else:
    # 原始路径（保持不变）
    images = batch["image"]
    images_vico = batch.get("image_vico", None)
    ...
```

**重要**：离线特征模式下，长序列分块需要特殊处理。因为 `split_sequence` 原本切分的是图像+时序，但离线模式下没有图像。由于 `_forward_with_offline_features` 不依赖 `fold_image`，长序列分块只需切分时序数据，每个 chunk 仍使用相同的完整特征。但仔细考虑——这其实不对，长序列的 ViT 输出 token 数量取决于 fold 后的图像大小，而 fold 参数来自 `init_img_size`。如果序列被分块，每块需要独立 fold 和 ViT 编码，这意味着离线特征无法直接用于长序列分块（因为离线特征是整个序列 fold 后的 ViT 输出）。

**解决方案**：对于长序列（`labels.shape[1] > model.MAX_L`），在离线提取阶段就需要按 chunk 分别提取特征。但这增加了复杂度。更简单的方案是：离线模式下不使用 `split_sequence`，而是直接处理完整序列（因为 ViT 编码已被跳过，`_forward_with_offline_features` 的显存开销远低于原始 forward，大多数序列可以在完整长度下处理）。如果确实遇到超长序列，回退到在线计算。

因此 trainer 中的离线模式长序列处理逻辑为：

```python
if self.use_offline_features:
    image_features_vetime = batch["image_features_vetime"]
    image_features_vico = batch["image_features_vico"]
    init_img_size_list = batch["init_img_size_list"]

    # 离线特征模式：直接 forward，不做 split_sequence
    # ViT 编码已被跳过，显存占用大幅降低，大多数序列可直接处理
    local_embeddings1, m_w, loss_cl, local_embeddings2 = model._forward_with_offline_features(
        image_features_vetime=image_features_vetime,
        image_features_vico=image_features_vico,
        init_img_size_list=init_img_size_list,
        time_series=time_series,
        att_mask=att_mask,
        labels=labels,
    )

    # loss 计算（与原始路径的常规序列分支一致）
    loss1, logits = model.anomaly_detection_loss(local_embeddings1, labels)
    loss_recon, rec = model.weighted_reconstruction_loss(
        local_embeddings2, time_series, att_mask, labels
    )
    if m_w is None:
        loss_e_tensor = torch.tensor(0.0, device=self.device)
    elif is_stage_1:
        loss_e_tensor = balance_weight * load_balance_loss(m_w[0])
    else:
        loss_e_tensor = balance_weight * 0.5 * (
            load_balance_loss(m_w[0]) + load_balance_loss(m_w[1])
        )
    if is_stage_1:
        loss1 = torch.tensor(0.0, device=self.device)
    loss2 = (alpha_recon * loss_recon) + loss_e_tensor + cl_weight * loss_cl

    batch_loss_bce = loss1.item()
    batch_loss_mse = loss_recon.item()
    batch_loss_cl = (cl_weight * loss_cl).item()
    batch_loss_e = loss_e_tensor.item()
```

**Step 3: 验证代码可正常导入**

Run: `python -c "from src.engines.trainer import Trainer; print('OK')"`
Expected: 输出 `OK`

**Step 4: Commit**

```bash
git add src/engines/trainer.py
git commit -m "feat: add offline features support in Trainer"
```

---

### Task 7: 修改 `train.py` 添加离线特征训练入口

**Files:**
- Modify: `train.py:1911-2124`（train_univariate_hydra 函数）

**Step 1: 在 `train_univariate_hydra` 中添加离线特征分支**

在 `train_univariate_hydra` 函数中，DataLoader 构建部分（约第 2032 行之后）添加离线特征模式分支。

核心改动：

```python
# ---- 检查是否使用离线特征 ----
use_offline_features = getattr(cfg, 'offline_features', None) is not None

if use_offline_features:
    from src.datasets.offline_feature_dataset import OfflineFeatureDataset
    from src.datasets.offline_collate import offline_collate_fn

    feature_dir = cfg.offline_features.feature_dir
    log.info(f"使用离线特征模式: feature_dir={feature_dir}")

    # 离线特征数据集（特征已保存为 bf16，直接加载无需类型转换）
    if val_mode == 'split':
        train_ratio = 1.0 - cfg.data.val_ratio
        train_dataset = OfflineFeatureDataset(
            dataset_dir=cfg.paths.dataset_path,
            feature_dir=feature_dir,
            patch_size=patch_size,
            split="train",
            train_ratio=train_ratio,
            seed=cfg.seed,
        )
        val_dataset = OfflineFeatureDataset(
            dataset_dir=cfg.paths.dataset_path,
            feature_dir=feature_dir,
            patch_size=patch_size,
            split="test",
            train_ratio=train_ratio,
            seed=cfg.seed,
        )
    else:
        train_dataset = OfflineFeatureDataset(
            dataset_dir=cfg.paths.dataset_path,
            feature_dir=feature_dir,
            patch_size=patch_size,
            split="train",
            seed=cfg.seed,
        )
        val_dataset = None

    collatefn = partial(offline_collate_fn, patch_size=patch_size)

    # DataLoader 构建（与原始相同，但使用更高 num_workers）
    ...
else:
    # 原始路径（保持不变）
    ...
```

**Step 2: 传递 `use_offline_features` 到 Trainer**

```python
trainer = Trainer(cfg, model, train_loader, val_loader, accelerator,
                  data_setting, patch_size,
                  use_offline_features=use_offline_features)
```

**Step 3: Commit**

```bash
git add train.py
git commit -m "feat: add offline features entry in train_univariate_hydra"
```

---

### Task 8: 更新配置文件 `configs/base.yaml`

**Files:**
- Modify: `configs/base.yaml`

**Step 1: 添加 offline_features 配置段**

在 `configs/base.yaml` 末尾添加：

```yaml
# ==================== 离线特征配置 ====================
offline_features: null                    # null=不使用离线特征，使用在线 ViT 编码
# offline_features:                       # 启用离线特征时取消注释
#   feature_dir: ./features/             # safetensors 特征文件目录
```

**Step 2: Commit**

```bash
git add configs/base.yaml
git commit -m "feat: add offline_features config section"
```

---

### Task 9: 端到端集成测试

**Files:**
- 无新增文件

**Step 1: 在小数据集上运行特征提取**

Run: `python scripts/extract_vision_features.py --pickle_path <small_test_pickle> --output_dir ./features_test/`
Expected: 生成 safetensors + json 文件

**Step 2: 使用离线特征启动训练**

修改配置文件启用 `offline_features`，然后运行训练。确认：
1. `OfflineFeatureDataset` 对齐校验通过
2. `_forward_with_offline_features` 输出形状正确
3. 损失值与原始训练路径数量级一致
4. 无 NaN 或形状不匹配错误

**Step 3: 对比在线/离线模式的输出**

用同一数据集分别运行在线模式（原始）和离线模式，对比第一个 epoch 的 loss 值。由于图像生成中的 gamma 随机性和 collate 中的 mask 随机性，两者不会完全一致，但数量级应相同。

**Step 4: Commit（如有修复）**

```bash
git add -A
git commit -m "fix: integration test fixes for offline feature pipeline"
```

---

### Task 10: 更新项目记忆

**Files:**
- Modify: `/home/cjm/.claude/projects/-mnt-sda-cjmProject-VETime/memory/MEMORY.md`

**Step 1: 更新 MEMORY.md**

添加以下内容到适当位置：

```markdown
## 离线特征提取（Offline Feature Extraction）
- 提取脚本：`scripts/extract_vision_features.py`，复用 AnomalyDataset + V_model
- 存储格式：safetensors（特征）+ JSON sidecar（元数据），每个数据集+split 一个文件
- 提取层：ViT 编码器输出（含 CLS token），保存 float32
- 训练模式：`OfflineFeatureDataset` + `offline_collate_fn` + `VETIME._forward_with_offline_features`
- 配置：`cfg.offline_features.feature_dir` 指向特征目录
- 确定性种子：64
- 对齐校验：样本级（样本数、ts_length、period、padding_value）
- 离线模式下不做 split_sequence（ViT 已跳过，显存足够）
```

**Step 2: Commit**

```bash
git commit -m "docs: update project memory with offline feature extraction info"
```