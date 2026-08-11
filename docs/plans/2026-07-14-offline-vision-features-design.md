# 离线视觉特征提取设计

> 日期: 2026-07-14
> 状态: 设计完成

## 1. 目标

将冻结的视觉编码器（ViT）的推理计算从训练循环中移出，通过离线预提取保存到磁盘，训练时直接加载特征，降低 GPU 计算开销和显存占用。

## 2. 核心约束

1. **流水线同源**：离线提取脚本最大程度复用现有项目的原生函数和类，不重写转换逻辑
2. **时序精准对齐**：视觉特征与原始时序在时间戳和样本维度 100% 对齐，含样本级校验机制
3. **确定性可复现**：固定随机种子 64，确保同一数据集的提取结果完全一致

## 3. 提取层截断

在 ViT 编码器输出层截断，保存：
- **VETime 分支**：`fold_image` → `vit_encoder.forward()` 的输出 `(num_patches+1, v_dim)`，含 CLS token
- **ViCO 分支**：`vit_encoder.forward()` 的输出 `(num_patches+1, v_dim)`，含 CLS token
- **fold 参数**：`init_img_size = [init_h, init_w, pad_patch, img_size, Num]`

训练时跳过 `fold_image` + `vit_encoder.forward()`，但保留 `unfold_image()`（因 batch padding 影响 token 数量）。

## 4. 存储格式

### 4.1 Safetensors 特征文件

每个数据集一个 `.safetensors` 文件，key 命名规则：

```
sample_{idx}_features_vetime    # (num_patches+1, v_dim) float32
sample_{idx}_features_vico     # (num_patches+1, v_dim) float32
sample_{idx}_init_img_size     # (5,) int64 — [init_h, init_w, pad_patch, img_size, Num]
sample_{idx}_padding_value     # (3, C, 1) float32
sample_{idx}_period            # scalar int64
sample_{idx}_ts_length         # scalar int64 — 原始时序长度
```

### 4.2 JSON 元数据 sidecar

同名 `.json` 文件，包含：
```json
{
  "dataset_name": "SMD",
  "pickle_path": "/path/to/dataset.pkl",
  "num_samples": 28,
  "vit_model_name": "mae_base",
  "patch_size": 16,
  "img_size": 224,
  "embed_dim": 768,
  "extraction_seed": 64,
  "extraction_timestamp": "2026-07-14T12:00:00",
  "sample_ids": ["sample_0", "sample_1", ...]
}
```

### 4.3 数据类型策略

- 保存时统一用 **bfloat16**（与训练混合精度一致，节省 50% 存储空间）
- 15 万样本场景：bf16 ≈ 82.5 GB vs float32 ≈ 165 GB，节省约 82.5 GB
- 加载时直接使用 bf16 tensor，无需类型转换，与 autocast 无缝兼容
- ViT 推理时使用 `torch.no_grad()` + `autocast(dtype=bfloat16)` 以加速提取
- `init_img_size`、`period`、`ts_length` 等整型数据保持 int64
- `padding_value` 保持 bfloat16（与特征一致）

## 5. 组件设计

### 5.1 离线提取脚本 `scripts/extract_vision_features.py`

**职责**：遍历数据集，提取并保存 ViT 特征。

**流程**：
1. 加载配置（pickle_path, vit_checkpoint, patch_size, output_dir）
2. 设置确定性种子 64
3. 初始化 `AnomalyDataset(gen_image=True)` — 复用现有图像生成逻辑
4. 初始化 `V_model`（ViT 编码器），加载预训练权重
5. 逐 batch 遍历：
   - `collate_fn` 组装 batch（固定种子保证确定性）
   - `fold_image()` 折叠图像
   - `vit_encoder.forward()` 提取 VETime 分支特征
   - `vit_encoder.forward()` 提取 ViCO 分支特征
   - 收集每个样本的特征 + 元数据
6. 保存为 safetensors + JSON

**复用清单**：
- `AnomalyDataset` — 数据加载 + 图像生成
- `collate_fn` — batch 组装
- `V_model.fold_image()` — 图像折叠
- `V_model.forward()` — ViT 编码

**确定性措施**：
- `random.seed(64)`, `np.random.seed(64)`, `torch.manual_seed(64)`
- gamma 校正的随机 shuffle 受种子控制
- collate_fn 中的随机 mask 受种子控制

### 5.2 离线特征数据集 `src/datasets/offline_feature_dataset.py`

**类**：`OfflineFeatureDataset(Dataset)`

**`__init__`**：
1. 加载同一个 pickle 文件，获取样本列表（与 `AnomalyDataset` 相同的排序/过滤逻辑）
2. 加载 safetensors 特征文件到内存（mmap 模式，按需读取）
3. 加载 JSON 元数据
4. **对齐校验**：
   - pickle 样本数 == safetensors 特征条目数
   - 逐样本：`len(time_series)` == `ts_length`
   - 逐样本：`period` 一致
   - 逐样本：`padding_value` 一致

**`__getitem__`** 返回 9 元组：
| 位置 | 内容 | 形状 | 说明 |
|---|---|---|---|
| 0 | time_series | (L, C) | 原始时序 |
| 1 | normal_time_series | (L, C) | 正常参考时序 |
| 2 | image_features_vetime | (197, v_dim) | VETime 分支 ViT 输出 |
| 3 | image_features_vico | (197, v_dim) | ViCO 分支 ViT 输出 |
| 4 | init_img_size | (5,) | fold 参数 |
| 5 | labels | (L,) | 异常标签 |
| 6 | attribute | dict | 元数据 |
| 7 | period | int | 周期 |
| 8 | padding_value | (3, C, 1) | 填充值 |

**对比原始 `AnomalyDataset.__getitem__`**：
- 移除：`image_vetime` (pos 2), `image_vico` (pos 3)
- 新增：`image_features_vetime` (pos 2), `image_features_vico` (pos 3), `init_img_size` (pos 4)
- 其余位置和内容不变

### 5.3 离线特征 Collate `src/datasets/offline_collate.py`

**函数**：`offline_collate_fn(batch, patch_size)`

与原始 `collate_fn` 的差异：
| 操作 | 原始 collate_fn | offline_collate_fn |
|---|---|---|
| 时序归一化 | batch 级 z-score | 相同 |
| 时序 padding | pad 到 patch_size 倍数 | 相同 |
| attention_mask | 生成 | 相同 |
| VETime 图像 padding | `image_right_padding()` | **跳过**（特征直接 stack） |
| ViCO 图像 stack | `torch.stack()` | **跳过**（特征直接 stack） |
| 随机 mask | `create_random_mask()` | **保留**（时序侧 mask 仍需） |

**输出 batch dict 新增字段**：
- `image_features_vetime`: `(B, 197, v_dim)` — 直接 stack
- `image_features_vico`: `(B, 197, v_dim)` — 直接 stack
- `init_img_size_list`: `list[(5,)]` — 逐样本 fold 参数

**移除字段**：
- `image` — 不再需要原始 VETime 图像
- `image_vico` — 不再需要原始 ViCO 图像

### 5.4 模型适配 `src/models/vetime.py`

新增方法 `_forward_with_offline_features`：

```python
def _forward_with_offline_features(
    self,
    image_features_vetime,  # (B, 197, v_dim) 预提取特征
    image_features_vico,    # (B, 197, v_dim) 预提取特征
    init_img_size_list,     # list[(5,)] fold 参数
    time_series,            # (B, L, C)
    att_mask,               # (B, L)
    labels=None,
):
    """使用预提取 ViT 特征，跳过 fold_image + ViT forward"""
    # VETime 分支：unfold_image 恢复 1D 时间顺序
    I_embeddings_vetime = self.vit_encoder.unfold_image(
        image_features_vetime, init_img_size_list
    )
    # ViCO 分支：去掉 CLS token
    K_V_tokens = image_features_vico[:, 1:, :]
    # 后续融合逻辑与 _forward_impl 完全相同
    ...
```

**关键**：`_forward_with_offline_features` 的融合部分（`mlp_i`, `V_Attention`, `GatedTimeFrequencyFusion`, `VTS_Alignment` 等）与 `_forward_impl` 完全相同，通过提取公共方法避免代码重复。

### 5.5 Trainer 适配 `src/engines/trainer.py`

`train_epoch` 中根据 `use_offline_features` 标志选择调用路径：

```python
if self.use_offline_features:
    local_embeddings1, m_w, loss_cl, local_embeddings2 = model._forward_with_offline_features(
        image_features_vetime=batch["image_features_vetime"],
        image_features_vico=batch["image_features_vico"],
        init_img_size_list=batch["init_img_size_list"],
        time_series=time_series,
        att_mask=att_mask,
        labels=labels,
    )
else:
    # 原始路径：fold_images → model.forward
    ...
```

### 5.6 多线程 DataLoader 配置

```python
dataloader = DataLoader(
    offline_dataset,
    batch_sampler=batch_sampler,
    collate_fn=offline_collate_fn,
    num_workers=4,            # 多线程预取
    pin_memory=True,          # GPU 直传
    prefetch_factor=2,        # 预取 batch 数
    persistent_workers=False,  # 常驻 worker 避免重启开销
)
```

离线特征是纯 CPU tensor（无图像解码开销），`num_workers` 可设更高，预取效率远高于原始图像 DataLoader。

## 6. 数据流对比

```
【原始流水线】
pickle → AnomalyDataset(图像+时序)
  → collate_fn(图像padding+时序padding+归一化)
    → model.fold_images(1D→2D)
      → vit_encoder.forward(ViT编码)        ← GPU 密集
        → unfold_image(2D→1D)
          → VTS融合 → 损失

【离线特征流水线】
pickle → OfflineFeatureDataset(特征+时序)  ← safetensors
  → offline_collate_fn(特征stack+时序padding+归一化)
    → model._forward_with_offline_features
      → unfold_image(2D→1D)                 ← 仍在线执行
        → VTS融合 → 损失
```

**节省的计算**：`fold_image`（CPU 重排 + resize）+ `vit_encoder.forward`（ViT Transformer 推理，最大计算瓶颈）

## 7. 对齐校验机制

### 初始化时校验（OfflineFeatureDataset.__init__）

| 校验项 | 方法 | 失败行为 |
|---|---|---|
| 样本数匹配 | `len(pickle_samples) == metadata["num_samples"]` | 抛出 ValueError |
| 时序长度匹配 | `len(ts) == ts_length_safetensors[idx]` | 抛出 ValueError |
| 周期匹配 | `period_pickle[idx] == period_safetensors[idx]` | 打印警告 |
| 填充值匹配 | `allclose(pad_pickle, pad_safetensors)` | 打印警告 |

### 运行时隐式保证

- `attention_mask` 与 `time_series` 的对齐由 `offline_collate_fn` 保证（逻辑与原始相同）
- `unfold_image` 的 1D 输出与 `time_series` 的时间对齐由 fold/unfold 逆操作数学保证
- `split_sequence` 同步切分特征和时序的时间维

## 8. 新增文件清单

| 文件 | 职责 |
|---|---|
| `scripts/extract_vision_features.py` | 离线特征提取脚本 |
| `src/datasets/offline_feature_dataset.py` | 离线特征数据集类 |
| `src/datasets/offline_collate.py` | 离线特征 collate 函数 |

## 9. 修改文件清单

| 文件 | 修改内容 |
|---|---|
| `src/models/vetime.py` | 新增 `_forward_with_offline_features` 方法 |
| `src/engines/trainer.py` | 新增 `use_offline_features` 分支 |
| `train.py` | 新增离线特征模式入口 |
| `configs/base.yaml` | 新增 `offline_features` 配置段 |

## 10. 非目标（明确不做）

- 不修改原始 `AnomalyDataset` 和 `collate_fn`（向后兼容）
- 不修改 ViT 编码器代码
- 不实现增量特征提取（数据集变更时需重新提取）
- 不实现特征压缩（float32 无损存储）
