# 模态非对称三任务解耦表示学习架构设计

## 概述

将 VETime 从统一前向路径重构为模态非对称的三任务解耦架构，利用数据集"纯正常序列 vs 人工注入异常序列"的成对优势，在模态维度进行物理切割。

## 三个任务

### Task A：单模态插值（Unimodal Interpolation）

- **输入**：`normal_time_series` + 随机掩码
- **前向路径**：纯1D时序编码器（`TimeSeriesEncoder`）→ `interpolation_head`
- **损失**：仅在掩码位置计算 MSE（`pred[mask]` vs `normal_ts[mask]`）
- **核心目的**：以极低计算代价，在无视觉噪声的环境中学习时序内在规律，构建稳固的正常基准特征空间
- **约束**：绝不触发 Vision Encoder / VTS_Alignment / MMoE

### Task B：多模态降噪重建（Multi-modal Denoising Reconstruction）

- **输入**：`time_series`（异常序列）+ 图像（从异常序列生成）
- **前向路径**：TS编码 → 图像编码 → 交叉融合 → MMoE(task_id=0) → `reconstruction_head`
- **损失**：全局 MSE（`reconstructed` vs `normal_time_series`），只在有效位置（排除 padding）计算，无 mask/label 过滤
- **核心目的**：强迫 MMoE 网络顶住异常信号干扰，利用多模态感受野将异常尖峰"压平"回正常态
- **MMoE task_id**：0（语义从"掩码重建"变为"降噪重建"）

### Task C：多模态异常分类（Multi-modal Anomaly Classification）

- **输入**：`time_series`（异常序列）+ 图像
- **前向路径**：与 Task B 共享前向计算，MMoE(task_id=1) → `anomaly_head`
- **损失**：Focal Loss（与现有实现一致，gamma=2.0，类别加权 1.2/0.8）
- **MMoE task_id**：1（语义不变）

## 关键设计决策

1. **共享编码器并回传梯度**：Task A 和 Task B/C 共享 `TimeSeriesEncoder`，Task A 的梯度通过共享参数回传，"正常世界观"帮助夯实编码器对时序规律的理解
2. **分阶段训练**：Stage1(A+B) → Stage2(A+B+C)，先建立正常世界观+降噪能力，再加入异常分类
3. **复用 reconstruction_head**：Task B 复用现有 `reconstruction_head`，仅改造损失计算逻辑
4. **前向路径分流**：在 `VETime.py` 中新增 `forward_task_a()`，`train.py` 分别调用两条路径

## 受影响文件

### `dataset/dataloader.py`（改动量：小）

修改 `collate_fn` 第252-253行：

```python
# 修改前
mask_time_series, mask = create_random_mask(padded_time_series, attention_mask, patch_size)
normal_time_series_tensors, mask = create_random_mask(normal_time_series_tensors, attention_mask, patch_size)

# 修改后
mask_normal_ts, normal_mask = create_random_mask(normal_time_series_tensors, attention_mask, patch_size)
```

- 只对 `normal_time_series` 做掩码（Task A）
- `time_series`（异常）保持原样（Task B/C）
- 返回 dict 中 `mask_time_series` → `mask_normal_ts`，`mask` → `normal_mask`

### `model/TS_encoder/ts_model.py`（改动量：中）

1. **新增 `interpolation_head`**：结构与 `reconstruction_head` 类似的逐点回归头
2. **`weighted_reconstruction_loss` → `denoising_reconstruction_loss`**：
   - 去掉 `effective_mask = mask & (~labels)` 过滤逻辑
   - 改为全局 MSE：`F.mse_loss(reconstructed[valid], normal_time_series[valid])`
   - 目标从 `time_series` 变为 `normal_time_series`
   - 只排除 padding 位置（`att_mask`）
3. **新增 `interpolation_loss`**：仅在掩码位置计算 MSE

### `model/VETime.py`（改动量：中）

1. **`__init__` 新增引用**：`self.interpolation_head = ts_model.interpolation_head`
2. **新增 `forward_task_a()`**：纯1D旁路，仅调用 `self.ts_encoder` + `self.interpolation_head`
3. `_forward_impl` / `_forward_with_checkpointing` 不改动

### `train.py`（改动量：中）

1. **训练循环改为三任务**：
   - Task A：`pred_A = model.forward_task_a(mask_normal_ts, att_mask)`
   - Task B/C：复用现有 `model(...)` 多模态前向
2. **两阶段策略**：
   - Stage1：loss_A + alpha*loss_B + loss_e + 0.1*loss_cl（loss_C 切断）
   - Stage2：loss_C + alpha*loss_B + beta*loss_A + loss_e + 0.1*loss_cl
3. **新增超参数**：`beta`（Task A 插值权重）

### 不改动的文件

| 文件 | 原因 |
|------|------|
| `model/VTS_module.py` | MMoE 保持2个task，仅语义重映射 |
| `loss/loss.py` | 对比学习只在 B/C 路径，逻辑不变 |
| `dataset/pre_image.py` | 图像生成逻辑不变 |
| `model/Vision_encoder/` 全部 | 视觉编码器本身不变 |
| `model/TS_encoder/ts_encoder.py` | 编码器接口不变 |
| `model/TS_encoder/encoding_utils.py` | 底层组件不变 |
| `evaluation/` 全部 | 推理只用 Task C |
| `Test_TSB.py` | 同上 |

## 数据流图

```
normal_time_series ──[mask]──→ TimeSeriesEncoder ──→ interpolation_head ──→ loss_A (MSE on masked)
                                       ↑
                                  (共享参数，梯度回传)
                                       ↓
time_series(异常) ──→ TimeSeriesEncoder ──┬──→ VTS_Alignment ──→ MMoE(task=0) ──→ reconstruction_head ──→ loss_B (全局MSE vs normal)
       │                                │         ↑
       └──[图像转换]──→ VisionEncoder ──┘         │
                                                  └──→ MMoE(task=1) ──→ anomaly_head ──→ loss_C (Focal)
```
