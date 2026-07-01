# 模态非对称三任务解耦架构 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 VETime 重构为模态非对称三任务解耦架构——Task A(单模态插值)、Task B(多模态降噪重建)、Task C(多模态异常分类)

**Architecture:** 在 VETime.py 中新增 `forward_task_a()` 轻量1D旁路，Task A 绝不触碰 Vision Encoder / VTS_Alignment / MMoE。Task B/C 走现有完整多模态路径。训练循环中分别调用两条路径，三任务损失加权组合，分阶段训练(Stage1: A+B → Stage2: A+B+C)。

**Tech Stack:** PyTorch, Accelerate, 现有 VETime 代码库

---

### Task 1: 新增 interpolation_head 到 ts_model.py

**Files:**
- Modify: `model/TS_encoder/ts_model.py:40-48`（在 anomaly_head 之后新增）

**Step 1: 在 `__init__` 中添加 interpolation_head**

在 `model/TS_encoder/ts_model.py` 第 48 行（`anomaly_head` 定义结束）之后，添加：

```python
        # Interpolation head for Task A (unimodal interpolation on normal sequences)
        self.interpolation_head = nn.Sequential(
            nn.Linear(self.d_proj, self.d_proj * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_proj * 4, self.d_proj * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_proj * 4, 1)
        )
```

**Step 2: 验证模型可以正常实例化**

Run: `python -c "from model.TS_encoder.ts_model import TS_Model; from model.TS_encoder.config import TimeSeriesConfig; m = TS_Model(TimeSeriesConfig()); print('interpolation_head:', type(m.interpolation_head)); print('OK')"`

Expected: 打印 `interpolation_head: <class 'torch.nn.modules.container.Sequential'>` 和 `OK`

**Step 3: Commit**

```bash
git add model/TS_encoder/ts_model.py
git commit -m "feat: 添加 interpolation_head 用于 Task A 单模态插值"
```

---

### Task 2: 改造 weighted_reconstruction_loss 为 denoising_reconstruction_loss

**Files:**
- Modify: `model/TS_encoder/ts_model.py:74-122`

**Step 1: 将 `weighted_reconstruction_loss` 方法重写为 `denoising_reconstruction_loss`**

将 `model/TS_encoder/ts_model.py` 第 74-122 行的 `weighted_reconstruction_loss` 方法**整体替换**为：

```python
    def denoising_reconstruction_loss(
        self,
        local_embeddings: torch.Tensor,
        normal_time_series: torch.Tensor,
        att_mask: torch.Tensor,
    ):
        """
        Task B: 降噪重建损失 — 全局 MSE。

        输入是异常序列的 embedding，目标是正常序列。
        强迫模型把异常尖峰"压平"回正常态。
        只排除 padding 位置，无 mask/label 过滤。

        Args:
            local_embeddings: [B, seq_len, num_features, d_proj]
            normal_time_series: [B, seq_len, num_features] 正常时序目标
            att_mask: [B, seq_len] bool，True=有效位置

        Returns:
            loss: scalar MSE loss
            reconstructed: [B, seq_len, num_features] 重建结果
        """
        reconstructed = self.reconstruction_head(local_embeddings).squeeze(-1)  # [B, seq_len, num_features]

        # 只在有效位置计算（排除 padding）
        if not att_mask.dtype == torch.bool:
            att_mask = att_mask.bool()
        valid = att_mask  # [B, seq_len]

        # 扩展 valid 到 num_features 维度
        num_features = normal_time_series.shape[-1] if normal_time_series.dim() == 3 else 1
        valid_expanded = valid.unsqueeze(-1).expand(-1, -1, num_features)  # [B, seq_len, num_features]

        loss = F.mse_loss(
            reconstructed[valid_expanded],
            normal_time_series[valid_expanded],
        )

        return loss, reconstructed
```

**Step 2: 新增 `interpolation_loss` 方法**

在 `denoising_reconstruction_loss` 之后、`anomaly_detection_loss` 之前，添加：

```python
    def interpolation_loss(self, local_embeddings, normal_time_series, mask):
        """
        Task A: 插值损失 — 只在被掩码位置计算 MSE。

        Args:
            local_embeddings: [B, seq_len, num_features, d_proj]
            normal_time_series: [B, seq_len, num_features] 原始正常时序
            mask: [B, seq_len] bool，True=被掩码的位置

        Returns:
            loss: scalar MSE loss on masked positions
        """
        pred = self.interpolation_head(local_embeddings).squeeze(-1)  # [B, seq_len, num_features]

        if not mask.dtype == torch.bool:
            mask = mask.bool()

        num_features = normal_time_series.shape[-1] if normal_time_series.dim() == 3 else 1
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_features)

        loss = F.mse_loss(
            pred[mask_expanded],
            normal_time_series[mask_expanded],
        )
        return loss
```

**Step 3: 验证语法正确**

Run: `python -c "from model.TS_encoder.ts_model import TS_Model; from model.TS_encoder.config import TimeSeriesConfig; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add model/TS_encoder/ts_model.py
git commit -m "feat: 改造 weighted_reconstruction_loss 为 denoising_reconstruction_loss，新增 interpolation_loss"
```

---

### Task 3: 修改 VETime.py — 新增 forward_task_a 和 interpolation_head 引用

**Files:**
- Modify: `model/VETime.py:39-45`（ts setting 区域）
- Modify: `model/VETime.py`（新增 forward_task_a 方法）

**Step 1: 在 `__init__` 的 ts setting 区域新增 interpolation_head 引用**

在 `model/VETime.py` 第 44 行（`self.anomaly_head = ts_model.anomaly_head`）之后，添加：

```python
        self.interpolation_head = ts_model.interpolation_head
```

**Step 2: 新增 `forward_task_a` 方法**

在 `model/VETime.py` 的 `forward` 方法之后（第 68 行之后），添加：

```python
    def forward_task_a(self, normal_time_series, att_mask):
        """Task A: 纯1D时序插值 — 轻量旁路，绝不触发 Vision Encoder / VTS_Alignment / MMoE"""
        _, local_embeddings, _ = self.ts_encoder(normal_time_series, att_mask)
        pred_interp = self.interpolation_head(local_embeddings)
        return pred_interp
```

**Step 3: 更新 `_forward_impl` 中的注释，反映 task_id 语义变化**

将 `model/VETime.py` 第 90-91 行的注释：

```python
        # 两路任务干净并行：task 1 -> anomaly head(local_emb1), task 0 -> reconstruction head(local_emb2)
        # 任务映射保持与原实现一致（原 mask=None 分支 = task 1 = anomaly）。
```

替换为：

```python
        # 两路任务：task 1 -> anomaly head(local_emb1), task 0 -> denoising reconstruction head(local_emb2)
        # task 0 语义：降噪重建 — 把融合了图文特征的隐变量拉回正常时序空间
        # task 1 语义：异常分类 — 点级别二分类
```

同样更新 `_forward_with_checkpointing` 中第 143 行的注释：

```python
            # 两路任务干净并行：task 1 -> anomaly, task 0 -> denoising reconstruction（与 _forward_impl 一致）
```

**Step 4: 验证语法正确**

Run: `python -c "from model.VETime import VETIME; print('OK')"`

Expected: `OK`（可能因缺少 vision checkpoint 打印警告，但不应报 ImportError/AttributeError）

**Step 5: Commit**

```bash
git add model/VETime.py
git commit -m "feat: VETime 新增 forward_task_a 轻量1D旁路和 interpolation_head 引用"
```

---

### Task 4: 修改 dataloader.py — 只对 normal_time_series 做掩码

**Files:**
- Modify: `dataset/dataloader.py:252-265`（collate_fn 的掩码和返回部分）

**Step 1: 修改掩码逻辑**

将 `dataset/dataloader.py` 第 252-265 行：

```python
    mask_time_series, mask = create_random_mask(padded_time_series, attention_mask, patch_size)
    normal_time_series_tensors, mask = create_random_mask(normal_time_series_tensors, attention_mask, patch_size)

    return {
        'time_series': padded_time_series,
        'normal_time_series': normal_time_series_tensors,
        'mask_time_series': mask_time_series,
        'image': image_inputs,
        'mask': mask,
        'labels': padded_labels,
        'attention_mask': attention_mask,
        'period': period,
        'padding_value': padding_value,
    }
```

替换为：

```python
    # Task A: 只对正常序列做掩码（插值任务）
    # Task B/C: 异常序列保持原样（不掩码）
    mask_normal_ts, normal_mask = create_random_mask(normal_time_series_tensors, attention_mask, patch_size)

    return {
        'time_series': padded_time_series,
        'normal_time_series': normal_time_series_tensors,
        'mask_normal_ts': mask_normal_ts,
        'normal_mask': normal_mask,
        'image': image_inputs,
        'labels': padded_labels,
        'attention_mask': attention_mask,
        'period': period,
        'padding_value': padding_value,
    }
```

**Step 2: 验证 dataloader 语法**

Run: `python -c "from dataset.dataloader import AnomalyDataset, collate_fn; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add dataset/dataloader.py
git commit -m "feat: collate_fn 只对 normal_time_series 做掩码，异常序列保持原样"
```

---

### Task 5: 修改 train.py — 训练循环改为三任务

**Files:**
- Modify: `train.py:474-611`（univariate 训练循环）
- Modify: `train.py:855-908`（univariate 验证循环）
- Modify: `train.py:1528-1639`（multivariate 训练循环）

这是改动量最大的 Task。`train.py` 中有三处训练/验证循环需要同步修改。以 univariate 训练循环为主，其余两处参照修改。

**Step 1: 修改 univariate 训练循环的 batch 解包（第 483-489 行）**

将：

```python
        for batch in progress_bar:
            labels = batch["labels"]
            images = batch["image"]  # (B, C, H, W)
            time_series, att_mask = batch['time_series'], batch['attention_mask']
            mask = batch['mask']
            period = batch['period']
            p_value = batch['padding_value']
```

替换为：

```python
        for batch in progress_bar:
            labels = batch["labels"]
            images = batch["image"]  # (B, C, H, W)
            time_series, att_mask = batch['time_series'], batch['attention_mask']
            normal_ts = batch['normal_time_series']
            mask_normal_ts = batch['mask_normal_ts']
            normal_mask = batch['normal_mask']
            period = batch['period']
            p_value = batch['padding_value']
```

**Step 2: 新增 Task A 超参数和训练阶段描述（第 464-492 行区域）**

在 `alpha_recon = 0.05` 之前新增：

```python
            # Task A 插值权重（新增超参数，需要调参）
            alpha_interp = 0.1
```

将 Stage 描述更新：

```python
        if is_stage_1:
            print(f"[Stage 1] Epoch {epoch+1}/{epochs}: 插值(A)+降噪重建(B)预训练 (异常分类损失已切断)")
        else:
            print(f"[Stage 2] Epoch {epoch+1}/{epochs}: 三任务联合训练 插值(A)+降噪重建(B)+异常分类(C)")
```

**Step 3: 修改训练循环核心 — 短序列分支（第 550-581 行）**

将 `else:` 分支（第 550 行起，`labels.shape[1] <= model.MAX_L`）整体替换为：

```python
            else:
                # === Task A: 纯1D插值（极快，无图像开销）===
                pred_A = model.forward_task_a(mask_normal_ts, att_mask)
                loss_A = model.interpolation_loss(pred_A, normal_ts, normal_mask)

                # === Task B & C: 多模态路径 ===
                images_folded, init_img_size = model.vit_encoder.fold_image(images, period, p_value, **data_setting)

                local_embeddings1, m_w, loss_cl, local_embeddings2 = model(images_folded, time_series, att_mask, init_img_size, labels)

                # Task B: 降噪重建 — 全局 MSE vs normal_time_series
                loss_B, rec = model.denoising_reconstruction_loss(local_embeddings2, normal_ts, att_mask)

                # Task C: 异常分类
                loss_C, logits = model.anomaly_detection_loss(local_embeddings1, labels)

                # ========== 两阶段训练范式 ==========
                if is_stage_1:
                    loss_C = torch.tensor(0.0, device=device)
                    loss_e_tensor = 0.01 * load_balance_loss(m_w[0])
                else:
                    loss_e_tensor = 0.01 * 0.5 * (load_balance_loss(m_w[0]) + load_balance_loss(m_w[1]))

                # 三任务损失组合
                loss2 = (alpha_recon * loss_B) + (alpha_interp * loss_A) + loss_e_tensor + 0.1 * loss_cl

                # 提取纯数值用于打 log
                batch_loss_bce = loss_C.item()
                batch_loss_mse = loss_B.item()
                batch_loss_interp = loss_A.item()
                batch_loss_cl = (0.1 * loss_cl).item()
                batch_loss_e = loss_e_tensor.item()
```

**Step 4: 修改训练循环核心 — 长序列 split 分支（第 494-548 行）**

在 `for data_part in data_splits:` 循环**之前**，新增 Task A（Task A 不受 split 影响，用完整序列）：

```python
                # === Task A: 纯1D插值（不受 split 影响）===
                pred_A = model.forward_task_a(mask_normal_ts, att_mask)
                loss_A = model.denoising_reconstruction_loss(pred_A, normal_ts, att_mask)
```

**注意**：Task A 的 loss 应该用 `interpolation_loss`，但这里 `pred_A` 已经是 `interpolation_head` 的输出。需要改为：

```python
                # === Task A: 纯1D插值（不受 split 影响，无需切分序列）===
                pred_A = model.forward_task_a(mask_normal_ts, att_mask)
                loss_A = model.interpolation_loss(pred_A, normal_ts, normal_mask)
```

然后在 split 循环**内部**，将 `weighted_reconstruction_loss` 替换为 `denoising_reconstruction_loss`：

```python
                        # Task B: 降噪重建
                        loss02, rec = model.denoising_reconstruction_loss(local_embeddings2, normal_ts_part, att_mask_part)
```

**注意**：split 循环中 `normal_ts` 也需要对应切分。在 split 循环之前准备 `normal_ts_splits`：

```python
                # Task A: 纯1D插值（不受 split 影响，无需切分序列）
                pred_A = model.forward_task_a(mask_normal_ts, att_mask)
                loss_A = model.interpolation_loss(pred_A, normal_ts, normal_mask)

                # 切分 normal_ts 以匹配 data_splits
                normal_ts_splits = []
                start = 0
                for data_part in data_splits:
                    chunk_len = data_part[1].shape[1]  # ts_part 的长度
                    normal_ts_splits.append(normal_ts[:, start:start+chunk_len, :])
                    start += chunk_len
```

然后在循环内部使用 `normal_ts_part = normal_ts_splits[idx]`：

```python
                for idx, data_part in enumerate(data_splits):
                    img_part, ts_part, att_mask_part, label_part = data_part
                    normal_ts_part = normal_ts_splits[idx]
                    images_folded, init_img_size = model.vit_encoder.fold_image(img_part, period, p_value, **data_setting)

                    local_embeddings1, m_w, loss_cl, local_embeddings2 = model(images_folded, ts_part, att_mask_part, init_img_size, label_part)

                    loss01, logit = model.anomaly_detection_loss(local_embeddings1, label_part)

                    # Task B: 降噪重建
                    loss02, rec = model.denoising_reconstruction_loss(local_embeddings2, normal_ts_part, att_mask_part)

                    if is_stage_1:
                        batch_loss_e_part = 0.01 * load_balance_loss(m_w[0])
                    else:
                        batch_loss_e_part = 0.01 * 0.5 * (load_balance_loss(m_w[0]) + load_balance_loss(m_w[1]))

                    if is_stage_1:
                        loss01 = torch.tensor(0.0, device=device)

                    batch_loss_bce += loss01.item()
                    batch_loss_mse += loss02.item()
                    batch_loss_cl += (0.1 * loss_cl).item()
                    batch_loss_e += batch_loss_e_part.item()

                    loss2 = loss2 + (alpha_recon * loss02) + 0.1 * loss_cl + batch_loss_e_part
                    loss1 = loss1 + loss01
                    logits_list.append(logit)
```

在 split 累加完成后，加入 Task A 的 loss：

```python
                # 加入 Task A 的损失（不除以 num_splits，因为只计算了一次）
                loss2 = loss2 + (alpha_interp * loss_A)
                if num_splits > 0:
                    loss1 = loss1 / num_splits
                    loss2 = loss2 / num_splits  # Task A 的 loss 也被平均了
                    batch_loss_bce /= num_splits
                    batch_loss_mse /= num_splits
                    batch_loss_cl /= num_splits
                    batch_loss_e /= num_splits
```

**Step 5: 更新 loss 统计和日志打印**

在第 475-479 行的 loss 累加器中新增：

```python
        total_loss_interp = 0  # 插值损失 (Task A)
```

在第 605-611 行的 loss 累加中新增：

```python
            total_loss_interp += batch_loss_interp
```

更新 progress_bar 的 `set_postfix`，加入 `Interp` 指标：

```python
            progress_bar.set_postfix({"Tot": f"{batch_loss:.3f}", "BCE": f"{batch_loss_bce:.3f}", "MSE": f"{batch_loss_mse:.3f}", "Interp": f"{batch_loss_interp:.3f}", "CL": f"{batch_loss_cl:.3f}", "Bal": f"{batch_loss_e:.4f}"})
```

更新 TensorBoard 日志（第 616-623 行），加入 Task A 指标：

```python
                accelerator.log({
                    "Loss/Total": batch_loss,
                    "Loss/BCE_Anomaly": batch_loss_bce,
                    "Loss/MSE_Denoise": batch_loss_mse,
                    "Loss/Interp_TaskA": batch_loss_interp,
                    "Loss/CL_Contrastive": batch_loss_cl,
                    "Loss/Balance": batch_loss_e,
                    "Train/LR": optimizer.param_groups[0]['lr'],
                }, step=global_step)
```

**Step 6: 修改 univariate 验证循环（第 855-908 行）**

验证循环同样需要将 `weighted_reconstruction_loss` 改为 `denoising_reconstruction_loss`，目标改为 `normal_time_series`。

在验证循环的 batch 解包中加入：

```python
            normal_ts = batch['normal_time_series']
```

将所有 `model.weighted_reconstruction_loss(local_embeddings2, ts_part, att_mask_part, label_part)` 替换为：

```python
model.denoising_reconstruction_loss(local_embeddings2, normal_ts_part, att_mask_part)
```

对于验证的 split 分支，同样需要切分 `normal_ts`。

**Step 7: 修改 multivariate 训练循环（第 1528-1639 行）**

与 univariate 训练循环完全相同的模式：
1. batch 解包加入 `normal_ts`, `mask_normal_ts`, `normal_mask`
2. 新增 `alpha_interp` 超参数
3. Task A 独立前向
4. `weighted_reconstruction_loss` → `denoising_reconstruction_loss`，目标改为 `normal_ts`
5. 三任务损失组合
6. 更新日志

**Step 8: Commit**

```bash
git add train.py
git commit -m "feat: 训练循环改为三任务解耦架构(A插值+B降噪+C分类)"
```

---

### Task 6: 修改 Test_TSB.py — 适配新 dataloader 返回格式

**Files:**
- Modify: `Test_TSB.py:154-165`（测试时的 collate 函数）

**Step 1: 测试 collate 函数中移除对异常序列的掩码**

测试时只需要 Task C（异常分类），不需要 Task A 的掩码。但 `mask_time_series` 在 `Test_TSB.py` 的返回 dict 中被引用。

检查 `Test_TSB.py` 第 154 行：

```python
    mask_time_series,mask  = create_random_mask(padded_ts, attention_mask,patch_size)
```

测试时可以保留这个掩码（测试阶段不关心掩码语义，只是为了接口兼容），或者直接移除。由于测试的 forward 路径不走 Task A，这里**保持不变**即可——`mask_time_series` 和 `mask` 在测试时不被使用（测试的 batch 解包只取 `time_series`, `image`, `labels`, `attention_mask`）。

**无需修改 Test_TSB.py。**

**Step 2: 验证** — 跳过，确认测试脚本不依赖 `mask_time_series`

Run: `grep -n 'mask_time_series' Test_TSB.py`

Expected: 只在 collate 函数的返回 dict 中出现（第 158 行），不在 batch 解包中使用。

**Step 3: Commit** — 无需 commit，此文件不改

---

### Task 7: 端到端验证

**Files:** 无修改，仅验证

**Step 1: 验证模型前向传播**

Run:
```bash
python -c "
import torch
from model.TS_encoder.ts_model import TS_Model
from model.TS_encoder.config import TimeSeriesConfig
config = TimeSeriesConfig()
model = TS_Model(config)
B, L, F = 2, 28, 1
ts = torch.randn(B, L, F)
att_mask = torch.ones(B, L, dtype=torch.bool)
mask = torch.zeros(B, L, dtype=torch.bool)
mask[:, 14:] = True

# Task A: forward_task_a (via ts_encoder + interpolation_head)
_, local_emb, _ = model(ts, att_mask)
pred = model.interpolation_head(local_emb)
loss_A = model.interpolation_loss(pred, ts, mask)
print(f'Task A loss: {loss_A.item():.4f}')

# Task B: denoising_reconstruction_loss
loss_B, rec = model.denoising_reconstruction_loss(local_emb, ts, att_mask)
print(f'Task B loss: {loss_B.item():.4f}')

# Task C: anomaly_detection_loss
labels = torch.zeros(B, L, dtype=torch.long)
loss_C, logits = model.anomaly_detection_loss(local_emb, labels)
print(f'Task C loss: {loss_C.item():.4f}')
print('All tasks OK!')
"
```

Expected: 三个 loss 都正常打印，无报错

**Step 2: 验证 dataloader 返回新格式**

Run:
```bash
python -c "
from dataset.dataloader import AnomalyDataset, collate_fn
import torch
# 检查 collate_fn 的返回 dict 包含新 key
print('collate_fn signature check: need real data to test, but import OK')
"
```

Expected: `import OK`

**Step 3: Commit** — 无需 commit，验证通过即可

---

### Task 8: 最终提交与合并准备

**Step 1: 检查所有修改**

Run: `git diff main --stat`

Expected: 只有以下文件被修改：
- `dataset/dataloader.py`
- `model/TS_encoder/ts_model.py`
- `model/VETime.py`
- `train.py`
- `docs/plans/` 下的设计文档

**Step 2: 确认没有遗漏的 `weighted_reconstruction_loss` 调用**

Run: `grep -rn 'weighted_reconstruction_loss' --include='*.py' .`

Expected: 只在 `ts_model.py` 的方法定义（如果保留了旧方法）或 `docs/plans/` 中出现，不在 `train.py` 中出现。

**Step 3: 确认没有遗漏的 `mask_time_series` 或 `batch['mask']` 引用**

Run: `grep -rn "batch\['mask'\]" --include='*.py' .` 和 `grep -rn 'mask_time_series' --include='*.py' .`

Expected: `train.py` 中不再有 `batch['mask']` 或 `mask_time_series` 引用。

**Step 4: 最终 Commit**

如果有零散修复：
```bash
git add -A
git commit -m "chore: 三任务解耦架构最终清理"
```
