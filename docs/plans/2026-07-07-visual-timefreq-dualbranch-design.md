# 视觉时频双分支架构设计文档

**日期**: 2026-07-07
**状态**: 设计确认完成，待实现
**目标**: 为 VETime 单变量时序异常检测实现视觉时频双分支架构

---

## 1. 设计背景

### 1.1 问题陈述
当前 VETime 使用单一时域视觉渲染（STL/SRD 分解），缺乏频域信息的利用。ViCO 论文提出的三视图渲染（STFT/Heatmap/Gradient）能够捕捉时序的频率和结构特征，但未与时域信息融合。

### 1.2 设计目标
将 VETime（时域渲染）和 ViCO（频域渲染）整合为统一的视觉双分支框架，通过纯交叉注意力实现时域特征自适应查询频域特征，废除刚性对角假设。

---

## 2. 核心架构

### 2.1 数据流概览

```
原始时间序列 [B, L, 1]
    │
    ├─────────────────────────────────────────────────────────────────┐
    │                                                                  │
    ▼                                                                  ▼
┌─────────────────────────────────────┐        ┌─────────────────────────────────────┐
│  分支 A: VETime 时域渲染              │        │  分支 B: ViCO 频域渲染               │
│  ───────────────────────────────    │        │  ───────────────────────────────    │
│  输入: time_series [B, L, 1]        │        │  输入: time_series [B, L, 1]        │
│                                     │        │                                     │
│  渲染流程:                           │        │  渲染流程:                           │
│  1. SRD 分解 → Trend/Residual       │        │  1. STFT → 时频谱图 [F, T]          │
│  2. 原始信号 + Trend + Residual     │        │  2. Heatmap → 周期折叠 2D           │
│     → 三个 1D 信号                   │        │  3. Gradient → 周期折叠 2D          │
│  3. 三个 1D → 周期折叠 → RGB        │        │  4. 三种 2D → 拼接 → RGB            │
│                                     │        │                                     │
│  方法: ts2image_1d (已实现)         │        │  方法: render_timeseries (移植)     │
│  输出: [B, 3, 224, 224]             │        │  输出: [B, 3, 224, 224]             │
└─────────────────────────────────────┘        └─────────────────────────────────────┘
    │                                                                  │
    ▼                                                                  ▼
┌─────────────────────────────────────┐        ┌─────────────────────────────────────┐
│  共享冻结 MAE 编码器                  │◄───────│  共享冻结 MAE 编码器                  │
│  vit_encoder(img_A)                 │        │  vit_encoder(img_B)                 │
│                                     │        │                                     │
│  Tokens_A_raw: [B, 196, d]          │        │  Tokens_B_raw: [B, 196, d]          │
│  (patch tokens, 14×14)              │        │  (patch tokens, 14×14)              │
└─────────────────────────────────────┘        └─────────────────────────────────────┘
    │                                                                  │
    ▼                                                                  │
┌─────────────────────────────────────┐        ┌─────────────────────────────────────┐
│  unfold_image (保持不变)             │        │                                     │
│  PTA 时序对齐                        │        │                                     │
│                                     │        │                                     │
│  Q_visual: [B, N_TS, d]             │        │  K, V: Tokens_B_raw [B, 196, d]    │
│  与时序 patch 数量对齐               │        │  保持原始 patch 数量 (不对齐)        │
└─────────────────────────────────────┘        └─────────────────────────────────────┘
    │                                                                  │
    │                      ┌─────────────────────────────────────────────┘
    │                      │
    ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    纯交叉注意力融合 (Cross-Attention Fusion)                   │
│  ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  Query (Q): Q_visual [B, N_TS, d]    ← 时域对齐的视觉特征                    │
│  Key   (K): Tokens_B [B, 196, d]     ← 频域原始 patch tokens                 │
│  Value (V): Tokens_B [B, 196, d]     ← 频域原始 patch tokens                 │
│                                                                              │
│  Attention Matrix: [B, N_TS, 196]   ← 时间点 × 频域 patch 的软对齐           │
│                                                                              │
│  公式: Attn = Softmax(Q·K^T / √d) · V                                       │
│         (无对角偏置，完全废除刚性假设)                                        │
│                                                                              │
│  输出: Fused_Visual [B, N_TS, d]    ← 替换原有 I_embeddings                  │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  VTS_Alignment                       │
│  ───────────────────────────────    │
│  Fused_Visual + TS_embeddings        │
│  → 双向交叉注意力融合                 │
│                                     │
│  I_out, TS_out: [B, N_TS, d]        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  M_moe (多模态融合)                  │
│  ───────────────────────────────    │
│  mix_out = cat([TS_out, I_out])     │
│                                     │
│  Router → 任务专属投影               │
│  → anomaly head / reconstruction    │
└─────────────────────────────────────┘
```

---

## 3. 设计决策记录

| 决策项 | 选择 | 理由 |
|--------|------|------|
| ViCO 来源 | 从 `/mnt/sda/cjmProject/vico` 移植 | 用户已有 ViCO 实现 |
| 融合输出用途 | 替换原有 I_embeddings | 保持现有 VTS_Alignment + MoE 流程不变 |
| 交叉注意力偏置 | 纯 Cross-Attention（无对角偏置） | 废除刚性对角假设，让时域自适应查询频域 |
| MAE 编码器 | 共享同一个冻结 MAE | 论文要求冻结；节省显存；两分支视觉先验一致 |
| PTA 混合池化 | 暂时保持不变，后续迭代 | 用户指示：PTA 过于复杂，先保持现有实现 |
| 实现方案 | 方案 A（最小侵入式） | 改动集中、风险可控、不影响下游任务 |
| 图像转换 | 同时保留 VETime 和 ViCO 两种方式 | 两分支数据来源相同但渲染逻辑不同 |
| 数据来源 | 两分支共享同一原始时间序列 | 简化数据流，无需额外输入 |

---

## 4. 关键模块设计

### 4.1 双分支渲染器

**职责**：将原始时间序列渲染为两个 3 通道 RGB 图像

**接口设计**：
```python
def dual_branch_render(
    time_series: np.ndarray,      # [L, 1] 单变量时序
    patch_size: int,
    img_size: int = 224,
    period: int,                  # 检测到的周期
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Returns:
        img_vetime: [3, H, W] VETime 时域渲染图像
        img_vico: [3, 224, 224] ViCO 频域渲染图像
        period: 检测到的周期
        pad_values: 用于 fold_image 的填充值
    """
```

**VETime 渲染（保持现有 ts2image_1d）**：
- 输入：三个 L 长的 1D 信号 [Trend, Residual, Original]
- 方法：周期折叠 → 三个 2D → 堆叠 RGB
- 特点：保留 VETime 原有的 SRD 分解逻辑

**ViCO 渲染（移植 render_timeseries）**：
- 输入：原始时序 → 直接生成三种 2D 表示
- 方法：
  - STFT Spectrogram: torch.stft → log-magnitude → resize
  - Heatmap: 周期折叠 + interpolate
  - Gradient: 一阶差分 → 周期折叠 + interpolate
- 特点：每个 2D 表示本身就是 RGB 一个通道

### 4.2 纯交叉注意力融合模块

**职责**：用时域对齐的视觉特征查询频域特征

**接口设计**：
```python
class VisualCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        Q_visual: torch.Tensor,    # [B, N_TS, d] 时域对齐的视觉特征
        K_V_tokens: torch.Tensor,  # [B, 196, d] 频域原始 patch tokens
        patch_mask: torch.Tensor,  # [B, N_TS] 时序有效性 mask
    ) -> torch.Tensor:
        """
        Returns:
            fused_visual: [B, N_TS, d] 融合后的视觉特征（替换 I_embeddings）
        """
        # 纯交叉注意力：无对角偏置
        attn_out, _ = self.cross_attn(
            query=Q_visual,
            key=K_V_tokens,
            value=K_V_tokens,
            need_weights=False
        )
        x = self.norm(Q_visual + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x
```

**关键设计**：
- Attention Matrix 形状 `[B, N_TS, 196]`：每个时间点与所有频域 patch 的软对齐
- 无刚性对角偏置：让时域 Query 自适应决定关注哪些频域区域

---

## 5. 实现计划

### 5.1 修改文件清单

| 文件 | 修改类型 | 修改内容 |
|------|----------|----------|
| `dataset/pre_image.py` | 新增函数 | 添加 ViCO 渲染函数 `vico_render_timeseries` |
| `model/VTS_module.py` | 新增类 | 添加 `VisualCrossAttention` 类 |
| `model/VETime.py` | 修改 forward | 双分支渲染 + 交叉注意力融合 |
| `train.py` | 修改 dataloader | 适配双分支图像生成（可选） |

### 5.2 实现步骤

**Phase 1: ViCO 渲染移植**
1. 从 `/mnt/sda/cjmProject/vico/models/vico/visual_teacher.py` 提取核心渲染函数
2. 适配 VETime 的数据格式和参数约定
3. 添加到 `dataset/pre_image.py`

**Phase 2: 交叉注意力模块**
1. 在 `VTS_module.py` 中实现 `VisualCrossAttention` 类
2. 包含 LayerNorm + FFN 的标准 Transformer block 结构

**Phase 3: VETime forward 重构**
1. 在 `__init__` 中初始化新模块
2. 修改 `_forward_impl`：
   - 调用双分支渲染
   - 两图像通过共享 MAE
   - VETime 分支 → unfold_image → Q_visual
   - ViCO 分支 → 保留原始 tokens → K, V
   - Cross-Attention 融合 → 替换 I_embeddings
3. 保持后续 VTS_Alignment + MoE 流程不变

**Phase 4: 验证与测试**
1. 单元测试双分支渲染输出形状
2. 验证交叉注意力输出与原有 I_embeddings 兼容
3. 检查下游任务（重构、异常检测）功能正常

---

## 6. 风险与考量

### 6.1 显存消耗
- 双分支渲染产生 2 张图像，显存增加约 2×
- MAE 编码器共享，避免额外参数

### 6.2 数据流兼容性
- ViCO 渲染需要 `periodicity` 参数，VETime 已有 `find_period` 函数
- 两分支共享周期检测结果

### 6.3 后续优化点
- PTA 混合池化（Max-Avg Concat）：保持点异常锐利度
- 可学习的周期感知偏置（可选）：如果发现纯交叉注意力效果不足

---

## 7. 参考文献

- VETime: 时域 STL/SRD 分解渲染
- ViCO: Vision-Augmented Time Series Anomaly Detection with Prototype-Guided Coordination
  - `visual_teacher.py` 三视图渲染实现