# 视觉时频双分支架构实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 VETime 单变量时序异常检测实现视觉时频双分支架构，用 ViCO 频域特征替换原有视觉特征，通过纯交叉注意力融合时域和频域信息。

**Architecture:** 双分支渲染（VETime 时域 + ViCO 频域）→ 共享冻结 MAE → PTA 对齐 → 纯 Cross-Attention → 替换 I_embeddings → VTS_Alignment + MoE

**Tech Stack:** PyTorch, MAE (ViT), torch.stft (STFT), numpy

**Design Doc:** `docs/plans/2026-07-07-visual-timefreq-dualbranch-design.md`

---

## 前置条件

**依赖检查：**
- ViCO 代码路径: `/mnt/sda/cjmProject/vico/models/vico/visual_teacher.py`
- VETime 现有渲染: `dataset/pre_image.py` 的 `ts2image_1d`
- MAE 编码器: `model/Vision_encoder/V_encoder.py`

---

## Task 1: 移植 ViCO 频域渲染函数

**Files:**
- Modify: `dataset/pre_image.py` (新增函数)
- Reference: `/mnt/sda/cjmProject/vico/models/vico/visual_teacher.py`

**目标:** 将 ViCO 的三视图渲染函数适配到 VETime 的数据格式。

**Step 1: 添加 ViCO 渲染核心函数**

在 `dataset/pre_image.py` 末尾添加以下函数（基于 ViCO `visual_teacher.py` 的 `_stft_spectrogram`, `_gradient_map`, `_normalise_01`）:

```python
# ============== ViCO 频域渲染函数 ==============

import math
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import acf


def _normalise_01_np(tensor: np.ndarray) -> np.ndarray:
    """将 tensor 归一化到 [0, 1] 范围（numpy 版本）"""
    vmin = tensor.min()
    vmax = tensor.max()
    return (tensor - vmin) / (vmax - vmin + 1e-8)


def _stft_spectrogram_np(
    x: np.ndarray,
    win_len: int = 64,
    hop_len: int = 16,
    n_fft: int = 128,
) -> np.ndarray:
    """
    计算 STFT 时频谱图（numpy 版本）。

    Args:
        x: 1D 时间序列 [L]
        win_len: 窗口长度
        hop_len: 跳跃长度
        n_fft: FFT 点数

    Returns:
        spec: 时频谱图 [F, T]，log-magnitude 归一化到 [0, 1]
    """
    from scipy.signal import stft
    # 使用 scipy 的 STFT
    f, t, Zxx = stft(x, fs=1.0, window='hann', nperseg=win_len, noverlap=win_len-hop_len, nfft=n_fft)
    spec = np.abs(Zxx)  # 取幅度
    spec = np.log1p(spec)  # log-magnitude
    return _normalise_01_np(spec)


def _heatmap_period_fold_np(
    x: np.ndarray,
    period: int,
    target_h: int = 224,
    target_w: int = 224,
) -> np.ndarray:
    """
    按周期折叠成热力图（numpy 版本）。

    Args:
        x: 1D 时间序列 [L]，已归一化
        period: 周期长度
        target_h: 目标高度
        target_w: 目标宽度

    Returns:
        heatmap: 2D 热力图 [H, W]，归一化到 [0, 1]
    """
    L = len(x)
    # 左侧填充使长度为周期的倍数
    pad_left = (period - L % period) % period
    x_padded = np.pad(x, (pad_left, 0), mode='edge')

    # 折叠成 2D: [num_segments, period]
    num_segments = len(x_padded) // period
    heatmap_2d = x_padded.reshape(num_segments, period)

    # resize 到目标尺寸（使用 scipy.ndimage 或 cv2）
    from scipy.ndimage import zoom
    scale_h = target_h / num_segments
    scale_w = target_w / period
    heatmap_resized = zoom(heatmap_2d, (scale_h, scale_w), order=1)

    return _normalise_01_np(heatmap_resized)


def _gradient_map_np(
    x: np.ndarray,
    period: int,
    target_h: int = 224,
    target_w: int = 224,
) -> np.ndarray:
    """
    计算梯度图并按周期折叠（numpy 版本）。

    Args:
        x: 1D 时间序列 [L]，已归一化
        period: 周期长度
        target_h: 目标高度
        target_w: 目标宽度

    Returns:
        gradient: 2D 梯度图 [H, W]，归一化到 [0, 1]
    """
    L = len(x)
    # 一阶差分
    grad = np.zeros_like(x)
    grad[1:] = x[1:] - x[:-1]
    grad = np.abs(grad)

    # 按周期折叠
    pad_left = (period - L % period) % period
    grad_padded = np.pad(grad, (pad_left, 0), mode='constant', constant_values=0)

    num_segments = len(grad_padded) // period
    grad_2d = grad_padded.reshape(num_segments, period)

    # resize
    from scipy.ndimage import zoom
    scale_h = target_h / num_segments
    scale_w = target_w / period
    grad_resized = zoom(grad_2d, (scale_h, scale_w), order=1)

    return _normalise_01_np(grad_resized)


def vico_render_timeseries(
    x: Union[np.ndarray, torch.Tensor],
    period: int,
    img_size: int = 224,
    norm_const: float = 0.4,
    stft_win: int = 64,
    stft_hop: int = 16,
    stft_fft: int = 128,
) -> np.ndarray:
    """
    ViCO 三视图频域渲染：STFT + Heatmap + Gradient → RGB 图像。

    Args:
        x: 输入时间序列 [L, C] 或 [L]（单变量）
        period: 检测到的周期长度
        img_size: 输出图像尺寸，默认 224
        norm_const: 归一化常数，默认 0.4
        stft_win: STFT 窗口长度
        stft_hop: STFT 跳跃长度
        stft_fft: STFT FFT 点数

    Returns:
        image: RGB 图像 [3, img_size, img_size]，dtype uint8
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.ndim == 1:
        x = x[:, np.newaxis]
    L, C = x.shape

    # 单变量：只处理第一个通道
    x_1d = x[:, 0].ravel()

    # RevIN 归一化（与 ViCO 保持一致）
    mean = x_1d.mean()
    std = x_1d.std() + 1e-8
    x_norm = (x_1d - mean) / (std / norm_const)

    # 周期保护
    period = max(1, period)

    # === 渲染三个视图 ===
    # R: STFT 时频谱图
    n_fft_cap = max(8, min(L - 1, 2 ** int(math.floor(math.log2(L)))))
    n_fft_eff = max(8, min(stft_fft, n_fft_cap))
    win_eff = max(8, min(stft_win, n_fft_eff))
    hop_eff = max(1, min(stft_hop, win_eff))
    stft_img = _stft_spectrogram_np(x_norm, win_len=win_eff, hop_len=hop_eff, n_fft=n_fft_eff)
    # resize 到 img_size × img_size
    from scipy.ndimage import zoom
    stft_resized = zoom(stft_img, (img_size / stft_img.shape[0], img_size / stft_img.shape[1]), order=1)
    R = _normalise_01_np(stft_resized)

    # G: Heatmap（周期折叠）
    G = _heatmap_period_fold_np(x_norm, period, target_h=img_size, target_w=img_size)

    # B: Gradient（梯度图）
    B = _gradient_map_np(x_norm, period, target_h=img_size, target_w=img_size)

    # 拼接为 RGB
    image = np.stack([R, G, B], axis=0)  # [3, img_size, img_size]
    image = (image * 255).clip(0, 255).astype(np.uint8)

    return image
```

**Step 2: 运行简单测试验证函数**

创建临时测试脚本：

```python
# test_vico_render.py (临时测试文件，测试后删除)
import numpy as np
from dataset.pre_image import vico_render_timeseries, find_period

# 测试数据
x = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.randn(1000) * 0.1
period = find_period(x)

# 渲染
img = vico_render_timeseries(x, period, img_size=224)
print(f"ViCO image shape: {img.shape}")  # 期望: (3, 224, 224)
print(f"ViCO image dtype: {img.dtype}")  # 期望: uint8
print(f"ViCO image value range: [{img.min()}, {img.max()}]")  # 期望: [0, 255]
```

运行:
```bash
python test_vico_render.py
```

Expected output:
```
ViCO image shape: (3, 224, 224)
ViCO image dtype: uint8
ViCO image value range: [0, 255]
```

**Step 3: 提交**

```bash
git add dataset/pre_image.py
git commit -m "feat: add ViCO frequency-domain rendering function"
```

---

## Task 2: 实现纯交叉注意力融合模块

**Files:**
- Modify: `model/VTS_module.py` (新增类)

**目标:** 实现 `VisualCrossAttention` 类，用时域对齐的视觉特征查询频域特征。

**Step 1: 添加 VisualCrossAttention 类**

在 `model/VTS_module.py` 末尾添加：

```python
class VisualCrossAttention(nn.Module):
    """
    纯交叉注意力融合模块：用 VETime 时域特征 (Q) 查询 ViCO 频域特征 (K, V)。

    废除刚性对角假设，让时间域 Query 自适应决定关注哪些频域 patch。
    Attention Matrix 形状 [B, N_TS, 196]，表示每个时间点与所有频域 patch 的软对齐。
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, ffn_ratio: float = 4.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Cross-Attention: Q 来自时域，K/V 来自频域
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 标准 Transformer block 结构
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(d_model * ffn_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * ffn_ratio), d_model),
        )

    def forward(
        self,
        Q_visual: torch.Tensor,    # [B, N_TS, d] 时域对齐的视觉特征 (来自 VETime 分支)
        K_V_tokens: torch.Tensor,  # [B, 196, d] 频域原始 patch tokens (来自 ViCO 分支)
    ) -> torch.Tensor:
        """
        Args:
            Q_visual: [B, N_TS, d] 时域对齐的视觉特征（经过 unfold_image + mlp_i）
            K_V_tokens: [B, 196, d] ViCO 分支的原始 patch tokens（不对齐）

        Returns:
            fused_visual: [B, N_TS, d] 融合后的视觉特征，替换原有 I_embeddings
        """
        # 纯交叉注意力：无对角偏置，完全废除刚性假设
        attn_out, attn_weights = self.cross_attn(
            query=Q_visual,
            key=K_V_tokens,
            value=K_V_tokens,
            need_weights=False  # 启用 Flash Attention
        )

        # Post-norm + residual
        x = self.norm1(Q_visual + self.dropout(attn_out))

        # FFN + residual
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x
```

**Step 2: 运行单元测试验证模块**

创建临时测试：

```python
# test_cross_attn.py
import torch
from model.VTS_module import VisualCrossAttention

# 测试参数
B, N_TS, d = 4, 100, 512
N_patches = 196

# 测试数据
Q_visual = torch.randn(B, N_TS, d)
K_V_tokens = torch.randn(B, N_patches, d)

# 测试模块
cross_attn = VisualCrossAttention(d_model=d, num_heads=8)

# Forward
fused = cross_attn(Q_visual, K_V_tokens)

print(f"Q_visual shape: {Q_visual.shape}")  # 期望: [4, 100, 512]
print(f"K_V_tokens shape: {K_V_tokens.shape}")  # 期望: [4, 196, 512]
print(f"Fused output shape: {fused.shape}")  # 期望: [4, 100, 512] (与 Q_visual 相同)
```

运行:
```bash
python test_cross_attn.py
```

Expected output:
```
Q_visual shape: torch.Size([4, 100, 512])
K_V_tokens shape: torch.Size([4, 196, 512])
Fused output shape: torch.Size([4, 100, 512])
```

**Step 3: 提交**

```bash
git add model/VTS_module.py
git commit -m "feat: add VisualCrossAttention module for time-frequency fusion"
```

---

## Task 3: 修改 VETime forward 实现双分支渲染

**Files:**
- Modify: `model/VETime.py`

**目标:** 在 `_forward_impl` 中实现双分支渲染和交叉注意力融合。

**Step 1: 修改 __init__ 初始化新模块**

在 `VETIME.__init__` 中添加（约 line 37 后）：

```python
        # === 视觉时频双分支 ===
        # 交叉注意力融合模块：用 VETime 时域特征查询 ViCO 频域特征
        self.visual_cross_attn = VisualCrossAttention(t_dim, num_heads=8, dropout=0.1)

        # ViCO 分支的 MLP（与 VETime 分支结构一致）
        self.mlp_vico = nn.Sequential(
            nn.Linear(v_dim, t_dim2),
            nn.GELU(),
            nn.Linear(t_dim2, t_dim),
            nn.LayerNorm(t_dim),
        )
```

同时在文件顶部添加导入：

```python
from model.VTS_module import V_Attention, VTS_Alignment, M_moe, VisualCrossAttention
```

**Step 2: 修改 _forward_impl 实现双分支**

替换 `_forward_impl` 方法（约 line 70-107）：

```python
    def _forward_impl(self, hidden_states: torch.Tensor, time_series: torch.Tensor,
                      att_mask: Optional[torch.Tensor] = None, init_img_size=None, labels=None):
        """
        实际的 forward 实现 - 视觉时频双分支架构

        流程:
        1. TS_encoder 提取时序特征
        2. hidden_states 包含 VETime 时域图像（预渲染）
        3. 从 time_series 实时渲染 ViCO 频域图像
        4. 两图像通过共享冻结 MAE
        5. VETime 分支 → unfold_image → Q_visual
        6. ViCO 分支 → 保留原始 tokens → K, V
        7. Cross-Attention 融合 → 替换 I_embeddings
        8. VTS_Alignment + MoE → 任务头
        """
        TS_embeddings0, local_embeddings0, patch_mask = self.ts_encoder(time_series, att_mask)
        B, seq_len, num_features = time_series.size()

        patch_num = patch_mask.size(1) // num_features
        temporal_pos_emb = self.pos_emb_v[:, :patch_num, :]
        multivariate_pos_emb = temporal_pos_emb.repeat(1, num_features, 1)

        # === 分支 A: VETime 时域 (现有流程) ===
        # hidden_states 已经是预渲染的 VETime 图像
        image_features_vetime, _ = self.vit_encoder(hidden_states)
        I_embeddings_vetime = self.vit_encoder.unfold_image(image_features_vetime, init_img_size)
        I_embeddings_vetime = self.mlp_i(I_embeddings_vetime + multivariate_pos_emb)
        # Q_visual: 时域对齐的视觉特征 [B, N_TS, t_dim]
        Q_visual = self.I_att(I_embeddings_vetime, patch_mask)

        # === 分支 B: ViCO 频域 (新增) ===
        # 从原始 time_series 实时渲染 ViCO 图像
        # 注意：这里需要 batch 级渲染，在 dataloader 层处理更合适
        # 当前假设 hidden_states_vico 已传入（后续 Task 4 修改 dataloader）
        # 暂时使用 hidden_states 作为 placeholder（与 VETime 共用图像）
        # 实际实现时需要单独传入 ViCO 图像

        # 方案：利用现有的 hidden_states，因为 dataloader 会同时渲染两分支
        # 假设 init_img_size 包含 ViCO 的 size 信息（需要扩展）
        # 简化实现：ViCO 分支与 VETime 分支共享同一输入图像的 tokens

        # TODO: Task 4 将在 dataloader 层实现 ViCO 渲染
        # 当前 placeholder：使用 VETime tokens 作为 K_V（后续替换）
        K_V_tokens = image_features_vetime[:, 1:, :]  # [B, 196, v_dim] 原始 patch tokens

        # ViCO tokens 投影到时序维度
        K_V_tokens_proj = self.mlp_vico(K_V_tokens)  # [B, 196, t_dim]

        # === 交叉注意力融合 ===
        # Q_visual [B, N_TS, t_dim] 查询 K_V_tokens_proj [B, 196, t_dim]
        I_embeddings = self.visual_cross_attn(Q_visual, K_V_tokens_proj)

        # === 后续流程保持不变 ===
        I_embeddings, TS_embeddings = self.fusion(I_embeddings, TS_embeddings0, patch_mask)
        loss_sc = self.compute_cl(I_embeddings, TS_embeddings, labels, num_features)
        mix_out0 = torch.cat([TS_embeddings, I_embeddings], dim=-1)

        # MoE + 任务头
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

**Step 3: 运行模型加载测试**

测试模型是否能正常初始化：

```python
# test_vetime_init.py
import torch
from model.Vision_encoder.V_encoder import V_model
from model.TS_encoder.ts_model import TS_Model
from model.TS_encoder.config import default_config_t
from model.VETime import VETIME

# 加载视觉编码器
vision_model = V_model(vision_name='mae_visualize_vit_base.pth', MAX_L=5000)

# 加载时序编码器
config_t = default_config_t()
ts_model = TS_Model(config_t)

# 创建 VETime 模型
model = VETIME(
    config_v=None,
    vision_model=vision_model,
    config_t=config_t,
    ts_model=ts_model
)

print(f"VisualCrossAttention initialized: {hasattr(model, 'visual_cross_attn')}")
print(f"mlp_vico initialized: {hasattr(model, 'mlp_vico')}")
```

运行:
```bash
python test_vetime_init.py
```

Expected output:
```
VisualCrossAttention initialized: True
mlp_vico initialized: True
```

**Step 4: 提交**

```bash
git add model/VETime.py
git commit -m "feat: implement dual-branch rendering in VETime forward"
```

---

## Task 4: 修改 dataloader 实现双分支图像生成

**Files:**
- Modify: `dataset/dataloader.py`
- Modify: `dataset/pre_image.py`

**目标:** 在数据加载时同时生成 VETime 和 ViCO 两张图像。

**Step 1: 修改 AnomalyDataset.generate_image**

在 `AnomalyDataset.generate_image` 方法中添加 ViCO 渲染（约 line 114-144）：

```python
    def generate_image(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        生成双分支图像：VETime 时域 + ViCO 频域
        """
        for idx, data0 in enumerate(data):
            target_length = ((len(data0['time_series']) + self.patch_size - 1) // self.patch_size) * self.patch_size

            # === VETime 时域渲染 (现有) ===
            img_vetime, period, padding_value = ts2image_1d(
                data0['time_series'], target_length, self.patch_size
            )

            # === ViCO 频域渲染 (新增) ===
            img_vico = vico_render_timeseries(
                data0['time_series'], period, img_size=224
            )

            # 存储两分支图像
            data[idx]['image_vetime'] = img_vetime  # [3, C*h_size, width]
            data[idx]['image_vico'] = img_vico      # [3, 224, 224]
            data[idx]['period'] = period
            data[idx]['padding_value'] = padding_value

            # 删除旧的 'image' 键（避免混淆）
            if 'image' in data[idx]:
                del data[idx]['image']

        return data
```

同时添加导入：

```python
from dataset.pre_image import ts2image_1d, vico_render_timeseries
```

**Step 2: 修改 __getitem__ 返回两分支图像**

修改 `__getitem__` 方法返回值：

```python
    def __getitem__(self, idx: int):
        # ... 现有代码 ...

        # 返回两分支图像
        image_vetime = torch.from_numpy(self.data[idx]['image_vetime']).float()
        image_vico = torch.from_numpy(self.data[idx]['image_vico']).float()

        return (
            time_series,
            labels,
            attribute,
            image_vetime,    # VETime 时域图像
            image_vico,      # ViCO 频域图像
            period,
            padding_value
        )
```

**Step 3: 修改 collate_fn 处理双分支图像**

在 `collate_fn` 中处理两个图像：

```python
def collate_fn(batch):
    # ... 现有处理 ...

    # 双分支图像
    images_vetime = torch.stack([item[3] for item in batch])  # [B, 3, C*h_size, width]
    images_vico = torch.stack([item[4] for item in batch])    # [B, 3, 224, 224]
    periods = [item[5] for item in batch]
    padding_values = [item[6] for item in batch]

    # fold_image 操作（对 VETime 分支）
    # ViCO 分支已经是 224×224，无需 fold

    return {
        'time_series': time_series,
        'labels': labels,
        'att_mask': att_mask,
        'hidden_states': images_vetime_folded,  # VETime 折叠后的图像
        'hidden_states_vico': images_vico,      # ViCO 图像（已 resize）
        'init_img_size': init_img_size,
        'init_img_size_vico': [(14, 14, 0, 224, 1)] * B,  # ViCO 固定参数
        'periods': periods,
        'padding_values': padding_values,
    }
```

**Step 4: 提交**

```bash
git add dataset/dataloader.py dataset/pre_image.py
git commit -m "feat: implement dual-branch image generation in dataloader"
```

---

## Task 5: 修改 VETime forward 接收 ViCO 图像

**Files:**
- Modify: `model/VETime.py`

**目标:** 修改 `forward` 和 `_forward_impl` 接收 ViCO 分支图像。

**Step 1: 修改 forward 签名**

```python
    def forward(
        self,
        hidden_states: torch.Tensor,       # VETime 时域图像
        hidden_states_vico: torch.Tensor,  # ViCO 频域图像 (新增参数)
        time_series: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        init_img_size=None,
        init_img_size_vico=None,           # ViCO size 信息 (新增参数)
        labels=None
    ):
```

**Step 2: 修改 _forward_impl 使用 ViCO 图像**

```python
    def _forward_impl(
        self,
        hidden_states: torch.Tensor,       # VETime 图像
        hidden_states_vico: torch.Tensor,  # ViCO 图像
        time_series: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        init_img_size=None,
        init_img_size_vico=None,
        labels=None
    ):
        # ... 时序编码 ...

        # === VETime 分支 ===
        image_features_vetime, _ = self.vit_encoder(hidden_states)
        I_embeddings_vetime = self.vit_encoder.unfold_image(image_features_vetime, init_img_size)
        I_embeddings_vetime = self.mlp_i(I_embeddings_vetime + multivariate_pos_emb)
        Q_visual = self.I_att(I_embeddings_vetime, patch_mask)

        # === ViCO 分支 (使用传入的 ViCO 图像) ===
        image_features_vico, _ = self.vit_encoder(hidden_states_vico)
        K_V_tokens = image_features_vico[:, 1:, :]  # [B, 196, v_dim]
        K_V_tokens_proj = self.mlp_vico(K_V_tokens)

        # === Cross-Attention 融合 ===
        I_embeddings = self.visual_cross_attn(Q_visual, K_V_tokens_proj)

        # ... 后续流程不变 ...
```

**Step 3: 修改 _forward_with_checkpointing**

同步修改 checkpoint 版本：

```python
    def _forward_with_checkpointing(
        self,
        hidden_states: torch.Tensor,
        hidden_states_vico: torch.Tensor,  # 新增
        time_series: torch.Tensor,
        att_mask: Optional[torch.Tensor],
        init_img_size,
        init_img_size_vico,                # 新增
        labels
    ):
        # ... 使用 checkpoint 包装双分支计算 ...
```

**Step 4: 提交**

```bash
git add model/VETime.py
git commit -m "feat: integrate ViCO image input in VETime forward"
```

---

## Task 6: 修改 train.py 适配双分支数据流

**Files:**
- Modify: `train.py`

**目标:** 训练脚本适配新的数据格式和模型接口。

**Step 1: 修改数据加载**

在 `train_univariate` 函数中：

```python
# 修改 DataLoader 调用（如果需要）
# collate_fn 已经返回 hidden_states_vico
```

**Step 2: 修改模型 forward 调用**

```python
# 原调用:
# local_embeddings1, m_w, loss_sc, local_embeddings2 = model(
#     hidden_states, time_series, att_mask, init_img_size, labels
# )

# 新调用:
local_embeddings1, m_w, loss_sc, local_embeddings2 = model(
    hidden_states=batch['hidden_states'],
    hidden_states_vico=batch['hidden_states_vico'],
    time_series=batch['time_series'],
    att_mask=batch['att_mask'],
    init_img_size=batch['init_img_size'],
    init_img_size_vico=batch['init_img_size_vico'],
    labels=batch['labels']
)
```

**Step 3: 提交**

```bash
git add train.py
git commit -m "feat: adapt train.py for dual-branch data flow"
```

---

## Task 7: 集成测试与验证

**Files:**
- None (验证阶段)

**目标:** 验证整个双分支架构正常工作。

**Step 1: 运行单样本测试**

```python
# test_dualbranch_integration.py
import torch
from torch.utils.data import DataLoader
from dataset.dataloader import AnomalyDataset, collate_fn
from model.VETime import VETIME
# ... 模型初始化 ...

# 加载单样本
dataset = AnomalyDataset('path/to/dataset.pkl', patch_size=16, gen_image=True)
batch = collate_fn([dataset[0]])

# Forward
output = model(
    hidden_states=batch['hidden_states'],
    hidden_states_vico=batch['hidden_states_vico'],
    time_series=batch['time_series'],
    att_mask=batch['att_mask'],
    init_img_size=batch['init_img_size'],
    init_img_size_vico=batch['init_img_size_vico'],
    labels=batch['labels']
)

print(f"Output shapes: local_emb1={output[0].shape}, local_emb2={output[3].shape}")
```

**Step 2: 运行完整训练一个 epoch**

```bash
python train.py --config configs/train_univariate.yaml --epochs 1
```

Expected: 训练正常运行，无 shape 错误。

**Step 3: 提交**

```bash
git add docs/plans/2026-07-07-visual-timefreq-dualbranch-design.md
git add docs/plans/2026-07-07-visual-timefreq-dualbranch-impl.md
git commit -m "docs: add dual-branch architecture design and implementation plan"
```

---

## 后续优化（不在本次实现范围）

1. **PTA 混合池化**: 将平均池化替换为 Max-Avg Concat，保持点异常锐利度
2. **ViCO 图像缓存**: 预渲染 ViCO 图像并存储，避免训练时实时渲染开销
3. **周期检测优化**: 统一两分支的周期检测逻辑

---

## 文件修改总结

| 文件 | 修改内容 |
|------|----------|
| `dataset/pre_image.py` | 新增 ViCO 渲染函数 |
| `model/VTS_module.py` | 新增 `VisualCrossAttention` 类 |
| `model/VETime.py` | 修改 `__init__` 和 `forward` 实现双分支 |
| `dataset/dataloader.py` | 修改图像生成和 collate |
| `train.py` | 适配新数据格式 |