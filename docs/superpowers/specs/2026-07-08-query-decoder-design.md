# Query-based 专家解码器设计文档

**日期**: 2026-07-08
**状态**: 待实现
**作者**: Claude Code

---

## 1. 背景与动机

当前 VETime 模型使用 M_moe（多门控专家混合）模块进行多任务特征融合。虽然有效，但存在以下局限性：

1. **显式路由限制**：软门控机制需要预定义的路由策略，缺乏自适应的任务-专家匹配能力
2. **梯度耦合**：重构和分类任务共享同一组专家特征，可能导致梯度冲突
3. **特征纯度不足**：融合特征同时包含重构和分类信息，缺乏任务专属性

本设计提出一种**基于 Query 的隐式路由解码器**，通过交叉注意力机制实现：
- 任务专属 Query 自动从专家库中提取最相关的特征
- 梯度驱动的隐空间自发分化
- 零额外 Token 开销

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                     Backbone (冻结/可训练)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  TS_encoder  │  │  Vit_encoder │  │   VTS_Alignment  │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
│         │                 │                    │             │
│         ▼                 ▼                    ▼             │
│    TS_embeddings0    I_embeddings         mix_out0          │
│    (B, N, D)         (B, N, D)           (B, N, 2D)         │
│         │                 │                    │             │
│         ▼                 ▼                    ▼             │
│     F_TS (专家1)      F_V (专家2)          F_A (专家3)       │
│    时间特征           视觉特征              融合特征          │
└─────────┬─────────────────┬────────────────────┬─────────────┘
          │                 │                    │
          └─────────────────┼────────────────────┘
                            ▼
              ┌─────────────────────────────┐
              │      专家存储库构建          │
              │  K = V = Cat(F_TS,F_V,F_A)  │
              │       (B, 3N, D)            │
              └─────────────┬───────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
  ┌───────────────┐                   ┌───────────────┐
  │  Q_rec 生成    │                   │  Q_cls 生成    │
  │ Task_Token_rec │                   │ Task_Token_cls │
  │    + RoPE     │                   │    + RoPE     │
  └───────┬───────┘                   └───────┬───────┘
          │                                   │
          ▼                                   ▼
  ┌───────────────┐                   ┌───────────────┐
  │ CrossAttn_rec │                   │ CrossAttn_cls │
  │ Q_rec × K,V   │                   │ Q_cls × K,V   │
  └───────┬───────┘                   └───────┬───────┘
          │                                   │
          ▼                                   ▼
      F'_rec (B,N,D)                    F'_cls (B,N,D)
      重构特征                          分类特征
          │                                   │
          ▼                                   ▼
  ┌───────────────┐                   ┌───────────────┐
  │ projection_   │                   │ projection_   │
  │ layer         │                   │ layer         │
  └───────┬───────┘                   └───────┬───────┘
          │                                   │
          ▼                                   ▼
  ┌───────────────┐                   ┌───────────────┐
  │ reconstruction│                   │ anomaly_head  │
  │ _head         │                   │               │
  └───────┬───────┘                   └───────┬───────┘
          │                                   │
          ▼                                   ▼
      MSE Loss                            BCE Loss
```

---

## 3. 核心模块设计

### 3.1 QueryDecoder 模块

**位置**: `model/VTS_module.py`（新增类）

**职责**:
- 接收三个专家特征矩阵
- 生成任务专属 Query
- 执行交叉注意力路由
- 输出任务解耦特征

**参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| d_model | int | 必填 | 特征维度 |
| num_heads | int | 8 | 注意力头数 |
| dropout | float | 0.1 | Dropout 概率 |
| num_tasks | int | 2 | 任务数量 |

**初始化**:
```python
class QueryDecoder(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 任务 Token（可学习参数）
        self.task_token_rec = nn.Parameter(torch.zeros(1, 1, d_model))
        self.task_token_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.task_token_rec, std=0.02)
        nn.init.normal_(self.task_token_cls, std=0.02)

        # RoPE 位置编码（复用现有 RotaryEmbedding）
        self.rope = RotaryEmbedding(d_model)

        # 交叉注意力层（每个任务独立）
        self.cross_attn_rec = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_cls = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # LayerNorm
        self.norm_rec = nn.LayerNorm(d_model)
        self.norm_cls = nn.LayerNorm(d_model)

        # FFN（可选，增加表达能力）
        self.ffn_rec = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn_cls = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm_ffn_rec = nn.LayerNorm(d_model)
        self.norm_ffn_cls = nn.LayerNorm(d_model)
```

**前向传播**:
```python
def forward(self, F_TS, F_V, F_A, patch_mask=None):
    """
    Args:
        F_TS: 时间专家特征 (B, N, D)
        F_V: 视觉专家特征 (B, N, D)
        F_A: 融合专家特征 (B, N, D)
        patch_mask: 有效位置掩码 (B, N)

    Returns:
        F_rec: 重构特征 (B, N, D)
        F_cls: 分类特征 (B, N, D)
    """
    B, N, D = F_TS.shape

    # 1. 构建专家存储库
    K = V = torch.cat([F_TS, F_V, F_A], dim=1)  # (B, 3N, D)

    # 2. 生成位置编码
    freqs = self.rope(N)  # (N, D//2)

    # 3. 生成任务 Query
    Q_rec_base = self.task_token_rec.expand(B, N, -1)  # (B, N, D)
    Q_cls_base = self.task_token_cls.expand(B, N, -1)  # (B, N, D)

    # 应用 RoPE
    Q_rec = self.apply_rope(Q_rec_base, freqs)
    Q_cls = self.apply_rope(Q_cls_base, freqs)

    # 4. 构建注意力掩码（如果需要）
    kv_mask = None
    if patch_mask is not None:
        # 专家存储库掩码：每个专家位置对应原始位置
        kv_mask = patch_mask.repeat(1, 3)  # (B, 3N)

    # 5. 交叉注意力
    F_rec_attn, _ = self.cross_attn_rec(
        query=Q_rec,
        key=K,
        value=V,
        key_padding_mask=~kv_mask if kv_mask is not None else None
    )
    F_cls_attn, _ = self.cross_attn_cls(
        query=Q_cls,
        key=K,
        value=V,
        key_padding_mask=~kv_mask if kv_mask is not None else None
    )

    # 6. 残差连接 + LayerNorm
    F_rec = self.norm_rec(Q_rec + F_rec_attn)
    F_cls = self.norm_cls(Q_cls + F_cls_attn)

    # 7. FFN + 残差
    F_rec = self.norm_ffn_rec(F_rec + self.ffn_rec(F_rec))
    F_cls = self.norm_ffn_cls(F_cls + self.ffn_cls(F_cls))

    return F_rec, F_cls

def apply_rope(self, x, freqs):
    """应用 RoPE 位置编码（复用现有实现逻辑）"""
    B, seq_len, embed_dim = x.shape
    x_ = x.view(B, seq_len, embed_dim // 2, 2)
    cos = freqs.cos().unsqueeze(0)
    sin = freqs.sin().unsqueeze(0)

    x_rot = torch.stack([
        x_[..., 0] * cos - x_[..., 1] * sin,
        x_[..., 0] * sin + x_[..., 1] * cos,
    ], dim=-1)
    return x_rot.view(B, seq_len, embed_dim)
```

---

### 3.2 VETIME 模型修改

**位置**: `model/VETime.py`

**修改策略**: 最小侵入式改造

**新增成员**:
```python
class VETIME(TS_Model):
    def __init__(self, ...):
        # ... 现有代码 ...

        # 新增：Query-based 解码器
        self.query_decoder = QueryDecoder(
            d_model=t_dim,
            num_heads=8,
            dropout=0.1
        )

        # 新增：融合特征投影（用于生成 F_A）
        self.fusion_proj = nn.Sequential(
            nn.Linear(t_dim * 2, t_dim),
            nn.LayerNorm(t_dim)
        )

        # 新增：模式选择标志
        self.use_query_decoder = True  # 可通过参数控制
```

**前向传播修改**:
```python
def _forward_impl(self, ...):
    # ... 现有编码逻辑 ...

    # 生成三个专家特征
    F_TS = TS_embeddings0  # (B, N, D)
    F_V = I_embeddings      # (B, N, D)

    # 融合专家特征
    mix_out0 = torch.cat([TS_embeddings, I_embeddings], dim=-1)
    F_A = self.fusion_proj(mix_out0)  # (B, N, D)

    if self.use_query_decoder:
        # 新路径：Query-based 解码
        F_rec, F_cls = self.query_decoder(F_TS, F_V, F_A, patch_mask)

        # 任务头投影
        local_embeddings1 = self.projection_layer(F_cls)
        local_embeddings1 = local_embeddings1.view(B, num_features, seq_len//self.patch_size,
                                                    self.patch_size, self.d_proj)
        local_embeddings1 = local_embeddings1.permute(0, 2, 3, 1, 4).contiguous()
        local_embeddings1 = local_embeddings1.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]

        local_embeddings2 = self.projection_layer(F_rec)
        local_embeddings2 = local_embeddings2.view(B, num_features, seq_len//self.patch_size,
                                                    self.patch_size, self.d_proj)
        local_embeddings2 = local_embeddings2.permute(0, 2, 3, 1, 4).contiguous()
        local_embeddings2 = local_embeddings2.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]

        m_w = None  # 新路径无路由权重
    else:
        # 原路径：M_moe 融合
        mix_out_a, m_w_a = self.mm_w(mix_out0, TS_embeddings0, I_embeddings0, mix_out0, task_id=1)
        mix_out_r, m_w_r = self.mm_w(mix_out0, TS_embeddings0, I_embeddings0, mix_out0, task_id=0)
        m_w = {0: m_w_r, 1: m_w_a}

        # ... 原投影逻辑 ...

    return local_embeddings1, m_w, loss_sc, local_embeddings2
```

---

## 4. 数据流详细说明

### 4.1 输入

| 变量 | 形状 | 来源 | 说明 |
|------|------|------|------|
| time_series | (B, N, F) | 用户输入 | 原始时间序列 |
| hidden_states | (B, L, ...) | 用户输入 | 视觉输入 |
| att_mask | (B, N) | 用户输入 | 有效位置掩码 |
| labels | (B, N) | 用户输入 | 异常标签 |

### 4.2 骨干网络输出

| 变量 | 形状 | 计算方式 | 专家类型 |
|------|------|----------|----------|
| F_TS | (B, N, D) | ts_encoder(time_series) | 时间专家 |
| F_V | (B, N, D) | vit_encoder + mlp_i + I_att | 视觉专家 |
| F_A | (B, N, D) | fusion_proj(cat(TS_emb, I_emb)) | 融合专家 |

### 4.3 Query 生成

```
Task_Token_rec: (1, 1, D) → expand → (B, N, D)
RoPE: RotaryEmbedding(N) → (N, D//2)
Q_rec = apply_rope(Task_Token_rec, RoPE) → (B, N, D)

Task_Token_cls: (1, 1, D) → expand → (B, N, D)
Q_cls = apply_rope(Task_Token_cls, RoPE) → (B, N, D)
```

### 4.4 交叉注意力

```
K = V = cat(F_TS, F_V, F_A) → (B, 3N, D)

Attention_rec = softmax(Q_rec @ K^T / sqrt(d)) @ V
F'_rec = Attention_rec → (B, N, D)

Attention_cls = softmax(Q_cls @ K^T / sqrt(d)) @ V
F'_cls = Attention_cls → (B, N, D)
```

### 4.5 任务头输出

| 分支 | 输入 | 输出形状 | 损失函数 |
|------|------|----------|----------|
| 重构 | F'_rec → projection_layer → reconstruction_head | (B, N, 1) | MSE |
| 分类 | F'_cls → projection_layer → anomaly_head | (B, N, 2) | BCE |

---

## 5. 梯度驱动分化机制

### 5.1 核心思想

通过多目标反向传播，让不同任务的梯度"拉扯"解码器学习不同的注意力模式：

- **重构梯度**（MSE）：惩罚高频突变，迫使 Q_rec 关注 F_TS 和 F_V 中的低频平滑特征
- **分类梯度**（BCE）：对正常波形零梯度，迫使 Q_cls 关注 F_A 中的异常判别特征

### 5.2 注意力矩阵演化

初始化时，Q_rec 和 Q_cls 的注意力分布相似。训练过程中：

1. 重构损失反向传播 → Q_rec 的注意力权重向专家1、2区域集中
2. 分类损失反向传播 → Q_cls 的注意力权重向专家3区域集中
3. 自发形成任务专属的特征路由模式

### 5.3 理论保证

由于 RoPE 的位置对齐特性：
- Q_rec[i] 与 K[i], K[N+i], K[2N+i] 具有较高的位置相似度
- 任务 Token 的语义差异驱使注意力权重分化

---

## 6. 向后兼容性

### 6.1 配置开关

```python
# config.py 新增参数
use_query_decoder: bool = False  # 默认使用原 M_moe 路径
```

### 6.2 权重加载

- `use_query_decoder=False`：可加载原模型权重，无缺失/多余参数
- `use_query_decoder=True`：新增参数随机初始化，可从预训练模型微调

### 6.3 推理一致性

两种模式的输出接口完全一致：
```python
local_embeddings1, m_w, loss_sc, local_embeddings2 = model(...)
```

当 `use_query_decoder=True` 时，`m_w` 返回 `None`。

---

## 7. 实现要点

### 7.1 RoPE 复用

直接使用 `model/TS_encoder/encoding_utils.py` 中的 `RotaryEmbedding` 类，确保位置编码与时序编码器一致。

### 7.2 掩码处理

专家存储库的掩码需要将原始掩码复制 3 份：
```python
kv_mask = patch_mask.repeat(1, 3)  # (B, 3N)
```

### 7.3 内存优化

对于长序列，可考虑：
- 使用 Flash Attention（PyTorch 2.0+ 已内置）
- 分块计算交叉注意力

---

## 8. 测试计划

### 8.1 单元测试

- [ ] QueryDecoder 输出形状正确
- [ ] RoPE 位置编码与 TS_encoder 一致
- [ ] 掩码正确传播

### 8.2 集成测试

- [ ] `use_query_decoder=False` 输出与原模型一致
- [ ] `use_query_decoder=True` 正常训练
- [ ] 梯度正确传播到骨干网络

### 8.3 性能验证

- [ ] 对比 M_moe 和 QueryDecoder 的训练损失曲线
- [ ] 验证注意力矩阵的分化程度
- [ ] 评估异常检测 F1 分数和重构 MSE

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 注意力矩阵不分化的 | 任务特征耦合 | 增加任务 Token 初始化差异；调整损失权重 |
| 训练不稳定 | 损失震荡 | 使用预训练任务 Token；降低学习率 |
| 内存占用增加 | 长序列 OOM | 启用 Flash Attention；梯度检查点 |
| 与原模型性能差距 | 需重新调参 | 保留 M_moe 路径作为备选 |

---

## 10. 未来扩展

1. **多任务扩展**：增加更多任务 Token（如预测、插值）
2. **动态专家数量**：根据数据复杂度自适应调整专家数量
3. **注意力可视化**：分析不同任务对专家的偏好模式
