# Query-based 专家解码器使用指南

## 快速开始

### 启用 QueryDecoder 模式

创建 VETIME 模型时设置 `use_query_decoder=True`:

```python
from model.VETime import VETIME

model = VETIME(
    config_v=vision_config,
    vision_model=vit_encoder,
    config_t=ts_config,
    ts_model=ts_model,
    use_query_decoder=True
)
```

### 训练示例

```python
local_embeddings1, m_w, loss_sc, local_embeddings2 = model(
    hidden_states=images,
    time_series=time_series,
    att_mask=attention_mask,
    init_img_size=img_size,
    labels=labels
)

# m_w 为 None 表示使用 QueryDecoder 模式
# local_embeddings1: 分类特征
# local_embeddings2: 重构特征

rec_loss, _ = model.masked_reconstruction_loss(
    local_embeddings2, time_series, mask, labels
)
cls_loss, _ = model.anomaly_detection_loss(local_embeddings1, labels)

total_loss = rec_loss + cls_loss + loss_sc
```

## 架构说明

### 专家特征来源

| 专家 | 变量 | 来源 |
|------|------|------|
| 时间专家 | F_TS | TS_embeddings0 |
| 视觉专家 | F_V | I_embeddings |
| 融合专家 | F_A | fusion_proj(cat(TS_emb, I_emb)) |

### 核心组件

1. **任务 Token**: 可学习参数 `task_token_rec` 和 `task_token_cls`，初始化标准差为 0.02
2. **RoPE 位置编码**: 与 TS_encoder 保持一致，确保位置对齐
3. **交叉注意力层**: 每个任务独立的 MultiheadAttention
4. **FFN 层**: 增强表达能力，包含 GELU 激活和 Dropout

### 工作流程

```
输入: F_TS, F_V, F_A (各 B x N x D)
  |
  v
构建专家存储库 K = V = cat([F_TS, F_V, F_A])  ->  (B, 3N, D)
  |
  v
生成位置编码 freqs = RoPE(N)
  |
  v
生成任务 Query:
  Q_rec = apply_rope(task_token_rec, freqs)
  Q_cls = apply_rope(task_token_cls, freqs)
  |
  v
交叉注意力:
  F_rec_attn = CrossAttn_rec(Q_rec, K, V)
  F_cls_attn = CrossAttn_cls(Q_cls, K, V)
  |
  v
残差 + LayerNorm + FFN
  |
  v
输出: F_rec, F_cls (各 B x N x D)
```

### 与 M_moe 对比

| 特性 | M_moe | QueryDecoder |
|------|-------|--------------|
| 路由机制 | 显式软门控 | 隐式注意力路由 |
| 任务解耦 | 共享特征 | 任务专属 Query |
| 梯度流 | 耦合 | 自发分化 |
| 路由权重 | 返回 m_w 供负载均衡 | 无显式权重 |
| 参数效率 | 任务共享投影层 | 任务独立注意力层 |

## 训练建议

1. **学习率**: 新参数（task_token, cross_attn, ffn）建议使用较小学习率，可通过参数分组实现：
   ```python
   # 示例：对新参数使用较小学习率
   base_params = [p for n, p in model.named_parameters()
                  if 'query_decoder' not in n and 'fusion_proj' not in n]
   new_params = [p for n, p in model.named_parameters()
                 if 'query_decoder' in n or 'fusion_proj' in n]

   optimizer = torch.optim.AdamW([
       {'params': base_params, 'lr': 1e-4},
       {'params': new_params, 'lr': 1e-5}  # 新参数使用 1/10 学习率
   ])
   ```

2. **任务 Token 初始化**: 默认 std=0.02，可根据任务调整：
   ```python
   # 自定义初始化（如需调整）
   nn.init.normal_(model.query_decoder.task_token_rec, std=0.01)
   nn.init.normal_(model.query_decoder.task_token_cls, std=0.01)
   ```

3. **损失权重**: 重构和分类损失权重建议 1:1，可根据任务优先级调整：
   ```python
   total_loss = alpha * rec_loss + beta * cls_loss + loss_sc
   # alpha=1.0, beta=1.0 为推荐起始值
   ```

4. **Gradient Checkpointing**: 已内置支持，启用方式：
   ```python
   model = VETIME(
       ...,
       use_query_decoder=True,
       use_gradient_checkpointing=True  # 节省显存
   )
   ```

## 推理示例

```python
model.eval()
with torch.no_grad():
    local_embeddings1, _, loss_sc, local_embeddings2 = model(
        hidden_states=images,
        time_series=time_series,
        att_mask=attention_mask,
        init_img_size=img_size
    )

    # 异常检测
    anomaly_scores = model.anomaly_head(local_embeddings1)

    # 时序重构
    reconstructed = model.reconstruction_head(local_embeddings2)
```

## 常见问题

### Q: 如何切换回 M_moe 模式？
A: 创建模型时不设置 `use_query_decoder` 或设置为 `False`：
```python
model = VETIME(..., use_query_decoder=False)  # 或直接省略
```

### Q: 返回值 m_w 为 None 是否正常？
A: 是的，QueryDecoder 模式下无显式路由权重，这是设计特性。路由通过注意力权重隐式实现。

### Q: 如何验证 QueryDecoder 正确工作？
A: 可通过测试文件验证：
```bash
pytest tests/test_query_decoder.py -v
```

## 技术细节

### RoPE 位置编码

QueryDecoder 使用与 TS_encoder 相同的 RotaryEmbedding 实现，确保位置编码一致性。位置编码应用于任务 Query，使其能够区分不同时间位置。

### 梯度流特性

每个任务拥有独立的交叉注意力层和任务 Token，使得：
- 重构任务梯度主要影响 `cross_attn_rec` 和 `task_token_rec`
- 分类任务梯度主要影响 `cross_attn_cls` 和 `task_token_cls`
- 专家特征（F_TS, F_V, F_A）接收两个任务的梯度叠加，实现自发分化

### 内存优化

- 支持 gradient checkpointing 以减少显存占用
- 交叉注意力使用 `need_weights=False` 启用 Flash Attention
