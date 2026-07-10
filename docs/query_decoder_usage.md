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

1. **任务原型 Token**: 可学习参数 `task_token_rec` 和 `task_token_cls`，初始化标准差为 0.02
2. **RoPE 位置编码**: 与 TS_encoder 保持一致，投影后应用旋转
3. **共享交叉注意力层**: Q/K/V 投影 + Flash Attention
4. **共享 FFN 层**: 两个任务共享，促进隐式正则化

### 工作流程

```
输入: F_TS, F_V, F_A (各 B x N x D)
  |
  v
构建专家存储库 K = V = cat([F_TS, F_V, F_A])  ->  (B, 3N, D)
构建任务 Queries Q = cat([task_token_rec, task_token_cls])  ->  (B, 2N, D)
  |
  v
线性投影: Q, K, V = q_proj(Q), k_proj(K), v_proj(V)
  |
  v
块对齐 RoPE:
  Q_rot = apply_rope(Q, freqs_2N)  # 重构群 + 分类群
  K_rot = apply_rope(K, freqs_3N)  # 时间专家 + 视觉专家 + 融合专家
  |
  v
Flash Attention: y = F.scaled_dot_product_attention(Q_rot, K_rot, V)
  |
  v
输出投影 + LayerNorm + 共享 FFN
  |
  v
任务解耦: F_rec = y[:, :N, :], F_cls = y[:, N:, :]
```

### 与 M_moe 对比

| 特性 | M_moe | QueryDecoder |
|------|-------|--------------|
| 路由机制 | 显式软门控 | 隐式注意力路由 |
| 任务解耦 | 共享特征 + 任务投影 | 任务专属 Query Token |
| 共享层 | Router 跨任务共享 | Cross-Attention + FFN 共享 |
| 梯度流 | 通过门控权重耦合 | 通过注意力权重自发分化 |
| 路由权重 | 返回 m_w 供负载均衡 | 无显式权重 |
| 参数效率 | 较高（任务共享 Router） | 极高（单层共享解码器） |

## 训练建议

### 训练模式选择

QueryDecoder 支持两种训练模式，通过命令行参数 `--query_decoder_training_mode` 选择：

**模式 1: 同时训练 (joint, 推荐)**
```bash
python train.py --query_decoder_training_mode joint
```
- 从 epoch 0 开始多任务联合训练
- 两个任务通过独立 Query 自然解耦，RoPE 让任务关注专家库不同位置
- 梯度驱动隐式路由，自发分化

**模式 2: 分阶段训练 (staged)**
```bash
python train.py --query_decoder_training_mode staged --stage1_epochs 1
```
- 阶段 1 (epoch < stage1_epochs): 仅训练重构任务
- 阶段 2 (epoch >= stage1_epochs): 加入分类任务
- 适用于重构任务需要优先收敛的场景

**对比建议**:
| 特性 | joint | staged |
|------|-------|--------|
| 收敛速度 | 较快 | 较慢 |
| 任务冲突 | RoPE 自动化解 | 需手动调节 |
| 适用场景 | 通用（推荐） | 重构优先级极高时 |

### 其他建议

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

QueryDecoder 使用与 TS_encoder 相同的 RotaryEmbedding 实现，关键设计：

**投影后应用**（与 VETime Encoder 对齐）：
```python
# 1. 先进行线性投影
q = self.q_proj(Q_combined)  # (B, 2N, D)
k = self.k_proj(K_expert)    # (B, 3N, D)

# 2. 再应用 RoPE（投影后、分头前）
q_rot = self.apply_rope(q, freqs_2N)
k_rot = self.apply_rope(k, freqs_3N)
```

**块对齐频率**：
- `freqs_2N = cat([freqs_N, freqs_N])` → 重构群 + 分类群
- `freqs_3N = cat([freqs_N, freqs_N, freqs_N])` → 时间专家 + 视觉专家 + 融合专家

### Flash Attention 加速

使用 `F.scaled_dot_product_attention` 原生支持 Flash Attention 2：
- 计算复杂度: O(2N × 3N) = O(6N²)
- 无需显式计算注意力权重矩阵
- 显存占用从 O(N²) 降至 O(N)

### 内存优化

- 支持 gradient checkpointing 以减少显存占用
- 共享解码器设计，参数量极低
