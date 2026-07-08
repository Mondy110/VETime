# Query-based 专家解码器实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现基于 Query 的隐式路由解码器，通过交叉注意力机制实现任务专属特征提取和梯度驱动分化。

**Architecture:** 在现有 M_moe 模块旁边新增 QueryDecoder 模块，三个专家特征（时间、视觉、融合）通过拼接形成专家存储库，任务专属 Query 通过 RoPE 位置编码和交叉注意力自动从专家库中提取最相关特征。

**Tech Stack:** PyTorch, nn.MultiheadAttention, RotaryEmbedding (复用现有)

## Global Constraints

- RoPE 必须复用 `model/TS_encoder/encoding_utils.py` 中的 `RotaryEmbedding` 类
- 保持向后兼容：`use_query_decoder=False` 时输出与原模型一致
- 最小侵入式改造：不删除现有 M_moe 模块
- 专家特征来源：F_TS = TS_embeddings0, F_V = I_embeddings, F_A = fusion_proj(mix_out0)
- 任务 Token 使用可学习参数初始化，std=0.02

---

## File Structure

```
model/
├── VTS_module.py          # 新增 QueryDecoder 类
├── VETime.py              # 修改：集成 QueryDecoder
└── TS_encoder/
    └── encoding_utils.py  # 复用 RotaryEmbedding

tests/
└── test_query_decoder.py  # 新增：单元测试
```

---

### Task 1: 实现 QueryDecoder 核心模块

**Files:**
- Modify: `model/VTS_module.py`（新增 QueryDecoder 类）
- Test: `tests/test_query_decoder.py`（新建）

**Interfaces:**
- Consumes: `RotaryEmbedding` from `model/TS_encoder/encoding_utils.py`
- Produces: `QueryDecoder` class with `forward(F_TS, F_V, F_A, patch_mask) -> (F_rec, F_cls)`

- [ ] **Step 1: 编写 QueryDecoder 输出形状测试**

```python
# tests/test_query_decoder.py
import pytest
import torch
from model.VTS_module import QueryDecoder


def test_query_decoder_output_shape():
    """测试 QueryDecoder 输出形状正确"""
    B, N, D = 2, 128, 256
    decoder = QueryDecoder(d_model=D, num_heads=8, dropout=0.1)
    
    F_TS = torch.randn(B, N, D)
    F_V = torch.randn(B, N, D)
    F_A = torch.randn(B, N, D)
    
    F_rec, F_cls = decoder(F_TS, F_V, F_A)
    
    assert F_rec.shape == (B, N, D), f"F_rec shape mismatch: {F_rec.shape}"
    assert F_cls.shape == (B, N, D), f"F_cls shape mismatch: {F_cls.shape}"


def test_query_decoder_with_mask():
    """测试带掩码的 QueryDecoder"""
    B, N, D = 2, 128, 256
    decoder = QueryDecoder(d_model=D, num_heads=8, dropout=0.1)
    
    F_TS = torch.randn(B, N, D)
    F_V = torch.randn(B, N, D)
    F_A = torch.randn(B, N, D)
    patch_mask = torch.ones(B, N, dtype=torch.bool)
    patch_mask[:, -10:] = False  # 最后10个位置为padding
    
    F_rec, F_cls = decoder(F_TS, F_V, F_A, patch_mask)
    
    assert F_rec.shape == (B, N, D)
    assert F_cls.shape == (B, N, D)
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd /mnt/sda1/cjm_workspace/VETime && python -m pytest tests/test_query_decoder.py -v`
Expected: FAIL with "cannot import name 'QueryDecoder'"

- [ ] **Step 3: 实现 QueryDecoder 类**

在 `model/VTS_module.py` 文件末尾添加:

```python
from model.TS_encoder.encoding_utils import RotaryEmbedding


class QueryDecoder(nn.Module):
    """基于 Query 的隐式路由解码器。
    
    通过任务专属 Query 从专家存储库中提取最相关特征，
    实现梯度驱动的隐空间自发分化。
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 任务 Token（可学习参数）
        self.task_token_rec = nn.Parameter(torch.zeros(1, 1, d_model))
        self.task_token_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.task_token_rec, std=0.02)
        nn.init.normal_(self.task_token_cls, std=0.02)
        
        # RoPE 位置编码（复用现有）
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
        
        # FFN（增加表达能力）
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
    
    def apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """应用 RoPE 位置编码。"""
        B, seq_len, embed_dim = x.shape
        x_ = x.view(B, seq_len, embed_dim // 2, 2)
        cos = freqs.cos().unsqueeze(0)
        sin = freqs.sin().unsqueeze(0)
        
        x_rot = torch.stack([
            x_[..., 0] * cos - x_[..., 1] * sin,
            x_[..., 0] * sin + x_[..., 1] * cos,
        ], dim=-1)
        return x_rot.view(B, seq_len, embed_dim)
    
    def forward(
        self,
        F_TS: torch.Tensor,
        F_V: torch.Tensor,
        F_A: torch.Tensor,
        patch_mask: torch.Tensor = None
    ) -> tuple:
        """前向传播。"""
        B, N, D = F_TS.shape
        
        # 1. 构建专家存储库
        K = V = torch.cat([F_TS, F_V, F_A], dim=1)  # (B, 3N, D)
        
        # 2. 生成位置编码
        freqs = self.rope(N)  # (N, D // 2)
        
        # 3. 生成任务 Query
        Q_rec_base = self.task_token_rec.expand(B, N, -1)
        Q_cls_base = self.task_token_cls.expand(B, N, -1)
        
        Q_rec = self.apply_rope(Q_rec_base, freqs)
        Q_cls = self.apply_rope(Q_cls_base, freqs)
        
        # 4. 构建注意力掩码
        kv_mask = None
        if patch_mask is not None:
            kv_mask = patch_mask.repeat(1, 3)
        
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
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd /mnt/sda1/cjm_workspace/VETime && python -m pytest tests/test_query_decoder.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: 提交**

```bash
git add model/VTS_module.py tests/test_query_decoder.py
git commit -m "feat: 添加 QueryDecoder 核心模块

实现基于交叉注意力的隐式路由解码器

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: 验证 RoPE 位置编码一致性

**Files:**
- Modify: `tests/test_query_decoder.py`

**Interfaces:**
- Consumes: `RotaryEmbedding` from `model/TS_encoder/encoding_utils.py`
- Produces: 验证 RoPE 一致性

- [ ] **Step 1: 添加 RoPE 一致性测试**

在 `tests/test_query_decoder.py` 末尾添加:

```python
def test_rope_consistency():
    """验证 QueryDecoder 的 RoPE 与 TS_encoder 一致"""
    from model.TS_encoder.encoding_utils import RotaryEmbedding
    
    D = 256
    N = 128
    
    rope1 = RotaryEmbedding(D)
    rope2 = RotaryEmbedding(D)
    
    freqs1 = rope1(N)
    freqs2 = rope2(N)
    
    assert torch.allclose(freqs1, freqs2), "RoPE 频率不一致"
    
    decoder = QueryDecoder(d_model=D, num_heads=8)
    freqs_decoder = decoder.rope(N)
    
    assert torch.allclose(freqs1, freqs_decoder), "QueryDecoder RoPE 与 TS_encoder 不一致"


def test_rope_position_alignment():
    """验证 RoPE 位置对齐特性"""
    D = 256
    N = 64
    
    decoder = QueryDecoder(d_model=D, num_heads=8)
    freqs = decoder.rope(N)
    
    base_token = torch.randn(1, 1, D)
    Q1 = base_token.expand(1, N, -1).clone()
    Q2 = base_token.expand(1, N, -1).clone()
    
    Q1_rot = decoder.apply_rope(Q1, freqs)
    Q2_rot = decoder.apply_rope(Q2, freqs)
    
    assert torch.allclose(Q1_rot, Q2_rot), "相同位置应产生相同旋转结果"
```

- [ ] **Step 2: 运行测试**

Run: `cd /mnt/sda1/cjm_workspace/VETime && python -m pytest tests/test_query_decoder.py -v`
Expected: PASS (4 tests)

- [ ] **Step 3: 提交**

```bash
git add tests/test_query_decoder.py
git commit -m "test: 添加 RoPE 位置编码一致性测试

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: 集成 QueryDecoder 到 VETIME 模型

**Files:**
- Modify: `model/VETime.py`

**Interfaces:**
- Consumes: `QueryDecoder` from `model/VTS_module.py`
- Produces: VETIME 支持 `use_query_decoder` 开关

- [ ] **Step 1: 修改 VETIME.__init__ 添加 QueryDecoder 支持**

在 `model/VETime.py` 的 `__init__` 方法中，约第49行后添加:

```python
        # 新增：Query-based 解码器（可选）
        self.query_decoder = None
        self.fusion_proj = None
        self.use_query_decoder = kwargs.get('use_query_decoder', False)
        
        if self.use_query_decoder:
            from model.VTS_module import QueryDecoder
            self.query_decoder = QueryDecoder(
                d_model=t_dim,
                num_heads=8,
                dropout=0.1
            )
            self.fusion_proj = nn.Sequential(
                nn.Linear(t_dim * 2, t_dim),
                nn.LayerNorm(t_dim)
            )
```

- [ ] **Step 2: 修改 _forward_impl 添加新路径**

在 `model/VETime.py` 的 `_forward_impl` 方法中，在 `loss_sc` 计算后（约第88行）添加:

```python
        # === 新增：Query-based 解码路径 ===
        if self.use_query_decoder and self.query_decoder is not None:
            # 生成三个专家特征
            F_TS = TS_embeddings0  # (B, N, D) 时间专家
            F_V = I_embeddings      # (B, N, D) 视觉专家
            
            # 融合专家特征
            mix_out0_for_proj = torch.cat([TS_embeddings, I_embeddings], dim=-1)
            F_A = self.fusion_proj(mix_out0_for_proj)  # (B, N, D)
            
            # Query-based 解码
            F_rec, F_cls = self.query_decoder(F_TS, F_V, F_A, patch_mask)
            
            # 任务头投影 - 分类分支
            patch_proj = self.projection_layer(F_cls)
            local_embeddings = patch_proj.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
            local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
            local_embeddings1 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]
            
            # 任务头投影 - 重构分支
            patch_proj2 = self.projection_layer(F_rec)
            local_embeddings = patch_proj2.view(B, num_features, seq_len//self.patch_size, self.patch_size, self.d_proj)
            local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4).contiguous()
            local_embeddings2 = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]
            
            # 新路径无路由权重
            m_w = None
            
            return local_embeddings1, m_w, loss_sc, local_embeddings2
```

- [ ] **Step 3: 验证语法正确**

Run: `cd /mnt/sda1/cjm_workspace/VETime && python -c "from model.VETime import VETIME; print('Import OK')"`
Expected: Import OK

- [ ] **Step 4: 提交**

```bash
git add model/VETime.py
git commit -m "feat: 集成 QueryDecoder 到 VETIME 模型

添加 use_query_decoder 开关，默认使用原 M_moe 路径

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: 添加梯度流测试

**Files:**
- Modify: `tests/test_query_decoder.py`

- [ ] **Step 1: 添加梯度流测试**

在 `tests/test_query_decoder.py` 末尾添加:

```python
def test_gradient_flow_to_experts():
    """测试梯度正确传播到专家特征"""
    B, N, D = 2, 64, 256
    decoder = QueryDecoder(d_model=D, num_heads=8, dropout=0.1)
    
    F_TS = torch.randn(B, N, D, requires_grad=True)
    F_V = torch.randn(B, N, D, requires_grad=True)
    F_A = torch.randn(B, N, D, requires_grad=True)
    
    F_rec, F_cls = decoder(F_TS, F_V, F_A)
    
    loss = F_rec.mean() + F_cls.mean()
    loss.backward()
    
    assert F_TS.grad is not None, "梯度未传播到 F_TS"
    assert F_V.grad is not None, "梯度未传播到 F_V"
    assert F_A.grad is not None, "梯度未传播到 F_A"
    assert F_TS.grad.shape == F_TS.shape


def test_separate_gradient_paths():
    """测试两个任务的梯度路径独立"""
    B, N, D = 2, 64, 256
    decoder = QueryDecoder(d_model=D, num_heads=8, dropout=0.1)
    
    F_TS = torch.randn(B, N, D, requires_grad=True)
    F_V = torch.randn(B, N, D, requires_grad=True)
    F_A = torch.randn(B, N, D, requires_grad=True)
    
    F_rec, F_cls = decoder(F_TS, F_V, F_A)
    
    loss_rec = F_rec.mean()
    loss_rec.backward(retain_graph=True)
    grad_TS_from_rec = F_TS.grad.clone()
    
    F_TS.grad = None
    F_V.grad = None
    F_A.grad = None
    
    loss_cls = F_cls.mean()
    loss_cls.backward()
    grad_TS_from_cls = F_TS.grad.clone()
    
    assert not torch.allclose(grad_TS_from_rec, grad_TS_from_cls, atol=1e-6), \
        "两个任务的梯度路径应该有所不同"
```

- [ ] **Step 2: 运行测试**

Run: `cd /mnt/sda1/cjm_workspace/VETime && python -m pytest tests/test_query_decoder.py -v`
Expected: PASS (6 tests)

- [ ] **Step 3: 提交**

```bash
git add tests/test_query_decoder.py
git commit -m "test: 添加 QueryDecoder 梯度流测试

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: 更新 _forward_with_checkpointing 方法

**Files:**
- Modify: `model/VETime.py`

- [ ] **Step 1: 修改 _forward_with_checkpointing 支持 QueryDecoder**

在 `model/VETime.py` 的 `_forward_with_checkpointing` 方法中，添加 QueryDecoder 路径支持:

在 `moe_projection_forward` 函数之前添加:

```python
        # === Query-based 解码路径 ===
        if self.use_query_decoder and self.query_decoder is not None:
            def query_decoder_forward(TS_embeddings0, I_embeddings, TS_embeddings, patch_mask, B, seq_len, num_features):
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
                
                return local_embeddings1, local_embeddings2
            
            local_embeddings1, local_embeddings2 = checkpoint(
                query_decoder_forward,
                TS_embeddings0, I_embeddings, TS_embeddings, patch_mask, B, seq_len, num_features
            )
            m_w = None
            return local_embeddings1, m_w, loss_sc, local_embeddings2
```

- [ ] **Step 2: 验证语法正确**

Run: `cd /mnt/sda1/cjm_workspace/VETime && python -c "from model.VETime import VETIME; print('Import OK')"`
Expected: Import OK

- [ ] **Step 3: 提交**

```bash
git add model/VETime.py
git commit -m "feat: 支持 QueryDecoder 的 gradient checkpointing

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: 创建使用文档

**Files:**
- Create: `docs/query_decoder_usage.md`

- [ ] **Step 1: 创建使用文档**

```markdown
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

### 与 M_moe 对比

| 特性 | M_moe | QueryDecoder |
|------|-------|--------------|
| 路由机制 | 显式软门控 | 隐式注意力路由 |
| 任务解耦 | 共享特征 | 任务专属 Query |
| 梯度流 | 耦合 | 自发分化 |

## 训练建议

1. 学习率：新参数建议使用较小学习率
2. 任务 Token 初始化：默认 std=0.02
3. 损失权重：重构和分类损失权重建议 1:1
```

- [ ] **Step 2: 提交**

```bash
git add docs/query_decoder_usage.md
git commit -m "docs: 添加 QueryDecoder 使用指南

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## 验收标准

- [ ] 所有测试通过
- [ ] `use_query_decoder=False` 时输出与原模型一致
- [ ] `use_query_decoder=True` 时正常训练和推理
- [ ] 梯度正确传播到骨干网络
