# TS Encoder Pretrain Checkpoint 结构分析

## 文件信息

- **文件路径**: `checkpoints/weight_ts/pretrain_checkpoint_best_multi.pth`
- **训练轮次**: Epoch 2
- **最终 Loss**: 0.1824

## 模型配置

```python
ts_config = {
    'd_model': 512,        # 隐藏层维度
    'd_proj': 256,         # 投影层维度
    'patch_size': 16,      # 时间序列 patch 大小
    'num_query_tokens': 1, # Query token 数量
    'num_layers': 8,       # Transformer 层数
    'num_heads': 8,        # 注意力头数
    'd_ff_dropout': 0.1,   # Dropout 率
    'use_rope': True,      # 使用旋转位置编码
    'activation': 'gelu',  # 激活函数
    'num_features': 8      # 输入特征数
}
```

## 网络架构

```
TS_Encoder
├── embedding_layer          # Patch Embedding
│   ├── weight: [512, 16]
│   └── bias: [512]
├── rope_embedder            # Rotary Position Embedding
│   └── inv_freq: [256]
├── transformer_encoder      # 8层 Transformer
│   └── layers[0-7] (结构相同)
│       ├── self_attn        # 自注意力
│       │   ├── q_proj.weight: [512, 512]
│       │   ├── k_proj.weight: [512, 512]
│       │   ├── v_proj.weight: [512, 512]
│       │   ├── out_proj.weight: [512, 512]
│       │   └── binary_attention_bias.emd.weight: [2, 8]
│       ├── input_norm       # 输入 RMSNorm
│       │   └── scale: [512]
│       ├── output_norm      # 输出 RMSNorm
│       │   └── scale: [512]
│       └── mlp              # SwiGLU MLP
│           ├── gate_proj.weight: [2048, 512]
│           ├── gate_proj.bias: [2048]
│           ├── up_proj.weight: [2048, 512]
│           ├── up_proj.bias: [2048]
│           ├── down_proj.weight: [512, 2048]
│           └── down_proj.bias: [512]
└── projection_layer         # 输出投影
    ├── weight: [4096, 512]
    └── bias: [4096]

Reconstruction_Head          # 重构头
├── 0: Linear [256 → 1024]
├── 3: Linear [1024 → 1024]
└── 6: Linear [1024 → 1]

Anomaly_Head                 # 异常检测头
├── 0: Linear [256 → 128]
└── 3: Linear [128 → 2]
```

## 详细参数列表

### TS Encoder Embedding

| 参数名 | 形状 | 参数量 |
|--------|------|--------|
| ts_encoder.embedding_layer.weight | [512, 16] | 8,192 |
| ts_encoder.embedding_layer.bias | [512] | 512 |
| ts_encoder.rope_embedder.inv_freq | [256] | 256 |

### Transformer Encoder Layers (×8 层，每层参数相同)

| 参数名 | 形状 | 参数量 |
|--------|------|--------|
| self_attn.q_proj.weight | [512, 512] | 262,144 |
| self_attn.k_proj.weight | [512, 512] | 262,144 |
| self_attn.v_proj.weight | [512, 512] | 262,144 |
| self_attn.out_proj.weight | [512, 512] | 262,144 |
| self_attn.binary_attention_bias.emd.weight | [2, 8] | 16 |
| input_norm.scale | [512] | 512 |
| output_norm.scale | [512] | 512 |
| mlp.gate_proj.weight | [2048, 512] | 1,048,576 |
| mlp.gate_proj.bias | [2048] | 2,048 |
| mlp.up_proj.weight | [2048, 512] | 1,048,576 |
| mlp.up_proj.bias | [2048] | 2,048 |
| mlp.down_proj.weight | [512, 2048] | 1,048,576 |
| mlp.down_proj.bias | [512] | 512 |
| **每层小计** | | **3,937,920** |

### Projection Layer

| 参数名 | 形状 | 参数量 |
|--------|------|--------|
| ts_encoder.projection_layer.weight | [4096, 512] | 2,097,152 |
| ts_encoder.projection_layer.bias | [4096] | 4,096 |

### Reconstruction Head

| 参数名 | 形状 | 参数量 |
|--------|------|--------|
| reconstruction_head.0.weight | [1024, 256] | 262,144 |
| reconstruction_head.0.bias | [1024] | 1,024 |
| reconstruction_head.3.weight | [1024, 1024] | 1,048,576 |
| reconstruction_head.3.bias | [1024] | 1,024 |
| reconstruction_head.6.weight | [1, 1024] | 1,024 |
| reconstruction_head.6.bias | [1] | 1 |

### Anomaly Head

| 参数名 | 形状 | 参数量 |
|--------|------|--------|
| anomaly_head.0.weight | [128, 256] | 32,768 |
| anomaly_head.0.bias | [128] | 128 |
| anomaly_head.3.weight | [2, 128] | 256 |
| anomaly_head.3.bias | [2] | 2 |

## 参数统计

| 模块 | 参数量 |
|------|--------|
| Embedding Layer | 8,704 |
| RoPE | 256 |
| Transformer Layers (×8) | 31,503,360 |
| Projection Layer | 2,101,248 |
| Reconstruction Head | 1,313,793 |
| Anomaly Head | 33,154 |
| **总计** | **~34.96M** |

## 关键设计特点

1. **SwiGLU MLP**: 使用 gate_proj + up_proj + down_proj 三层结构，激活函数为 GELU
2. **RMSNorm**: 使用 input_norm 和 output_norm（只有 scale 参数，无 bias）
3. **Rotary Position Embedding**: 使用 RoPE 进行位置编码
4. **Binary Attention Bias**: 用于注意力机制的二元偏置嵌入
5. **多头注意力**: 8 个注意力头，每个头维度 64 (512/8)

## 与 LoRA 模式的键名映射关系

当使用 `--ts_finetune_type lora` 时，预训练权重的键名需要映射：

```
原始键名                              →  LoRA 键名
ts_encoder.xxx.q_proj.weight         →  ts_encoder.xxx.q_proj.original_linear.weight
ts_encoder.xxx.k_proj.weight         →  ts_encoder.xxx.k_proj.original_linear.weight
ts_encoder.xxx.v_proj.weight         →  ts_encoder.xxx.v_proj.original_linear.weight
ts_encoder.xxx.out_proj.weight       →  ts_encoder.xxx.out_proj.original_linear.weight
ts_encoder.xxx.gate_proj.weight      →  ts_encoder.xxx.gate_proj.original_linear.weight
ts_encoder.xxx.up_proj.weight        →  ts_encoder.xxx.up_proj.original_linear.weight
ts_encoder.xxx.down_proj.weight      →  ts_encoder.xxx.down_proj.original_linear.weight
```

其他参数（embedding、norm、projection、heads）保持不变。