# VETime 多变量顺序训练设计文档

**日期**: 2026-06-12
**作者**: Claude
**状态**: 待实现

---

## 1. 概述

### 1.1 背景

VETime 项目目前仅支持单变量（Univariate）时间序列异常检测。为了扩展到多变量数据集（如 38 维的 SMD、55 维的 MSL），需要引入 TimeRCD 论文中的"分维度顺序训练（Dimension-wise Sequential Training）"策略。

### 1.2 核心目标

1. **支持多变量数据集**：通过分维度顺序训练策略处理高维多变量时间序列
2. **零回归承诺**：绝对保留并完美兼容原有的单变量数据加载逻辑
3. **知识延续**：模型权重在不同维度的数据集之间平滑传递

### 1.3 设计约束

- 单变量训练流程必须完全不受影响
- 代码风格与现有项目保持一致
- 遵循 TimeRCD 的"销毁重建 + strict=False"范式

---

## 2. 架构设计

### 2.1 分维度顺序训练策略

```
┌─────────────────────────────────────────────────────────────┐
│                    多变量顺序训练流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  循环外（只执行一次）                                         │
│  ├── Accelerator 初始化                                     │
│  ├── Vision Encoder 初始化 + to(device) + eval()            │
│  └── prev_checkpoint_path = None                            │
│                                                             │
│  循环内（每个数据集）                                         │
│  ├── 1. 刷新 config.num_features = current_dim              │
│  ├── 2. create_model(args, dim, vision_model, config_v)     │
│  ├── 3. strict=False 加载 prev_checkpoint                   │
│  ├── 4. 重建 Optimizer                                      │
│  ├── 5. 创建 DataLoader                                     │
│  ├── 6. accelerator.prepare(model, optimizer, loader)       │
│  ├── 7. model.train() + vision_model.eval()                 │
│  ├── 8. 内层 Epoch 训练                                     │
│  ├── 9. 保存 checkpoint                                     │
│  └── 10. accelerator.free_memory() + 清理显存               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 关键设计决策

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 数据集存储方式 | 按维度分组存储 | 同一文件内样本维度一致，简化 DataLoader |
| 图像拼接策略 | 全通道垂直拼接 | `ts2image_1d` 已原生支持 |
| 权重延续机制 | 销毁重建 + strict=False | 符合 TimeRCD 范式，避免参数纳管问题 |
| Batch Size 策略 | 用户自定义映射表 | 灵活可控 |
| 训练顺序 | 建议递增但允许灵活配置 | 平衡约束与灵活性 |

---

## 3. 文件修改清单

### 3.1 新增文件

| 文件路径 | 用途 |
|----------|------|
| `config/multivariate_config.yaml` | 多变量训练配置文件 |

### 3.2 修改文件

| 文件路径 | 改动类型 | 具体内容 |
|----------|----------|----------|
| `train.py` | 重构 | 新增 `train_multivariate()`、`create_model()`、`train_univariate()` 函数 |
| `model/TS_encoder/config.py` | 无需修改 | `num_features` 参数已支持动态设置 |

### 3.3 无需修改的文件

| 文件路径 | 原因 |
|----------|------|
| `dataset/dataloader.py` | `ts2image_1d` 已原生支持多变量拼接 |
| `dataset/pre_image.py` | 图像生成逻辑已支持 (L, C) 输入 |
| `model/VETime.py` | 模型初始化时已根据 `num_features` 自动构建组件 |

---

## 4. 详细设计

### 4.1 配置文件设计

```yaml
# config/multivariate_config.yaml

# 训练模式: univariate(单变量) 或 multivariate(多变量顺序训练)
training_mode: multivariate

# 多变量顺序训练配置
multivariate:
  # 数据集列表，按训练顺序排列
  datasets:
    - path: ./dataset/dim_1.pkl
      dim: 1
    - path: ./dataset/dim_38.pkl
      dim: 38
    - path: ./dataset/dim_55.pkl
      dim: 55

  # 维度到批次大小的映射
  dim_batch_size_map:
    1: 32
    38: 8
    55: 4

  # checkpoint 保存目录
  checkpoint_dir: ./output/multivariate_checkpoints

  # 是否严格按维度递增顺序检查
  enforce_dim_order: false

# 单变量训练配置（兼容原有方式）
univariate:
  dataset_path: ./dataset/univariate.pkl
  batch_size: 32
```

### 4.2 train.py 改造设计

#### 4.2.1 主入口函数

```python
def main(args):
    """主入口：根据配置选择训练模式"""
    # 加载配置文件
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        training_mode = config.get('training_mode', 'univariate')
    else:
        config = None
        training_mode = 'univariate'

    if training_mode == 'univariate':
        return train_univariate(args)
    elif training_mode == 'multivariate':
        return train_multivariate(args, config)
```

#### 4.2.2 多变量训练函数核心逻辑

```python
def train_multivariate(args, config: Dict[str, Any]):
    """
    多变量顺序训练（参考 TimeRCD 范式）

    核心策略：
    1. 销毁重建：每次切换维度时全新实例化模型
    2. 知识延续：strict=False 加载 checkpoint
    3. 参数纳管：重建 Optimizer 确保所有参数可训练
    4. 显存安全：Accelerator 循环外初始化，Vision Encoder 单次实例化
    """

    # ========== [关键] Accelerator 全局初始化（只执行一次）==========
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=4
    )
    device = accelerator.device

    # ========== [关键] Vision Encoder 只实例化一次 ==========
    vision_model = V_model(args.vision_name, MAX_L=5000, unpatch=True, finetune_type='none')
    vision_model = vision_model.to(device)
    vision_model.eval()

    # 【防御性代码】显式冻结所有参数，防止意外梯度计算
    for param in vision_model.parameters():
        param.requires_grad = False

    config_v = vision_model.config

    prev_checkpoint_path = None

    # ========== 外层维度循环 ==========
    for dataset_idx, dataset_info in enumerate(datasets):
        current_dim = dataset_info['dim']
        batch_size = dim_batch_size_map.get(str(current_dim), 4)

        # 1. 刷新配置
        default_config_t.num_features = current_dim

        # 2. 全新实例化模型（复用 vision_model）
        model, ts_model = create_model(args, current_dim, vision_model, config_v)

        # 3. 知识延续（strict=False 是灵魂）
        if prev_checkpoint_path is not None:
            state_dict = torch.load(prev_checkpoint_path, map_location='cpu')

            # 【防御性代码】剥离 DDP 的 'module.' 前缀，适配不同保存格式
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict, strict=False)

        # 4. 重建 Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # 5. 创建 DataLoader
        train_loader = create_dataloader(dataset_path, batch_size, args)

        # 6. prepare 动态组件
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

        # 7. 设置训练模式（注意：vision_model 需保持 eval）
        model.train()
        vision_model.eval()

        # 8. 内层 Epoch 训练
        for epoch in range(args.num_epochs):
            train_one_epoch(model, train_loader, optimizer, accelerator, args, device)

        # 9. 保存 Checkpoint
        unwrapped_model = accelerator.unwrap_model(model)
        checkpoint_path = os.path.join(checkpoint_dir, f'vetime_dim{current_dim}.pth')
        torch.save(unwrapped_model.state_dict(), checkpoint_path)
        prev_checkpoint_path = checkpoint_path

        # 10. 彻底清理显存（注意：不删除 vision_model）
        del model, optimizer, train_loader, ts_model
        accelerator.free_memory()
        torch.cuda.empty_cache()
        gc.collect()
```

#### 4.2.3 模型创建函数

```python
def create_model(args, num_features: int, vision_model, config_v):
    """
    创建 VETime 模型（复用已存在的 vision_model）

    Args:
        args: 命令行参数
        num_features: 当前数据集的特征维度
        vision_model: 已实例化的视觉编码器（复用，不重建）
        config_v: 视觉模型配置

    Returns:
        model: VETIME 模型
        ts_model: 时间序列编码器
    """
    # 更新 TS Encoder 配置
    default_config_t.num_features = num_features

    # 仅实例化 TS Encoder（根据 num_features 自动构建 BinaryAttentionBias）
    ts_model = TS_Model(default_config_t)

    # 构建 VETime 完整模型
    model = VETIME(config_v, vision_model, default_config_t, ts_model, args.model_name)

    return model, ts_model
```

---

## 5. 关键技术点

### 5.1 为什么"动态打补丁"是反模式？

| 问题 | 原因 | 后果 |
|------|------|------|
| `hasattr` 动态添加模块 | 新参数不在 Optimizer 的参数图中 | 死权重，永远不更新 |
| 运行时修改 `num_features` | 不改变已初始化的网络结构 | 前向传播维度不匹配 |
| 不重建 Optimizer | 模型参数图已改变 | 梯度计算错误 |

### 5.2 TimeRCD 范式优势

```
维度切换生命周期：
1. 更新 config.num_features = new_dim
2. 销毁旧模型 → 重建新模型（自动构建正确结构）
3. strict=False 加载 checkpoint（继承 Backbone，新模块随机初始化）
4. 重建 Optimizer（新参数图被正确纳管）
5. 创建新 DataLoader
6. 训练 → 保存 checkpoint
```

### 5.3 分布式训练安全要点

| 问题 | 修复 |
|------|------|
| Accelerator 在循环内初始化 | 移到循环外，只初始化一次 |
| Vision Encoder 重复重建 | 循环外实例化，作为参数传入 |
| model.train() 覆盖 vision_model 状态 | model.train() 后调用 vision_model.eval() |
| 计算图残留引用 | 使用 accelerator.free_memory() |

---

## 6. 验证计划

### 6.1 单变量兼容性验证

1. 运行原有单变量训练脚本，确保输出与改造前完全一致
2. 检查单变量路径的代码覆盖率

### 6.2 多变量训练验证

1. 创建测试数据集（1维、8维、16维）
2. 验证维度间 checkpoint 正确传递
3. 检查 BinaryAttentionBias 模块在多变量时被正确创建

### 6.3 分布式验证

1. 单卡训练测试
2. 多卡 DDP 训练测试
3. 显存泄漏检测

---

## 7. 参考资料

- TimeRCD 论文：分维度顺序训练策略
- TimeRCD 源码：`/mnt/sda/cjmProject/Time-RCD/training.py`
- HuggingFace Accelerate 文档
