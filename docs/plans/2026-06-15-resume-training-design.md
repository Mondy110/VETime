# 继续训练功能设计文档

> 日期: 2026-06-15

## 背景

多变量训练过程中容易出现OOM（显存不足）问题，需要支持从断点继续训练。同时希望支持加载单变量训练好的权重用于多变量训练的初始化。

## 需求

1. **从epoch中断恢复**：在某个维度训练过程中OOM时，能从该维度已保存的checkpoint恢复
2. **从维度中断恢复**：能从某个维度的final checkpoint恢复，继续后续维度的训练
3. **加载单变量权重启动**：使用单变量训练好的权重作为起点，从头开始多变量顺序训练

## 设计方案

### 1. 新增命令行参数

```python
--resume          # 从checkpoint继续训练（完整状态恢复）
--pretrain_from   # 使用预训练权重启动多变量训练（仅模型权重）
```

### 2. 配置文件支持

在 `multivariate_config.yaml` 中新增：

```yaml
multivariate:
  # ... 现有配置 ...
  resume_from: null      # 可在配置中指定resume路径
  pretrain_from: null    # 可在配置中指定pretrain路径
```

命令行参数优先级高于配置文件。

### 3. Checkpoint 格式

#### 完整状态Checkpoint

保存路径: `vetime_dim{N}_epoch{E}_full.pth`

```python
checkpoint = {
    # 核心状态
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,                    # 当前epoch（已完成）
    'global_step': global_step,        # 全局步数

    # 多变量训练特有
    'dataset_idx': dataset_idx,        # 当前数据集索引
    'current_dim': current_dim,        # 当前维度
    'prev_checkpoint_path': prev_checkpoint_path,  # 用于维度间传递

    # 早停状态
    'best_val_loss': best_val_loss,
    'patience_counter': patience_counter,

    # 随机状态（确保可复现）
    'random_state': {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
}
```

#### 仅模型权重Checkpoint

保持现有格式，用于维度间传递和最终保存：

- `vetime_dim{N}_best.pth` - 验证集最佳模型
- `vetime_dim{N}_final.pth` - 维度训练完成时的最终模型

### 4. Checkpoint 保存时机

| 时机 | 文件名 | 内容 |
|------|--------|------|
| 每个epoch结束 | `vetime_dim{N}_epoch{E}_full.pth` | 完整状态 |
| 验证集最佳 | `vetime_dim{N}_best.pth` | 仅模型权重 |
| 维度训练完成 | `vetime_dim{N}_final.pth` | 仅模型权重 |

### 5. 恢复逻辑

#### `--resume` 恢复流程

1. 加载checkpoint文件
2. 验证维度匹配：
   - 一致：从该维度的epoch中断处继续
   - 不一致：作为预训练权重加载（使用strict=False）
3. 恢复训练状态：
   - 加载模型权重
   - 加载optimizer状态
   - 设置起始epoch、global_step
   - 恢复早停计数器
   - 恢复随机状态

#### `--pretrain_from` 加载流程

1. 加载预训练权重（仅模型权重）
2. 使用 `strict=False` 加载，缺失参数随机初始化
3. 从配置文件中的第一个数据集开始训练

### 6. 代码修改

```
train.py
├── save_full_checkpoint()      # 新增：保存完整状态
├── load_full_checkpoint()      # 新增：加载完整状态
├── load_pretrain_weights()     # 新增：加载预训练权重
└── train_multivariate()        # 修改：集成上述功能
```

## 使用示例

```bash
# 场景1: 正常训练
python train.py --config config/multivariate_config.yaml

# 场景2: 从epoch中断恢复
python train.py --config config/multivariate_config.yaml \
    --resume ./output/multivariate_checkpoints/vetime_dim10_epoch5_full.pth

# 场景3: 使用单变量权重启动
python train.py --config config/multivariate_config.yaml \
    --pretrain_from ./checkpoints/weight_ts/pretrain_checkpoint_best_uni.pth

# 场景4: 从维度最终检查点恢复（跨维度）
python train.py --config config/multivariate_config.yaml \
    --resume ./output/multivariate_checkpoints/vetime_dim6_final.pth

# 场景5: 通过配置文件指定
# config/multivariate_config.yaml 中设置:
# resume_from: ./output/multivariate_checkpoints/vetime_dim10_epoch5_full.pth
python train.py --config config/multivariate_config.yaml
```

## 参数对比

| 参数 | 用途 | 恢复状态 | 典型场景 |
|------|------|----------|----------|
| `--pretrain_from` | 加载预训练权重启动新训练 | 仅模型权重 | 单变量→多变量 |
| `--resume` | 从中断点继续训练 | 完整训练状态 | OOM恢复、继续训练 |
