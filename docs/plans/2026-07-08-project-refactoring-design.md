# VETime 项目重构设计文档

日期：2026-07-08
状态：已确认

## 1. 背景与动机

VETime 项目当前耦合度极高，核心问题：

| 耦合问题 | 严重度 | 具体表现 |
|---|---|---|
| train.py 深度访问模型内部 | **严重** | `model.vit_encoder.fold_image()`、`model.anomaly_detection_loss()`、`model.weighted_reconstruction_loss()` |
| train_univariate 是 750 行 God Function | **高** | 模型构建、optimizer、checkpoint、训练循环、日志、验证全部交织 |
| 损失函数散布 3 处 | **高** | loss/loss.py、ts_model.py、train.py 各持有部分损失逻辑 |
| VETIME 继承 TS_Model 并偷取子组件 | **高** | `self.projection_layer = self.ts_encoder.ts_encoder.projection_layer`（2 层深度访问） |
| Test_TSB.py 重复数据管线 | **中** | `dataloader_TSB()` 重复了 normalization/padding/masking 逻辑 |
| utils/pcgrad.py 是死代码 | **中** | 已设计但从未被导入 |

## 2. 重构决策

| 决策项 | 选择 | 理由 |
|---|---|---|
| 重构范围 | 仅 train_univariate + Test_TSB | train_multivariate 暂不重构 |
| 配置管理 | Hydra + OmegaConf | PyTorch 生态主流，支持配置组合与命令行覆盖 |
| 模型修改 | 允许调整 VETime.py 封装 | 新增方法（compute_loss、fold_images、split_sequence），不改 nn.Module 命名，权重 100% 兼容 |
| PCGrad | 保持现状 | 死代码暂不激活 |
| 日志系统 | logging + TensorBoard | 替换所有 print()，TensorBoard 保持不变 |
| 重构策略 | 渐进式抽离（方案 A） | 先数据→再模型→最后训练，每步可验证 |

## 3. 目标目录结构

```
VETime/
├── configs/
│   ├── base.yaml              # 通用配置（seed、设备、精度等）
│   ├── univariate.yaml        # univariate 训练超参
│   ├── multivariate.yaml      # multivariate 配置（暂不重构，仅迁移）
│   └── model/
│       ├── vetime.yaml        # 模型超参（从 TS_encoder/config.py 抽离）
│       └── vision.yaml        # 视觉编码器超参
│
├── src/
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── anomaly_dataset.py # AnomalyDataset（从 dataset/dataloader.py 迁移）
│   │   ├── collate.py         # collate_fn + DynamicLengthBatchSampler
│   │   ├── masking.py         # create_random_mask（从 collate_fn 中分离）
│   │   └── pre_image.py       # ts2image_1d, vico_render（从 dataset/pre_image.py 迁移）
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vetime.py          # VETIME（从 model/VETime.py 迁移，加统一接口）
│   │   ├── ts_encoder/
│   │   │   ├── __init__.py
│   │   │   ├── config.py      # TimeSeriesConfig（从 model/TS_encoder/config.py 迁移）
│   │   │   ├── ts_model.py    # TS_Model
│   │   │   ├── ts_encoder.py  # TimeSeriesEncoder
│   │   │   └── encoding_utils.py  # LoRA, RoPE, BinaryAttention
│   │   ├── vision_encoder/
│   │   │   ├── __init__.py
│   │   │   ├── v_encoder.py   # V_model
│   │   │   ├── models_mae.py  # MaskedAutoencoderViT
│   │   │   └── vit4ad.py      # MAETS_AD, VitTS_AD
│   │   └── vts_module.py      # V_Attention, VTS_Alignment, M_moe, GatedTimeFrequencyFusion
│   │
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Trainer 类：训练循环、阶段切换、分类预热
│   │   ├── evaluator.py       # Evaluator 类：TSB-AD 测试、指标计算
│   │   └── hooks.py           # 分类预热 freeze/restore、阶段过渡逻辑
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── contrastive.py     # win_Contrastive_Loss
│   │   ├── reconstruction.py  # denoising_reconstruction_loss, weighted_reconstruction_loss
│   │   ├── anomaly.py         # anomaly_detection_loss (Masked Focal Loss)
│   │   └── balance.py         # load_balance_loss
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py          # 日志管理（logging + TensorBoard 封装）
│       ├── seed.py            # seed_everything()
│       ├── checkpoint.py      # save/load checkpoint 通用逻辑
│       └── metrics.py         # 评估指标（从 evaluation/ 迁移）
│
├── evaluation/                 # 保留，evaluation/basic_metrics.py 不动
├── train.py                    # 薄壳入口：Hydra 配置 → 构建组件 → Trainer.run()
├── evaluate.py                 # 薄壳入口：替代 Test_TSB.py
├── Test_TSB.py                 # 保留为兼容入口（内部调用 evaluate.py）
└── ...
```

## 4. VETIME 统一接口

在 VETIME 类上新增以下方法，**不修改任何已有的 `__init__` 和 `forward`**：

### 4.1 compute_loss()

```python
def compute_loss(self, outputs, batch, stage, labels=None):
    """
    统一损失计算入口，train.py 不再直接调用 TS_Model 的内部损失方法。

    Args:
        outputs: self.forward() 的返回值 (local_embeddings1, m_w, loss_cl, local_embeddings2)
        batch:   DataLoader 输出的 dict（含 time_series, mask, labels 等）
        stage:   1=仅重构, 2=重构+分类
        labels:  可选，覆盖 batch 中的 labels

    Returns:
        dict: {
            'loss_recon': Tensor,     # 重构损失（已乘 alpha_recon）
            'loss_anomaly': Tensor,   # 分类损失（stage 1 时为 0）
            'loss_cl': Tensor,        # 对比损失（已乘 cl_weight）
            'loss_balance': Tensor,   # 专家平衡损失（已乘 balance_weight）
            'loss_total': Tensor,     # 总损失（可直接 backward）
            'logits': Tensor,         # 分类 logits
            'reconstruction': Tensor, # 重构输出
        }
    """
```

### 4.2 fold_images()

```python
def fold_images(self, images, images_vico, period, padding_value, **data_setting):
    """
    封装 vit_encoder.fold_image 调用。

    Returns:
        (images_folded, init_img_size)
    """
```

### 4.3 split_sequence()

```python
def split_sequence(self, images, time_series, att_mask, labels):
    """
    封装 self.split_data 调用。

    Returns:
        list of (sub_images, sub_ts, sub_att_mask, sub_labels) chunks
    """
```

## 5. Trainer 类设计

```python
class Trainer:
    """单变量训练引擎，承载两阶段课程训练循环。"""

    def __init__(self, config, model, train_loader, val_loader, accelerator):
        ...

    def setup(self):
        """构建 optimizer/scheduler/early_stopping"""
        ...

    def train_epoch(self, epoch):
        """单个 epoch 训练，根据 stage 切换策略"""
        is_stage_1 = epoch < self.cfg.training.stage1_epochs
        is_cls_warmup = (not is_stage_1) and (epoch == self.cfg.training.stage1_epochs) and (self.cfg.training.cls_warmup_ratio > 0)
        ...

    def validate(self, epoch):
        """验证循环"""
        ...

    def run(self):
        """完整训练流程：遍历所有 epoch，自动切换 stage"""
        self.setup()
        for epoch in range(self.cfg.training.total_epochs):
            self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            if self.early_stopping(val_loss, self.model):
                break
```

**train.py 薄壳**：

```python
@hydra.main(config_path="configs", config_name="base")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    accelerator = Accelerator(...)
    model = build_model(cfg.model)
    train_loader, val_loader = build_dataloaders(cfg.data)
    trainer = Trainer(cfg, model, train_loader, val_loader, accelerator)
    trainer.run()
```

## 6. Evaluator 设计

```python
class Evaluator:
    """TSB-AD 基准测试引擎"""

    def __init__(self, config, model, accelerator):
        ...

    def evaluate_dataset(self, dataset_name, test_data, test_labels):
        """单个数据集推理 + 异常分数计算"""
        ...

    def evaluate_benchmark(self, dataset_list):
        """遍历 TSB-AD 所有数据集，汇总结果"""
        ...

    def compute_metrics(self, scores, labels):
        """调用 evaluation.metrics 计算指标"""
        ...
```

## 7. 配置管理（Hydra + OmegaConf）

### configs/base.yaml

```yaml
# ==================== 基础配置 ====================
seed: 2024                          # 随机种子，保证实验可复现
device: cuda                        # 计算设备
mixed_precision: bf16               # 混合精度模式（bf16/fp16/no）

# ==================== 训练配置 ====================
training:
  stage1_epochs: 1                  # Stage 1 持续的 epoch 数（仅重构预训练）
  total_epochs: 30                  # 总 epoch 数
  cls_warmup_ratio: 0.5             # Stage 2 首个 epoch 的分类预热比例（0=跳过）

  optimizer:
    lr: 1e-4                        # 学习率
    weight_decay: 0.01              # 权重衰减
  scheduler:
    type: cosine_with_warmup        # 学习率调度器类型
    warmup_ratio: 0.1              # 预热比例
  early_stopping:
    patience: 5                     # 早停耐心值

# ==================== 损失配置 ====================
loss:
  alpha_recon: 0.05                 # 重构损失缩放系数，降低重构量级避免淹没分类梯度
  cl_weight: 0.1                    # 对比学习损失权重
  balance_weight: 0.01              # MoE 专家负载均衡损失权重
  focal_gamma: 2.0                  # Focal Loss gamma，增大对难样本的关注
  w_anomaly: 1.2                    # 异常点分类权重
  w_normal: 0.8                     # 正常点分类权重

# ==================== 数据配置 ====================
data:
  batch_size: 32                    # 批大小
  mask_ratio: 0.3                   # 自监督掩码比例
  gradient_accumulation_steps: 2    # 梯度累积步数
  max_seq_length: 5000              # 最大序列长度
  effective_batch_size: 64          # 动态批大小的有效批大小
```

### configs/model/vetime.yaml

```yaml
# ==================== VETIME 时序编码器配置 ====================
d_model: 512                        # Transformer 隐藏维度
d_proj: 256                         # 投影维度
patch_size: 16                      # Patch 大小
num_layers: 8                       # Transformer 层数
num_heads: 8                        # 注意力头数
d_ff_dropout: 0.1                   # FFN dropout 率
max_total_tokens: 8192              # 最大 token 数
use_rope: true                      # 是否使用旋转位置编码
activation: gelu                    # 激活函数
num_features: 1                     # 输入特征数
use_lora: true                      # 是否使用 LoRA 微调
lora_r: 8                           # LoRA 秩
lora_alpha: 16                      # LoRA alpha
```

命令行覆盖示例：
```bash
python train.py training.optimizer.lr=1e-4 loss.alpha_recon=0.1
```

## 8. 损失函数归拢

| 现有位置 | 目标位置 | 处理方式 |
|---|---|---|
| `loss/loss.py` → `win_Contrastive_Loss` | `src/losses/contrastive.py` | 直接迁移 |
| `loss/loss.py` → `load_balance_loss` | `src/losses/balance.py` | 直接迁移 |
| `ts_model.py` → `anomaly_detection_loss` | `src/losses/anomaly.py` | 迁移计算逻辑 |
| `ts_model.py` → `weighted_reconstruction_loss` | `src/losses/reconstruction.py` | 迁移计算逻辑 |
| `train.py` → 损失聚合逻辑 | `VETIME.compute_loss()` | 聚合逻辑移入统一接口 |

ts_model.py 上的损失方法暂时保留（标注 `@deprecated`），内部改为调用 `src/losses/` 中的函数，保证向后兼容。

## 9. 渐进式迁移步骤

| 步骤 | 内容 | 验证方式 |
|---|---|---|
| **Step 1** | 创建 `src/utils/`（logger.py、seed.py、checkpoint.py） | 单元测试：日志输出、种子可复现、checkpoint 读写 |
| **Step 2** | 迁移 `dataset/` → `src/datasets/`，分离 masking | 跑一个 batch 验证输出格式一致 |
| **Step 3** | 迁移 `model/` → `src/models/`，VETime.py 加统一接口 | 虚假 Tensor forward + compute_loss，验证输出形状 |
| **Step 4** | 抽离训练循环 → `src/engines/trainer.py`，配置 Hydra | 端到端训练 1 epoch，对比旧 train.py 的 loss 曲线 |
| **Step 5** | 抽离评估 → `src/engines/evaluator.py` + evaluate.py | 跑 TSB-AD 测试，对比指标数值一致 |

每步完成后旧代码仍可用，新旧可并行运行验证一致性。

## 10. 规范遵守

- **配置与代码分离**：所有超参写进 YAML，核心代码不硬编码
- **数据-模型-训练三方解耦**：datasets/ 只输出 Tensor，models/ 只做 forward，engines/ 负责缝合
- **防御性编程**：VETIME 新接口方法加形状注释，后续补充 assert
- **日志正规化**：print() → logging，TensorBoard 保持
- **可复现性**：seed_everything() 在 train.py 最开头调用
