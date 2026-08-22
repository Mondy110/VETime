# VETime 清洁架构重构设计

**日期：** 2026-08-22  
**状态：** 已设计，待评审  
**范围：** 在不改变现有训练和测试命令的前提下，重构 VETime 的内部结构；将 Hydra 设为标准配置入口；保证时序预训练权重 `checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth` 可完整加载。

## 1. 目标与边界

### 1.1 目标

1. 采用常见的深度学习工程布局，将模型、数据、训练引擎、损失、指标、checkpoint 与入口代码分离。
2. 将 `train_hydra.py` 设为推荐入口，同时保持 `train.py` 的现有 CLI 参数与 `Test_TSB.py` 的现有测试方式可用。
3. 让 CLI、Hydra 与测试入口调用同一套训练、评估、模型构建与 checkpoint 加载实现。
4. 通过显式兼容层完整加载现有时序预训练 checkpoint，而非仅依靠 `strict=False`。
5. 将时序预训练、VETime 模型保存与训练恢复三种 checkpoint 的语义和校验规则分开。
6. 保留现有 LoRA、冻结模式、CMRG、QueryDecoder、梯度检查点、多变量恢复训练和 TSB-AD 评估功能。

### 1.2 非目标

1. 不修改或重训练既有 checkpoint。
2. 不改变 MAE 视觉编码器的默认冻结策略。
3. 不在本次重构中改变模型计算语义、损失公式或指标定义。
4. 不强制用户改变现有的脚本命令、参数名或默认 checkpoint 路径。

## 2. 已确认的输入与兼容约束

| 资源 | 固定路径 | 角色 |
|---|---|---|
| 时序预训练权重 | `checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth` | 时序编码器、重构头、异常头初始化与合约测试 |
| MAE 权重 | `checkpoints/weight_v/mae_visualize_base.pth` | 冻结视觉编码器与端到端 smoke test |

当前时序预训练模型的 checkpoint 键空间为：

```text
ts_encoder.*
reconstruction_head.*
anomaly_head.*
```

现有 `VETIME(TS_Model)` 同时使用继承和重复子模块引用，导致模型职责和 checkpoint 命名耦合。重构后，时序模型必须改为由多模态模型组合，而不是被继承。

## 3. 目录与依赖结构

项目采用 `src/` 布局，生产代码通过 `vetime` 包导入：

```text
VETime/
├── configs/
│   ├── base.yaml
│   ├── data/
│   ├── model/
│   └── experiment/
├── src/vetime/
│   ├── application/
│   │   ├── build_model.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── data/
│   │   ├── datasets.py
│   │   ├── collate.py
│   │   └── dataloaders.py
│   ├── engine/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── phases.py
│   │   └── callbacks.py
│   ├── infrastructure/
│   │   ├── checkpointing/
│   │   │   ├── temporal_legacy.py
│   │   │   ├── model_checkpoint.py
│   │   │   └── resume.py
│   │   └── logging/
│   ├── interfaces/
│   │   ├── cli.py
│   │   └── hydra.py
│   ├── losses/
│   ├── metrics/
│   └── models/
│       ├── temporal/
│       │   ├── config.py
│       │   ├── encoder.py
│       │   ├── heads.py
│       │   └── model.py
│       ├── vision/
│       ├── multimodal/
│       └── factory.py
├── scripts/
│   ├── train_cli.py
│   ├── train_hydra.py
│   └── evaluate_tsb.py
├── tests/
├── train.py                 # 兼容转发入口
├── train_hydra.py           # 兼容转发入口
└── Test_TSB.py              # 兼容转发入口
```

依赖只能从外向内：

```text
interfaces -> application -> engine / factories -> models, data, losses, metrics
                                      |
                                      -> infrastructure
```

`models`、`data`、`losses` 与 `metrics` 不得导入 CLI、Hydra 或文件系统路径；checkpoint、日志、路径与配置加载属于 `infrastructure` 或 `interfaces`。

## 4. 模型边界

```text
TemporalModel
├── TimeSeriesEncoder
├── ReconstructionHead
└── AnomalyHead

VETimeMultimodalModel
├── temporal: TemporalModel
├── vision_encoder: FrozenVisionEncoder
├── alignment / image attention / MoE
├── query_decoder（可选）
└── cmrg modules（可选）
```

`VETimeMultimodalModel` 继承 `nn.Module`，但不继承 `TemporalModel`。它通过 `self.temporal` 调用时序编码器与两个任务头；不再将同一子模块注册在多个属性下。模型工厂是创建、配置 LoRA、安装 CMRG、装入预训练权重和应用冻结策略的唯一装配位置。

## 5. 入口与统一配置

### 5.1 入口职责

| 入口 | 长期定位 | 仅负责 |
|---|---|---|
| `scripts/train_hydra.py` | 推荐入口 | Hydra 配置组合、override、调用训练用例 |
| `scripts/train_cli.py` | CLI 实现入口 | 参数解析、转换配置、调用训练用例 |
| `scripts/evaluate_tsb.py` | TSB 评估实现入口 | 参数解析、转换配置、调用评估用例 |
| 根目录 `train.py` / `train_hydra.py` / `Test_TSB.py` | 向后兼容转发 | 导入并调用对应 `scripts/` 入口 |

### 5.2 统一配置

定义不可变的 `TrainingConfig` 与 `EvaluationConfig` 数据类。CLI adapter 与 Hydra adapter 分别将输入转换为这两类配置；`application` 不再接收 `argparse.Namespace` 或 Hydra `DictConfig`。

Hydra 是标准配置入口，但现有参数名保持可用。例如两种调用最终都获得等价 `TrainingConfig`：

```bash
python train.py --ts_path checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth
python train_hydra.py paths.ts_path=checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth
```

## 6. Checkpoint 协议

### 6.1 三类 checkpoint

| 类型 | 服务 | 使用场景 | 加载策略 |
|---|---|---|---|
| `temporal_pretrain` | `TemporalLegacyCheckpointLoader` | 加载旧时序预训练权重 | 必需时序参数严格校验 |
| `vetime_model` | `ModelCheckpointService` | 推理或继续微调 VETime | 版本化模型加载 |
| `training_resume` | `ResumeCheckpointService` | 恢复训练进度 | 模型、优化器、调度器与训练状态分别校验 |

新保存格式使用下列顶层字段：

```python
{
    "format_version": 2,
    "kind": "temporal_pretrain" | "vetime_model" | "training_resume",
    "model_state_dict": {...},
    "metadata": {
        "architecture": "vetime-clean",
        "temporal_config": {...},
        "source_checkpoint": "...",
    },
    # 仅 training_resume：
    "optimizer_state_dict": {...},
    "scheduler_state_dict": {...},
    "training_state": {...},
}
```

### 6.2 旧时序 checkpoint 映射

兼容层接收裸 `state_dict` 或 `{"model_state_dict": state_dict}`；先移除可选的 `module.` 前缀，再按下表映射。

| 旧前缀 | 新前缀 |
|---|---|
| `ts_encoder.` | `temporal.encoder.` |
| `reconstruction_head.` | `temporal.reconstruction_head.` |
| `anomaly_head.` | `temporal.anomaly_head.` |

LoRA 模式中的线性层应由兼容层映射到 `.original_linear.`，而不在训练入口中维护字符串替换逻辑。

`strict=False` 只能用于装配含新增模块的多模态模型，且必须配合 `LoadReport`。对于旧时序 checkpoint，任何缺失的必需时序键、未消费的旧时序键或 shape 不匹配都必须失败，并列出旧键、新键与两侧 shape。视觉、融合、CMRG、QueryDecoder 等新模块的缺失仅在“从时序预训练初始化”流程中允许。

### 6.3 加载报告

所有加载函数返回 `LoadReport`：已加载键数、映射列表、缺失必需键、允许的随机初始化键、意外键、shape 冲突和来源 metadata。训练日志与测试都以该报告做断言，避免静默漏加载。

## 7. 训练与评估流程

```text
入口 -> 配置校验 -> UseCase -> ModelFactory / DataFactory
    -> checkpoint 服务 -> Trainer 或 Evaluator -> 结构化结果
```

`TrainUseCase` 负责协调数据、模型、checkpoint、优化器和 `Trainer`，但不实现 epoch 内部细节；`Trainer` 负责训练阶段、反向传播、梯度累积、早停与 callback 调用。`EvaluateUseCase` 组装模型和评估数据；`Evaluator` 负责前向、后处理和指标计算。

冻结策略、LoRA、CMRG、QueryDecoder 训练阶段与梯度检查点迁入 `engine/phases.py` 或模型工厂的显式策略对象，避免散落在入口脚本。

## 8. 错误处理

1. checkpoint 路径不存在、顶层格式不合法、`kind` 不匹配时立即失败，并说明期望格式与实际格式。
2. 旧时序 checkpoint 的必需参数键缺失、未消费或 shape 不一致时立即失败；不允许被 `strict=False` 隐藏。
3. 预训练初始化时，仅新加的视觉以外多模态模块可以随机初始化，并在 `LoadReport` 与日志中列出。
4. `resume` checkpoint 不能作为预训练权重使用；`temporal_pretrain` checkpoint 不能恢复 optimizer、scheduler 或 epoch。
5. 参数配置在进入模型构建前校验，包括路径、模型维度、LoRA 选择、CMRG 依赖和恢复模式。

## 9. 测试策略

1. **兼容层单元测试：** checkpoint 包装解析、`module.` 清理、前缀映射、LoRA 映射、未知键、缺失键与 shape 不匹配。
2. **真实 checkpoint 合约测试：** 加载 `full_mask_anomaly_head_pretrain_checkpoint_best.pth`；断言所有旧时序键均被消费、全部必需新键完成加载、加载后的 tensor 与 checkpoint 对应 tensor 相等。
3. **模型 smoke test：** 使用小尺寸配置验证纯时序前向、普通 VETime 前向、CMRG、QueryDecoder、LoRA、冻结策略和梯度检查点。
4. **真实 MAE 端到端 smoke test：** 从 `checkpoints/weight_v/mae_visualize_base.pth` 构建冻结视觉编码器，运行最小 VETime 前向。
5. **入口集成测试：** 以等价配置分别经 CLI 和 Hydra 启动，断言生成相同统一配置、调用相同用例且加载报告一致；对 TSB 评估入口执行最小数据夹具。
6. **回归测试：** 现有 `tests/` 必须保持通过；新测试不得依赖网络下载。

## 10. 迁移原则

采用渐进式迁移而非一次性搬迁：先建立新接口和测试，再逐段迁移现有实现，最后将旧根脚本收缩为兼容转发器。每一步应维持可运行状态，且先将真实时序 checkpoint 的兼容测试变为绿灯，才移动时序模型代码。

该设计不会修改两个既有权重文件；它们均为运行时输入，并应保持在 `.gitignore` 覆盖的 `checkpoints/` 目录中。
