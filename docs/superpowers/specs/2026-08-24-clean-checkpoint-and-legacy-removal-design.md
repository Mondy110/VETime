# VETime 新 Checkpoint 与遗留实现移除设计

**日期：** 2026-08-24  
**状态：** 已确认，待实施  
**范围：** 保留时序预训练权重和 MAE 权重兼容；停止支持旧完整 `VETIME` checkpoint；将训练、恢复和 TSB 评估切换到新组合模型与版本化 checkpoint，并逐步移除旧实现目录。

## 1. 已确认决策

1. 必须继续加载 `checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth`。
2. 必须继续加载 `checkpoints/weight_v/mae_visualize_base.pth`，并保持视觉编码器冻结。
3. 不再支持旧完整 `VETIME` checkpoint；删除 `--vetime_path` 和 `paths.vetime_path`。
4. 新训练产生的模型必须可保存、用于推理/评估加载，并可完整恢复训练状态。
5. 重构后的 `src/vetime` 不能继续依赖 `model/`、`dataset/`、`loss/`、`evaluation/` 下的实现。

## 2. 新 Checkpoint 协议

### 2.1 模型 checkpoint

模型 checkpoint 用于评估、推理或继续微调，只保存组合模型参数：

```python
{
    "format_version": 3,
    "kind": "vetime_model",
    "model_state_dict": {...},
    "metadata": {
        "architecture": "vetime-clean",
        "model_config": {...},
        "temporal_pretrain": "..." | None,
        "vision_checkpoint": "...",
        "epoch": int,
        "best_val_loss": float | None,
    },
}
```

`load_model_checkpoint` 必须接受仅 `kind == "vetime_model"` 的版本化 payload，并严格加载新组合模型 state dict。旧裸 state dict 或旧 `VETIME` namespace 必须给出明确错误，不进行静默转换。

### 2.2 恢复 checkpoint

恢复 checkpoint 用于精确续训：

```python
{
    "format_version": 3,
    "kind": "training_resume",
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "scheduler_state_dict": {...},
    "training_state": {
        "epoch": int,
        "global_step": int,
        "best_val_loss": float | None,
        "patience_counter": int,
    },
    "metadata": {
        "architecture": "vetime-clean",
        "model_config": {...},
    },
    "random_state": {...},
}
```

`resume` 与 `ts_path`、`model_checkpoint` 互斥。resume 加载必须严格校验版本和 `kind`，然后恢复模型、优化器、scheduler、随机状态和训练游标。

### 2.3 初始化来源

模型只能从下列三种来源之一创建：

| 来源 | 配置字段 | 行为 |
|---|---|---|
| 时序预训练 | `paths.ts_path` | 严格映射旧 `ts_encoder.*`、两个 head 以及 LoRA 原始线性层。 |
| 新模型 checkpoint | `paths.model_checkpoint` | 严格加载 `vetime_model`。 |
| 训练恢复 | `paths.resume` | 严格加载 `training_resume` 和训练状态。 |

MAE 始终由 `paths.vision_path` 和 `model.vision_name` 加载，不作为初始化来源选择。

## 3. 训练与评估路径

训练入口先将 CLI/Hydra 输入转成 `TrainingConfig`，再由模型工厂构建 `VETimeMultimodalModel`。训练结束保存一个 `vetime_model` 最佳权重；周期性和最终恢复点保存 `training_resume`。根目录 CLI 与 Hydra 不允许再触发旧 `VETIME` 类。

TSB 评估使用同一模型工厂、`EvaluationConfig` 与 `paths.model_checkpoint`。其模型构建和加载不能导入 `model.VETime`、`TS_Model` 或旧 `V_model`。

## 4. 遗留代码迁移与删除

真实实现按职责迁入 `src/vetime`：

```text
model/TS_encoder/*       -> src/vetime/models/temporal/
model/Vision_encoder/*   -> src/vetime/models/vision/
model/CMRG.py            -> src/vetime/models/multimodal/cmrg.py
model/VTS_module.py      -> src/vetime/models/multimodal/modules.py
dataset/*                -> src/vetime/data/
loss/loss.py             -> src/vetime/losses/
evaluation/*             -> src/vetime/metrics/
```

迁移后的 `src/vetime` 仅能导入同包模块和第三方依赖。`model/VETime.py`、`model/cmrg_training.py`、`model/`、`dataset/`、`loss/`、`evaluation/` 在所有入口、测试和 `src/vetime` 脱离后删除。根目录 `train.py`、`train_hydra.py`、`Test_TSB.py` 改为薄转发入口；其参数兼容仅覆盖新 checkpoint 协议。

## 5. 可观察性与验收

每次训练启动打印运行拓扑：真实 CUDA 设备、外层架构、时序初始化来源、视觉冻结状态、CMRG/QueryDecoder 状态和各可训练模块参数量。

完成条件：

1. `python scripts/validate_checkpoints.py --full-model` 成功加载时序 111 个参数和 MAE。
2. 新模型 checkpoint 和 resume checkpoint 均通过保存/恢复 round-trip 测试。
3. CLI、Hydra、TSB 评估均从新工厂构建组合模型。
4. `rg "from (model|dataset|loss|evaluation)" src/vetime` 无结果。
5. 旧完整 checkpoint、`--vetime_path` 和 `paths.vetime_path` 均不存在于生产入口。
6. 完整 pytest 通过；服务器完成一个 epoch 的 Hydra smoke training。
