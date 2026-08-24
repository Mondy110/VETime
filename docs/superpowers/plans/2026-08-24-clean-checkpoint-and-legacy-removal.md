# VETime 新 Checkpoint 与遗留实现移除 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以版本化组合模型 checkpoint 取代旧完整 `VETIME` checkpoint，并迁移仍在旧目录的运行时实现到 `src/vetime`。

**Architecture:** 模型只能从时序预训练、`vetime_model` 或 `training_resume` 三种互斥来源初始化。训练和 TSB 评估均通过模型工厂创建 `VETimeMultimodalModel`；完成迁移后，`src/vetime` 不再依赖旧目录。

**Tech Stack:** Python、PyTorch、Accelerate、Hydra、pytest。

**Spec:** `docs/superpowers/specs/2026-08-24-clean-checkpoint-and-legacy-removal-design.md`

## Global Constraints

- 保留旧时序预训练权重和 MAE 权重兼容。
- 删除 `--vetime_path`、`paths.vetime_path` 和旧完整 `VETIME` checkpoint 转换逻辑。
- 新 checkpoint 固定为 `format_version=3`。
- `paths.ts_path`、`paths.model_checkpoint`、`paths.resume` 互斥。
- 不提交 `tests/test_postprocess_runtime.py`、`tests/test_training_logging.py`。

---

### Task 1: 建立 v3 checkpoint 和初始化来源契约

**Files:**

- Modify: `src/vetime/config.py`
- Modify: `src/vetime/interfaces/cli.py`
- Modify: `src/vetime/interfaces/hydra.py`
- Modify: `src/vetime/infrastructure/checkpointing/model_checkpoint.py`
- Modify: `src/vetime/infrastructure/checkpointing/resume.py`
- Test: `tests/test_config_adapters.py`
- Test: `tests/test_checkpoint_services.py`

**Interfaces:** `CheckpointPaths.model_checkpoint`；严格的 `save_model_checkpoint`、`load_model_checkpoint`、`save_resume_checkpoint`、`load_resume_checkpoint`。

- [ ] **Step 1: 写失败测试**

~~~python
def test_config_rejects_multiple_initialization_sources():
    paths = CheckpointPaths("ts.pth", "weights", "mae.pth", model_checkpoint="model.pth")
    with pytest.raises(ValueError, match="exactly one initialization source"):
        TrainingConfig(seed=1, batch_size=1, paths=paths)

def test_model_loader_rejects_legacy_raw_state_dict(tmp_path):
    path = tmp_path / "legacy.pth"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(CheckpointCompatibilityError, match="format_version"):
        load_model_checkpoint(TinyModel(), path)
~~~

- [ ] **Step 2: 运行 RED 测试**

Run: `python -m pytest -q tests/test_config_adapters.py tests/test_checkpoint_services.py`  
Expected: FAIL，因为 `model_checkpoint` 和严格 v3 校验不存在。

- [ ] **Step 3: 实现最小契约**

~~~python
@dataclass(frozen=True)
class CheckpointPaths:
    temporal: str | None
    vision_dir: str
    vision_name: str
    model_checkpoint: str | None = None
    resume: str | None = None

def _require_payload(payload, kind):
    if payload.get("format_version") != 3:
        raise CheckpointCompatibilityError("expected checkpoint format_version=3")
    if payload.get("kind") != kind:
        raise CheckpointCompatibilityError(f"expected {kind}, got {payload.get('kind')!r}")
~~~

删除 `vetime`、`pretrain_from` adapter 字段；使模型加载仅接受 `kind="vetime_model"` 且 `strict=True`，resume 仅接受 `kind="training_resume"`。

- [ ] **Step 4: GREEN 与提交**

Run: `python -m pytest -q tests/test_config_adapters.py tests/test_checkpoint_services.py`  
Expected: PASS.

~~~bash
git add src/vetime/config.py src/vetime/interfaces src/vetime/infrastructure/checkpointing tests/test_config_adapters.py tests/test_checkpoint_services.py
git commit -m "feat: enforce v3 checkpoint protocol"
~~~

### Task 2: 训练接入新保存、加载和运行拓扑报告

**Files:**

- Create: `src/vetime/infrastructure/logging/topology.py`
- Modify: `src/vetime/infrastructure/logging/__init__.py`
- Modify: `src/vetime/models/factory.py`
- Modify: `src/vetime/application/train.py`
- Modify: `train.py`
- Test: `tests/test_topology_logging.py`
- Test: `tests/test_model_factory.py`

**Interfaces:** `format_runtime_topology(model, *, device, initialization_source) -> str`；模型工厂只返回组合模型。

- [ ] **Step 1: 写失败测试**

~~~python
def test_topology_declares_clean_model_and_enabled_modules():
    text = format_runtime_topology(model, device=torch.device("cuda:0"), initialization_source="temporal")
    assert "VETimeMultimodalModel (clean composition)" in text
    assert "CMRG: enabled" in text
    assert "QueryDecoder: enabled" in text
~~~

- [ ] **Step 2: 运行 RED 测试**

Run: `python -m pytest -q tests/test_topology_logging.py`  
Expected: FAIL，因为 formatter 不存在。

- [ ] **Step 3: 实现装配与保存路径**

~~~python
def format_runtime_topology(model, *, device, initialization_source):
    return "\n".join((
        "[INFO] ===== VETime Runtime Topology =====",
        f"[INFO] Device: {device}",
        "[INFO] Architecture: VETimeMultimodalModel (clean composition)",
    ))
~~~

输出真实 CUDA 名称、MAE 冻结状态、时序来源、LoRA、CMRG、QueryDecoder 和训练参数分组。用 `accelerator.device` 代替按进程数推断设备。最佳模型调用 `save_model_checkpoint`，周期保存调用 `save_resume_checkpoint`，恢复调用 `load_resume_checkpoint`。移除 `VETIME` 分支。

- [ ] **Step 4: 写模型 checkpoint round-trip 测试**

~~~python
def test_factory_loads_clean_model_checkpoint(tmp_path):
    source = build_vetime_model(config, temporal_config=tiny_config, vision_encoder=TinyVision())
    path = tmp_path / "model.pth"
    save_model_checkpoint(source, path, {"architecture": "vetime-clean"})
    target = build_vetime_model(config_with_model_checkpoint(path), temporal_config=tiny_config, vision_encoder=TinyVision())
    assert_state_dict_equal(source, target)
~~~

- [ ] **Step 5: GREEN 与提交**

Run: `python -m pytest -q tests/test_topology_logging.py tests/test_model_factory.py tests/test_checkpoint_services.py`  
Expected: PASS.

~~~bash
git add src/vetime/infrastructure/logging src/vetime/models/factory.py src/vetime/application/train.py train.py tests/test_topology_logging.py tests/test_model_factory.py
git commit -m "refactor: use versioned checkpoints in training"
~~~

### Task 3: 新 TSB 入口并删除旧完整模型兼容

**Files:**

- Create: `src/vetime/application/evaluate_tsb.py`
- Create: `scripts/evaluate_tsb.py`
- Modify: `Test_TSB.py`
- Modify: `train.py`
- Modify: `configs/univariate.yaml`
- Modify: `src/vetime/interfaces/cli.py`
- Test: `tests/test_tsb_clean_entry.py`
- Delete: `model/VETime.py`
- Delete: `model/cmrg_training.py`

**Interfaces:** `evaluate_tsb(config: EvaluationConfig) -> float`; CLI/Hydra 使用 `model_checkpoint`，不再有 `vetime_path`。

- [ ] **Step 1: 写失败入口测试**

~~~python
def test_clean_tsb_entry_builds_factory_model(monkeypatch):
    calls = []
    monkeypatch.setattr("vetime.application.evaluate_tsb.build_vetime_model", lambda config: calls.append(config) or TinyModel())
    evaluate_tsb(evaluation_config)
    assert calls

def test_train_parser_has_model_checkpoint_not_vetime_path(parser):
    options = {action.dest for action in parser._actions}
    assert "model_checkpoint" in options
    assert "vetime_path" not in options
~~~

- [ ] **Step 2: 运行 RED 测试**

Run: `python -m pytest -q tests/test_tsb_clean_entry.py`  
Expected: FAIL，因为新 TSB 应用入口不存在。

- [ ] **Step 3: 实现新评估入口**

~~~python
def evaluate_tsb(config: EvaluationConfig) -> float:
    model = build_vetime_model(training_config_for_evaluation(config))
    load_model_checkpoint(model, config.paths.model_checkpoint)
    return run_tsb_benchmark(model, config)
~~~

将 `Test_TSB.py` 收缩为脚本转发器。删除所有生产代码中的 `model.VETime`、`model.cmrg_training` 和 `vetime_path` 引用，再删除两个源文件。

- [ ] **Step 4: GREEN 与提交**

Run: `python -m pytest -q tests/test_tsb_clean_entry.py tests/test_application_entry_equivalence.py tests/test_cmrg.py`  
Expected: PASS.

~~~bash
git add src/vetime/application scripts Test_TSB.py train.py configs src/vetime/interfaces tests
git rm model/VETime.py model/cmrg_training.py
git commit -m "refactor: remove legacy full VETime checkpoint path"
~~~

### Task 4: 迁移剩余运行时实现并删除旧目录

**Files:**

- Create: `src/vetime/models/temporal/encoding_utils.py`
- Create: `src/vetime/models/vision/{encoder.py,vit4ad.py,models_mae.py}`
- Create: `src/vetime/models/multimodal/{cmrg.py,modules.py}`
- Create: `src/vetime/data/{dataloader_impl.py,pre_image.py}`
- Create: `src/vetime/losses/core.py`
- Create: `src/vetime/metrics/{core.py,basic_metrics.py,sliding_windows.py,affiliation/}`
- Modify: all imports under `src/vetime`, root entries and tests
- Delete: legacy Python source under `model/`, `loss/`, `evaluation/`, and only `dataset/*.py`
- Test: `tests/test_import_boundaries.py`

**Interfaces:** `src/vetime` 不含 `from model.`、`from dataset.`、`from loss.`、`from evaluation.` import。

- [ ] **Step 1: 写失败边界测试**

~~~python
def test_vetime_package_has_no_legacy_runtime_imports():
    forbidden = ("from model.", "from dataset.", "from loss.", "from evaluation.")
    offenders = [p for p in Path("src/vetime").rglob("*.py") if any(x in p.read_text() for x in forbidden)]
    assert offenders == []
~~~

- [ ] **Step 2: 运行 RED 测试**

Run: `python -m pytest -q tests/test_import_boundaries.py`  
Expected: FAIL，并列出旧目录依赖。

- [ ] **Step 3: 使用 `git mv` 迁移实现**

按顺序迁移时序编码器、视觉 MAE、CMRG/VTS、数据预处理、损失、指标，并更新 import。公开函数必须保持 tensor shape、返回 tuple 和数值语义。不要移动 `dataset/TSB-AD/`、CSV、PKL 或 checkpoint 文件。

- [ ] **Step 4: 更新测试与删除旧源码**

先运行：

~~~bash
rg -n "from (model|dataset|loss|evaluation)\\.|import (model|dataset|loss|evaluation)" src train.py train_hydra.py Test_TSB.py scripts tests
~~~

结果不再显示源码引用后才删除旧 Python 源文件；保留数据资产目录和 `dataset/...` 配置路径。

- [ ] **Step 5: GREEN 与提交**

Run: `python -m pytest -q`  
Expected: PASS.

~~~bash
git add src tests train.py train_hydra.py Test_TSB.py scripts configs
git rm -r model loss evaluation
git rm dataset/__init__.py dataset/dataloader.py dataset/pre_image.py
git commit -m "refactor: remove legacy runtime implementation directories"
~~~

### Task 5: 文档化服务器运行与最终验收

**Files:**

- Modify: `README.md`
- Modify: `scripts/validate_checkpoints.py`
- Create: `tests/test_checkpoint_cli_contract.py`

- [ ] **Step 1: 写失败 CLI 契约测试**

~~~python
def test_train_help_advertises_model_checkpoint_not_vetime_path():
    result = subprocess.run([sys.executable, "train.py", "--help"], text=True, capture_output=True, check=True)
    assert "--model_checkpoint" in result.stdout
    assert "--vetime_path" not in result.stdout
~~~

- [ ] **Step 2: 运行 RED 测试**

Run: `python -m pytest -q tests/test_checkpoint_cli_contract.py`  
Expected: FAIL，因为旧参数仍存在。

- [ ] **Step 3: 文档化服务器命令**

~~~bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 python scripts/validate_checkpoints.py --full-model
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 python train_hydra.py paths.ts_path=checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 python train_hydra.py paths.model_checkpoint=output/models/best.pth
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 python train_hydra.py paths.resume=output/resume/latest.pth
~~~

说明 `model_checkpoint` 只加载模型；`resume` 同时恢复优化器、调度器、epoch 和随机状态。

- [ ] **Step 4: 最终验证与提交**

Run: `python -m pytest -q`  
Run: `python scripts/validate_checkpoints.py --full-model`  
Run: `python train_hydra.py --cfg job`  
Expected: 测试通过、真实权重加载成功、Hydra 只显示新 checkpoint 字段。

~~~bash
git add README.md scripts/validate_checkpoints.py tests/test_checkpoint_cli_contract.py
git commit -m "docs: document clean checkpoint server workflow"
~~~
