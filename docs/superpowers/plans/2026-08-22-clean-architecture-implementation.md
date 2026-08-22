# VETime Clean Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert VETime to a conventional deep-learning `src/` architecture while keeping its CLI/Hydra/TSB entry points and loading the existing temporal pretraining checkpoint exactly.

**Architecture:** Add a `vetime` package with pure model modules, application use cases, training engine, configuration adapters and checkpoint services. Refactor `VETIME` from inheritance to composition via `VETimeMultimodalModel.temporal`; decode old checkpoint keys in one strict temporal compatibility service. Keep root scripts as compatibility forwarding entry points while Hydra and CLI both call one `TrainUseCase`.

**Tech Stack:** Python 3, PyTorch >=2.3, Accelerate, Hydra/OmegaConf, pytest, MAE vision backbone, PEFT-style LoRA wrappers.

**Spec:** `docs/superpowers/specs/2026-08-22-clean-architecture-design.md`

## Global Constraints

- Preserve the existing paths `checkpoints/weight_ts/full_mask_anomaly_head_pretrain_checkpoint_best.pth` and `checkpoints/weight_v/mae_visualize_base.pth`; never rewrite either file.
- Preserve existing CLI flag names in `train.py` and `Test_TSB.py`; make Hydra the recommended configuration entry without removing the CLI path.
- Do not import Hydra, argparse, filesystem paths or Accelerate into `vetime.models`.
- A legacy temporal checkpoint may be a naked state dict or a mapping containing `model_state_dict`; support both.
- The legacy prefixes `ts_encoder.`, `reconstruction_head.` and `anomaly_head.` are mandatory. Missing, unconsumed, or shape-incompatible legacy temporal parameters must raise an error.
- `strict=False` is allowed only when building a larger VETime model from temporal pretraining and the resulting `LoadReport` has no required failures.
- Keep LoRA, freeze mode, CMRG, QueryDecoder, gradient checkpointing, resume training and TSB evaluation behaviour.
- Do not stage or commit existing user-owned files, including `tests/test_postprocess_runtime.py` and `tests/test_training_logging.py`.
- All commands below run from repository root. On PowerShell set `$env:PYTHONPATH = 'src;.'` before pytest commands.

---

## Target file map

| Path | Responsibility |
|---|---|
| `src/vetime/config.py` | Immutable training/evaluation/model configuration dataclasses and validation. |
| `src/vetime/interfaces/cli.py` | Legacy argparse namespace to `TrainingConfig` / `EvaluationConfig` conversion. |
| `src/vetime/interfaces/hydra.py` | Hydra mapping to the same configurations. |
| `src/vetime/models/temporal/` | Encoder wrapper, task heads, and standalone `TemporalModel`. |
| `src/vetime/models/vision/mae.py` | Frozen MAE loading and vision adapter. |
| `src/vetime/models/multimodal/model.py` | Composed `VETimeMultimodalModel`; no inheritance from the temporal model. |
| `src/vetime/models/factory.py` | Model assembly, LoRA/CMRG options and freeze policy application. |
| `src/vetime/infrastructure/checkpointing/` | Legacy temporal decoding, model save/load and training resume services. |
| `src/vetime/data/` | Existing datasets, collate functions and loader construction migrated unchanged in semantics. |
| `src/vetime/losses/` / `src/vetime/metrics/` | Existing loss and evaluation code under package namespaces. |
| `src/vetime/engine/` | Trainer, evaluator, training phases and callback protocols. |
| `src/vetime/application/` | Model/data orchestration for training and evaluation. |
| `scripts/` | Thin executable implementations; root scripts forward into these modules. |
| `tests/` | Unit, contract, smoke and entry integration coverage. |

## Task 1: Establish the package, configuration contract, and entry adapters

**Files:**
- Create: `src/vetime/__init__.py`
- Create: `src/vetime/config.py`
- Create: `src/vetime/interfaces/__init__.py`
- Create: `src/vetime/interfaces/cli.py`
- Create: `src/vetime/interfaces/hydra.py`
- Create: `tests/test_config_adapters.py`
- Modify: `hydra_config.py`

**Interfaces:**
- Produces `TrainingConfig`, `EvaluationConfig`, `ModelConfig`, `CheckpointPaths`, `training_config_from_namespace(namespace)` and `training_config_from_mapping(mapping)`.
- `TrainingConfig` owns primitive values and nested dataclasses only; it contains no Hydra or argparse object.

- [ ] **Step 1: Write failing adapter-equivalence tests**

```python
from argparse import Namespace
from vetime.interfaces.cli import training_config_from_namespace
from vetime.interfaces.hydra import training_config_from_mapping

def test_cli_and_hydra_create_equivalent_training_configuration():
    cli = Namespace(seed=64, batch_size=32, ts_path="checkpoints/weight_ts/x.pth",
                    vision_path="checkpoints/weight_v", vision_name="mae_visualize_base.pth")
    hydra = {"seed": 64, "data": {"batch_size": 32},
             "paths": {"ts_path": "checkpoints/weight_ts/x.pth", "vision_path": "checkpoints/weight_v"},
             "model": {"vision_name": "mae_visualize_base.pth"}}
    assert training_config_from_namespace(cli) == training_config_from_mapping(hydra)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_config_adapters.py -v`  
Expected: FAIL because `vetime` does not exist.

- [ ] **Step 3: Implement immutable configuration and adapters**

```python
@dataclass(frozen=True)
class CheckpointPaths:
    temporal: str | None
    vision_dir: str
    vision_name: str
    vetime: str | None = None
    resume: str | None = None

@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    batch_size: int
    paths: CheckpointPaths
    model: ModelConfig
    training: OptimizerConfig
    data: DataConfig

@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "VETime"
    vision_name: str = "mae_visualize_base.pth"
    ts_finetune_type: str = "lora"
    cmrg_enabled: bool = False
    use_query_decoder: bool = True

@dataclass(frozen=True)
class OptimizerConfig:
    num_epochs: int = 10
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5

@dataclass(frozen=True)
class DataConfig:
    num_workers: int = 4
    val_ratio: float = 0.1

def training_config_from_mapping(raw: Mapping[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        seed=_get(raw, "seed", 64),
        batch_size=_get(raw, "data.batch_size", 32),
        paths=CheckpointPaths(
            temporal=_get(raw, "paths.ts_path"),
            vision_dir=_get(raw, "paths.vision_path", "./checkpoints/weight_v"),
            vision_name=_get(raw, "model.vision_name", "mae_visualize_base.pth"),
            vetime=_get(raw, "paths.vetime_path"),
            resume=_get(raw, "paths.resume"),
        ),
        model=ModelConfig(model_name=_get(raw, "model.model_name", "VETime"),
                          vision_name=_get(raw, "model.vision_name", "mae_visualize_base.pth"),
                          ts_finetune_type=_get(raw, "model.ts_finetune_type", "lora"),
                          cmrg_enabled=_get(raw, "model.cmrg.enabled", False),
                          use_query_decoder=_get(raw, "model.use_query_decoder", True)),
        training=OptimizerConfig(num_epochs=_get(raw, "training.total_epochs", 10),
                                 learning_rate=_get(raw, "training.optimizer.lr", 5e-4),
                                 weight_decay=_get(raw, "training.optimizer.weight_decay", 1e-5)),
        data=DataConfig(num_workers=_get(raw, "data.num_workers", 4),
                        val_ratio=_get(raw, "data.val_ratio", 0.1)),
    )
```

Copy every existing `hydra_config.namespace_from_config` default into a named dataclass field. Make `hydra_config.namespace_from_config` call `training_config_from_mapping` and convert the result to the legacy namespace only until Task 9 removes that dependency.

- [ ] **Step 4: Add configuration validation tests and implementation**

```python
def test_config_rejects_non_positive_batch_size():
    paths = CheckpointPaths(temporal=None, vision_dir="checkpoints/weight_v", vision_name="mae_visualize_base.pth")
    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainingConfig(seed=64, batch_size=0, paths=paths, model=ModelConfig(), training=OptimizerConfig(), data=DataConfig())
```

Validate positive batch size/epochs/workers, non-empty vision name, and mutually exclusive `resume` versus temporal-pretrain initialization at construction time.

- [ ] **Step 5: Run tests and commit**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_config_adapters.py -v`  
Expected: PASS.

```bash
git add src/vetime/config.py src/vetime/interfaces hydra_config.py tests/test_config_adapters.py
git commit -m "feat: add unified training configuration adapters"
```

## Task 2: Add strict legacy temporal checkpoint decoding

**Files:**
- Create: `src/vetime/infrastructure/__init__.py`
- Create: `src/vetime/infrastructure/checkpointing/__init__.py`
- Create: `src/vetime/infrastructure/checkpointing/temporal_legacy.py`
- Create: `tests/test_temporal_legacy_checkpoint.py`

**Interfaces:**
- Produces `LoadReport`, `CheckpointCompatibilityError`, `extract_state_dict(payload)`, `map_legacy_temporal_state_dict(state_dict, *, lora=False)`, and `load_legacy_temporal_checkpoint(model, path, *, lora=False)`.
- Later tasks pass a `TemporalModel` or `VETimeMultimodalModel` target to this service.

- [ ] **Step 1: Write failing key-map tests**

```python
def test_maps_all_required_legacy_prefixes_and_removes_module_prefix():
    source = {
        "module.ts_encoder.embedding_layer.weight": torch.ones(2, 3),
        "reconstruction_head.0.bias": torch.ones(4),
        "anomaly_head.3.bias": torch.ones(2),
    }
    mapped, report = map_legacy_temporal_state_dict(source)
    assert set(mapped) == {
        "temporal.encoder.embedding_layer.weight",
        "temporal.reconstruction_head.0.bias",
        "temporal.anomaly_head.3.bias",
    }
    assert report.unconsumed_legacy_keys == ()
```

- [ ] **Step 2: Run the tests to verify failure**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_temporal_legacy_checkpoint.py -v`  
Expected: FAIL because the checkpointing module does not exist.

- [ ] **Step 3: Implement extraction, mapping, and mandatory validation**

```python
LEGACY_PREFIXES = {
    "ts_encoder.": "temporal.encoder.",
    "reconstruction_head.": "temporal.reconstruction_head.",
    "anomaly_head.": "temporal.anomaly_head.",
}

def extract_state_dict(payload: Mapping[str, Any]) -> Mapping[str, Tensor]:
    candidate = payload.get("model_state_dict", payload)
    if not isinstance(candidate, Mapping):
        raise CheckpointCompatibilityError("checkpoint does not contain a state dictionary")
    return candidate
```

For each source key, remove one leading `module.`, map exactly one required prefix, and record unknown keys. With `lora=True`, replace the terminal projection layer owner with `.original_linear.` only for `q_proj`, `k_proj`, `v_proj`, `out_proj`, `gate_proj`, `up_proj`, and `down_proj`. Keep `LoadReport` immutable and include `loaded_keys`, `mapped_pairs`, `unconsumed_legacy_keys`, `missing_required_keys`, `unexpected_target_keys`, and `shape_conflicts`.

- [ ] **Step 4: Add negative and LoRA tests**

```python
def test_loader_rejects_shape_conflict():
    with pytest.raises(CheckpointCompatibilityError, match="shape"):
        load_legacy_temporal_checkpoint(model, checkpoint_path)

def test_lora_maps_attention_weight_to_original_linear():
    mapped, _ = map_legacy_temporal_state_dict(
        {"ts_encoder.transformer_encoder.layers.0.self_attn.q_proj.weight": torch.ones(2, 2)},
        lora=True,
    )
    assert next(iter(mapped)).endswith("q_proj.original_linear.weight")
```

- [ ] **Step 5: Run tests and commit**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_temporal_legacy_checkpoint.py -v`  
Expected: PASS.

```bash
git add src/vetime/infrastructure/checkpointing tests/test_temporal_legacy_checkpoint.py
git commit -m "feat: add strict temporal checkpoint compatibility loader"
```

## Task 3: Extract standalone temporal model without changing numerical behavior

**Files:**
- Create: `src/vetime/models/__init__.py`
- Create: `src/vetime/models/temporal/__init__.py`
- Create: `src/vetime/models/temporal/config.py`
- Create: `src/vetime/models/temporal/encoder.py`
- Create: `src/vetime/models/temporal/heads.py`
- Create: `src/vetime/models/temporal/model.py`
- Create: `tests/test_temporal_model_contract.py`
- Modify: `model/TS_encoder/ts_model.py`

**Interfaces:**
- Produces `TemporalModel(config)`, with attributes `encoder`, `reconstruction_head`, `anomaly_head`; `forward(time_series, mask)` returns the legacy three-value tuple.
- Provides `legacy_state_dict()` only for transitional export when a legacy consumer needs old key names.

- [ ] **Step 1: Write failing temporal forward and state-key tests**

```python
def test_temporal_model_keeps_pretraining_forward_contract(tiny_temporal_config):
    model = TemporalModel(tiny_temporal_config)
    patch, local, full_mask = model(torch.randn(2, 8, 1), torch.ones(2, 8, dtype=torch.bool))
    assert patch.shape[0] == local.shape[0] == full_mask.shape[0] == 2

def test_temporal_model_has_single_canonical_module_path(tiny_temporal_config):
    assert all(key.startswith(("encoder.", "reconstruction_head.", "anomaly_head."))
               for key in TemporalModel(tiny_temporal_config).state_dict())
```

- [ ] **Step 2: Run failing tests**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_temporal_model_contract.py -v`  
Expected: FAIL because `TemporalModel` does not exist.

- [ ] **Step 3: Implement by relocating existing code without algorithm edits**

```python
class TemporalModel(nn.Module):
    def __init__(self, config: TemporalModelConfig) -> None:
        super().__init__()
        self.encoder = TimeSeriesEncoder(
            d_model=config.d_model, d_proj=config.d_proj, patch_size=config.patch_size,
            num_layers=config.num_layers, num_heads=config.num_heads,
            d_ff_dropout=config.d_ff_dropout, use_rope=config.use_rope,
            num_features=config.num_features, activation=config.activation,
            cmrg_injection_mode=config.cmrg_injection_mode,
        )
        self.reconstruction_head = build_reconstruction_head(config.d_proj)
        self.anomaly_head = build_anomaly_head(config.d_proj)

    def forward(self, time_series: Tensor, mask: Tensor | None = None):
        return self.encoder(time_series, mask)
```

Copy the exact loss methods and head layer ordering from `model/TS_encoder/ts_model.py`. `model/TS_encoder/ts_model.py` becomes a deprecated import alias that exposes `TS_Model = TemporalModel` through a compatibility constructor translating the old config object.

- [ ] **Step 4: Add the real temporal checkpoint contract test**

```python
@pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason="real checkpoint absent")
def test_real_pretrain_checkpoint_loads_every_required_tensor():
    model = VETimeMultimodalModel(
        temporal=TemporalModel(real_temporal_config),
        vision_encoder=FakeVisionEncoder(hidden_size=768),
        options=VETimeOptions(vision_dim=768, temporal_dim=512, cmrg_enabled=False, use_query_decoder=False),
    )
    report = load_legacy_temporal_checkpoint(model, REAL_CHECKPOINT)
    assert not report.missing_required_keys
    assert not report.shape_conflicts
    assert report.loaded_keys
```

After loading, compare every legacy tensor with `model.state_dict()[mapped_key]` using `torch.testing.assert_close(legacy_tensor, model.state_dict()[mapped_key], rtol=0, atol=0)`.

- [ ] **Step 5: Run focused tests and commit**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_temporal_model_contract.py tests/test_temporal_legacy_checkpoint.py -v`  
Expected: PASS; real-checkpoint test PASS in this repository because the confirmed file exists.

```bash
git add src/vetime/models/temporal model/TS_encoder/ts_model.py tests/test_temporal_model_contract.py
git commit -m "refactor: extract standalone temporal model"
```

## Task 4: Compose the multimodal model and preserve optional features

**Files:**
- Create: `src/vetime/models/vision/__init__.py`
- Create: `src/vetime/models/vision/mae.py`
- Create: `src/vetime/models/multimodal/__init__.py`
- Create: `src/vetime/models/multimodal/model.py`
- Create: `src/vetime/models/multimodal/cmrg.py`
- Create: `tests/test_multimodal_composition.py`
- Modify: `model/VETime.py`

**Interfaces:**
- Produces `FrozenMAEEncoder`, `VETimeMultimodalModel`, `VETimeForwardOutput`, and `CMRGOptions`.
- `VETimeMultimodalModel.temporal` is the only registered temporal submodule; no model subclasses `TemporalModel`.

- [ ] **Step 1: Write failing composition tests**

```python
def test_multimodal_model_composes_instead_of_inherits(tiny_vetime):
    assert isinstance(tiny_vetime.temporal, TemporalModel)
    assert not isinstance(tiny_vetime, TemporalModel)
    keys = tuple(tiny_vetime.state_dict())
    assert any(key.startswith("temporal.encoder.") for key in keys)
    assert not any(key.startswith("ts_encoder.") for key in keys)

def test_frozen_mae_has_no_trainable_parameters(real_mae_path):
    assert not any(p.requires_grad for p in FrozenMAEEncoder(real_mae_path).parameters())
```

- [ ] **Step 2: Run tests to verify failure**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_multimodal_composition.py -v`  
Expected: FAIL because the composed model does not exist.

- [ ] **Step 3: Implement composed model and vision adapter**

```python
class VETimeMultimodalModel(nn.Module):
    def __init__(self, temporal: TemporalModel, vision_encoder: nn.Module, options: VETimeOptions):
        super().__init__()
        self.temporal = temporal
        self.vision_encoder = vision_encoder
        self.fusion = VTS_Alignment(options.vision_dim, options.temporal_dim)
        self.mm_w = M_moe(options.temporal_dim)
        self.query_decoder = build_query_decoder(options)
        self.cmrg = build_cmrg(options)
```

Relocate the existing `VETIME.forward`, CMRG initialization and gradient-checkpointing paths exactly, changing only references from `self.ts_encoder` / duplicated heads to `self.temporal.encoder` / `self.temporal.*`. Implement `FrozenMAEEncoder` by reusing the existing MAE construction and loading behavior from `model/Vision_encoder/Vit4AD.py` and `V_encoder.py`.

- [ ] **Step 4: Add feature-combination smoke tests**

```python
@pytest.mark.parametrize("cmrg,query,checkpointing", [(False, False, False), (True, False, False), (False, True, True)])
def test_feature_combinations_preserve_forward_contract(cmrg, query, checkpointing):
    output = build_tiny_model(cmrg=cmrg, query=query, checkpointing=checkpointing)(images, series, labels=labels)
    assert output is not None
```

- [ ] **Step 5: Run tests and commit**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_multimodal_composition.py tests/test_cmrg.py -v`  
Expected: PASS.

```bash
git add src/vetime/models/vision src/vetime/models/multimodal model/VETime.py tests/test_multimodal_composition.py
git commit -m "refactor: compose VETime multimodal model"
```

## Task 5: Centralize model assembly, LoRA, CMRG and freeze policies

**Files:**
- Create: `src/vetime/models/factory.py`
- Create: `src/vetime/engine/__init__.py`
- Create: `src/vetime/engine/phases.py`
- Create: `tests/test_model_factory.py`
- Modify: `model/cmrg_training.py`

**Interfaces:**
- Produces `build_vetime_model(config) -> VETimeMultimodalModel`, `apply_temporal_finetune_policy(model, mode)`, `freeze_for_classification_warmup(model)`, and `restore_requires_grad(model, snapshot)`.

- [ ] **Step 1: Write failing policy tests**

```python
def test_freeze_policy_keeps_cmrg_trainable(model):
    apply_temporal_finetune_policy(model, "freeze")
    assert not model.temporal.encoder.embedding_layer.weight.requires_grad
    assert all(p.requires_grad for n, p in model.named_parameters() if n.startswith("cmrg"))

def test_lora_factory_loads_base_projections_before_lora_parameters(model_config):
    model = build_vetime_model(model_config.with_lora())
    assert any("original_linear" in name for name, _ in model.named_parameters())
```

- [ ] **Step 2: Run tests to verify failure**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_model_factory.py -v`  
Expected: FAIL because model factory does not exist.

- [ ] **Step 3: Implement factory and policies**

```python
def build_vetime_model(config: TrainingConfig) -> VETimeMultimodalModel:
    temporal = TemporalModel(temporal_config_from(config))
    install_lora_if_requested(temporal, config.model.ts_finetune_type)
    model = VETimeMultimodalModel(temporal, build_frozen_vision_encoder(config.paths), options_from(config))
    if config.paths.temporal:
        load_legacy_temporal_checkpoint(model, Path(config.paths.temporal), lora=config.model.ts_finetune_type == "lora")
    apply_temporal_finetune_policy(model, config.model.ts_finetune_type)
    return model
```

Move `apply_cmrg_config`, `configure_freeze_mode`, warmup freeze/restore and monitoring from `train.py` / `model/cmrg_training.py` into the factory and phase functions. Retain `model/cmrg_training.py` as import-compatible forwarding functions during migration.

- [ ] **Step 4: Run tests and commit**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_model_factory.py tests/test_cmrg.py -v`  
Expected: PASS.

```bash
git add src/vetime/models/factory.py src/vetime/engine/phases.py model/cmrg_training.py tests/test_model_factory.py
git commit -m "feat: centralize VETime model assembly policies"
```

## Task 6: Move data, loss, metrics, and logging under package boundaries

**Files:**
- Create: `src/vetime/data/__init__.py`
- Create: `src/vetime/data/datasets.py`
- Create: `src/vetime/data/collate.py`
- Create: `src/vetime/data/dataloaders.py`
- Create: `src/vetime/losses/__init__.py`
- Create: `src/vetime/losses/contrastive.py`
- Create: `src/vetime/metrics/__init__.py`
- Create: `src/vetime/metrics/tsb.py`
- Create: `src/vetime/infrastructure/logging/__init__.py`
- Create: `src/vetime/infrastructure/logging/training.py`
- Create: `tests/test_package_data_and_metrics.py`
- Modify: `loss/loss.py`, `training_logging.py`, `postprocess_runtime.py`, `evaluation/*.py`

**Interfaces:**
- Produces `build_training_loaders(config)`, `win_Contrastive_Loss`, `evaluate_tsb_predictions(scores, labels, config)`, `DeferredLossMetrics`, `log_batch_metrics`, and current postprocess-worker helpers under `vetime`.

- [ ] **Step 1: Write failing import and equivalence tests**

```python
def test_packaged_loss_matches_legacy_loss_for_fixed_inputs():
    torch.manual_seed(9)
    assert torch.allclose(NewLoss(8)(left, right), LegacyLoss(8)(left, right))

def test_packaged_postprocess_worker_cap_matches_existing_behavior():
    assert resolve_postprocess_workers(None, cpu_count=24) == 4
```

- [ ] **Step 2: Run tests to verify failure**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_package_data_and_metrics.py -v`  
Expected: FAIL because packaged modules do not exist.

- [ ] **Step 3: Relocate code with compatibility re-exports**

Move implementation bodies without formula changes. Replace legacy files with explicit imports such as:

```python
from vetime.losses.contrastive import win_Contrastive_Loss, load_balance_loss
__all__ = ["win_Contrastive_Loss", "load_balance_loss"]
```

Do the same for metrics, post-processing and logging so old imports in existing tests continue to resolve.

- [ ] **Step 4: Run existing and new focused tests**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_dataloader_collate.py tests/test_deferred_loss_metrics.py tests/test_package_data_and_metrics.py tests/test_postprocess_runtime.py tests/test_training_logging.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vetime/data src/vetime/losses src/vetime/metrics src/vetime/infrastructure/logging loss/loss.py training_logging.py postprocess_runtime.py evaluation tests/test_package_data_and_metrics.py
git commit -m "refactor: package data loss metrics and logging services"
```

## Task 7: Implement checkpoint save/load and resume services

**Files:**
- Create: `src/vetime/infrastructure/checkpointing/model_checkpoint.py`
- Create: `src/vetime/infrastructure/checkpointing/resume.py`
- Create: `tests/test_checkpoint_services.py`
- Modify: `train.py`

**Interfaces:**
- Produces `save_model_checkpoint(model, path, metadata)`, `load_model_checkpoint(model, path)`, `save_resume_checkpoint(path, model, optimizer, scheduler, state, metadata)`, and `load_resume_checkpoint(path, model, optimizer, scheduler) -> ResumeState`.

- [ ] **Step 1: Write failing checkpoint-kind tests**

```python
def test_resume_loader_rejects_temporal_pretrain_checkpoint(tmp_path):
    path = tmp_path / "temporal.pth"
    torch.save({"format_version": 2, "kind": "temporal_pretrain", "model_state_dict": {}}, path)
    with pytest.raises(CheckpointCompatibilityError, match="training_resume"):
        load_resume_checkpoint(path, model, optimizer, scheduler)

def test_new_model_checkpoint_contains_version_and_kind(tmp_path):
    path = tmp_path / "model.pth"
    save_model_checkpoint(model, path, metadata={})
    assert torch.load(path, weights_only=False)["kind"] == "vetime_model"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_checkpoint_services.py -v`  
Expected: FAIL because service functions do not exist.

- [ ] **Step 3: Implement versioned saving and explicit loaders**

```python
def save_resume_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, state: ResumeState, metadata: Mapping[str, Any]) -> None:
    torch.save({"format_version": 2, "kind": "training_resume", "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "training_state": asdict(state), "metadata": dict(metadata)}, path)
```

Load old full checkpoints as an explicitly supported legacy branch. Restore model, optimizer, scheduler and RNG independently; a failed optimizer restore returns a warning record, while model incompatibility is fatal. Replace the save/resume helper definitions in `train.py` with compatibility re-exports.

- [ ] **Step 4: Run tests and commit**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_checkpoint_services.py tests/test_cmrg.py -v`  
Expected: PASS.

```bash
git add src/vetime/infrastructure/checkpointing/model_checkpoint.py src/vetime/infrastructure/checkpointing/resume.py train.py tests/test_checkpoint_services.py
git commit -m "feat: add versioned model and resume checkpoints"
```

## Task 8: Extract trainer and evaluator without changing optimization semantics

**Files:**
- Create: `src/vetime/engine/trainer.py`
- Create: `src/vetime/engine/evaluator.py`
- Create: `src/vetime/engine/callbacks.py`
- Create: `tests/test_trainer_smoke.py`
- Create: `tests/test_evaluator_smoke.py`
- Modify: `train.py`, `Test_TSB.py`

**Interfaces:**
- Produces `Trainer.fit(max_epochs, max_train_steps) -> TrainingResult` and `Evaluator.evaluate_tsb(model, loader, config) -> EvaluationResult`.
- The engine receives model, loaders, optimizer, scheduler, Accelerator-like protocol and callbacks; it does not parse arguments or construct file paths.

- [ ] **Step 1: Write failing one-batch trainer test**

```python
def test_trainer_updates_a_trainable_parameter_on_one_batch(tiny_training_fixture):
    before = tiny_training_fixture.model.temporal.anomaly_head[0].weight.detach().clone()
    Trainer(tiny_training_fixture.dependencies).fit(max_epochs=1, max_train_steps=1)
    assert not torch.equal(before, tiny_training_fixture.model.temporal.anomaly_head[0].weight)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_trainer_smoke.py tests/test_evaluator_smoke.py -v`  
Expected: FAIL because `Trainer` and `Evaluator` do not exist.

- [ ] **Step 3: Extract the loops in behavior-preserving order**

First copy the univariate training epoch code from `train_univariate` into `Trainer`; retain gradient accumulation, warmup phase changes, loss components, early stop, Accelerate synchronization and TensorBoard callbacks. Then move `evaluate_univariate` to `Evaluator`. Finally move `TSB_test`, parallel post-processing and CSV result creation from `Test_TSB.py` into the evaluator. Every moved function accepts typed dependencies rather than global arguments.

```python
class Trainer:
    def fit(self, *, max_epochs: int | None = None, max_train_steps: int | None = None) -> TrainingResult:
        for epoch in range(self.start_epoch, resolved_epochs):
            self._run_training_epoch(epoch, max_train_steps)
            validation = self.evaluator.evaluate_validation(self.model, self.validation_loader, epoch)
            self.callbacks.on_epoch_end(epoch=epoch, validation=validation, global_step=self.global_step)
        return TrainingResult(last_epoch=epoch, global_step=self.global_step, best_validation_loss=self.best_validation_loss)
```

- [ ] **Step 4: Add regression fixture comparing old and new one-step losses**

```python
def test_trainer_one_step_loss_matches_legacy_univariate_path(fixed_batch):
    assert new_one_step_loss == pytest.approx(legacy_one_step_loss, rel=1e-6, abs=1e-7)
```

- [ ] **Step 5: Run tests and commit**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_trainer_smoke.py tests/test_evaluator_smoke.py tests/test_deferred_loss_metrics.py -v`  
Expected: PASS.

```bash
git add src/vetime/engine train.py Test_TSB.py tests/test_trainer_smoke.py tests/test_evaluator_smoke.py
git commit -m "refactor: extract training and evaluation engines"
```

## Task 9: Add application use cases and make Hydra/CLI share them

**Files:**
- Create: `src/vetime/application/__init__.py`
- Create: `src/vetime/application/build_model.py`
- Create: `src/vetime/application/train.py`
- Create: `src/vetime/application/evaluate.py`
- Create: `tests/test_application_entry_equivalence.py`
- Modify: `train_hydra.py`, `hydra_config.py`, `train.py`, `Test_TSB.py`

**Interfaces:**
- Produces `TrainUseCase.run(config: TrainingConfig) -> TrainingResult` and `EvaluateUseCase.run(config: EvaluationConfig) -> EvaluationResult`.
- Both legacy entry adapters call these exact functions.

- [ ] **Step 1: Write failing common-use-case test**

```python
def test_cli_and_hydra_call_the_same_train_use_case(monkeypatch, minimal_config):
    calls = []
    monkeypatch.setattr("vetime.application.train.TrainUseCase.run", lambda self, cfg: calls.append(cfg))
    cli_main(namespace_from(minimal_config))
    hydra_run(mapping_from(minimal_config))
    assert calls[0] == calls[1]
```

- [ ] **Step 2: Run test to verify failure**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_application_entry_equivalence.py -v`  
Expected: FAIL because entry points still invoke legacy training directly.

- [ ] **Step 3: Implement use cases and update entries**

```python
class TrainUseCase:
    def run(self, config: TrainingConfig) -> TrainingResult:
        model, report = build_training_model(config)
        loaders = build_training_loaders(config)
        trainer = build_trainer(config, model, loaders, report)
        return trainer.fit()
```

`train_hydra.py` converts `DictConfig` with `training_config_from_mapping` and calls `TrainUseCase`; `train.py` converts parsed arguments with `training_config_from_namespace` and calls the same object. Maintain the previous JSON result location and contents in the Hydra script. `Test_TSB.py` calls `EvaluateUseCase` after its compatibility adapter.

- [ ] **Step 4: Run tests and smoke both interfaces**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_application_entry_equivalence.py tests/test_config_adapters.py -v`  
Expected: PASS.

Run: `python train_hydra.py --cfg job`  
Expected: Hydra prints the composed configuration and exits without training.

Run: `python train.py --help`  
Expected: Existing training flags, including `--ts_path`, remain listed.

- [ ] **Step 5: Commit**

```bash
git add src/vetime/application train.py train_hydra.py Test_TSB.py hydra_config.py tests/test_application_entry_equivalence.py
git commit -m "refactor: route CLI and Hydra through training use case"
```

## Task 10: Convert root scripts to compatibility forwarders and complete end-to-end verification

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/train_cli.py`
- Create: `scripts/train_hydra.py`
- Create: `scripts/evaluate_tsb.py`
- Create: `tests/test_real_checkpoint_end_to_end.py`
- Modify: `train.py`, `train_hydra.py`, `Test_TSB.py`, `README.md`

**Interfaces:**
- Root scripts expose the same executable behavior but import a `main` or `run` function from `scripts/`.
- `scripts/train_hydra.py` is documented as the recommended entry.

- [ ] **Step 1: Write failing forwarding and real-weight smoke tests**

```python
def test_root_cli_forwards_to_script_main(monkeypatch):
    called = []
    monkeypatch.setattr("scripts.train_cli.main", lambda: called.append(True))
    runpy.run_path("train.py", run_name="__main__")
    assert called == [True]

@pytest.mark.skipif(not TEMPORAL.exists() or not MAE.exists(), reason="local checkpoints absent")
def test_real_temporal_and_mae_weights_build_a_vetime_forward_model():
    model, report = build_training_model(real_config)
    assert report.loaded_keys
    assert not report.missing_required_keys
```

- [ ] **Step 2: Run tests to verify failure**

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_real_checkpoint_end_to_end.py -v`  
Expected: FAIL because root forwarding has not been installed.

- [ ] **Step 3: Implement forwarders and user documentation**

```python
# train.py
from scripts.train_cli import main

if __name__ == "__main__":
    main()
```

Move actual top-level logic to `scripts/`, retain no model/training/checkpoint implementation in root scripts, and update README examples to recommend `python train_hydra.py` while retaining the CLI example. Document both checkpoint locations and the strict temporal compatibility guarantee.

- [ ] **Step 4: Run complete verification**

Run: `$env:PYTHONPATH='src;.'; python -m pytest -q`  
Expected: PASS, with only tests explicitly skipped for unavailable optional external datasets.

Run: `$env:PYTHONPATH='src;.'; python -m pytest tests/test_real_checkpoint_end_to_end.py -v`  
Expected: PASS because both confirmed local checkpoint files exist.

Run: `python train_hydra.py --cfg job`  
Expected: PASS and prints the Hydra configuration.

Run: `python train.py --help`  
Expected: PASS and includes legacy arguments.

- [ ] **Step 5: Review the change set and commit**

Run: `git diff --check HEAD~1..HEAD`  
Expected: no whitespace errors.

```bash
git add scripts train.py train_hydra.py Test_TSB.py README.md tests/test_real_checkpoint_end_to_end.py
git commit -m "refactor: finalize clean architecture entry points"
```

## Final acceptance checklist

- [ ] Both real files remain unmodified in `checkpoints/weight_ts/` and `checkpoints/weight_v/`.
- [ ] The legacy temporal checkpoint loads all required temporal tensors with zero shape conflicts.
- [ ] `VETimeMultimodalModel` composes, rather than inherits from, `TemporalModel`.
- [ ] CLI and Hydra construct equivalent `TrainingConfig` objects and invoke `TrainUseCase`.
- [ ] `train.py --help`, `train_hydra.py --cfg job`, and the TSB evaluation compatibility entry execute.
- [ ] LoRA, freeze, CMRG, QueryDecoder, gradient checkpointing and resume tests pass.
- [ ] `python -m pytest -q` passes in the configured PyTorch environment.
- [ ] Each task is committed independently, and user-owned untracked tests are never staged.
