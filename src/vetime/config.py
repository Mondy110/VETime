"""Framework-independent configuration contracts for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class CheckpointPaths:
    temporal: str | None
    vision_dir: str
    vision_name: str
    vetime: str | None = None
    resume: str | None = None
    pretrain_from: str | None = None
    keep_idx_path: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "VETime"
    vision_name: str = "mae_visualize_base.pth"
    ts_finetune_type: str = "lora"
    use_vectorized_fold: bool = False
    query_decoder_training_mode: str = "joint"
    use_query_decoder: bool = True
    use_gradient_checkpointing: bool = False
    cmrg_enabled: bool = False
    cmrg_num_relation_tokens: int = 16
    cmrg_guide_dim: int = 512
    cmrg_num_heads: int = 8
    cmrg_metric_init: str = "identity"
    cmrg_gate_init: float = 0.0
    cmrg_injection_mode: str = "all_layers"
    cmrg_factorized: bool = True
    cmrg_log_interval: int = 100


@dataclass(frozen=True)
class OptimizerConfig:
    num_epochs: int = 10
    stage1_epochs: int = 1
    cls_warmup_ratio: float = 0.5
    early_stop_patience: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5


@dataclass(frozen=True)
class DataConfig:
    num_workers: int = 4
    effective_batch_size: int = 256
    dynamic_batch: bool = False
    max_batch_size: int = 256
    padding_ratio: float = 1.5
    shuffle_bucket: bool = False
    val_ratio: float = 0.1
    val_mode: str = "split"
    data_setting: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_setting", _frozen_mapping(self.data_setting))
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.effective_batch_size <= 0 or self.max_batch_size <= 0:
            raise ValueError("batch sizes must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    paths: CheckpointPaths
    model: ModelConfig = field(default_factory=ModelConfig)
    data_setting: Mapping[str, Any] = field(default_factory=dict)
    postprocess_workers: int = 4
    cpu_threads_per_worker: int = 1
    device: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_setting", _frozen_mapping(self.data_setting))
        if self.postprocess_workers <= 0 or self.cpu_threads_per_worker <= 0:
            raise ValueError("postprocess workers and CPU threads must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    batch_size: int
    paths: CheckpointPaths
    model: ModelConfig = field(default_factory=ModelConfig)
    training: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "auto"
    dataset_path: str | None = None
    dataset_test_dir: str | None = None
    file_list: str | None = None
    output_file_path: str = "./output/result.json"
    tsb_postprocess_workers: int = 4
    tsb_worker_cpu_threads: int = 1

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.training.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.tsb_postprocess_workers <= 0 or self.tsb_worker_cpu_threads <= 0:
            raise ValueError("TSB worker settings must be positive")
        if self.paths.resume and self.paths.temporal:
            raise ValueError("resume and temporal pretraining paths are mutually exclusive")
