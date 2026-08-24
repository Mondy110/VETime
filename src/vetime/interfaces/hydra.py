"""Hydra/OmegaConf mapping adapter with no dependency on the training engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vetime.config import (
    CheckpointPaths,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


def _get(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def training_config_from_mapping(config: Mapping[str, Any]) -> TrainingConfig:
    paths = CheckpointPaths(
        temporal=_get(config, "paths.ts_path"),
        vision_dir=_get(config, "paths.vision_path", "./checkpoints/weight_v"),
        vision_name=_get(config, "model.vision_name", "mae_visualize_base.pth"),
        model_checkpoint=_get(config, "paths.model_checkpoint"),
        resume=_get(config, "paths.resume"),
        keep_idx_path=_get(config, "paths.keep_idx_path"),
    )
    model = ModelConfig(
        model_name=_get(config, "model.model_name", "VETime"),
        vision_name=_get(config, "model.vision_name", "mae_visualize_base.pth"),
        ts_finetune_type=_get(config, "model.ts_finetune_type", "lora"),
        use_vectorized_fold=_get(config, "model.use_vectorized_fold", False),
        query_decoder_training_mode=_get(config, "model.query_decoder_training_mode", "joint"),
        use_query_decoder=_get(config, "model.use_query_decoder", True),
        use_gradient_checkpointing=_get(config, "model.use_gradient_checkpointing", False),
        cmrg_enabled=_get(config, "model.cmrg.enabled", False),
        cmrg_num_relation_tokens=_get(config, "model.cmrg.num_relation_tokens", 16),
        cmrg_guide_dim=_get(config, "model.cmrg.guide_dim", 512),
        cmrg_num_heads=_get(config, "model.cmrg.num_heads", 8),
        cmrg_metric_init=_get(config, "model.cmrg.metric_init", "identity"),
        cmrg_gate_init=_get(config, "model.cmrg.gate_init", 0.0),
        cmrg_injection_mode=_get(config, "model.cmrg.injection_mode", "all_layers"),
        cmrg_factorized=_get(config, "model.cmrg.factorized", True),
        cmrg_log_interval=_get(config, "model.cmrg.log_interval", 100),
    )
    training = OptimizerConfig(
        num_epochs=_get(config, "training.total_epochs", 10),
        stage1_epochs=_get(config, "training.stage1_epochs", 1),
        cls_warmup_ratio=_get(config, "training.cls_warmup_ratio", 0.5),
        early_stop_patience=_get(config, "training.early_stopping.patience", 4),
        learning_rate=_get(config, "training.optimizer.lr", 5e-4),
        weight_decay=_get(config, "training.optimizer.weight_decay", 1e-5),
    )
    data = DataConfig(
        num_workers=_get(config, "data.num_workers", 4),
        effective_batch_size=_get(config, "data.effective_batch_size", 256),
        dynamic_batch=_get(config, "data.dynamic_batch", False),
        max_batch_size=_get(config, "data.max_batch_size", 256),
        padding_ratio=_get(config, "data.padding_ratio", 1.5),
        shuffle_bucket=_get(config, "data.shuffle_bucket", False),
        val_ratio=_get(config, "data.val_ratio", 0.1),
        val_mode=_get(config, "data.val_mode", "split"),
        data_setting=_get(config, "data.data_setting", {"img_size": 224, "T_sqrt": False}),
    )
    return TrainingConfig(
        seed=_get(config, "seed", 64),
        batch_size=_get(config, "data.batch_size", 32),
        paths=paths,
        model=model,
        training=training,
        data=data,
        device=_get(config, "device", "auto"),
        dataset_path=_get(config, "paths.dataset_path"),
        dataset_test_dir=_get(config, "paths.dataset_test_dir"),
        file_list=_get(config, "paths.file_list"),
        output_file_path=_get(config, "paths.output_file_path", "./output/result.json"),
        tsb_postprocess_workers=_get(config, "evaluation.postprocess_workers", 4),
        tsb_worker_cpu_threads=_get(config, "evaluation.cpu_threads_per_worker", 1),
    )
