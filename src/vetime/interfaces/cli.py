"""Adapter from the existing argparse namespace to the typed config contract."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict

from vetime.config import TrainingConfig
from vetime.interfaces.hydra import training_config_from_mapping


def training_config_from_namespace(namespace: Namespace) -> TrainingConfig:
    """Convert legacy CLI fields through the same mapping used by Hydra."""
    values = asdict(namespace) if hasattr(namespace, "__dataclass_fields__") else vars(namespace)
    model = {
        "model_name": values.get("model_name", "VETime"),
        "vision_name": values.get("vision_name", "mae_visualize_base.pth"),
        "ts_finetune_type": values.get("ts_finetune_type", "lora"),
        "use_vectorized_fold": values.get("use_vectorized_fold", False),
        "query_decoder_training_mode": values.get("query_decoder_training_mode", "joint"),
        "use_gradient_checkpointing": values.get("use_gradient_checkpointing", False),
        "cmrg": {
            "enabled": values.get("cmrg_enabled", False),
            "num_relation_tokens": values.get("cmrg_num_relation_tokens", 16),
            "guide_dim": values.get("cmrg_guide_dim", 512),
            "num_heads": values.get("cmrg_num_heads", 8),
            "metric_init": values.get("cmrg_metric_init", "identity"),
            "gate_init": values.get("cmrg_gate_init", 0.0),
            "injection_mode": values.get("cmrg_injection_mode", "all_layers"),
            "factorized": values.get("cmrg_factorized", True),
            "log_interval": values.get("cmrg_log_interval", 100),
        },
    }
    mapping = {
        "seed": values.get("seed", 64),
        "device": values.get("device", "auto"),
        "data": {
            "batch_size": values.get("batch_size", 32),
            "effective_batch_size": values.get("effective_batch_size", 256),
            "num_workers": values.get("num_workers", 4),
            "dynamic_batch": values.get("dynamic_batch", False),
            "max_batch_size": values.get("max_batch_size", 256),
            "padding_ratio": values.get("padding_ratio", 1.5),
            "shuffle_bucket": values.get("shuffle_bucket", False),
            "val_ratio": values.get("val_ratio", 0.1),
            "val_mode": values.get("val_mode", "split"),
            "data_setting": values.get("data_setting", {"img_size": 224, "T_sqrt": False}),
        },
        "model": model,
        "training": {
            "total_epochs": values.get("num_epochs", 10),
            "stage1_epochs": values.get("stage1_epochs", 1),
            "cls_warmup_ratio": values.get("cls_warmup_ratio", 0.5),
            "early_stopping": {"patience": values.get("early_stop_patience", 4)},
            "optimizer": {
                "lr": values.get("learning_rate", 5e-4),
                "weight_decay": values.get("weight_decay", 1e-5),
            },
        },
        "evaluation": {
            "postprocess_workers": values.get("tsb_postprocess_workers", 4),
            "cpu_threads_per_worker": values.get("tsb_worker_cpu_threads", 1),
        },
        "paths": {
            "dataset_path": values.get("dataset_path"),
            "dataset_test_dir": values.get("dataset_test_dir"),
            "file_list": values.get("file_list"),
            "output_file_path": values.get("output_file_path", "./output/result.json"),
            "vision_path": values.get("vision_path", "./checkpoints/weight_v"),
            "ts_path": values.get("ts_path"),
            "model_checkpoint": values.get("model_checkpoint"),
            "resume": values.get("resume"),
            "keep_idx_path": values.get("keep_idx_path"),
        },
    }
    return training_config_from_mapping(mapping)


def namespace_from_training_config(config: TrainingConfig) -> Namespace:
    """Flatten a typed config for the legacy training loop during migration."""
    return Namespace(
        config=None,
        seed=config.seed,
        batch_size=config.batch_size,
        effective_batch_size=config.data.effective_batch_size,
        num_workers=config.data.num_workers,
        tsb_postprocess_workers=config.tsb_postprocess_workers,
        tsb_worker_cpu_threads=config.tsb_worker_cpu_threads,
        dynamic_batch=config.data.dynamic_batch,
        max_batch_size=config.data.max_batch_size,
        padding_ratio=config.data.padding_ratio,
        shuffle_bucket=config.data.shuffle_bucket,
        val_ratio=config.data.val_ratio,
        val_mode=config.data.val_mode,
        data_setting=dict(config.data.data_setting),
        model_name=config.model.model_name,
        vision_name=config.model.vision_name,
        ts_finetune_type=config.model.ts_finetune_type,
        use_vectorized_fold=config.model.use_vectorized_fold,
        query_decoder_training_mode=config.model.query_decoder_training_mode,
        num_epochs=config.training.num_epochs,
        stage1_epochs=config.training.stage1_epochs,
        cls_warmup_ratio=config.training.cls_warmup_ratio,
        early_stop_patience=config.training.early_stop_patience,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        dataset_path=config.dataset_path,
        dataset_test_dir=config.dataset_test_dir,
        file_list=config.file_list,
        output_file_path=config.output_file_path,
        vision_path=config.paths.vision_dir,
        ts_path=config.paths.temporal,
        model_checkpoint=config.paths.model_checkpoint,
        resume=config.paths.resume,
        keep_idx_path=config.paths.keep_idx_path,
        device=config.device,
        cmrg_enabled=config.model.cmrg_enabled,
        cmrg_num_relation_tokens=config.model.cmrg_num_relation_tokens,
        cmrg_guide_dim=config.model.cmrg_guide_dim,
        cmrg_num_heads=config.model.cmrg_num_heads,
        cmrg_metric_init=config.model.cmrg_metric_init,
        cmrg_gate_init=config.model.cmrg_gate_init,
        cmrg_injection_mode=config.model.cmrg_injection_mode,
        cmrg_factorized=config.model.cmrg_factorized,
        cmrg_log_interval=config.model.cmrg_log_interval,
        use_gradient_checkpointing=config.model.use_gradient_checkpointing,
    )
