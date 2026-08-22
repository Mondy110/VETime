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
            "vetime_path": values.get("vetime_path"),
            "resume": values.get("resume"),
            "pretrain_from": values.get("pretrain_from"),
            "keep_idx_path": values.get("keep_idx_path"),
        },
    }
    return training_config_from_mapping(mapping)
