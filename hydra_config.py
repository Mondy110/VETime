"""Adapt layered experiment configuration to the legacy training entry point.

The experiment branch intentionally retains its established ``train.py``
training loop.  This module provides a narrow boundary between a Hydra
``DictConfig`` (or a regular nested mapping in tests) and that loop's flat
``argparse.Namespace`` interface.
"""

from argparse import Namespace
from collections.abc import Mapping
from typing import Any

from vetime.interfaces.hydra import training_config_from_mapping


def _get(config: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Return a dotted config value without treating false-y values as absent."""
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def namespace_from_config(config: Mapping[str, Any]) -> Namespace:
    """Create the legacy trainer arguments from the layered experiment config."""
    return Namespace(
        config=None,
        seed=_get(config, "seed", 64),
        batch_size=_get(config, "data.batch_size", 32),
        effective_batch_size=_get(config, "data.effective_batch_size", 256),
        num_workers=_get(config, "data.num_workers", 5),
        tsb_postprocess_workers=_get(config, "evaluation.postprocess_workers", 4),
        tsb_worker_cpu_threads=_get(config, "evaluation.cpu_threads_per_worker", 1),
        dynamic_batch=_get(config, "data.dynamic_batch", False),
        max_batch_size=_get(config, "data.max_batch_size", 256),
        padding_ratio=_get(config, "data.padding_ratio", 1.5),
        shuffle_bucket=_get(config, "data.shuffle_bucket", False),
        val_ratio=_get(config, "data.val_ratio", 0.1),
        val_mode=_get(config, "data.val_mode", "tsb"),
        data_setting=_get(config, "data.data_setting", {"img_size": 224, "T_sqrt": False}),
        model_name=_get(config, "model.model_name", "VETime"),
        vision_name=_get(config, "model.vision_name", "mae_visualize_base.pth"),
        ts_finetune_type=_get(config, "model.ts_finetune_type", "lora"),
        use_vectorized_fold=_get(config, "model.use_vectorized_fold", False),
        query_decoder_training_mode=_get(config, "model.query_decoder_training_mode", "joint"),
        num_epochs=_get(config, "training.total_epochs", 25),
        stage1_epochs=_get(config, "training.stage1_epochs", 1),
        cls_warmup_ratio=_get(config, "training.cls_warmup_ratio", 0.5),
        early_stop_patience=_get(config, "training.early_stopping.patience", 4),
        learning_rate=_get(config, "training.optimizer.lr", 5e-4),
        weight_decay=_get(config, "training.optimizer.weight_decay", 1e-5),
        dataset_path=_get(config, "paths.dataset_path", "./dataset"),
        dataset_test_dir=_get(config, "paths.dataset_test_dir", "./dataset/TSB-AD/Datasets/TSB-AD-U"),
        file_list=_get(config, "paths.file_list", "./dataset/TSB-AD/Datasets/File_List/TSB-AD-U.csv"),
        output_file_path=_get(config, "paths.output_file_path", "./output/result.json"),
        vision_path=_get(config, "paths.vision_path", "./checkpoints/weight_v"),
        ts_path=_get(config, "paths.ts_path"),
        vetime_path=_get(config, "paths.vetime_path"),
        resume=_get(config, "paths.resume"),
        pretrain_from=_get(config, "paths.pretrain_from"),
        keep_idx_path=_get(config, "paths.keep_idx_path"),
        device=_get(config, "device", "auto"),
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


def training_config_from_hydra(config: Mapping[str, Any]):
    """Return the typed configuration used by the clean application layer."""
    return training_config_from_mapping(config)
