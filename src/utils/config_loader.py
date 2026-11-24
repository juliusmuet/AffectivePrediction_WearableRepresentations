from pathlib import Path
from typing import Any

import yaml


def load_config_file(config_path: str | Path) -> dict[str, Any]:
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_seed(config: dict[str, Any]) -> int | None:
    return config.get('seed', None)


def get_study_configs(config: dict[str, Any]) -> list[dict[str, Any]] | None:
    return config.get('studies', None)


def get_extractor_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get('extractor', {})


def get_label_transformer_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get('label_transformer', {'transformer_name': "Identity",
                                            'transformer_args': {}})


def get_dataset_split_config(config: dict[str, Any]) -> dict[str, Any] | None:
    return config.get('dataset_split', None)


def get_dataloader_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get('dataloader', {})


def get_data_scaler_config(config: dict[str, Any]) -> dict[str, Any] | None:
    return config.get('data_scaler', None)


def get_data_scheduler_config(config: dict[str, Any]) -> dict[str, Any] | None:
    return config.get('data_scheduler', None)


def get_train_config(config: dict[str, Any]) -> dict[str, Any] | None:
    return config.get('train_args', None)


def get_loss_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get('loss', {'loss_name': "SmoothL1Loss",
                                'loss_args': {'beta': 0.1}})


def get_frontend_config(config: dict[str, Any]) -> dict[str, Any] | None:
    return config.get('frontend', None)


def get_backend_config(config: dict[str, Any]) -> dict[str, Any] | None:
    return config.get('backend', None)


def get_optimiser_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get('optimiser', {'optimiser_name': "AdamW",
                                    'optimiser_args': {}})


def get_lr_scheduler_config(config: dict[str, Any]) -> dict[str, Any] | None:
    return config.get('lr_scheduler', None)


def get_shap_config(config: dict[str, Any]) -> dict[str, Any] | None:
    return config.get('shap', None)


def get_finetune_config(config: dict[str, Any]) -> dict[str, Any] | None:
    return config.get('finetune', None)
