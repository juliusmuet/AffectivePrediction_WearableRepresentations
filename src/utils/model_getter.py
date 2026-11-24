import torch
import logging
import preprocessing.label_transformer as label_transformer
import datasets.data_scheduler as data_schedulers
import train.lr_scheduler as lr_schedulers
import datasets.data_splitter as dataset_splitters
import models_backend.BiMamba as BiMamba
import models_backend.PatchTST as PatchTST
import models_backend.ModernTCN as ModernTCN
import models_backend.TimesNet as TimesNet
import models_backend.TSMixer as TSMixer
import models_backend.FTTransformer as FTTransformer
import models_backend.XGBoostRegressors as XGBoostRegressors

logger = logging.getLogger(__name__)

"""
Module for dynamically resolving and loading classes by name prefixes.
"""

def get_class_by_prefix(name: str, prefix_module_mapping: str, item_type: str ="Item"):
    """
    Fetches a class from the appropriate module based on the provided name prefix.

    Args:
        name (str): The name of the class to load.
        prefix_module_mapping (dict): A dictionary where keys are prefixes and values are modules.
        item_type (str): A string for identifying the item type in error messages (default: "item").

    Returns:
        class: The resolved class from the module.

    Raises:
        SystemExit: If the name does not match any known prefix.
    """
    for prefix, module in prefix_module_mapping.items():
        if name.startswith(prefix):
            return getattr(module, name)
    logger.info(f"{item_type} '{name}' not found!")
    exit(1)


label_transformer_prefixes = {
    "Identity": label_transformer,
    "MinMax": label_transformer,
    "Logit": label_transformer
}

data_scheduler_prefixes = {
    "KDELabelScheduler": data_schedulers
}

optimiser_prefixes = {
    "Adam": torch.optim,
    "SGD": torch.optim
}

lr_scheduler_prefixes = {
    "Exponential": lr_schedulers,
    "Cosine": lr_schedulers
}

loss_prefixes = {
    "MSELoss": torch.nn,
    "SmoothL1Loss": torch.nn,
    "BCEWithLogitsLoss": torch.nn
}

dataset_splitter_prefixes = {
    "BetweenSubjectSplitter": dataset_splitters,
    "BetweenSubjectMonteCarloSplitter": dataset_splitters,
    "WithinSubjectSplitter": dataset_splitters
}

backend_model_prefixes = {
    "BiMamba": BiMamba,
    "PatchTST": PatchTST,
    "ModernTCN": ModernTCN,
    "TimesNet": TimesNet,
    "TSMixer": TSMixer,
    "FTTransformer": FTTransformer,
    "XGBoostRegressor": XGBoostRegressors,
    "XGBoostRandomForestRegressor": XGBoostRegressors
}


def get_label_transformer(name: str):
    """
    Retrieves an label transformer class from preprocessing.label_transformer based on its name prefix.

    Supported label transformer names:
    - Identity
    - MinMax
    - Logit

    Args:
        name (str): The name of the label transformer.

    Returns:
        class: The label transformer class.
    """
    return get_class_by_prefix(name, label_transformer_prefixes, "Label Transformer")


def get_data_scheduler(name: str):
    """
    Retrieves an data scheduler class from torch.optim based on its name prefix.

    Supported data scheduler names:
    - KDELabelScheduler

    Args:
        name (str): The name of the data scheduler.

    Returns:
        class: The data scheduler class.
    """
    return get_class_by_prefix(name, data_scheduler_prefixes, "Data Scheduler")


def get_optimiser(name: str):
    """
    Retrieves an optimiser class from torch.optim based on its name prefix.

    Supported optimiser names:
    - Adam
    - SGD

    Args:
        name (str): The name of the optimiser.

    Returns:
        class: The optimiser class.
    """
    return get_class_by_prefix(name, optimiser_prefixes, "Optimiser")


def get_lr_scheduler(name: str):
    """
    Retrieves a learning rate scheduler from a scheduler module based on its name prefix.

    Supported scheduler names:
    - ExponentialDecrease
    - CosineDecrease

    Args:
        name (str): The name of the learning rate scheduler.

    Returns:
        class: The learning rate scheduler class.
    """
    return get_class_by_prefix(name, lr_scheduler_prefixes, "Learning Rate Scheduler")


def get_loss(name: str):
    """
    Retrieves a loss from torch.nn based on its name prefix.

    Supported loss names:
    - MSELoss
    - SmoothL1Loss
    - BCEWithLogitsLoss

    Args:
        name (str): The name of the loss.

    Returns:
        class: The loss class.
    """
    return get_class_by_prefix(name, loss_prefixes, "Loss")


def get_dataset_splitter(name: str):
    """
    Retrieves a dataset splitter from a dataset splitter module based on its name prefix.

    Supported scheduler names:
    - BetweenSubjectSplitter
    - BetweenSubjectMonteCarloSplitter
    - WithinSubjectSplitter

    Args:
        name (str): The name of the dataset splitter.

    Returns:
        class: The dataset splitter class.
    """
    return get_class_by_prefix(name, dataset_splitter_prefixes, "Dataset Splitter")


def get_backend_model(name: str):
    """
    Retrieves a backend model from a models_backend module based on its name prefix.

    Supported scheduler names:
    - CatPooledGRU
    - GRUSelfAttention
    - ConvGRUSelfAttention
    - XGBoostRegressor
    - XGBoostRandomForestRegressor

    Args:
        name (str): The name of the backend model.

    Returns:
        class: The backend model class.
    """
    return get_class_by_prefix(name, backend_model_prefixes, "Backend Model")