import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from explainability.shap_strategies import BaseSHAPStrategy
from preprocessing.label_transformer import BaseTargetTransform


logger = logging.getLogger(__name__)


class BaseBackendModel(ABC, nn.Module):
    """
    Abstract Base Class enforcing structure across all backend models.
    Neural Network based model directly inherit this class.
    For non Neural Network based models use BaseNonNNBackendModel.

    Args:
        input_dim (int): Number of features of the input.
        output_dim (int): Number of features of the output.
        seq_len (int): Size of lookback time window.
        label_transformer (BaseTargetTransform): Instance of a label transformer.
    """
    def __init__(self, input_dim: int, output_dim: int, seq_len: int, label_transformer: BaseTargetTransform):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.label_transformer = label_transformer


    @property
    @abstractmethod
    def is_neural_network(self) -> bool:
        """Flag identifying neural networks to determine training routine."""
        pass


    @property
    @abstractmethod
    def requires_one_hot_encoding(self) -> bool:
        """Flag identifying whether categorical input features should be one-hot-encoded."""
        pass


    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the inference forward pass call."""
        pass


    @abstractmethod
    def get_shap_strategy(self) -> BaseSHAPStrategy:
        """Returns its own compatible SHAP explainability strategy."""
        pass



class BaseNNBackendModel(BaseBackendModel):
    """
    Extended Abstract Base Class for models which are a Neural Network.
    """
    @property
    def is_neural_network(self) -> bool:
        """
        Flag identifying neural networks to determine training routine.

        Returns:
            bool: Always True for Neural Network architectures.
        """
        return True
    

    @property
    def requires_one_hot_encoding(self) -> bool:
        """
        Flag identifying whether categorical input features should be one-hot-encoded.

        Returns:
            bool: Always True since Neural Networks cannot processes categories internally, 
                  but there are exceptions due to tokenization (e.g. FT-Transformer).
        """
        return True



class BaseNonNNBackendModel(BaseBackendModel):
    """
    Extended Abstract Base Class for models which are not a Neural Network.
    """
    @property
    def is_neural_network(self) -> bool:
        """
        Flag identifying neural networks to determine training routine.

        Returns:
            bool: Always False for non-gradient architectures.
        """
        return False
    

    @abstractmethod
    def run_full_training(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, binary_mask: list, **kwargs):
        """
        Executes the entire training and validation loop for the non-NN model.
        
        Args:
            train_loader (DataLoader): PyTorch DataLoader streaming training batches.
            val_loader (DataLoader): PyTorch DataLoader streaming validation batches.
            test_loader (DataLoader): PyTorch DataLoader streaming test batches.
            binary_mask (list): Numeric binary mask representing base frame feature types (1 for continuous, 0 for categorical).
        
        Returns:
            Tuple[dict, dict, dict]: Metrics of best model for train, validation and test subset (see BaseMetricsTracker)
        """
        pass