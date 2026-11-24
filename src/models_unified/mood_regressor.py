import logging

import numpy as np
import torch
import torch.nn as nn

from models_backend.base_backend_model import BaseBackendModel


logger = logging.getLogger(__name__)


class MoodRegressor(nn.Module):
    """
    Unified model container designed for modular mood prediction.
    Combines an optional feature extraction Frontend with a required prediction Backend.

    Args:
        backend_model (BaseBackendModel): Backend model, which inherits from BaseBackendModel, for mood regression.
        frontend_model (BaseFrontendModel): Frontend model, which inherits from BaseFrontendModel, for feature extraction. Defaults to None.
    """
    def __init__(self, backend_model: BaseBackendModel, frontend_model=None):
        super().__init__()
        self.frontend = frontend_model
        self.backend = backend_model

        if not self.backend.is_neural_network and self.frontend is not None:
            raise ValueError("Cannot create model with frontend model and non Neural Network backend model.")

        # Calculate parameter counts
        backend_params_total, backend_params_trainable = self._count_parameters(self.backend)
        
        if self.has_frontend:
            frontend_params_total, frontend_params_trainable = self._count_parameters(self.frontend)
        else:
            frontend_params_total, frontend_params_trainable = 0, 0

        total_params = frontend_params_total + backend_params_total
        total_trainable_params = frontend_params_trainable + backend_params_trainable

        logger.info(f"Initialized {self}")
        logger.info(
            f"Model Parameter Count:\n"
            f"  Frontend:  {frontend_params_trainable:,} trainable / {frontend_params_total:,} total\n"
            f"  Backend:   {backend_params_trainable:,} trainable / {backend_params_total:,} total\n"
            f"  Combined:  {total_trainable_params:,} trainable / {total_params:,} total\n"
        )
        
    
    def __str__(self):
        """Returns a string representation of the model configuration."""
        return f"UnifiedModel {'Direct-to-Backend' if self.frontend is None else self.frontend.__class__.__name__} -> {self.backend.__class__.__name__}"


    def _count_parameters(self, model: nn.Module) -> tuple[int, int]:
        """
        Calculates total and trainable parameters for a given PyTorch module or backend.

        Args:
            model (nn.Module): The model to count the parameters of.

        Returns:
            tuple (total, trainable)
        """
        if not hasattr(model, "parameters"):
            return 0, 0
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable


    @property
    def has_frontend(self) -> bool:
        """Returns True if a frontend model is used."""
        return self.frontend is not None
    

    def get_trainable_parameters_backend(self) -> list:
        """
        Returns only the trainable parameters of the backend model.
        """
        return list(filter(lambda p: p.requires_grad, self.backend.parameters()))


    def get_trainable_parameters_frontend(self) -> list:
        """
        Returns only the trainable parameters of the frontend model.
        """
        if not self.has_frontend:
            return []
        
        # TODO: add logic if frontend should not be trained
        
        return list(filter(lambda p: p.requires_grad, self.frontend.parameters()))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes inference across the frontend-backend pipeline.
        
        Args:
            x (torch.Tensor): Input data of shape (Batch, Window_Size, Features).
            
        Returns:
            torch.Tensor: Predicted multi-dimensional target mood vectors of shape (Batch, 3).
        """
        if self.frontend is not None:
            x = self.frontend(x)
        
        return self.backend(x)
    

    def calculate_shap(
        self, 
        X_train_raw: np.ndarray, 
        X_test_raw: np.ndarray, 
        n_samples: int = 100, 
        seed: int = 42,
    ) -> np.ndarray:
        """
        Computes out-of-fold signed SHAP values for all output dimensions.

        Args:
            X_train_raw (np.ndarray): Background reference tensor of shape (Samples_Train, Window, Features).
            X_test_raw (np.ndarray): Target evaluation data tensor of shape (Samples_Test, Window, Features).
            n_samples (int): Numeric size limit for background baseline reference sampling. Defaults to 100.
            seed (int): Seed for the sampling of background data. Defaults to 42. Placeholder for interface compatability.

        Returns:
            np.ndarray: Array of signed SHAP attribution values with shape (Samples_Test, Window, Features, Outputs).
        """
        logger.info(f"Calculating SHAP values...")
        strategy = self.backend.get_shap_strategy()
        return strategy.compute_shap(self, X_train_raw, X_test_raw, n_samples, seed)