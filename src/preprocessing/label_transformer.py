from abc import ABC, abstractmethod
from typing import Tuple, Union, Sequence
import logging

import numpy as np
from scipy.special import expit, logit
import torch


logger = logging.getLogger(__name__)


class BaseTargetTransform(ABC):
    """
    Abstract base class for continuous target transformation strategies.

    Defines a common interface for encoding targets and decoding model predictions or encoded ground-truth values 
    between the original continuous target space and a transformation-specific target space.
    """
    @abstractmethod
    def encode(self, y_0_100: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Transforms labels from original continuous [0, 100] space into target space.

        Args:
            y_0_100 (Union[np.ndarray, torch.Tensor]): Native target array or tensor bounded in [0, 100].

        Returns:
            Union[np.ndarray, torch.Tensor]: Transformed target data matching input type framework.
        """
        pass


    @abstractmethod
    def decode_predicted(self, y_pred: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverts model predictions (which may be raw logits or probabilities) back to original continuous space [0, 100].

        Args:
            y_pred (Union[np.ndarray, torch.Tensor]): Model predictions or latent outputs.

        Returns:
            Union[np.ndarray, torch.Tensor]: Restored continuous predictions bounded in original [0, 100] range.
        """
        pass


    @abstractmethod
    def decode_truth(self, y_true_encoded: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverts encoded ground truth labels (already in target space) back to original continuous space [0, 100].

        Does not apply logit activations (such as sigmoid or softmax).

        Args:
            y_true_encoded (Union[np.ndarray, torch.Tensor]): Encoded ground truth targets.

        Returns:
            Union[np.ndarray, torch.Tensor]: Restored continuous ground truth bounded in original [0, 100] range.
        """
        pass



class IdentityTransform(BaseTargetTransform):
    """
    Identity transformation passing continuous targets unchanged [0, 100].
    Maintains the raw continuous scale without normalization or bounded domain shifts.
    """
    def __init__(self) -> None:
        logger.info(f"Initialized {self}\n")


    def __str__(self) -> str:
        return "IdentityTransform()"


    def encode(self, y_0_100: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Passes target values through without alteration.

        Args:
            y_0_100 (Union[np.ndarray, torch.Tensor]): Target values in native continuous space [0, 100].

        Returns:
            Union[np.ndarray, torch.Tensor]: Unaltered continuous target array or tensor.
        """
        return y_0_100


    def decode_predicted(self, y_pred: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Passes model predictions through, ensuring bounds are clamped within [0, 100].
        Identical to `decode_truth`.

        Args:
            y_pred (Union[np.ndarray, torch.Tensor]): Unscaled target predictions.

        Returns:
            Union[np.ndarray, torch.Tensor]: Clamped predictions bounded strictly in range [0, 100].
        """
        return self.decode_truth(y_pred)


    def decode_truth(self, y_true_encoded: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Passes encoded ground truth values through, ensuring bounds are clamped within [0, 100].

        Args:
            y_true_encoded (Union[np.ndarray, torch.Tensor]): Target values in encoded space.

        Returns:
            Union[np.ndarray, torch.Tensor]: Clamped ground truth values bounded strictly in range [0, 100].
        """
        if isinstance(y_true_encoded, torch.Tensor):
            return torch.clamp(y_true_encoded, 0.0, 100.0)
        return np.clip(y_true_encoded, 0.0, 100.0)



class MinMaxTransform(BaseTargetTransform):
    """
    Linear Min-Max transformation scaling targets between [0, 100] and [0.0, 1.0].

    Args:
        scale_range (Sequence[float]): Native scale boundary range sequence (e.g., tuple or list). Defaults to (0.0, 100.0).
        from_logits (bool): Whether outputs of the models are raw logits. Defaults to True.
    """
    def __init__(self, scale_range: Sequence[float] = (0.0, 100.0), from_logits: bool = True) -> None:
        self.scale_range: Tuple[float, float] = tuple(scale_range)
        self.min_val, self.max_val = scale_range
        self.range_width = self.max_val - self.min_val
        self.from_logits = from_logits

        logger.info(f"Initialized {self}\n")


    def __str__(self) -> str:
        return f"MinMaxTransform(scale_range={self.scale_range}, from_logits={self.from_logits})"


    def encode(self, y_0_100: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Scales [0, 100] continuous values linearly to normalized range [0.0, 1.0].

        Args:
            y_0_100 (Union[np.ndarray, torch.Tensor]): Target values in native continuous space [0, 100].

        Returns:
            Union[np.ndarray, torch.Tensor]: Min-max normalized target values bounded in [0.0, 1.0].
        """
        # Linearly scale input down to unit range [0.0, 1.0]
        scaled = (y_0_100 - self.min_val) / self.range_width
        if isinstance(scaled, torch.Tensor):
            return torch.clamp(scaled, 0.0, 1.0)
        return np.clip(scaled, 0.0, 1.0)


    def decode_predicted(self, y_pred: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverts scaled [0.0, 1.0] linear predictions back to continuous scale [0, 100].
        Applies Sigmoid activation if `from_logits` is True.

        Args:
            y_pred (Union[np.ndarray, torch.Tensor]): Unit-scaled predictions or raw logits in range [0.0, 1.0].

        Returns:
            Union[np.ndarray, torch.Tensor]: Restored and clamped continuous predictions in range [0, 100].
        """
        if self.from_logits:
            if isinstance(y_pred, torch.Tensor):
                y_pred = torch.sigmoid(y_pred)
            else:
                y_pred = expit(y_pred)

        return self.decode_truth(y_pred)


    def decode_truth(self, y_true_encoded: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverts normalized [0.0, 1.0] ground truth labels directly back to continuous scale [0, 100].
        Bypasses logit activations.

        Args:
            y_true_encoded (Union[np.ndarray, torch.Tensor]): Normalized target values in range [0.0, 1.0].

        Returns:
            Union[np.ndarray, torch.Tensor]: Restored and clamped continuous target values in range [0, 100].
        """
        y_0_100 = (y_true_encoded * self.range_width) + self.min_val

        if isinstance(y_0_100, torch.Tensor):
            return torch.clamp(y_0_100, self.min_val, self.max_val)

        return np.clip(y_0_100, self.min_val, self.max_val)



class LogitTransform(BaseTargetTransform):
    """
    Logit Space Transformation with boundary epsilon scaling.

    Scales input variables from continuous range [0.0, 100.0] into bounded internal probability range
    [eps, 1 - eps] before mapping into an unconstrained real domain (-inf, inf) via inverse logistic functions.

    Args:
        eps (float): Epsilon smoothing boundary buffer. Defaults to 1e-3.
    """

    def __init__(self, eps: float = 1e-3) -> None:
        self.eps = eps

        logger.info(f"Initialized {self}\n")


    def __str__(self) -> str:
        return f"LogitTransform(eps={self.eps})"


    def encode( self, y_0_100: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Encodes bounded continuous labels into unconstrained logit space.

        Args:
            y_0_100 (Union[np.ndarray, torch.Tensor]): Target values bounded in range [0, 100].

        Returns:
            Union[np.ndarray, torch.Tensor]: Logit-transformed unbounded representations.
        """
        # Scale [0, 100] -> [eps, 1 - eps] to prevent numerical infinity in logit evaluations
        y_scaled = self.eps + (1.0 - 2 * self.eps) * (y_0_100 / 100.0)

        # PyTorch Logit Computation
        if isinstance(y_scaled, torch.Tensor):
            return torch.logit(y_scaled)

        # NumPy Logit Computation via scipy.special
        return logit(y_scaled).astype(y_0_100.dtype)


    def decode_predicted(self, y_pred_logit: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Decodes unconstrained logit predictions back to original bounded continuous range [0, 100].
        Identical to `decode_truth` as both model predictions and encoded ground truth live in logit space (-inf, inf).

        Args:
            y_pred_logit (Union[np.ndarray, torch.Tensor]): Model predictions in logit space.

        Returns:
            Union[np.ndarray, torch.Tensor]: Inverted continuous target values clamped within [0, 100].
        """
        return self.decode_truth(y_pred_logit)


    def decode_truth(self, y_true_logit: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverts logit-space ground truth labels back to original continuous range [0, 100].
        Applies Sigmoid activation to convert logit targets in (-inf, inf) back into probability
        space [eps, 1 - eps], then inverts the epsilon scaling.

        Args:
            y_true_logit (Union[np.ndarray, torch.Tensor]): Ground truth labels in logit space.

        Returns:
            Union[np.ndarray, torch.Tensor]: Restored continuous target values clamped within [0,
              100].
        """
        if isinstance(y_true_logit, torch.Tensor):
            y_scaled = torch.sigmoid(y_true_logit)
            y_0_100 = ((y_scaled - self.eps) / (1.0 - 2 * self.eps)) * 100.0
            return torch.clamp(y_0_100, 0.0, 100.0)

        else:
            y_scaled = expit(y_true_logit)
            y_0_100 = ((y_scaled - self.eps) / (1.0 - 2 * self.eps)) * 100.0
            return np.clip(y_0_100, 0.0, 100.0)
