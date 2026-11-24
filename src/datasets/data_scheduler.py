import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.stats import gaussian_kde
from torch.utils.data import WeightedRandomSampler


logger = logging.getLogger(__name__)


class BaseLabelScheduler(ABC):
    """
    Abstract base class for dataset label schedulers.

    Subclasses must implement get_sampler() to generate a PyTorch-compatible
    sampler for handling imbalanced regression or classification targets.
    """
    @abstractmethod
    def get_sampler(self, labels: np.ndarray) -> WeightedRandomSampler:
        """
        Generate a weighted sampler based on the target label distribution.

        Args:
            labels: Target labels of shape (N,) or (N, D), where N is the number of samples and D is the number of target dimensions.

        Returns:
            A PyTorch WeightedRandomSampler configured with sampling weights derived from the label distribution.
        """
        pass



class KDELabelScheduler(BaseLabelScheduler):
    """
    Generate density-aware sampling weights using kernel density estimation.

    Fits a multivariate Gaussian kernel density estimator (KDE) to the target labels and assigns sampling weights inversely proportional to the
    estimated label density. This increases the sampling probability of observations in sparsely populated target regions.

    The alpha parameter controls the strength of density-based balancing.

    Args:
        alpha: Temperature parameter in the range [0, 1] controlling the influence of inverse-density weighting. A value of 0.0
            results in uniform sampling, 0.5 applies moderate density-aware balancing, and 1.0 applies full inverse-density weighting. 
            Defaults to 0.5.
        bandwidth: Bandwidth selection method passed to scipy.stats.gaussian_kde. Accepts 'scott', 'silverman', or a positive scalar. 
            Defaults to 'scott'.
        num_samples: Number of samples drawn per epoch. If None, defaults to the number of input observations. Defaults to None.
        eps: Small positive constant added to estimated densities to improve numerical stability when calculating inverse-density weights.
            Defaults to 1e-6.
    """
    def __init__(
        self,
        alpha: float = 0.5,
        bandwidth: str | float = "scott",
        num_samples: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in range [0.0, 1.0], got {alpha}")

        if not eps > 0:
            raise ValueError(f"Eps must be a positive number, got {eps}")

        self.alpha = alpha
        self.bandwidth = bandwidth
        self.num_samples = num_samples
        self.eps = eps

        logger.info(f"Initialized {self}")


    def __str__(self):
        return f"KDELabelScheduler(alpha={self.alpha}, bandwidth={self.bandwidth}, num_samples={self.num_samples}, eps={self.eps})"
    

    def get_sampler(self, labels: np.ndarray) -> WeightedRandomSampler:
        """
        Fit a KDE model and generate a density-aware weighted sampler.

        Fits a multivariate Gaussian KDE to the input labels, evaluates the estimated density at each observation, and assigns inverse-density
        sampling weights controlled by alpha. The resulting weights are normalized before being passed to the PyTorch sampler.

        Args:
            labels: Target labels of shape (N,) or (N, D), where N is the number of observations and D is the number
                of target dimensions. Labels must contain finite numerical values.

        Returns:
            A WeightedRandomSampler configured with normalized inverse-density weights. Sampling is performed with replacement,
            and the number of draws is num_samples if specified, otherwise N.
        """
        labels = np.asarray(labels)
        
        # scipy.stats.gaussian_kde expects shape (features, samples) -> (D, N)
        if labels.ndim == 1:
            data_pts = labels[np.newaxis, :]
        else:
            data_pts = labels.T

        # Fit Multivariate Gaussian Kernel Density Estimator
        kde = gaussian_kde(data_pts, bw_method=self.bandwidth)

        # Evaluate label density at each data point p(y_i)
        densities = kde(data_pts)

        # Calculate inverse-density sampling weights via power-law scaling
        sampling_weights = 1.0 / (densities + self.eps) ** self.alpha

        # Normalize weights to construct valid probability distribution
        sampling_weights /= np.sum(sampling_weights)

        # Convert to PyTorch DoubleTensor for precision stability during sampling
        weights_tensor = torch.from_numpy(sampling_weights).double()

        # Determine total sample count per epoch
        draw_count = self.num_samples if self.num_samples is not None else len(labels)

        return WeightedRandomSampler(weights=weights_tensor, num_samples=draw_count, replacement=True)
