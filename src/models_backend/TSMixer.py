import logging
import math

import torch
import torch.nn as nn

from models_backend.layers.layer_utils import get_activation_function
from models_backend.base_backend_model import BaseNNBackendModel
from models_backend.layers.layers import SmallLinearRegressionHead
from explainability.shap_strategies import ShortSeqNeuralSHAPStrategy


logger = logging.getLogger(__name__)


# Code inspired by: https://github.com/ditschuk/pytorch-tsmixer

"""
Dimension Legend
B: Batch Size (number of samples processed concurrently)
S: Sequence Length
M: Num Variates / Channels / Features (e.g., self.input_dim)
"""


def init_tsmixer_weights(module: nn.Module) -> None:
    """
    Weight initialization for TSMixer architecture.
    
    Applies fan-in adjusted initialization for Linear layers.
    """
    if isinstance(module, nn.Linear):
        # Kaiming Normal for Linear layers (assuming ReLU/GELU activations)
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)



def apply_tsmixer_residual_scaling(model: nn.Module, num_blocks: int) -> None:
    """Scale down the output linear layers of mixing blocks to stabilize deep residual paths."""
    res_scale = 1.0 / math.sqrt(2.0 * num_blocks)
    
    for name, module in model.named_modules():
        # Target second FC in FeatureMixing and the linear projection in TimeMixing
        if name.endswith("feature_mixing.fc2") or name.endswith("time_mixing.fc"):
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight.mul_(res_scale)



class FeatureMixing(nn.Module):
    """
    Feature-Mixing block for TSMixer.
    Applies a Pre-Layer Normalization Multi-Layer Perceptron across feature channels
    independently at each time step, preserving temporal spatial locations via residual connection.

    Args:
        channels (int): Number of feature dimensions / channels (M).
        ff_dim (int): Hidden expansion dimension within the channel feed-forward network.
        activation (str): Non-linear activation method ("relu", "gelu", or "silu"). Defaults to "relu".
        dropout_rate (float): Dropout probability applied after activation and output projection. Defaults to 0.1.
    """
    def __init__(
        self, 
        channels: int, 
        ff_dim: int, 
        activation: str = "relu", 
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.ff_dim = ff_dim
        self.activation_name = activation.lower()
        self.dropout_rate = dropout_rate

        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, ff_dim)
        self.fc2 = nn.Linear(ff_dim, channels)
        self.activation = get_activation_function(activation)
        self.dropout = nn.Dropout(dropout_rate)

        logger.info(f"Initialized {self}")


    def __str__(self) -> str:
        return (f"FeatureMixing(channels={self.channels}, ff_dim={self.ff_dim}, activation={self.activation_name}, dropout_rate={self.dropout_rate})")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, S, M]

        Returns:
            torch.Tensor: Feature-mixed output tensor of shape [B, S, M]
        """
        res = x                          # Identity residual path [B, S, M]
        x_norm = self.norm(x)            # Pre-Layer Normalization [B, S, M]

        x_out = self.fc1(x_norm)         # Project channels to hidden dimension: [B, S, ff_dim]
        x_out = self.activation(x_out)   # Non-linear activation [B, S, ff_dim]
        x_out = self.dropout(x_out)      # Regularization [B, S, ff_dim]
        x_out = self.fc2(x_out)          # Project back to channel dimension: [B, S, M]
        x_out = self.dropout(x_out)      # Regularization [B, S, M]

        return res + x_out               # Residual connection [B, S, M]



class TimeMixing(nn.Module):
    """
    Time-Mixing block for TSMixer.
    Applies Pre-Layer Normalization and a shared linear projection across the sequence length 
    dimension to share temporal information independently across each feature channel.

    Args:
        sequence_length (int): Length of input sequence time steps (S).
        channels (int): Number of feature dimensions / channels (M or D).
        activation (str): Non-linear activation method ("relu", "gelu", or "silu"). Defaults to "relu".
        dropout_rate (float): Dropout probability applied after temporal projection. Defaults to 0.1.
    """
    def __init__(
        self, 
        sequence_length: int, 
        channels: int, 
        activation: str = "relu", 
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.channels = channels
        self.activation_name = activation.lower()
        self.dropout_rate = dropout_rate

        self.norm = nn.LayerNorm(channels)
        self.fc = nn.Linear(sequence_length, sequence_length)
        self.activation = get_activation_function(activation)
        self.dropout = nn.Dropout(dropout_rate)

        logger.info(f"Initialized {self}")


    def __str__(self) -> str:
        return (f"TimeMixing(sequence_length={self.sequence_length}, channels={self.channels}, activation={self.activation_name}, dropout_rate={self.dropout_rate})")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, S, M]

        Returns:
            torch.Tensor: Time-mixed output tensor of shape [B, S, M]
        """
        res = x                          # Identity residual path [B, S, M]
        x_norm = self.norm(x)            # Pre-Layer Normalization [B, S, M]

        # Transpose sequence and channel dimensions for temporal mixing
        x_temp = x_norm.transpose(1, 2)  # [B, M, S]
        x_temp = self.fc(x_temp)         # Temporal projection across S: [B, M, S]
        x_temp = self.activation(x_temp) # Non-linear activation [B, M, S]
        x_temp = self.dropout(x_temp)    # Regularization [B, M, S]
        x_temp = x_temp.transpose(1, 2)  # Transpose back: [B, S, M]

        return res + x_temp               # Residual connection [B, S, M]



class MixerLayer(nn.Module):
    """
    Single TSMixer Layer combining sequential Time-Mixing and Feature-Mixing operations.

    Args:
        sequence_length (int): Length of input sequence time steps (S).
        channels (int): Number of feature dimensions / channels (M).
        ff_dim (int): Hidden expansion dimension within the FeatureMixing block.
        activation (str): Non-linear activation method ("relu", "gelu", or "silu"). Defaults to "relu".
        dropout_rate (float): Dropout connection probability. Defaults to 0.1.
    """
    def __init__(
        self, 
        sequence_length: int, 
        channels: int, 
        ff_dim: int, 
        activation: str = "relu", 
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.channels = channels
        self.ff_dim = ff_dim
        self.activation_name = activation.lower()
        self.dropout_rate = dropout_rate

        self.time_mixing = TimeMixing(sequence_length, channels, activation, dropout_rate)
        self.feature_mixing = FeatureMixing(channels, ff_dim, activation, dropout_rate)

        logger.info(f"Initialized {self}")


    def __str__(self) -> str:
        return (f"MixerLayer(sequence_length={self.sequence_length}, channels={self.channels}, "
                f"ff_dim={self.ff_dim}, activation={self.activation_name}, dropout_rate={self.dropout_rate})")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, S, M]

        Returns:
            torch.Tensor: Mixed output representation tensor of shape [B, S, M]
        """
        x = self.time_mixing(x)          # Apply temporal mixing: [B, S, M]
        x = self.feature_mixing(x)       # Apply channel/feature mixing: [B, S, M]
        return x



class TSMixer(BaseNNBackendModel):
    """
    Time-Series Mixer (TSMixer) for multivariate regression.
    Utilizes stacked sequential time-mixing and feature-mixing MLPs under Pre-Layer Normalization 
    to extract cross-time and cross-feature interactions without compute-heavy attention mechanisms.

    Args:
        input_dim (int): Number of input features per time step (M).
        output_dim (int): Target dimension size for the final regression output.
        seq_len (int): Lookback window sequence length (S).
        label_transformer (BaseTargetTransform): Instance of a label transformer.
        activation (str): Non-linear activation function ("relu", "gelu", or "silu"). Defaults to "relu".
        num_blocks (int): Number of stacked MixerLayer blocks. Defaults to 5.
        dropout_rate (float): Dropout connection probability. Defaults to 0.1.
        ff_dim_factor (int): Factor of the hidden expansion dimension inside FeatureMixing blocks. Defaults to 2.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        label_transformer,
        activation: str = "relu",
        num_blocks: int = 5,
        dropout_rate: float = 0.1,
        ff_dim_factor: int = 2,
    ):
        super().__init__(input_dim, output_dim, seq_len, label_transformer)
        self.activation_name = activation.lower()
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.ff_dim_factor = ff_dim_factor

        # Stack of TSMixer blocks
        self.mixer_layers = nn.ModuleList([
            MixerLayer(
                sequence_length=seq_len,
                channels=input_dim,
                ff_dim=ff_dim_factor * input_dim,
                activation=activation,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_blocks)
        ])

        # Regression Head
        self.flatten = nn.Flatten(start_dim=1)
        self.head = SmallLinearRegressionHead(output_dim, input_dim * seq_len)

        # Weight initilization
        self.apply(init_tsmixer_weights)
        apply_tsmixer_residual_scaling(self, self.num_blocks)

        logger.info(f"Initialized {self}  \n")


    def __str__(self) -> str:
        return (f"TSMixer(\n"
                f"  input_dim={self.input_dim}, output_dim={self.output_dim}, seq_len={self.seq_len},\n"
                f"  num_blocks={self.num_blocks}, ff_dim_factor={self.ff_dim_factor}, activation={self.activation_name}, dropout_rate={self.dropout_rate}\n"
                f")")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input multivariate time-series tensor of shape [B, S, M]

        Returns:
            torch.Tensor: Projected regression output tensor of shape [B, output_dim]
        """
        # Sequential feature and temporal mixing
        for mixer in self.mixer_layers:
            x = mixer(x)                 # [B, S, M]

        x = self.flatten(x)              # Flatten [B, S * M]
        out = self.head(x)               # [B, output_dim]
        return out


    def get_shap_strategy(self) -> ShortSeqNeuralSHAPStrategy:
        """Returns its own compatible SHAP explainability strategy."""
        return ShortSeqNeuralSHAPStrategy()
