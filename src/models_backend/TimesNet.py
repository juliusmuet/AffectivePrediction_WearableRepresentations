import logging
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from models_backend.base_backend_model import BaseNNBackendModel
from models_backend.layers.layers import MultiScalePatchEmbedding, MultiHeadAttentionPooling, LinearRegressionHead
from models_backend.layers.layer_utils import get_activation_function, init_norm
from explainability.shap_strategies import LongSeqNeuralSHAPStrategy


logger = logging.getLogger(__name__)


# Code inspired by: https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py

"""
Dimension Legend
B: Batch Size (number of samples processed concurrently)
M: Num Variates / Channels / Features (e.g., self.input_dim)
L: Raw Sequence Length / Lookback Window (e.g., self.seq_len)
D: d_model (the embedding projection dimension , e.g., self.d_model)
K: top_k (number of dominant frequencies / periods selected via FFT)
"""


class Inception_Block_V1(nn.Module):
    """
    Multi-scale 2D convolutional Inception block.

    Applies multiple 2D convolutions with varying kernel sizes to capture 2D
    variations across different temporal scales, then averages their outputs.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        num_kernels (int, optional): Number of parallel 2D kernels. Defaults to 3.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_kernels: int = 3,
    ):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        kernels = []
        for i in range(self.num_kernels):
            # Kernel size increases odd-wise: 1, 3, 5, 7...
            # Padding preserves temporal/spatial resolution
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)

        self._init_weights()

        logger.info(f"Initialized {self}")


    def __str__(self) -> str:
        return f"Inception_Block_V1(in_channels={self.in_channels}, out_channels={self.out_channels}, num_kernels={self.num_kernels})"


    def _init_weights(self):
        """Applies Kaiming normal initialization to Conv2d layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input 2D tensor of shape [B, in_channels, H, W].

        Returns:
            torch.Tensor: Aggregated output tensor of shape [B, out_channels, H, W].
        """
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x)) # res_list[i]: [B, out_channels, H, W]

        res = torch.stack(res_list, dim=-1) # Stack along new dim: [B, out_channels, H, W, num_kernels]
        res = res.mean(-1)  # Average across kernel outputs: [B, out_channels, H, W]
        return res
    


class TimesBlock(nn.Module):
    """
    TimesBlock module for transforming 1D temporal sequences into 2D intra/inter-period tensor blocks.

    Uses rFFT to dynamically discover top periodicities, reshapes 1D series
    into 2D blocks, processes them using Inception 2D convolutions, and performs
    adaptive aggregation based on frequency amplitudes.

    Args:
        seq_len (int): Input sequence lookback length (L).
        top_k (int): Number of dominant periods to select (K).
        d_model (int): Input/Output model hidden dimension (D).
        d_ff (int): Intermediate feed-forward expansion dimension.
        num_kernels (int): Number of Inception kernels.
        activation (str): Linear activation choice profiles ("silu" or "gelu"). Defaults to "gelu".
    """
    def __init__(
        self, 
        seq_len: int, 
        top_k: int, 
        d_model: int, 
        d_ff: int, 
        num_kernels: int, 
        activation: str = "gelu",
    ):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.d_model = d_model
        self.d_ff = d_ff

        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            get_activation_function(activation),
            Inception_Block_V1(d_ff, d_model,num_kernels=num_kernels)
        )

        logger.info(f"Initialized {self}")


    def __str__(self) -> str:
        return f"TimesBlock(seq_len={self.seq_len}, top_k={self.k}, d_model={self.d_model}, d_ff={self.d_ff})"


    def FFT_for_Period(self, x: torch.Tensor, k: int = 2) -> tuple[np.ndarray, torch.Tensor]:
        """
        Discovers top-k periodicities using Real Fast Fourier Transform (rFFT).

        Args:
            x (torch.Tensor): Input sequence [B, L, D].
            k (int, optional): Number of periodic signals to extract. Defaults to 2.

        Returns:
            tuple:
                - period (np.ndarray): Array of period lengths calculated as L // freq_indices.
                - period_weights (torch.Tensor): Frequency energy weights of shape [B, K].
        """
        xf = torch.fft.rfft(x, dim=1)  # [B, L//2 + 1, D]

        # Calculate frequency amplitudes averaged across batch and channels
        frequency_list = abs(xf).mean(0).mean(-1)  # [L//2 + 1]
        frequency_list[0] = 0  # Ignore static DC component

        # Find top k frequency indices
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()

        # Convert frequency index to period length
        period = x.shape[1] // top_list

        # Extract per-sample weights for top-k periods: [B, K]
        period_weights = abs(xf).mean(-1)[:, top_list]

        return period, period_weights


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence [B, L, D].

        Returns:
            torch.Tensor: Transformed representation [B, L, D].
        """
        B, T, D = x.size()
        period_list, period_weight = self.FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # 1. Padding sequence if not evenly divisible by period length
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]], device=x.device)
                out = torch.cat([x, padding], dim=1)  # [B, length, D]
            else:
                length = self.seq_len
                out = x

            # 2. Reshape 1D sequence into 2D variation space:
            # [B, length, D] -> [B, length//period, period, D] -> permute -> [B, D, length//period, period]
            out = out.reshape(B, length // period, period, D).permute(0, 3, 1, 2).contiguous()

            # 3. Apply 2D Inception convolutions: [B, D, length//period, period]
            out = self.conv(out)

            # 4. Reshape back to 1D:
            # [B, D, length//period, period] -> permute -> [B, length//period, period, D] -> [B, length, D]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, D)

            # 5. Crop out any added padding
            res.append(out[:, :self.seq_len, :])

        # Stack top-k periodic representations: [B, L, D, K]
        res = torch.stack(res, dim=-1)

        # Adaptive Aggregation via Softmax normalized frequency weights
        period_weight = F.softmax(period_weight, dim=1)  # [B, K]
        period_weight = (period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, D, 1))    # [B, L, D, K]

        # Weighted combination across k periodicities
        res = torch.sum(res * period_weight, dim=-1)  # [B, L, D]

        # Residual Connection
        return res + x



class TimesNet(BaseNNBackendModel):
    """
    TimesNet architecture for classification/regression temporal modeling.

    Args:
        input_dim (int): Number of features of the input.
        output_dim (int): Number of features of the output.
        seq_len (int): Size of lookback time window.
        label_transformer (BaseTargetTransform): Instance of a label transformer.
        patch_len (int): Length of patches. Defaults to 32.
        stride (int): Stepping increments used during sliding window patch parsing. Defaults to 16.
        d_model (int): Inner model depth size dimensions. Defaults to 32.
        d_ff (int): Hidden dimension size inside FFN layer feeds. Defaults to 64.
        top_k (int): Number of dominant periods to select. Defaults to 5.
        num_kernels (int): num_kernels (int): Number of Inception kernels. Defaults to 3.
        e_layers (int): Number of encoder blocks. Defaults to 1.
        att_heads (int): Number of attention heads for temporal aggregation. Defaults to 2.
        dropout (float): Dropout regularization factor. Defaults to 0.1.
        activation (str): Linear activation choice profiles ("silu" or "gelu"). Defaults to "gelu".
        init_values (float): Initial value for layer scale. Defaults to 1e-4.
        use_instance_norm (bool): If True, normalizes local input windows and re-injects statistics into the regression head. Defaults to True.
    """
    def __init__(self,
        input_dim: int,
        output_dim: int ,
        seq_len: int,
        label_transformer,
        patch_len: int = 32,
        stride: int = 16,
        d_model: int = 32,
        d_ff: int = 64,
        top_k: int = 5,
        num_kernels: int = 3,
        e_layers: int = 1,
        att_heads: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        init_values: float = 1e-4,
        use_instance_norm: bool = True,
    ):
        super().__init__(input_dim, output_dim, seq_len, label_transformer)
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.e_layers = e_layers
        self.att_heads = att_heads
        self.dropout_prob = dropout
        self.activation = activation
        self.init_values = init_values
        self.use_instance_norm = use_instance_norm

        # Instance Norm Epsilon to avoid divide-by-zero
        self.eps = 1e-5

        # Input Data Embedding
        self.embedding = MultiScalePatchEmbedding(seq_len, input_dim, d_model, patch_len, stride, dropout, activation, False, True)
        num_patches = (seq_len - patch_len) // stride + 2

        # Stack of TimesBlocks
        self.encoder = nn.ModuleList(
            [
                TimesBlock(
                    num_patches, top_k, d_model, d_ff, num_kernels, activation
                )
                for _ in range(e_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(d_model)
        init_norm(self.layer_norm)

        self.act = get_activation_function(activation)
        self.dropout = nn.Dropout(dropout)

        # Attention Pooling Head to pool across the temporal dimension (L)
        self.temporal_pooling = MultiHeadAttentionPooling(d_model, att_heads, dropout, init_values)

        # MLP for output regression
        head_in_dim = d_model + (2 * input_dim) if use_instance_norm else d_model
        self.regression_head = LinearRegressionHead(output_dim, head_in_dim, activation, dropout)

        logger.info(f"Initialized {self}  \n")


    def __str__(self) -> str:
        return (
            f"TimesNet(\n"
            f"  input_dim={self.input_dim}, output_dim={self.output_dim}, seq_len={self.seq_len},\n"
            f"  patch_len={self.patch_len},, stride={self.stride}, d_model={self.d_model}, d_ff={self.d_ff}, top_k={self.top_k},\n"
            f"  num_kernels={self.num_kernels}, e_layers={self.e_layers}, att_heads={self.att_heads}, dropout={self.dropout_prob},\n"
            f"  activation={self.activation}, init_values={self.init_values}, use_instance_norm={self.use_instance_norm}\n"
            f")"
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Raw sequence input tensor of shape [B, L, M].

        Returns:
            torch.Tensor: Output of shape [B, output_dim].
        """
        # Local Window Instance Normalization & Statistics Extraction
        if self.use_instance_norm:
            # Calculate per-window mean and std across temporal dimension L -> shape [B, 1, M]
            seq_mean = x.mean(dim=1, keepdim=True)
            seq_std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
            
            # Normalize sequence window to zero-mean, unit-variance
            x_norm = (x - seq_mean) / seq_std
            
            # Reshape statistics for head concatenation: [B, 1, M] -> [B, M]
            seq_mean_flat = seq_mean.squeeze(1)
            seq_std_flat = seq_std.squeeze(1)
        else:
            x_norm = x

        # Extract & embed patches along time dimension L
        enc_out, _, _ = self.embedding(x_norm.permute(0, 2, 1)) # [B, L, M] -> [B, M, L] -> [B, N, D]

        # Extract features via TimesBlock Backbone
        for i in range(self.e_layers):
            enc_out = self.layer_norm(self.encoder[i](enc_out))   # [B, N, D]

        # 3. Output Activation
        output = self.act(enc_out)  # [B, N, D]
        output = self.dropout(output)   # [B, N, D]

        # Attentional Temporal Pooling across time dimension
        pooled_channels = self.temporal_pooling(output)   # Pool across the temporal sequence (L): [B, N, D] -> [B, D]

        # Re-inject Instance Statistics (if enabled)
        if self.use_instance_norm:
            # Concatenate temporal representations with raw window energy statistics -> [B, D + 2*M]
            head_input = torch.cat([pooled_channels, seq_mean_flat, seq_std_flat], dim=-1)
        else:
            head_input = pooled_channels

        # Final regression mapping: [B, D] -> [B, output_dim]
        return self.regression_head(head_input)


    def get_shap_strategy(self) -> LongSeqNeuralSHAPStrategy:
        """Returns its own compatible SHAP explainability strategy."""
        return LongSeqNeuralSHAPStrategy()
