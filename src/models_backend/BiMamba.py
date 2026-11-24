import logging
import torch

import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba

from models_backend.base_backend_model import BaseNNBackendModel
from models_backend.layers.layers import LayerScale, MultiScalePatchEmbedding, FFNResidualBlock, MultiHeadAttentionPooling, ContextualChannelAggregation, LinearRegressionHead
from models_backend.layers.layer_utils import init_norm
from explainability.shap_strategies import LongSeqNeuralSHAPStrategy


logger = logging.getLogger(__name__)


# Code inspired by https://github.com/Leopold2333/Bi-Mamba4TS

"""
Dimension Legend
B: Batch Size (number of samples processed concurrently)
M: Num Variates / Channels / Features (e.g., self.input_dim)
L: Raw Sequence Length / Lookback Window without patching (e.g., self.seq_len)
N: Num Patches (the number of sliced window segments, calculated as self.patch_num)
P: Patch Length (the temporal size of each slice, e.g., self.patch_len)
D: d_model (the embedding projection dimension, e.g., self.d_model)
S: Sequence Length (universal) (mostly N)
"""


class BiMambaResidualBlock(nn.Module):
    """
    Bidirectional Mamba Residual Block with Pre-Layer Normalization and LayerScale.
    Executes concurrent forward and backward temporal sweeps across sequence tokens, 
    averaging their output trajectories to support non-causal contextual modeling.

    Args:
        mamba_forward (nn.Module): Mamba module instance for the forward temporal sweep.
        mamba_backward (nn.Module): Mamba module instance for the backward temporal sweep. If None: unidirectional MAMBA.
        d_model (int): Inner representation depth dimensions (D).
        dropout (float): Regularization drop probability applied to combined outputs. Defaults to 0.1.
        init_value (float): Initial value assignment for LayerScale coefficients. Defaults to 1e-2.
    """
    def __init__(
        self, 
        mamba_forward: nn.Module, 
        mamba_backward: nn.Module, 
        d_model: int, 
        dropout: float = 0.1, 
        init_value: float = 1e-2,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout_prob = dropout
        self.init_value = init_value

        self.norm = nn.LayerNorm(d_model)

        self.mamba_forward = mamba_forward
        self.mamba_backward = mamba_backward

        self.dropout = nn.Dropout(dropout)
        self.layer_scale = LayerScale(d_model, init_value)

        # Apply layer weights
        init_norm(self.norm)

        logger.info(f"Initialized {self}")
    

    def __str__(self) -> str:
        return (f"BiMambaResidualBlock(mamba_forward={self.mamba_forward}, mamba_backward={self.mamba_backward}, d_model={self.d_model}, "
                f"dropout={self.dropout_prob}, init_value={self.init_value}, bidirectional={self.mamba_backward is not None})")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input token sequences of shape [B, S, D]

        Returns:
            torch.Tensor: Residual blended state trajectories of shape [B, S, D]
        """
        # Apply pre-activation Layer Normalization
        x_norm = self.norm(x)

        # Process standard forward temporal sequence flow
        forward = self.mamba_forward(x_norm.contiguous())  # [B, S, D]

        if self.mamba_backward is not None:
            # Flip along temporal sequence dimension for the backward processing pass
            backward = self.mamba_backward(x_norm.flip(dims=[1]).contiguous()).flip(dims=[1]).contiguous()  # [B, S, D]
            processed = 0.5 * (forward + backward)  # [B, S, D]
        else:
            processed = forward # [B, S, D]

        # Regularize combined output spaces
        processed = self.dropout(processed) # [B, S, D]

        # Stabilize convergence via LayerScale coefficient modification before residual addition
        return x + self.layer_scale(processed)  # [B, S, D]



class EncoderLayer(nn.Module):
    """
    Bidirectional State Space model layer executing synchronized parallel sweeps over inputs.
    Combines bidirectional Mamba temporal processing pipelines with standard positional 
    feed-forward layers into a unified sequence modeling stage.

    Args:
        mamba_forward (nn.Module): Mamba module instance for the forward temporal sweep.
        mamba_backward (nn.Module): Mamba module instance for the backward temporal sweep.
        d_model (int): Inner model depth size dimensions. Defaults to 128.
        d_ff (int): Hidden dimension size inside FFN layer feeds. Defaults to 256.
        dropout (float): Inner block dropout regularization factor. Defaults to 0.1.
        activation (str): Linear activation choice profiles ("relu", "gelu", or "silu"). Defaults to "gelu".
        init_values (float): Initial values of the scaling coefficients. Defaults to 1e-2.
    """
    def __init__(
        self, 
        mamba_forward: nn.Module, 
        mamba_backward: nn.Module, 
        d_model: int = 128, 
        d_ff: int = 256, 
        dropout: float = 0.1, 
        activation: str = "gelu", 
        init_values: float = 1e-2,
    ):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_prob = dropout
        self.activation_name = activation

        self.mamba_block = BiMambaResidualBlock(mamba_forward, mamba_backward, d_model, dropout, init_values)
        self.ffn_block = FFNResidualBlock(d_model, d_ff, activation, dropout, init_values)

        logger.info(f"Initialized {self}")
    

    def __str__(self) -> str:
        return (f"EncoderLayer(d_model={self.d_model}, d_ff={self.d_ff}, dropout={self.dropout_prob}, activation={self.activation_name})")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input token sequences of shape [B, S, D]
        
        Returns:
            torch.Tensor: Output token sequences of shape [B, S, D] 
        """
        x = self.mamba_block(x) # [B, S, D]
        x = self.ffn_block(x)   # [B, S, D]
        return x    # [B, S, D]



class Encoder(nn.Module):
    """
    Sequential execution container for stacking layers of Bidirectional Mamba encoders.

    Args:
        mamba_layers (list of nn.Module): Contains the Encoder Layers.
        norm_layer (nn.Module): Final normalisation layer after all encoder layers. Defaults to None.
    """
    def __init__(self, mamba_layers: list, norm_layer: nn.Module = None):
        super(Encoder, self).__init__()
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.norm = norm_layer

        # Apply layer weights
        if self.norm is not None:
            init_norm(self.norm)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, S, D]
        
        Returns: 
            x (torch.Tensor): Output tensor of shape [B, S, D]
        """
        # Loop through each EncoderLayer
        for mamba_block in self.mamba_layers:
            x = mamba_block(x) # [B, S, D]
            
        if self.norm is not None:
            x = self.norm(x) # [B, S, D]
        
        return x    # [B, S, D]



class BiMamba(BaseNNBackendModel):
    """
    BiMamba model modified for Multi-Parameter Regression.
    Accepts data windows matching shapes of [Batch, time_window, features].

    Args:
        input_dim (int): Number of features of the input.
        output_dim (int): Number of features of the output.
        seq_len (int): Size of lookback time window.
        label_transformer (BaseTargetTransform): Instance of a label transformer.
        patch_len (int): Length of patches. Defaults to 64.
        stride (int): Stepping increments used during sliding window patch parsing. Defaults to 32.
        d_model (int): Inner model depth size dimensions. Defaults to 64.
        d_ff (int): Hidden dimension size inside FFN layer feeds. Defaults to 128.
        d_state (int): State Space sizing allocation. Defaults to 16.
        d_conv (int): Internal Mamba convolution width parameters. Defaults to 4.
        e_fact (int): Channel expansion scaling multipliers inside Mamba blocks. Defaults to 2.
        e_layers (int): Number of encoder blocks. Defaults to 2.
        att_heads (int): Number of attention heads for temporal aggregation. Defaults to 4.
        dropout (float): Dropout regularization factor. Defaults to 0.1.
        activation (str): Linear activation choice profiles ("silu" or "gelu"). Defaults to "silu".
        init_values (float): Initial value for layer scale. Defaults to 1e-4.
        channel_independence (bool): If True, treats channels independently until output head. Defaults to True.
        use_instance_norm (bool): If True, normalizes local input windows and re-injects statistics into the regression head. Defaults to True.
    """
    def __init__(self,
        input_dim: int,
        output_dim: int ,
        seq_len: int,
        label_transformer,
        patch_len: int = 64,
        stride: int = 32,
        d_model: int = 64,
        d_ff: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        e_fact: int = 2,
        e_layers: int = 2,
        att_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "silu",
        init_values: float = 1e-4,
        channel_independence: bool = True,
        use_instance_norm: bool = True,
    ):
        super().__init__(input_dim, output_dim, seq_len, label_transformer)
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_state = d_state
        self.d_conv = d_conv
        self.e_fact = e_fact
        self.e_layers = e_layers
        self.att_heads = att_heads
        self.dropout_prob = dropout
        self.activation = activation
        self.init_values = init_values
        self.channel_independence = channel_independence
        self.use_instance_norm = use_instance_norm

        # Instance Norm Epsilon to avoid divide-by-zero
        self.eps = 1e-5

        # Patch Extraction Layer   
        self.patch_embedding = MultiScalePatchEmbedding(seq_len, input_dim, d_model, patch_len, stride, dropout, activation, channel_independence, False)
        
        # MAMBA encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=e_fact),
                    Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=e_fact),
                    d_model,
                    d_ff,
                    dropout,
                    activation,
                    init_values
                ) for _ in range(e_layers)
            ],
            norm_layer=None
        )

        # Attention Pooling Head to pool across the temporal dimension (N)
        self.temporal_pooling = MultiHeadAttentionPooling(d_model, att_heads, dropout, init_values)

        # Channel scoring to aggregate channel dimensions (M)
        self.channel_aggregation = ContextualChannelAggregation(self.input_dim, d_model, activation, dropout, init_values)
        
        # MLP for output regression
        head_in_dim = d_model + (2 * input_dim) if use_instance_norm else d_model
        self.regression_head = LinearRegressionHead(self.output_dim, head_in_dim, activation, dropout)

        logger.info(f"Initialized {self}  \n")

    
    def __str__(self) -> str:
        return (f"BiMamba(\n"
                f"  input_dim={self.input_dim}, output_dim={self.output_dim}, seq_len={self.seq_len},\n"
                f"  patch_len={self.patch_len}, stride={self.stride}, d_model={self.d_model}, d_ff={self.d_ff},\n"
                f"  d_state={self.d_state}, d_conv={self.d_conv}, expand={self.e_fact}, layers={self.e_layers},\n"
                f"  att_heads={self.att_heads}, dropout={self.dropout_prob}, activation='{self.activation},'\n"
                f"  init_values={self.init_values}, channel_independence={self.channel_independence}, use_instance_norm={self.use_instance_norm}\n"
                f")")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input data of shape [B, L, M]
            
        Returns:
            torch.Tensor: Evaluated continuous parameter matrix of shape [B, output_dim]
        """
        B, L, M = x.shape  # Batch, Time Window (seq_len), Features (input_dim)

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
        enc_out, _, _ = self.patch_embedding(x_norm.permute(0, 2, 1)) # [B, L, M] -> [B, M, L] -> [B * M, N, D] (Channel Independence) or [B, N, D] (Channel Mixing)

        # Run bidirectional MAMBA encoder with sequence length = N (number of patches)
        enc_out = self.encoder(enc_out) # [B * M, N, D] (Channel Independence) or [B, N, D] (Channel Mixing)

        # Attentional Temporal Pooling across patch dimension
        pooled_channels = self.temporal_pooling(enc_out)   # Pool across the temporal sequence (N): [B * M, N, D] -> [B * M, D] (Channel Independence) or [B, N, D] -> [B, D] (Channel Mixing)
        
        # Channel Scoring & Aggregation
        if self.channel_independence:
            pooled_channels = rearrange(pooled_channels, '(B M) D -> B M D', B=B, M=M)  # [B * M, D] -> [B, M, D]
            pooled_out = self.channel_aggregation(pooled_channels)  # [B, M, D] * [B, M, 1] -> [B, D]
        else:
            # In Channel Mixing, the convolutions have already blended M into D, so features are already shaped as [B, D]
            pooled_out = pooled_channels

        # Re-inject Instance Statistics if enabled
        if self.use_instance_norm:
            # Concatenate temporal representations with raw window energy statistics -> [B, D + 2*M]
            head_input = torch.cat([pooled_out, seq_mean_flat, seq_std_flat], dim=-1)
        else:
            head_input = pooled_out
        
        # Final regression mapping: [B, D] -> [B, output_dim]
        return self.regression_head(head_input)


    def get_shap_strategy(self) -> LongSeqNeuralSHAPStrategy:
        """Returns its own compatible SHAP explainability strategy."""
        return LongSeqNeuralSHAPStrategy()
