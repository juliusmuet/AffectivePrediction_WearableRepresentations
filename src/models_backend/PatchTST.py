import logging
import torch

from torch import nn
from einops import rearrange

from models_backend.base_backend_model import BaseNNBackendModel
from models_backend.layers.layers import LayerScale, MultiScalePatchEmbedding, FFNResidualBlock, MultiHeadAttentionPooling, ContextualChannelAggregation, LinearRegressionHead
from models_backend.layers.layer_utils import init_standard_weights, init_norm
from explainability.shap_strategies import LongSeqNeuralSHAPStrategy


logger = logging.getLogger(__name__)


# Code inspired by: https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py

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


class TriangularCausalMask():
    """
    Generates an upper-triangular boolean mask matrix to enforce causal constraints 
    in attention blocks, preventing positions from attending to subsequent tokens.

    Args:
        B (int): Batch size dimension.
        S (int): Sequence length dimension.
        device (str, optional): Target hardware device for tensor residency. Defaults to "cpu".
    """
    def __init__(self, B: int, S: int, device:str = "cpu"):
        mask_shape = [B, S, S]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """
        Exposes the generated upper-triangular boolean causal mask.

        Returns:
            torch.Tensor: Boolean mask of shape [B, S, S] where True indicates positions to be masked out.
        """
        return self._mask



class ResidualAttention(nn.Module):
    """
    Applies multi-head self-attention with pre-layer normalization (Pre-LN), 
    LayerScale stabilization and a residual skip connection.

    Args:
        d_model (int): Size of hidden model representations (D).
        n_heads (int): Number of parallel attention heads. Must divide `d_model` evenly.
        dropout (float, optional): Dropout probability applied to attention features. Defaults to 0.1.
        init_values (float, optional): Initial scaling value for the LayerScale parameter `gamma`. Defaults to 1e-2.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1, 
        init_values: float = 1e-2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_prob = dropout
        self.init_values = init_values

        self.norm = nn.LayerNorm(d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout, 
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.ls = LayerScale(d_model, init_value=init_values)

        # Apply layer weights
        self.apply(init_standard_weights)

        logger.info(f"Initialized {self}")
    

    def __str__(self) -> str:
        return (f"ResidualAttention(d_model={self.d_model}, n_heads={self.n_heads}, dropout={self.dropout_prob}, init_values={self.init_values})")


    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, S, D].
            attn_mask (torch.Tensor, optional): Boolean or additive attention mask.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - Output tensor of shape [B, S, D].
                - Attention weights matrix.
        """
        # Apply pre-activation Layer Normalization
        norm_x = self.norm(x)

        # Unwrap internal tensor if passed an instance of TriangularCausalMask
        mask_tensor = attn_mask.mask if hasattr(attn_mask, 'mask') else attn_mask
        
        # Execute native PyTorch attention
        # Passes (query, key, value) all as x for Self-Attention
        # attn_mask expects True at positions that should be ignored during attention mapping
        attn_out, attn_weights = self.attention(
            query=norm_x, 
            key=norm_x, 
            value=norm_x, 
            attn_mask=mask_tensor,
            need_weights=True # Set to False if you don't need to return the attention map matrix
        )
        
        # Apply LayerScale and Dropout to the attention output before the residual add
        x = x + self.ls(self.dropout(attn_out))
        return x, attn_weights



class EncoderLayer(nn.Module):
    """
    Standard Transformer Encoder block wrapper. Composes a multi-head attention 
    sub-layer and a point-wise 1D convolutional feed-forward network (FFN) utilizing 
    residual bypass connections and Layer Normalization.

    Args:
        d_model (int): d_model (int): Size of hidden model representations (D).
        num_heads (int, optional): Number of parallel attention heads. Must divide `d_model` evenly.
        d_ff (int, optional): Hidden expand dimension inside the feed-forward network.
        dropout (float): Dropout connectivity factor applied over internal sequences. Defaults to 0.1.
        activation (str): Non-linear activation method ("gelu", "relu", or "silu"). Defaults to "gelu".
        init_values (float): Initial values of the scaling coefficients. Defaults to 1e-2.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        dropout: float = 0.1, 
        activation: str = "gelu", 
        init_values: float = 1e-2,
    ):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_prob = dropout
        self.activation_name = activation
        
        self.attention_block = ResidualAttention(d_model, n_heads, dropout, init_values)
        self.ffn_block = FFNResidualBlock(d_model, d_ff, activation, dropout, init_values)

        logger.info(f"Initialized {self}")
    

    def __str__(self) -> str:
        return (f"EncoderLayer(d_model={self.d_model}, n_heads={self.n_heads}, d_ff={self.d_ff}, dropout={self.dropout_prob}, activation={self.activation_name})")


    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, S, D]
            attn_mask (object, optional): Masking configuration structures.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape [B, S, D].
                - Collected attention score mapping matrix from this layer's execution pass.
        """
        # Unwrap internal tensor if passed an instance of TriangularCausalMask
        mask_tensor = attn_mask.mask if hasattr(attn_mask, 'mask') else attn_mask
        
        x, attn = self.attention_block(x, mask_tensor)
        x = self.ffn_block(x)
        
        return x, attn



class Encoder(nn.Module):
    """
    Transformer Encoder stack composing multiple cascaded attention layers.

    Args:
        attn_layers (list of nn.Module): Stack of sequential EncoderLayer blocks.
        norm_layer (nn.Module, optional): Final normalization block applied at the end of the stack.
    """
    def __init__(self, attn_layers: list, norm_layer: nn.Module = None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

        if self.norm is not None:
            init_norm(self.norm)


    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, S, D]
            attn_mask (torch.Tensor, optional): Masking configuration structures.
        
        Returns: 
            tuple[torch.Tensor, list[torch.Tensor]]:
                - Output tensor of shape [B, S, D]
                - Collected attention matrices list from each layer in the pipeline.
        """
        attns = []

        # Loop through each EncoderLayer
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask)
            attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)

        return x, attns



class PatchTST(BaseNNBackendModel):
    """
    PatchTST model modified for Multi-Parameter Regression.
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
        e_layers (int): Number of encoder blocks. Defaults to 2.
        att_heads (int): Number of attention heads for temporal aggregation. Defaults to 4.
        dropout (float): Dropout regularization factor. Defaults to 0.1.
        activation (str): Linear activation choice profiles ("relu" or "gelu"). Defaults to "gelu".
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
        e_layers: int = 2,
        att_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        init_values: float = 1e-4,
        channel_independence: bool = True,
        use_instance_norm: bool = True,
    ):
        super().__init__(input_dim, output_dim, seq_len, label_transformer)
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.d_ff = d_ff
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
        self.patch_embedding = MultiScalePatchEmbedding(seq_len, input_dim, d_model, patch_len, stride, dropout, activation, channel_independence, True)

        # Transformer Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(d_model, att_heads, d_ff, dropout, activation, init_values) for _ in range(e_layers)
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
        return (f"PatchTST(\n"
                f"  input_dim={self.input_dim}, output_dim={self.output_dim}, seq_len={self.seq_len},\n"
                f"  patch_len={self.patch_len}, stride={self.stride}, d_model={self.d_model}, d_ff={self.d_ff}, layers={self.e_layers},\n"
                f"  att_heads={self.att_heads}, dropout={self.dropout_prob}, activation='{self.activation}', init_values={self.init_values},\n"
                f"  channel_independence={self.channel_independence}, use_instance_norm={self.use_instance_norm}\n"
                f")")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input data of shape [B, L, M]
            
        Returns:
            torch.Tensor: Evaluated continuous parameter matrix of shape [B, output_dim]
        """
        B, L, M = x.shape   # Batch, Time Window (seq_len), Features (input_dim)

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

        # Run bidirectional PatchTST encoder with sequence length = N (number of patches)
        enc_out, attns = self.encoder(enc_out) # [B * M, N, D] (Channel Independence) or [B, N, D] (Channel Mixing)

        # Attentional Temporal Pooling across patch dimension
        pooled_channels = self.temporal_pooling(enc_out)   # Pool across the temporal sequence (N): [B * M, N, D] -> [B * M, D] (Channel Independence) or [B, N, D] -> [B, D] (Channel Mixing)
        
        # Channel Scoring & Aggregation
        if self.channel_independence:
            pooled_channels = rearrange(pooled_channels, '(B M) D -> B M D', B=B, M=M)  # [B * M, D] -> [B, M, D]
            pooled_out = self.channel_aggregation(pooled_channels)  # [B, M, D] * [B, M, 1] -> [B, D]
        else:
            # In Channel Mixing, the convolutions have already blended M into D, so features are already shaped as [B, D]
            pooled_out = pooled_channels

        # Re-inject Instance Statistics (if enabled)
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
