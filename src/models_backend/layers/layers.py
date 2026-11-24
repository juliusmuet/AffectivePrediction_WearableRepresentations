import logging
import math

import torch
import torch.nn as nn

from models_backend.layers.layer_utils import init_standard_weights, init_norm, init_mha, get_activation_function


logger = logging.getLogger(__name__)


# Code inspired by: https://github.com/thuml/Time-Series-Library

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


class LayerScale(nn.Module):
    """
    Applies per-channel learnable scaling coefficients (LayerScale) 
    to stabilize deep encoder residual branches.

    Args:
        d_model (int): Size of hidden model representations (D).
        init_value (float): Initial values of the scaling coefficients. Defaults to 1e-2.
    """
    def __init__(self, d_model: int, init_value: float = 1e-2):
        super().__init__()
        self.d_model = d_model
        self.init_value = init_value
        self.gamma = nn.Parameter(init_value * torch.ones(d_model))

        logger.info(f"Initialized {self}")

    
    def __str__(self) -> str:
        return f"LayerScale(d_model={self.d_model}, init_value={self.init_value})"


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Output from a process block of shape [B, S, D]
        
        Returns:
            torch.Tensor: Scaled output from a process block of shape [B, S, D]
        """
        # Element-wise broadcasting multiplication along the D channel dimension
        return x * self.gamma



class PositionalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding module for Time Series Patches.
    Computes static, non-learnable multi-dimensional sinusoidal wave encodings 
    across sequence positions to preserve temporal order information.

    Args:
        d_model (int): Size of hidden model representations (D).
        max_len (int): Upper bound allocation for sequence length padding (S).
    """
    def __init__(self, d_model: int, max_len: int):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model).float()  # position-by-feature matrix [max_len, d_model]
        pe.requires_grad = False    # autograd tracking disabled since these weights are static

        # Create a sequence of position indices from 0 up to max_len - 1.
        position = torch.arange(0, max_len).float().unsqueeze(1)    # [max_len, 1]

        # Compute the geometric progression scaling factors (frequencies) for the sinusoidal waves.
        # Targets only the even dimension indices (0, 2, ... up to d_model) in log space to ensure numerical stability before applying the exponential expansion
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # Fill even feature dimensions (0, 2, 4, ...) with sine waves across all positions.
        pe[:, 0::2] = torch.sin(position * div_term)    # [max_len, 1] * [d_model // 2] -> [max_len, d_model // 2]

        # Fill odd feature dimensions (1, 3, 5, ...) with cosine waves across all positions
        pe[:, 1::2] = torch.cos(position * div_term)    # [max_len, 1] * [d_model // 2] -> [max_len, d_model // 2]

        # Prepend a batch dimension slot so the matrix shape aligns for tensor addition
        pe = pe.unsqueeze(0)    # [max_len, d_model] -> [1, max_len, d_model]
        
        # Register the embedding matrix as a persistent model buffer.
        # This keeps it out of optimizer parameters while ensuring it automatically migrates 
        # to the correct device (CPU/GPU) alongside the model during state transfers.
        self.register_buffer('pe', pe)

        logger.info(f"Initialized {self}")


    def __str__(self) -> str:
        return (f"PositionalEmbedding(d_model={self.d_model}, max_len={self.max_len})")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence tensors of shape [B, S, D]
            
        Returns:
            torch.Tensor: Static positional encoding matrix sliced to shape [1, S, D]
        """
        return self.pe[:, :x.size(1)]



class MultiScalePatchEmbedding(nn.Module):
    """
    Multi-Scale Gated Patch Embedding for long-context time-series.
    Extracts features across three temporal granularities (Fine, Medium, Coarse),
    fuses them via a Gated Linear Unit (SwiGLU) variant.

    Supports both Channel Independence (weight-shared 1D Convolutions) and 
    Channel Mixing (Factorized Depthwise-Separable Convolutions) paradigms.

    Args:
        seq_len (int): Size of lookback time window (L).
        in_channels (int): Number of input channels (M).
        d_model (int): Size of hidden model representations (D).
        patch_len (int): Length of each patch (P).
        stride (int): Stepping offset increment used across windows.
        dropout (float): Dropout connection regularizer probability.
        activation (str): Non-linear activation method for the channel mixing case ("gelu" or "silu"). Defaults to "silu".
        channel_independence (bool): If True, treats channels independently (ouput shape [B * M, N, D]).
                                     If False, mixes channels via convolution (ouput shape [B, N, D]).
                                     Defaults to True.
        use_pos_embed (bool): If True, adds static sinusoidal positional encodings to the fused multi-scale patch representations. Defaults to False.
    """
    def __init__(
        self, 
        seq_len: int, 
        in_channels: int, 
        d_model: int, 
        patch_len: int, 
        stride: int, 
        dropout: float, 
        activation: str = "silu", 
        channel_independence: bool = True, 
        use_pos_embed: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.dropout_prob = dropout
        self.activation = activation
        self.channel_independence = channel_independence
        self.use_pos_embed = use_pos_embed

        # Determine padding for stride
        if seq_len % stride == 0:
            self.process_layer = nn.Identity()
        else:
            padding_length = stride - (seq_len % stride)
            self.process_layer = nn.ReflectionPad1d((0, padding_length))

        # Multi-scale patch branches: different receptive fields, identical stride and exact padding to match sequence length N
        k_fine = patch_len // 2
        k_med = patch_len
        k_coarse = patch_len * 2
        
        # Exact padding math: Padding = (Kernel - Stride) // 2
        p_fine = (k_fine - stride) // 2 if k_fine > stride else 0
        p_med = (k_med - stride) // 2 if k_med > stride else 0
        p_coarse = (k_coarse - stride) // 2 if k_coarse > stride else 0

        # Define multi-scale branches dynamically based on independence paradigm
        if self.channel_independence:
            # --------------------------------------------------------------------
            # Channel Independence Mode: 1D Convolutions over [B * M, 1, L]
            # No pointwise layer needed; Conv1d projects directly from 1 -> d_model // 2
            # --------------------------------------------------------------------
            self.fine_branch = nn.Conv1d(in_channels=1, out_channels=d_model // 2, kernel_size=k_fine, stride=stride, padding=p_fine)
            self.med_branch = nn.Conv1d(in_channels=1, out_channels=d_model // 2, kernel_size=k_med, stride=stride, padding=p_med)
            self.coarse_branch = nn.Conv1d(in_channels=1, out_channels=d_model // 2, kernel_size=k_coarse, stride=stride, padding=p_coarse)
        else:
            # --------------------------------------------------------------------
            # Channel Mixing Mode: Factorized Depthwise-Separable Convolutions over [B, M, L]
            # Step 1: Depthwise temporal filtering per channel (groups = M)
            # Step 2: Pointwise 1x1 channel fusion across channels (M -> d_model // 2)
            # --------------------------------------------------------------------
            activation_layer = get_activation_function(activation)
            
            self.fine_branch = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=k_fine, stride=stride, padding=p_fine, groups=in_channels),
                nn.GroupNorm(num_groups=in_channels, num_channels=in_channels), 
                activation_layer,
                nn.Conv1d(in_channels, d_model // 2, kernel_size=1)
            )
            self.med_branch = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=k_med, stride=stride, padding=p_med, groups=in_channels),
                nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
                activation_layer,
                nn.Conv1d(in_channels, d_model // 2, kernel_size=1)
            )
            self.coarse_branch = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=k_coarse, stride=stride, padding=p_coarse, groups=in_channels),
                nn.GroupNorm(num_groups=in_channels, num_channels=in_channels),
                activation_layer,
                nn.Conv1d(in_channels, d_model // 2, kernel_size=1)
            )

        # Gated Feature Fusion (SwiGLU Variant): dynamic selection between multi-scale representations
        self.norm = nn.LayerNorm((d_model // 2) * 3)
        self.gate_proj = nn.Linear((d_model // 2) * 3, d_model)
        self.value_proj = nn.Linear((d_model // 2) * 3, d_model)
        self.act = nn.SiLU() 
        
        # Allocate optional sinusoidal positional space dynamically matching convolution length metrics
        if self.use_pos_embed:
            dummy_len = seq_len if seq_len % stride == 0 else seq_len + (stride - (seq_len % stride))
            max_patches = ((dummy_len - patch_len) // stride) + 1
            self.pos_embedding = PositionalEmbedding(d_model=d_model, max_len=max_patches + 10)
        else:
            self.pos_embedding = None

        self.dropout = nn.Dropout(dropout)

        # Apply layer weights
        self.apply(init_standard_weights)

        logger.info(f"Initialized {self}")
    

    def __str__(self) -> str:
        return (f"MultiScalePatchEmbedding(seq_len={self.seq_len}, in_channels={self.in_channels}, d_model={self.d_model}, "
                f"patch_len={self.patch_len}, stride={self.stride}, dropout={self.dropout_prob}, activation={self.activation}, "
                f"channel_independence={self.channel_independence}, use_pos_embed={self.use_pos_embed}, process_layer={self.process_layer})")


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x (torch.Tensor): Input signal of shape [B, M, L]
            
        Returns:
            tuple[torch.Tensor, int, int]:
                - Embedded multi-scale patch tensor of shape
                  [B * M, N, D] if channel_independence=True
                  [B, N, D] if channel_independence=False
                - Number of variates/channels (M)
                - Number of patches (N)
        """
        # It is assumed that x is already normalised

        B, M, _ = x.shape
        x = self.process_layer(x)  # Tail alignment [B, M, L_processed]

        # Route tensor shape based on independence flag
        if self.channel_independence:
            # Reshape to treat each channel as a single-channel 1D signal: [B * M, 1, L_processed]
            x_input = x.reshape(B * M, 1, -1)
        else:
            # Convolutions naturally interpret M as input channels: [B, M, L_processed]
            x_input = x

        # Extract features across scales
        feat_fine = self.fine_branch(x_input)      # [B * M, D//2, N] OR [B, D//2, N]
        feat_med = self.med_branch(x_input)        # [B * M, D//2, N] OR [B, D//2, N]
        feat_coarse = self.coarse_branch(x_input)  # [B * M, D//2, N] OR [B, D//2, N]

        # Force exact sequence alignment if sequence lengths differ by ±1 due to odd divisions
        N_target = feat_med.shape[-1]
        if feat_fine.shape[-1] != N_target:
            feat_fine = torch.nn.functional.interpolate(feat_fine, size=N_target, mode='linear', align_corners=False)
            logger.warning(f"feat_fine does not match target length. Got {feat_fine.shape[-1]}, need {N_target}")
        if feat_coarse.shape[-1] != N_target:
            feat_coarse = torch.nn.functional.interpolate(feat_coarse, size=N_target, mode='linear', align_corners=False)
            logger.warning(f"feat_coarse does not match target length. Got {feat_coarse.shape[-1]}, need {N_target}")

        # Concatenate features along the channel dimension: [B * M, (D//2)*3, N] or [B, (D//2)*3, N]
        fused = torch.cat([feat_fine, feat_med, feat_coarse], dim=1)
        
        # Transpose to layout: [B * M, N, (D//2)*3] or [B, N, (D//2)*3]
        fused = fused.transpose(1, 2)

        # SwiGLU Gated Fusion Step
        fused_norm = self.norm(fused)
        gate = self.act(self.gate_proj(fused_norm))
        value = self.value_proj(fused_norm)
        output = value * gate   # [B * M, N, d_model] or [B, N, d_model]

        # Inject positional context if enabled
        if self.pos_embedding is not None:
            output = output + self.pos_embedding(output)
        
        return self.dropout(output), M, N_target



class FFNResidualBlock(nn.Module):
    """
    Position-Wise Feed-Forward Network Block implemented via 1D Convolutions.
    Transforms individual sequence embeddings via standard projection mappings 
    under Pre-Layer Normalization and LayerScale tuning configurations.

    Args:
        d_model (int): Inner representation depth dimensions (D).
        d_ff (int): Expansion hidden dimension boundary inside FFN layer feeds.
        activation (str): Non-linear activation method ("gelu" or "silu"). Defaults to "silu".
        dropout (float): Dropout connection regularizer probability. Defaults to 0.1.
        init_value (float): Initial value assignment for LayerScale coefficients. Defaults to 1e-2.
    """
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        activation: str = "silu", 
        dropout: float = 0.1, 
        init_value: float = 1e-2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation_name = activation.lower()
        self.dropout_prob = dropout
        self.init_value = init_value

        self.norm = nn.LayerNorm(d_model)

        activation_layer = get_activation_function(activation)

        self.ffn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
            activation_layer,
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        )

        self.ls = LayerScale(d_model, init_value)

        # Apply layer weights
        self.apply(init_standard_weights)
    
        logger.info(f"Initialized {self}")
    

    def __str__(self) -> str:
        return (f"FFNResidualBlock(d_model={self.d_model}, d_ff={self.d_ff}, activation={self.activation_name}, dropout={self.dropout_prob}, init_value={self.init_value})")
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence states of shape [B, S, D]

        Returns:
            torch.Tensor: Projected feed-forward sequence spaces of shape [B, S, D]
        """
        # Pre-Normalization
        x_norm = self.norm(x)
        
        # FFN processing
        processed = self.ffn(x_norm.transpose(1, 2)).transpose(1, 2)    # [B, D, S]

        # Apply LayerScale to the attention output before the residual add
        return x + self.ls(processed)  # [B, S, D]



class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-Head Attention Temporal Pooling Head implemented via native nn.MultiheadAttention.
    Compresses a variable or temporal sequence dimension into a single sequence-agnostic vector representation
    by learning a global context query token that dynamically aggregates information across time steps.

    Args:
        query_dim (int): Dimensionality of the input features per time step.
        num_heads (int, optional): Number of parallel attention heads. Must divide `query_dim` evenly. Defaults to 4.
        dropout (float, optional): Dropout probability applied to the attention matrix weights. Defaults to 0.1.
        init_values (float, optional): Initial scaling factor for LayerScale. Defaults to 1e-2.
    """
    def __init__(
        self, 
        query_dim: int, 
        num_heads: int = 4, 
        dropout: float =0.1, 
        init_values: float = 1e-2,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout
        self.init_values = init_values
        
        self.norm = nn.LayerNorm(query_dim)

        # Native PyTorch MultiheadAttention
        self.mha = nn.MultiheadAttention(
            embed_dim=query_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True    # ensures input and output tensor formats remain (B, Seq_Len, D)
        )
        
        # A learnable global context token serving as the target query sequence (1 token, D features)
        self.static_query = nn.Parameter(torch.randn(1, 1, query_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.ls = LayerScale(query_dim, init_values)

        # Apply standard norm init
        init_norm(self.norm)
        # Apply targeted MHA init
        init_mha(self.mha)
        # Apply custom parameter init
        nn.init.trunc_normal_(self.static_query, mean=0.0, std=0.02)

        logger.info(f"Initialized {self}")

    
    def __str__(self) -> str:
        return f"MultiHeadAttentionPooling(query_dim={self.query_dim}, num_heads={self.num_heads}, dropout={self.dropout_prob}, init_values={self.init_values})"


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Combined flattened spatial-temporal sequences of shape [B, S, D]

        Returns:
            torch.Tensor: Compressed global context feature vector of shape [B, D]
        """
        b, _, _ = x.size()  # Get batch size

        # Pre-Normalization
        x_norm = self.norm(x)
        
        # Expand static query token to match the current batch size: [1, 1, D] -> [B, 1, D]
        q = self.static_query.expand(b, -1, -1)
        
        # Native MHA forward pass: 
        # - query: q -> [B, 1, D]
        # - key:   x -> [B, S, D]
        # - value: x -> [B, S, D]
        # Returns: 
        # - attn_out:       [B, 1, D]
        # - attn_out_weights: [B, 1, S]
        attn_out, attn_out_weights = self.mha(query=q, key=x_norm, value=x_norm)
        context = attn_out.squeeze(1)   # [B, 1, D] -> [B, D]
        
        # Apply LayerScale and Dropout to the attention output before the residual add
        return x.mean(dim=1) + self.ls(self.dropout(context))



class ContextualChannelAggregation(nn.Module):
    """
    Contextual Channel Aggregation module for Multi-Parameter Time Series.
    Projects temporally pooled channel tokens down to scalar values and learns 
    cross-channel dependency dynamics before computing a weighted reduction.

    Args:
        input_dim (int): Number of input features/channels (M).
        d_model (int): Size of hidden model representations (D).
        activation (str): Non-linear activation method ("gelu" or "silu"). Defaults to "silu".
        dropout (float): Inner block dropout regularization factor. Defaults to 0.2.
        init_values (float, optional): Initial scaling factor for LayerScale. Defaults to 1e-2.
    """
    def __init__(
        self, 
        input_dim: int, 
        d_model: int, 
        activation: str = "silu", 
        dropout: float = 0.2, 
        init_values: float = 1e-2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.activation = activation
        self.dropout_prob = dropout if input_dim > 12 else 0.0  # Disable dropout for small channel dimensions (M <= 12)
        self.init_values = init_values

        activation_layer = get_activation_function(activation)

        self.norm = nn.LayerNorm(d_model)

        # Operates on the channel space across all hidden channels D
        self.channel_interaction = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            activation_layer,
            nn.Dropout(self.dropout_prob),
            nn.Linear(input_dim * 2, input_dim) # Cross-channel relationship mapping
        )
        self.ls1 = LayerScale(d_model, init_values)
        
        # Operates on the feature space [d_model] to distill down to a final weight
        self.channel_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            activation_layer,
            nn.Linear(d_model // 2, 1)  # Outputs 1 scalar score per channel
        )

        self.ls2 = LayerScale(d_model, init_values)

        self.apply(init_standard_weights)

        logger.info(f"Initialized {self}")

    
    def __str__(self) -> str:
        return (f"ContextualChannelAggregation(input_dim={self.input_dim}, d_model={self.d_model}, activation={self.activation}, dropout={self.dropout_prob}, init_values={self.init_values})")
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Temporally pooled tokens of shape [B, M, D]
            
        Returns:
            torch.Tensor: Channel-aggregated context tensor of shape [B, D]
        """
        # Pre-Normalization
        x_norm = self.norm(x)
        
        # Mix channel dynamics
        out_channel_mix = self.channel_interaction(x_norm.transpose(1, 2)).transpose(1, 2)  # [B, M, D] -> [B, D, M] -> [B, M, D]
        mixed_x = x_norm + self.ls1(out_channel_mix)  # [B, M, D]
        
        # Compute per-channel attention weights
        contextual_channel_scores = self.channel_projection(mixed_x).squeeze(-1) # [B, M, D] -> [B, M, 1] -> [B, M]
        channel_attn_weights = torch.softmax(contextual_channel_scores, dim=1).unsqueeze(-1)    # [B, M, 1]

        # Aggregate channels using weighted sum across channels [B, M, D] * [B, M, 1] -> [B, D]
        aggregated = torch.sum(x_norm * channel_attn_weights, dim=1)

        # Apply LayerScale to the attention output before the residual add
        return x.mean(dim=1) + self.ls2(aggregated)



class LinearRegressionHead(nn.Module):
    """
    Multi-Parameter Linear Regression Head for Regression.
    Maps context vectors down to specific regression targets.

    Args:
        output_dim (int): Number of features of the output regression target.
        d_model (int): Size of hidden model representations (D).
        activation (str): Non-linear activation method ("gelu" or "silu"). Defaults to "silu".
        dropout (float): Inner block dropout regularization factor. Defaults to 0.1.
    """
    def __init__(
        self, 
        output_dim: int, 
        d_model: int, 
        activation: str = "silu", 
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.d_model = d_model
        self.activation = activation
        self.dropout_prob = dropout

        activation_layer = get_activation_function(activation)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

        # Apply layer weights
        self.apply(init_standard_weights)

        logger.info(f"Initialized {self}")


    def __str__(self) -> str:
        return (f"LinearRegressionHead(output_dim={self.output_dim}, d_model={self.d_model}, activation = {self.activation}, dropout={self.dropout_prob})")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Channel-aggregated vector of shape [B, D]
            
        Returns:
            torch.Tensor: Evaluated continuous parameter matrix of shape [B, output_dim]
        """
        return self.regression_head(x)



class SmallLinearRegressionHead(nn.Module):
    """
    Simplified Parameter-Efficient Regression Head.
    Directly maps normalized context vectors [B, d_model] to regression targets [B, output_dim].

    Args:
        output_dim (int): Number of target features.
        d_model (int): Hidden dimension size of input representations.
    """
    def __init__(self, output_dim: int, d_model: int):
        super().__init__()
        self.output_dim = output_dim
        self.d_model = d_model

        # Single Linear projection with pre-LayerNorm
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, output_dim)

        logger.info(f"Initialized {self}")


    def __str__(self) -> str:
        return f"LinearRegressionHead(output_dim={self.output_dim}, d_model={self.d_model})"


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Feature representations of shape [B, d_model]
            
        Returns:
            torch.Tensor: Target predictions of shape [B, output_dim]
        """
        x = self.norm(x)
        return self.proj(x)
