import logging
import math

import torch
from torch import nn

from models_backend.base_backend_model import BaseNNBackendModel
from models_backend.layers.layers import MultiHeadAttentionPooling, ContextualChannelAggregation, LinearRegressionHead
from explainability.shap_strategies import LongSeqNeuralSHAPStrategy


logger = logging.getLogger(__name__)


# Code inspired by: https://github.com/luodhhh/ModernTCN/blob/main/ModernTCN-classification/models/ModernTCN.py

"""
Dimension Legend
B: Batch Size (number of samples processed concurrently)
M: Num Variates / Channels / Features (e.g., self.input_dim)
L: Raw Sequence Length / Lookback Window without patching (e.g., self.seq_len)
N: Num Patches (the number of sliced window segments, calculated as self.patch_num)
D: d_model (the embedding projection dimension, e.g., self.d_model)
"""


def init_moderntcn_weights(module: nn.Module, std: float = 0.02) -> None:
    """
    Weight initialization routine optimized for ModernTCN.
    
    - Linear: Truncated normal (std=0.02), zero bias.
    - Conv1d (1x1 Pointwise & Depthwise): Truncated normal (std=0.02), zero bias.
    - Conv1d (Standard Temporal/Spatial > 1): Kaiming Normal (fan_out), zero bias.
    - Norms (BatchNorm1d, LayerNorm, GroupNorm): Identity scale (1.0), zero bias.

    Args:
        module (nn.Module): Target PyTorch module.
        std (float): Standard deviation scale for truncated normal distribution. Defaults to 0.02.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Conv1d):
        # 1x1 Pointwise Convolutions
        if module.kernel_size == (1,):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        # Depthwise Convolutions (groups == in_channels)
        elif module.groups == module.in_channels and module.in_channels > 1:
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        # Standard Spatial/Temporal Convolutions
        else:
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)



class LargeKernelConv(nn.Module):
    """
    Large Kernel 1D Depthwise Convolution block. Leverages parallel large and small kernel branches for multi-scale temporal context extraction.

    Args:
        in_channels (int): Input feature channels (nvars * dmodel).
        out_channels (int): Output feature channels (nvars * dmodel).
        kernel_size (int): Large kernel size (K_large).
        stride (int): Temporal convolution stride step.
        groups (int): Depthwise grouping parameter (nvars * dmodel).
        small_kernel (int): Small parallel kernel size (K_small).
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int, 
        groups: int, 
        small_kernel: int,
    ):
        super(LargeKernelConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.small_kernel = small_kernel

        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2

        # Dual multi-scale training branches
        self.lkb_origin = self._conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
        if small_kernel is not None:
            assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
            self.small_conv = self._conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                        stride=stride, padding=small_kernel // 2, groups=groups, dilation=1, bias=False)

        logger.info(f"Initialized {self}")


    def __str__(self) -> str:
        return (f"LargeKernelConv(in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, groups={self.groups}, small_kernel={self.small_kernel}")
    

    def _conv_bn(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int, 
        padding: int = None, 
        groups: int = 1, 
        dilation: int = 1, 
        bias: bool = False,
    ) -> nn.Sequential:
        """
        Constructs a coupled standard 1D Convolution + Batch Normalization block.

        Args:
            in_channels (int): Input feature depth.
            out_channels (int): Output feature depth.
            kernel_size (int): Temporal filter size.
            stride (int): Downsampling stride.
            padding (int, optional): Spatial boundary padding. Defaults to same padding (k // 2).
            groups (int, optional): Depthwise group split count. Defaults to 1.
            dilation (int, optional): Dilation spacing rate. Defaults to 1.
            bias (bool, optional): If True, enables convolutional bias parameter. Defaults to False.

        Returns:
            nn.Sequential: Sequential Conv1d + BatchNorm1d container module.
        """
        if padding is None:
            padding = kernel_size // 2
        result = nn.Sequential()
        result.add_module('conv', nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
        result.add_module('bn', nn.BatchNorm1d(out_channels))
        return result


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Flattened spatial tensor of shape [B, M * D, N].

        Returns:
            torch.Tensor: Convolved temporal trajectory tensor of shape [B, M * D, N].
        """
        out = self.lkb_origin(inputs)
        if hasattr(self, 'small_conv'):
            out += self.small_conv(inputs)
        return out



class Block(nn.Module):
    """
    ModernTCN Structural Processing Block consisting of:
    1. Depthwise Large-Kernel Reparameterizable Conv1D (Temporal Aggregation)
    2. Batch Normalization over feature spaces
    3. ConvFFN 1: Intra-Variable Channel-Mixer Feed-Forward Network
    4. ConvFFN 2: Inter-Variable Cross-Channel Mixer Feed-Forward Network
    5. Residual skip connection across the block

    Args:
        large_size (int): Primary large kernel convolution size.
        small_size (int): Secondary small kernel convolution size.
        dmodel (int): Hidden representation dimension per variable (D).
        dff (int): Expanded feed-forward inner representation dimension (D_ff).
        nvars (int): Total number of multivariate input variables (M).
        drop (float, optional): Regularization dropout probability. Defaults to 0.1.
    """
    def __init__(
        self, 
        large_size: int, 
        small_size: int, 
        dmodel: int, 
        dff: int,
        nvars: int, 
        drop: float = 0.1,
    ):
        super(Block, self).__init__()

        self.large_size = large_size
        self.small_size = small_size
        self.dmodel = dmodel
        self.dff = dff
        self.nvars = nvars
        self.drop = drop
        
        # 1. Depthwise Temporal Conv Layer
        self.dw = LargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel, kernel_size=large_size, stride=1, groups=nvars * dmodel, small_kernel=small_size)
        self.norm = nn.BatchNorm1d(dmodel)

        # 2. ConvFFN 1: Intra-Variable Feature Mixing (groups = nvars)
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1, padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1, padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # 3. ConvFFN 2: Inter-Variable Cross-Channel Mixing (groups = dmodel)
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1, padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1, padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel

        logger.info(f"Initialized {self}")


    def __str__(self) -> str:
        return (f"Block(large_size={self.large_size}, small_size={self.small_size}, dmodel={self.dmodel}, dff={self.dff}, "
                f"ffn_ratio={self.ffn_ratio}, nvars={self.nvars}, drop={self.drop})")
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, M, D, N].

        Returns:
            torch.Tensor: Processed tensor with residual addition of shape [B, M, D, N].
        """
        input_x = x
        B, M, D, N = x.shape

        # Step 1: Depthwise Large-Kernel Temporal Convolution
        x = x.reshape(B, M * D, N)  # [B, M * D, N]
        x = self.dw(x)  # [B, M * D, N]
        
        # Step 2: Batch Normalization along feature dimension D
        x = x.reshape(B, M, D, N)   # [B, M, D, N]
        x = x.reshape(B * M, D, N)  # [B * M, D, N]
        x = self.norm(x)    # [B * M, D, N]
        x = x.reshape(B, M, D, N)   # [B, M, D, N]
        x = x.reshape(B, M * D, N)  # [B, M * D, N]

        # Step 3: ConvFFN 1 (Intra-Variable Mixing)
        x = self.ffn1drop1(self.ffn1pw1(x)) # [B, M * D_ff, N]
        x = self.ffn1act(x) # [B, M * D_ff, N]
        x = self.ffn1drop2(self.ffn1pw2(x)) # [B, M * D, N]
        x = x.reshape(B, M, D, N)   # [B, M, D, N]

        # Step 4: ConvFFN 2 (Inter-Variable Cross-Channel Mixing)
        x = x.permute(0, 2, 1, 3)   # [B, D, M, N]
        x = x.reshape(B, D * M, N)  # [B, D * M, N]
        x = self.ffn2drop1(self.ffn2pw1(x)) # [B, D * M_ff, N]
        x = self.ffn2act(x) # [B, D * M_ff, N]
        x = self.ffn2drop2(self.ffn2pw2(x)) # [B, D * M, N]
        x = x.reshape(B, D, M, N)   # [B, D, M, N]
        x = x.permute(0, 2, 1, 3)   # [B, M, D, N]

        # Step 5: Global Residual Skip Connection
        x = input_x + x # [B, M, D, N]
        return x



class Stage(nn.Module):
    """
    Hierarchical ModernTCN backbone stage grouping multiple sequential `Block` layers.

    Args:
        ffn_ratio (int): Expansion factor for FFN hidden inner dimension (dff = dmodel * ffn_ratio).
        num_blocks (int): Total number of cascaded Block layers in this stage.
        large_size (int): Large kernel convolution size for this stage.
        small_size (int): Small kernel convolution size for this stage.
        dmodel (int): Feature depth dimension (D).
        nvars (int): Total count of multivariate target channels (M).
        drop (float, optional): Regularization dropout probability. Defaults to 0.1.
    """
    def __init__(
        self, 
        ffn_ratio: int, 
        num_blocks: int, 
        large_size: int, 
        small_size: int, 
        dmodel: int, 
        nvars: int, 
        drop: float = 0.1,
    ):
        super(Stage, self).__init__()

        self.ffn_ratio = ffn_ratio
        self.num_blocks = num_blocks
        self.large_size = large_size
        self.small_size = small_size
        self.dmodel = dmodel
        self.nvars = nvars
        self.drop = drop

        d_ffn = dmodel * ffn_ratio
        blks = []
        for _ in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

        logger.info(f"Initialized {self}")


    def __str__(self) -> str:
        return (f"Stage(num_blocks={self.num_blocks}, ffn_ratio={self.ffn_ratio}, large_size={self.large_size}, small_size={self.small_size}, "
                f"dmodel={self.dmodel}, nvars={self.nvars}, drop={self.drop})")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Stage input tensor of shape [B, M, D, N].

        Returns:
            torch.Tensor: Stage processed tensor of shape [B, M, D, N].
        """
        for blk in self.blocks:
            x = blk(x)

        return x
    


class ModernTCNEncoder(nn.Module):
    """
    Encoder module for ModernTCN. Performs multi-stage stem patching,
    temporal downsampling, boundary padding, and stage block operations.

    Args:
        patch_len (int): Temporal token patch projection kernel width.
        stride (int): Patch projection stride step size.
        downsample_ratio (int): Spatial downsampling stride factor between stages.
        ffn_ratio (int): ConvFFN hidden expansion factor.
        num_blocks (list): List containing block counts per stage.
        large_size (list): Large conv kernel sizes per stage.
        small_size (list): Small conv kernel sizes per stage.
        dims (list): Feature depth dimensions (d_model) per stage.
        nvars (int): Count of input multivariate channel variables (M).
        dropout (float, optional): Backbone block dropout probability. Defaults to 0.1.
    """
    def __init__(
        self, 
        patch_len: int, 
        stride: int, 
        downsample_ratio: int, 
        ffn_ratio: int, 
        num_blocks: list, 
        large_size: list, 
        small_size: list, 
        dims: list, 
        nvars: int, 
        dropout: float = 0.1,
    ):
        super(ModernTCNEncoder, self).__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.downsample_ratio = downsample_ratio
        self.ffn_ratio = ffn_ratio
        self.num_blocks = num_blocks
        self.large_size = large_size
        self.small_size = small_size
        self.dims = dims
        self.nvars = nvars
        self.dropout = dropout
        self.num_stages = len(num_blocks)

        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Build Stem and Inter-Stage Downsampling / Stage Layers
        for i in range(self.num_stages):
            if i == 0:
                downsample_layer = nn.Sequential(
                    nn.Conv1d(1, dims[0], kernel_size=patch_len, stride=stride),
                    nn.BatchNorm1d(dims[0])
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i - 1]),
                    nn.Conv1d(dims[i - 1], dims[i], kernel_size=downsample_ratio, stride=downsample_ratio),
                )
            self.downsample_layers.append(downsample_layer)

            stage = Stage(ffn_ratio=ffn_ratio, num_blocks=num_blocks[i], large_size=large_size[i], small_size=small_size[i], dmodel=dims[i], nvars=nvars, drop=dropout)
            self.stages.append(stage)

        logger.info(f"Initialized {self}")


    def __str__(self) -> str:
        return (f"ModernTCNEncoder(num_stages={self.num_stages}, patch_len={self.patch_len}, stride={self.stride}, "
                f"downsample_ratio={self.downsample_ratio}, ffn_ratio={self.ffn_ratio}, num_blocks={self.num_blocks}, large_size={self.large_size}, "
                f"small_size={self.small_size}, dims={self.dims}, nvars={self.nvars}, backbone_dropout={self.dropout})")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts multi-stage representation feature maps from raw time-series inputs.

        Args:
            x (torch.Tensor): Input series tensor of shape [B, M, L].

        Returns:
            torch.Tensor: Hierarchical feature map representation of shape [B, M, D_last, N_last].
        """
        B, M, L = x.shape
        x = x.unsqueeze(-2)  # [B, M, 1, L]

        for i in range(self.num_stages):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)  # [B * M, D, N]

            # Boundary Padding Logic
            if i == 0:
                if self.patch_len != self.stride:
                    pad_len = self.patch_len - self.stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)    # [B * M, D, pad_len]
                    x = torch.cat([x, pad], dim=-1) # [B * M, D, L + pad_len]
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)

            # Apply Downsampling / Stem projection
            x = self.downsample_layers[i](x)  # [B * M, D_stage, N_stage]
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)  # [B, M, D_stage, N_stage]

            # Stage Feature Processing
            x = self.stages[i](x)  # [B, M, D_stage, N_stage]

        return x



class ModernTCN(BaseNNBackendModel):
    """
    ModernTCN Architecture for time-series modeling. Features multi-stage hierarchical stem patching,
    downsampling convolutions, variable-agnostic depthwise temporal processing, and reparameterizable kernels.

    Args:
        input_dim (int): Number of features of the input.
        output_dim (int): Number of features of the output.
        seq_len (int): Size of lookback time window.
        label_transformer (BaseTargetTransform): Instance of a label transformer.
        patch_len (int): Temporal token patch projection kernel width. Defaults to 64.
        stride (int): Patch projection stride step size. Defaults to 32.
        downsample_ratio (int): Spatial downsampling stride factor between stages. Defaults to 2.
        ffn_ratio (int): ConvFFN hidden expansion factor. Defaults to 2.
        num_blocks (List[int]): Number of processing Blocks per architectural stage. Defaults to [1, 1].
        large_size (List[int]): Large conv kernel sizes per stage. Defaults to [51, 31].
        small_size (List[int]): Small conv kernel sizes per stage. Defaults to [5, 3].
        dims (List[int]): Feature depth dimensions (d_model) per stage. Defaults to [32, 64].
        dropout (float, optional): Dropout regularization factor. Defaults to 0.1.
        att_heads (int): Number of attention heads for temporal aggregation. Defaults to 4.
        activation (str): Linear activation choice profiles ("silu" or "gelu"). Defaults to "gelu".
        init_values (float): Initial value for layer scale. Defaults to 1e-4.
        use_instance_norm (bool): If True, normalizes local input windows and re-injects statistics into the regression head. Defaults to True.
    """
    def __init__(self, 
        input_dim: int, 
        output_dim: int, 
        seq_len: int, 
        label_transformer,
        patch_len: int = 64, 
        stride: int = 32, 
        downsample_ratio: int = 2, 
        ffn_ratio: int = 2, 
        num_blocks: list = [1, 1], 
        large_size: list = [51, 31], 
        small_size: list = [5, 3], 
        dims: list = [32, 64], 
        dropout: float = 0.1,
        att_heads: int = 4, 
        activation: str = "gelu", 
        init_values: float = 1e-4,
        use_instance_norm: bool = True,
    ):
        super().__init__(input_dim, output_dim, seq_len, label_transformer)
        self.patch_len = patch_len
        self.stride = stride
        self.downsample_ratio = downsample_ratio
        self.ffn_ratio = ffn_ratio
        self.num_blocks = num_blocks
        self.large_size = large_size
        self.small_size = small_size
        self.dims = dims
        self.dropout_prob = dropout
        self.att_heads = att_heads
        self.activation = activation
        self.init_values = init_values
        self.use_instance_norm = use_instance_norm

        # Instance Norm Epsilon to avoid divide-by-zero
        self.eps = 1e-5

        # MondernTCN Feature Extraction Backbone
        self.encoder = ModernTCNEncoder(
            patch_len=patch_len, stride=stride, downsample_ratio=downsample_ratio, ffn_ratio=ffn_ratio, num_blocks=num_blocks, 
            large_size=large_size, small_size=small_size, dims=dims, nvars=input_dim, dropout=dropout
        )

        # Attention Pooling Head to pool across the temporal dimension (N)
        self.temporal_pooling = MultiHeadAttentionPooling(dims[-1], att_heads, dropout, init_values)

       # Channel scoring to aggregate channel dimensions (M)
        self.channel_aggregation = ContextualChannelAggregation(input_dim, dims[-1], activation, dropout, init_values)

        # MLP for output regression
        head_in_dim = dims[-1] + (2 * input_dim) if use_instance_norm else dims[-1]
        self.regression_head = LinearRegressionHead(output_dim, head_in_dim, activation, dropout)

        # Weight initialization only for the Encoder
        # The weights for MultiHeadAttentionPooling, ContextualChannelAggregation and LinearRegressionHead are initialized in the respective modules
        self.encoder.apply(init_moderntcn_weights)
        total_blocks = sum(num_blocks)
        self._apply_residual_scaling(total_blocks)

        logger.info(f"Initialized {self}  \n")


    def __str__(self) -> str:
        return (f"ModernTCN(\n"
                f"  input_dim={self.input_dim}, output_dim={self.output_dim}, seq_len={self.seq_len},\n"
                f"  patch_len={self.patch_len}, stride={self.stride}, downsample_ratio={self.downsample_ratio},\n"
                f"  ffn_ratio={self.ffn_ratio}, num_blocks={self.num_blocks}, large_size={self.large_size},\n"
                f"  small_size={self.small_size}, dims={self.dims}, dropout={self.dropout_prob}, att_heads={self.att_heads},\n"
                f"  activation='{self.activation}', init_values={self.init_values}, use_instance_norm={self.use_instance_norm}\n"
                f")")


    def _apply_residual_scaling(self, total_blocks: int) -> None:
        """
        Scales down the weights of output projections in residual blocks to maintain 
        constant activation variance at initialization.
        """
        if total_blocks == 0:
            return
        scale = 1.0 / math.sqrt(2 * total_blocks)
        with torch.no_grad():
            for stage in self.encoder.stages:
                for block in stage.blocks:
                    block.ffn1pw2.weight.mul_(scale)
                    block.ffn2pw2.weight.mul_(scale)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Raw sequence input tensor of shape [B, L, M].

        Returns:
            torch.Tensor: Output of shape [B, class_num].
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

        x_norm = x_norm.permute(0, 2, 1)  # Permute to [B, M, L] for PyTorch conv dynamics
        
        # Extract features via ModernTCN Encoder
        enc_out = self.encoder(x_norm)  # [B, M, D_last, N_last]
        B, M, D, N = enc_out.shape

        # Attentional Temporal Pooling across patch dimension
        pooled_channels = enc_out.permute(0, 1, 3, 2).reshape(B * M, N, D)  # [B, M, D, N] -> [B * M, N, D]
        pooled_channels = self.temporal_pooling(pooled_channels)            # [B * M, D]

        # Channel Scoring & Aggregation
        pooled_channels = pooled_channels.reshape(B, M, D)                 # [B * M, D] -> [B, M, D]
        pooled_out = self.channel_aggregation(pooled_channels)             # [B, M, D] * [B, M, 1] -> [B, D]

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
