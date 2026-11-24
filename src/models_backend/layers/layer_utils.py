import torch.nn as nn


def init_linear(m: nn.Module, std: float = 0.02) -> None:
    """
    Initializes PyTorch `nn.Linear` layers with truncated normal weights and zero biases.

    Args:
        m (nn.Module): Target PyTorch module to evaluate and initialize.
        std (float, optional): Standard deviation scale for truncated normal distribution. Defaults to 0.02.
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_conv(m: nn.Module, std: float = 0.02) -> None:
    """
    Initializes `nn.Conv1d` layers using standard truncated normal for 1x1 convolutions
    or Xavier normal for temporal kernel sizes > 1.

    Args:
        m (nn.Module): Target PyTorch module to evaluate and initialize.
        std (float, optional): Standard deviation scale for truncated normal distribution. Defaults to 0.02.
    """
    if isinstance(m, nn.Conv1d):
        if m.kernel_size == (1,):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std)
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_norm(m: nn.Module) -> None:
    """
    Initializes normalization layers (`nn.LayerNorm` or `nn.GroupNorm`) to identity scale 
    parameters (ones) and zero shift biases.

    Args:
        m (nn.Module): Target PyTorch module to evaluate and initialize.
    """
    if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_mha(mha: nn.MultiheadAttention, std: float = 0.02) -> None:
    """
    Handles sub-projection parameter initialization for PyTorch native `nn.MultiheadAttention`.
    Configures combined or split input projection tensors and output linear weights.

    Args:
        mha (nn.MultiheadAttention): Target attention layer to initialize.
        std (float, optional): Standard deviation scale for truncated normal distribution. Defaults to 0.02.
    """
    if isinstance(mha, nn.MultiheadAttention):
        # Combined projection weights
        if mha.in_proj_weight is not None:
            nn.init.trunc_normal_(mha.in_proj_weight, mean=0.0, std=std)
        else:
            if mha.q_proj_weight is not None:
                nn.init.trunc_normal_(mha.q_proj_weight, mean=0.0, std=std)
            if mha.k_proj_weight is not None:
                nn.init.trunc_normal_(mha.k_proj_weight, mean=0.0, std=std)
            if mha.v_proj_weight is not None:
                nn.init.trunc_normal_(mha.v_proj_weight, mean=0.0, std=std)

        if mha.in_proj_bias is not None:
            nn.init.zeros_(mha.in_proj_bias)

        if mha.out_proj is not None:
            init_linear(mha.out_proj, std=std)


def init_standard_weights(module: nn.Module, std: float = 0.02) -> None:
    """
    Universal recursive initializer for Conv, Linear, and Norm sub-modules.
    Can be passed directly into PyTorch modules via `module.apply(init_standard_weights)`.

    Args:
        module (nn.Module): Root module or sub-layer node to initialize.
        std (float, optional): Standard deviation scale for truncated normal weight distributions. Defaults to 0.02.
    """
    init_linear(module, std=std)
    init_conv(module, std=std)
    init_norm(module)


def get_activation_function(activation: str) -> nn.Module:
    """
    Factory helper mapping string identifier keys to instantiated PyTorch activation function modules.

    Args:
        activation (str): Target activation function name ('gelu', 'silu' or 'relu).

    Returns:
        nn.Module: Instantiated PyTorch activation layer.

    Raises:
        ValueError: If `activation` does not match supported functional choices.
    """
    activation = activation.lower() 
    if activation == "gelu": 
        return nn.GELU() 
    elif activation == "silu": 
        return nn.SiLU()
    elif activation == "relu":
        return nn.ReLU()
    else: 
        raise ValueError(f"Unsupported activation '{activation}'. Choose from: gelu, silu.")
