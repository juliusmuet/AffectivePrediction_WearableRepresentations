import logging
from typing import List, Optional

import torch
from rtdl_revisiting_models import FTTransformer as RTDLFTTransformer

from models_backend.base_backend_model import BaseNNBackendModel
from explainability.shap_strategies import ShortSeqNeuralSHAPStrategy


logger = logging.getLogger(__name__)


"""
Dimension Legend
B: Batch Size (number of samples processed concurrently)
S: Sequence Length
M: Num Variates / Channels / Features (e.g., self.input_dim)
"""


class FTTransformer(BaseNNBackendModel):
    """
    Feature Tokenizer Transformer (FT-Transformer) for Multi-Parameter Regression on short lookback windows.

    Args:
        input_dim (int): Number of continuous features per time step (M_cont).
        output_dim (int): Number of target continuous output dimensions.
        seq_len (int): Size of lookback window.
        label_transformer (BaseTargetTransform): Instance of a label transformer.
        cardinality (List[int], optional): Cardinalities for categorical features per time step. Defaults to None.
        d_block (int): Feature embedding vector dimension. Defaults to 32.
        n_blocks (int): Number of Transformer Encoder blocks. Defaults to 2.
        att_heads (int): Number of attention heads. Defaults to 2.
        attention_dropout (float): Attention layer dropout probability. Defaults to 0.1.
        ffn_d_hidden_multiplier (float): FFN hidden size expansion multiplier. Defaults to 2.
        ffn_dropout (float): FFN dropout probability. Defaults to 0.1.
        residual_dropout (float): Residual connection dropout. Defaults to 0.1.
        linformer_kv_compression_ratio (float, optional): Linformer compression factor for high feature counts. Defaults to None.
        linformer_kv_compression_sharing (str, optional): Linformer sharing mode ("headwise" or "layerwise"). Defaults to 'headwise'.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        label_transformer,
        cardinality: List[int] = None,
        d_block: int = 32,
        n_blocks: int = 2,
        att_heads: int = 2,
        attention_dropout: float = 0.1,
        ffn_d_hidden_multiplier: float = 2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        linformer_kv_compression_ratio: Optional[float] = None,
        linformer_kv_compression_sharing: Optional[str] = "headwise",
    ):
        super().__init__(input_dim, output_dim, seq_len, label_transformer)
        self.cardinality = cardinality if cardinality is not None else []
        self.n_cont = input_dim - len(self.cardinality)
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.att_heads = att_heads
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.linformer_kv_compression_ratio = linformer_kv_compression_ratio
        self.linformer_kv_compression_sharing = linformer_kv_compression_sharing

        # Flatten counts across sequence length L
        self.total_cont_flat = seq_len * self.n_cont
        self.flattened_cardinality = self.cardinality * seq_len if len(self.cardinality) > 0 else None

        # Configure FT-Transformer
        rtdl_kwargs = {
            "n_cont_features": self.total_cont_flat,
            "cat_cardinalities": self.flattened_cardinality,
            "d_out": output_dim,  # Directly outputs target regression dimensions
            "n_blocks": n_blocks,
            "d_block": d_block,
            "attention_n_heads": att_heads,
            "attention_dropout": attention_dropout,
            "ffn_d_hidden": None,
            "ffn_d_hidden_multiplier": ffn_d_hidden_multiplier,
            "ffn_dropout": ffn_dropout,
            "residual_dropout": residual_dropout,
        }

        if linformer_kv_compression_ratio is not None:
            rtdl_kwargs["linformer_kv_compression_ratio"] = linformer_kv_compression_ratio
            rtdl_kwargs["linformer_kv_compression_sharing"] = linformer_kv_compression_sharing

        # Initialize FT-Transformer
        self.backbone = RTDLFTTransformer(**rtdl_kwargs)

        logger.info(f"Initialized {self}  \n")

    
    def __str__(self) -> str:
        return (
            f"FTTransformer(\n"
            f"  input_dim={self.input_dim}, output_dim={self.output_dim}, seq_len={self.seq_len},\n"
            f"  n_cat={len(self.cardinality)}, d_block={self.d_block}, n_blocks={self.n_blocks}, att_heads={self.att_heads}\n"
            f"  attention_dropout={self.attention_dropout}, ffn_d_hidden_multiplier={self.ffn_d_hidden_multiplier}\n"
            f"  ffn_dropout={self.ffn_dropout}, residual_dropout={self.residual_dropout}\n"
            f"  linformer_kv_compression_ratio={self.linformer_kv_compression_ratio}, linformer_kv_compression_sharing={self.linformer_kv_compression_sharing}\n"
            f")"
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Combined input [B, L, M_cont + M_cat]

        Returns:
            torch.Tensor: Continuous regression predictions [B, output_dim]
        """
        B, _, _ = x.shape

        # Slice continuous floats -> Flatten [B, L * M_cont]
        x_cont_flat = x[:, :, :self.n_cont].reshape(B, -1).to(torch.float32)

        # Slice categorical indices if present, otherwise pass None
        if self.cardinality:
            x_cat_flat = x[:, :, self.n_cont:].reshape(B, -1).to(torch.int64)   # [B, L * M_cat]
        else:
            x_cat_flat = None

        # Pass through backbone to get raw logits [B, output_dim]
        return self.backbone(x_cont_flat, x_cat_flat)


    def make_parameter_groups(self) -> List[dict]:
        """Exposes paper-accurate parameter groups for AdamW weight decay exclusion."""
        return self.backbone.make_parameter_groups()


    def requires_one_hot_encoding(self) -> bool:
        """Flag identifying whether categorical input features should be one-hot-encoded."""
        return False


    def get_shap_strategy(self) -> ShortSeqNeuralSHAPStrategy:
        """Returns dedicated SHAP explainability strategy."""
        return ShortSeqNeuralSHAPStrategy()
