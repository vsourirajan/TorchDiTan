from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class DiffusionModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    image_size: Tuple[int, int] = (32, 32)
    num_classes: int = 1000

    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    patch_size: int = 2

    #for context mode
    # condition_mode: str = "context"
    # norm_type: str = "rmsnorm"

    #for adaLN mode
    condition_mode: str = "adaLN"
    norm_type: str = "np_layernorm_bias"

    input_channels: int = 16 #for cosmos tokens