from dataclasses import dataclass, field


@dataclass
class ssm_780m_config:
    d_model: int = 1536
    d_intermediate: int = 0
    n_layer: int = 48
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=lambda: {"layer": "Mamba2"})
    attn_layer_idx: list[int] = field(default_factory=lambda: [])
    attn_cfg: dict = field(default_factory=lambda: {})
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 16
    tie_embeddings: bool = True


@dataclass
class ssm_attention_2_7b_config:
    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=lambda: {"layer": "Mamba2"})
    attn_layer_idx: list[int] = field(default_factory=lambda: [9, 18, 27, 36, 45, 56])
    attn_cfg: dict = field(
        default_factory=lambda: {
            "causal": True,
            "d_conv": 4,
            "head_dim": 128,
            "num_heads": 30,
            "out_proj_bias": False,
            "qkv_proj_bias": False,
            "rotary_emb_dim": 64,
        }
    )
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 16
    tie_embeddings: bool = True
