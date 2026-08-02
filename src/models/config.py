"""Model configuration classes for LLM and VLM.

This module follows the HuggingFace Transformers configuration pattern:
- Simple dataclass-based design with flat structure
- Standard serialization methods (to_json/from_json)
- Predefined model sizes for quick prototyping
- HuggingFace model compatibility

Example usage:
    # Create config with defaults
    config = LLMConfig()

    # Create config with custom params
    config = LLMConfig(n_embd=1024, n_layers=24)

    # Load from JSON
    config = LLMConfig.from_json("config.json")

    # Save to JSON
    config.save("config.json")

    # Load from HuggingFace model
    config = LLMConfig.from_pretrained("gpt2")
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """Configuration for Large Language Models.

    Follows HuggingFace Transformers style with flat structure.
    All fields are directly accessible as config.field_name.
    """

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    model_type: str = "gpt2"  # Model type identifier
    n_embd: int = 768  # Embedding/hidden dimension
    n_layers: int = 12  # Number of transformer layers
    n_heads: int = 12  # Number of attention heads
    n_kv_heads: int | None = None  # Number of key-value heads (GQA/MQA)
    n_intermediate: int | None = None  # FFN intermediate dimension
    n_positions: int = 1024  # Maximum context length
    vocab_size: int = 50304  # Vocabulary size
    window_pattern: str = "SSSL"  # Window attention pattern

    # -------------------------------------------------------------------------
    # Activation & Normalization
    # -------------------------------------------------------------------------
    activation_function: str = "gelu"  # Activation function
    layer_norm_epsilon: float = 1e-5  # LayerNorm epsilon
    bias: bool = False  # Use bias in linear layers

    # -------------------------------------------------------------------------
    # Dropout (Regularization)
    # -------------------------------------------------------------------------
    dropout: float = 0.0  # General dropout rate
    attn_pdrop: float = 0.1  # Attention dropout
    resid_pdrop: float = 0.1  # Residual dropout
    embd_pdrop: float = 0.1  # Embedding dropout

    # -------------------------------------------------------------------------
    # Attention & Inference
    # -------------------------------------------------------------------------
    use_cache: bool = True  # Enable KV caching
    scale_attn_weights: bool = True  # Scale attention weights
    scale_attn_by_inverse_layer_idx: bool = True  # Scale by 1/layer_idx
    reorder_and_upcast_attn: bool = False  # Upcast attention
    rotary_emb_base: float = 10000.0  # RoPE base frequency
    attn_scaling: float = 1.0  # Additional attention scaling

    # -------------------------------------------------------------------------
    # Special Tokens
    # -------------------------------------------------------------------------
    bos_token_id: int = 50256  # Beginning of sequence token
    eos_token_id: int = 50256  # End of sequence token
    pad_token_id: int | None = None  # Padding token

    # -------------------------------------------------------------------------
    # Architecture Flags
    # -------------------------------------------------------------------------
    add_cross_attention: bool = False  # Enable cross-attention
    tie_word_embeddings: bool = True  # Tie input/output embeddings

    # -------------------------------------------------------------------------
    # Language Model Specific
    # -------------------------------------------------------------------------
    lm_use_tokens: bool = False  # Input is token IDs or embeddings
    lm_tie_weights: bool = True  # Tie LM head weights
    lm_model_type: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # -------------------------------------------------------------------------
    # Pooling (for classification)
    # -------------------------------------------------------------------------
    summary_type: str = "cls_index"  # Pooling strategy
    summary_use_proj: bool = True  # Use projection layer
    summary_activation: str | None = None  # Activation after pooling
    summary_proj_to_labels: bool = True  # Project to label space
    summary_first_dropout: float = 0.1  # Dropout after pooling

    def __post_init__(self) -> None:
        """Set defaults for dependent fields."""
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.n_intermediate is None:
            self.n_intermediate = 4 * self.n_embd

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "LLMConfig":
        """Load configuration from HuggingFace model."""
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name_or_path)
        config = cls()
        config.update_from_hf_config(hf_config)
        return config

    @classmethod
    def from_json(cls, file_path: str) -> "LLMConfig":
        """Load configuration from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMConfig":
        """Create config from dictionary."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, file_path: str | None = None, indent: int = 4) -> str | None:
        """Convert config to JSON string or save to file."""
        data_dict = asdict(self)
        json_str = json.dumps(data_dict, ensure_ascii=False, indent=indent)

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=indent)
            print(f">>> Config saved: {file_path}")
            return None

        return json_str

    def save(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        self.to_json(file_path)

    def update_from_hf_config(self, hf_config: Any) -> None:
        """Update configuration from HuggingFace model config."""
        # hf_config is a transformers.PretrainedConfig; transformers has no
        # type stubs, so it must be treated as a dynamic interface.
        mapping = {
            "hidden_size": "n_embd",
            "intermediate_size": "n_intermediate",
            "num_hidden_layers": "n_layers",
            "num_attention_heads": "n_heads",
            "num_key_value_heads": "n_kv_heads",
            "max_position_embeddings": "n_positions",
            "vocab_size": "vocab_size",
            "hidden_dropout_prob": "dropout",
            "attention_dropout_prob": "attn_pdrop",
            "layer_norm_epsilon": "layer_norm_epsilon",
            "rope_theta": "rotary_emb_base",
            "bos_token_id": "bos_token_id",
            "eos_token_id": "eos_token_id",
            "pad_token_id": "pad_token_id",
        }

        for hf_key, our_key in mapping.items():
            if hasattr(hf_config, hf_key):
                value = getattr(hf_config, hf_key)
                if value is not None:
                    setattr(self, our_key, value)


@dataclass
class VLMConfig(LLMConfig):
    """Configuration for Vision-Language Models.

    Extends LLMConfig with vision-specific parameters.
    """

    # -------------------------------------------------------------------------
    # Vision Encoder
    # -------------------------------------------------------------------------
    vit_model_type: str = "google/siglip2-base-patch16-512"
    n_channels: int = 3  # Input channels (RGB)
    image_size: int = 512  # Input image size
    patch_size: int = 16  # Patch size for ViT

    vit_n_embd: int = 768  # ViT embedding dimension
    vit_n_intermediate: int = 3072  # ViT FFN intermediate dimension
    vit_n_heads: int = 12  # ViT attention heads
    vit_n_layers: int = 12  # ViT layers
    vit_layernorm_eps: float = 1e-6
    vit_pdropout: float = 0.0
    vit_cls_flag: bool = False

    # -------------------------------------------------------------------------
    # Modality Projector
    # -------------------------------------------------------------------------
    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64
    projection_size: int | None = None

    # -------------------------------------------------------------------------
    # Image Processing
    # -------------------------------------------------------------------------
    max_img_size: int = 2048
    resize_to_max_side_len: bool = True

    # -------------------------------------------------------------------------
    # VLM Specific
    # -------------------------------------------------------------------------
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = "checkpoints"
    hf_repo_name: str = "VLM"
    vlm_base_vocab_size: int = 49152

    vlm_extra_tokens: dict[str, str] = field(
        default_factory=lambda: {
            "image_token": "<|image|>",
            "global_image_token": "<|global_image|>",
            **{
                f"r{i}c{j}": f"<row_{i}_col_{j}>"
                for i in range(1, 9)
                for j in range(1, 9)
            },
        }
    )

    def __post_init__(self) -> None:
        """VLM-specific initialization."""
        super().__post_init__()
        # VLM defaults override LLM defaults
        self.lm_model_type = "HuggingFaceTB/SmolLM2-360M-Instruct"
        self.lm_tokenizer = "HuggingFaceTB/SmolLM2-360M-Instruct"
        self.model_type = "vlm"

        # Architecture overrides
        self.n_embd = 960
        self.n_intermediate = 2560
        self.n_layers = 32
        self.n_heads = 15
        self.n_kv_heads = 5
        self.n_positions = 4096

        # Token overrides
        self.vocab_size = self.vlm_base_vocab_size + len(self.vlm_extra_tokens)
        self.pad_token_id = 1
        self.bos_token_id = 49406
        self.eos_token_id = 49407

        # Other overrides
        self.activation_function = "gelu_pytorch_tanh"
        self.layer_norm_epsilon = 1e-6
        self.attn_pdrop = 0.0


# Predefined model configurations
def get_llm_config(model_size: str = "base") -> LLMConfig:
    """Get predefined LLM configuration.

    Args:
        model_size: One of "tiny", "small", "base", "medium", "large", "xl"

    Returns:
        LLMConfig with predefined parameters
    """
    configs = {
        "tiny": LLMConfig(n_embd=512, n_layers=6, n_heads=8, n_positions=1024),
        "small": LLMConfig(n_embd=768, n_layers=12, n_heads=12, n_positions=2048),
        "base": LLMConfig(n_embd=768, n_layers=12, n_heads=12, n_positions=2048),
        "medium": LLMConfig(n_embd=1024, n_layers=24, n_heads=16, n_positions=2048),
        "large": LLMConfig(n_embd=1536, n_layers=24, n_heads=16, n_positions=2048),
        "xl": LLMConfig(n_embd=2048, n_layers=24, n_heads=24, n_positions=4096),
    }
    return configs.get(model_size.lower(), configs["base"])
