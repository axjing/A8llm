"""Shared utilities for transformer models (GPT and LlamaTransformer).

Extracts common logic that is identical across both architectures:
- Weight initialization
- Parameter counting for scaling laws
- FLOP estimation
- Optimizer setup
"""

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from src.trainer.optim import MuonAdamW


# ---------------------------------------------------------------------------
# Weight initialization

def init_weights(module: nn.Module, *, rms_norm: bool = True,
                 std: float = 0.02, n_layers: int = 1,
                 residual_proj_suffix: str = "c_proj.weight") -> None:
    """Standard transformer weight initialization.

    Matches both GPT (with residual scaling) and LlamaTransformer.

    - ``nn.Linear``: ``N(0, std)`` weight, zero bias
    - ``nn.Embedding``: ``N(0, std)`` weight
    - ``RMSNorm``: weight = 1.0 (only when ``rms_norm=True``)

    After the recursive ``apply``, the caller should separately scale residual
    projection weights (GPT-2 style): iterate ``named_parameters`` and rescale
    any param whose name ends with *residual_proj_suffix*.

    Args:
        module: Module to initialise (passed by ``self.apply``).
        rms_norm: Whether the model uses ``RMSNorm`` (Llama-style) or ``LayerNorm`` (GPT-style).
        std: Standard deviation for the normal init.
        n_layers: Number of transformer layers (used for residual scaling).
        residual_proj_suffix: Name suffix identifying the residual-projection weights to scale.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)

    elif rms_norm and "RMSNorm" in type(module).__name__:
        module.weight.data.fill_(1.0)


def scale_residual_projections(model: nn.Module, n_layers: int,
                               suffix: str = "c_proj.weight",
                               std: float = 0.02) -> None:
    """Apply GPT-2-style residual projection scaling.

    ``std_rescaled = std * (2 * n_layers) ** (-0.5)``

    Args:
        model: Model whose named parameters to scan.
        n_layers: Number of transformer layers.
        suffix: Parameter name suffix to target.
        std: Base standard deviation.
    """
    scaled_std = std * (2 * n_layers) ** (-0.5)
    for _, p in model.named_parameters():
        if p.ndim == 2 and p.requires_grad:
            # Heuristic: scale any 2-D param ending with the suffix
            # (In GPT this is specifically c_proj; LlamaTransformer has none.)
            if _.endswith(suffix):
                nn.init.normal_(p, mean=0.0, std=scaled_std)


# ---------------------------------------------------------------------------
# Parameter counting for scaling laws

def count_scaling_params(*, n_layers: int, n_embd: int, n_heads: int,
                         n_kv_heads: int, n_intermediate: int,
                         vocab_size: int, n_positions: int,
                         bias: bool) -> dict[str, int]:
    """Compute parameter counts by category for scaling laws analysis.

    Works for both GPT (LayerNorm, Conv1D) and LlamaTransformer (RMSNorm, Linear)
    because the parameter counts are determined by the config, not the layer type.

    Args:
        n_layers: Number of transformer blocks.
        n_embd: Hidden dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key-value heads (GQA).
        n_intermediate: FFN intermediate dimension.
        vocab_size: Vocabulary size.
        n_positions: Maximum context length.
        bias: Whether bias is used in linear layers.

    Returns:
        Dict with keys ``transformer_matrices``, ``embeddings``, ``lm_head``, ``total``.
    """
    head_dim = n_embd // n_heads

    # Per-layer parameter counts
    # Attention: QKV + output projection
    qkv = n_embd * (n_embd + 2 * n_kv_heads * head_dim)
    out_proj = n_embd * n_embd

    # MLP: up + down (gate_proj is linear projection too)
    mlp_up = n_embd * n_intermediate
    mlp_down = n_intermediate * n_embd

    # LayerNorm: weight (+ optional bias)
    ln_params = n_embd * (1 + int(bias))
    # Two norms per block
    per_layer = qkv + out_proj + mlp_up + mlp_down + 2 * ln_params

    transformer_matrices = n_layers * per_layer
    embeddings = vocab_size * n_embd + n_positions * n_embd
    lm_head = vocab_size * n_embd  # tied with token embedding, counted separately
    total = transformer_matrices + embeddings + lm_head

    return {
        'transformer_matrices': transformer_matrices,
        'embeddings': embeddings,
        'lm_head': lm_head,
        'total': total,
    }


# ---------------------------------------------------------------------------
# FLOP estimation

def estimate_flops(n_params: int, n_positions: int) -> int:
    """Estimate FLOPs per training iteration (forward + backward).

    Standard transformer estimate: ``6 * N * T``.

    Args:
        n_params: Total parameter count (from ``get_num_params`` or similar).
        n_positions: Context length (``T``).

    Returns:
        Estimated FLOPs per iteration.
    """
    return 6 * n_params * n_positions


# ---------------------------------------------------------------------------
# Optimizer setup helper

def classify_params(model: nn.Module, *,
                    embed_keywords: tuple[str, ...] = ("wte", "wpe", "token_embedding"),
                    head_keywords: tuple[str, ...] = ("lm_head", "head"),
                    min_dim_for_muon: int = 2) -> tuple[list, list, list, list]:
    """Classify a model's parameters into four groups.

    Args:
        model: A ``nn.Module`` with named parameters.
        embed_keywords: Name fragments identifying embedding parameters (AdamW).
        head_keywords: Name fragments identifying lm_head parameters (AdamW).
        min_dim_for_muon: Minimum tensor dimensionality to qualify for Muon.

    Returns:
        ``(embed_params, head_params, scalar_params, matrix_params)`` — lists of
        ``(name, param)`` tuples.
    """
    embed_params: list[tuple[str, nn.Parameter]] = []
    head_params: list[tuple[str, nn.Parameter]] = []
    scalar_params: list[tuple[str, nn.Parameter]] = []
    matrix_params: list[tuple[str, nn.Parameter]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in embed_keywords):
            embed_params.append((name, param))
        elif any(k in name for k in head_keywords):
            head_params.append((name, param))
        elif param.dim() < min_dim_for_muon:
            scalar_params.append((name, param))
        else:
            matrix_params.append((name, param))

    return embed_params, head_params, scalar_params, matrix_params


def build_muon_optimizer(
    model: nn.Module,
    *,
    embed_lr: float = 0.3,
    head_lr: float = 0.008,
    scalar_lr: float = 0.5,
    matrix_lr: float = 0.02,
    weight_decay: float = 0.28,
    momentum: float = 0.95,
    ns_steps: int = 5,
) -> "MuonAdamW":
    """Create a combined MuonAdamW optimizer.

    AdamW is used for embeddings, head, and scalar parameters.
    Muon is used for matrix parameters.

    Args:
        model: The model to optimise.
        embed_lr: Learning rate for embedding parameters.
        head_lr: Learning rate for the LM head.
        scalar_lr: Learning rate for scalar-like parameters.
        matrix_lr: Learning rate for matrix parameters (Muon).
        weight_decay: Cautious weight decay for Muon.
        momentum: Muon momentum.
        ns_steps: Muon Newton-Schulz / Polar Express steps.

    Returns:
        Configured ``MuonAdamW`` optimizer.
    """
    from src.trainer.optim import MuonAdamW

    embed_params, head_params, scalar_params, matrix_params = classify_params(model)

    param_groups = []
    if embed_params:
        param_groups.append({
            'params': [p for _, p in embed_params],
            'kind': 'adamw',
            'lr': embed_lr,
            'betas': (0.9, 0.95),
            'eps': 1e-8,
            'weight_decay': 0.0,
        })
    if head_params:
        param_groups.append({
            'params': [p for _, p in head_params],
            'kind': 'adamw',
            'lr': head_lr,
            'betas': (0.9, 0.95),
            'eps': 1e-8,
            'weight_decay': weight_decay,
        })
    if scalar_params:
        param_groups.append({
            'params': [p for _, p in scalar_params],
            'kind': 'adamw',
            'lr': scalar_lr,
            'betas': (0.9, 0.95),
            'eps': 1e-8,
            'weight_decay': weight_decay,
        })
    if matrix_params:
        param_groups.append({
            'params': [p for _, p in matrix_params],
            'kind': 'muon',
            'lr': matrix_lr,
            'momentum': momentum,
            'ns_steps': ns_steps,
            'beta2': 0.0,
            'weight_decay': weight_decay,
        })

    return MuonAdamW(param_groups)
