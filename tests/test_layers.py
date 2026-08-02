"""Test script for GroupedQueryAttention module."""

import os
import sys

import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.config import LLMConfig
from src.models.layers import GroupedQueryAttention
from src.models.position_embedding import RotaryEmbedding


def test_grouped_query_attention() -> None:
    """Test GroupedQueryAttention with various scenarios."""
    print("Testing GroupedQueryAttention...")

    # Create config
    cfg = LLMConfig(
        n_embd=768,
        n_heads=12,
        n_kv_heads=4,  # Grouped Query Attention: 12 query heads, 4 key-value heads
        n_positions=1024,
        dropout=0.0,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        rotary_emb_base=10000.0,
        attn_scaling=1.0,
    )

    # Create model
    gqa = GroupedQueryAttention(cfg)
    print("\nGroupedQueryAttention initialized:")
    print(f"  n_head: {gqa.n_head}")
    print(f"  n_kv_head: {gqa.n_kv_head}")
    print(f"  n_kv_groups: {gqa.n_kv_groups}")
    print(f"  head_dim: {gqa.head_dim}")

    # Create dummy input
    batch_size = 2
    seq_len = 64
    x = torch.randn(batch_size, seq_len, cfg.n_embd)  # (B, seq_len, C)

    # Create rotary embeddings
    rotary_emb = RotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    cos, sin = rotary_emb(position_ids)

    # Test prefill (no cache)
    print("\nTesting prefill mode (no cache)...")
    output, cache = gqa(x, cos, sin, attention_mask=None, block_kv_cache=None)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Cache key shape: {cache['key'].shape}")
    print(f"  Cache value shape: {cache['value'].shape}")

    # Verify output shape
    assert output.shape == x.shape, (
        f"Output shape mismatch: {output.shape} vs {x.shape}"
    )
    assert cache["key"].shape == (
        batch_size,
        cfg.n_kv_heads,
        seq_len,
        cfg.n_embd // cfg.n_heads,
    ), f"Cache key shape mismatch: {cache['key'].shape}"
    print("  ✓ Prefill mode test passed!")

    # Test decode (with cache)
    print("\nTesting decode mode (with cache)...")
    decode_seq_len = 32
    x_decode = torch.randn(batch_size, decode_seq_len, cfg.n_embd)
    decode_pos_ids = (
        torch.arange(seq_len, seq_len + decode_seq_len)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )
    cos_decode, sin_decode = rotary_emb(decode_pos_ids)

    output_decode, cache_update = gqa(
        x_decode, cos_decode, sin_decode, attention_mask=None, block_kv_cache=cache
    )
    print(f"  Decode input shape: {x_decode.shape}")
    print(f"  Decode output shape: {output_decode.shape}")
    print(f"  Updated cache key shape: {cache_update['key'].shape}")

    # Verify output shape
    assert output_decode.shape == x_decode.shape, (
        f"Decode output shape mismatch: {output_decode.shape} vs {x_decode.shape}"
    )
    assert cache_update["key"].shape == (
        batch_size,
        cfg.n_kv_heads,
        seq_len + decode_seq_len,
        cfg.n_embd // cfg.n_heads,
    ), f"Updated cache key shape mismatch: {cache_update['key'].shape}"
    print("  ✓ Decode mode test passed!")

    # Test with attention mask
    print("\nTesting with attention mask...")
    attention_mask = torch.ones(batch_size, seq_len)  # (B, seq_len)
    attention_mask[:, -10:] = 0  # Mask last 10 tokens
    output_masked, _ = gqa(
        x, cos, sin, attention_mask=attention_mask, block_kv_cache=None
    )
    print(f"  Input shape: {x.shape}")
    print(f"  Attention mask shape: {attention_mask.shape}")
    print(f"  Output shape with mask: {output_masked.shape}")
    assert output_masked.shape == x.shape, (
        f"Output shape mismatch with mask: {output_masked.shape} vs {x.shape}"
    )
    print("  ✓ Attention mask test passed!")

    # Test with different configurations
    print("\nTesting with different n_kv_head configurations...")
    cfg2 = LLMConfig(
        n_embd=512,
        n_heads=8,
        n_kv_heads=2,  # 8 query heads, 2 key-value heads (4 groups)
        n_positions=512,
        dropout=0.1,
        rotary_emb_base=10000.0,
        attn_scaling=1.0,
    )
    gqa2 = GroupedQueryAttention(cfg2)
    x2 = torch.randn(1, 32, cfg2.n_embd)
    pos_ids2 = torch.arange(32).unsqueeze(0)
    cos2, sin2 = RotaryEmbedding(cfg2)(pos_ids2)
    output2, _ = gqa2(x2, cos2, sin2, attention_mask=None, block_kv_cache=None)
    assert output2.shape == x2.shape, (
        f"Output shape mismatch: {output2.shape} vs {x2.shape}"
    )
    print("  ✓ Test with n_head=8, n_kv_head=2 passed!")

    print("\n✅ All GroupedQueryAttention tests passed!")


if __name__ == "__main__":
    test_grouped_query_attention()
