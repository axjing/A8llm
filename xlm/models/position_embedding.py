from typing import cast

import torch
from torch import nn

from xlm.models.config import LLMConfig


class RotaryEmbedding(nn.Module):
    """
    目前旋转位置编码已有多种衍生变体，本版本是可对上下文长度做线性缩放的基础实现方案
    # ref:https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L190

    计算旋转位置编码，借助角度旋转的方式，在不新增训练参数的前提下为输入序列引入位置依赖以及token位置编号的相对距离信息。

    参数说明：
        cfg：配置对象，包含如下字段：
            - n_embd (int)：隐藏层维度大小
            - n_head (int)：注意力头数量
            - rotary_emb_base (float)：旋转位置编码频率基数
            - n_positions (int)：旋转位置编码支持的最大序列长度
            - attn_scaling (float)：注意力缩放系数
    """

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()

        assert cfg.n_embd % cfg.n_heads == 0, (
            "Hidden dimension must be divisible by number of heads"
        )

        self.head_dim = cfg.n_embd // cfg.n_heads  # 每个head的维度

        self.r_embd_base = cfg.rotary_emb_base
        self.max_seg_len = cfg.n_positions

        # Standard RoPE implementation - create frequencies for each dimension
        # freq_i = 1 / (base^(2i/dim)) where i is the dimension index

        inv_freq = 1.0 / (
            self.r_embd_base
            ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.n_positions
        self.attn_scaling = cfg.attn_scaling

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary positional embeddings (cosine and sine components).

        Args:
            position_ids (torch.Tensor): Tensor of shape (batch_size, seq_len) containing position indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of two tensors (cos, sin), each of shape
                                  (batch_size, seq_len, dim), representing rotary embeddings.
        """

        bsz, seq_len = position_ids.shape

        # Buffers registered at runtime are typed ``Tensor | Module`` by the
        # torch stubs, so narrow explicitly before arithmetic.
        inv_freq = cast(torch.Tensor, self.inv_freq)

        # Dynamic scaling for longer sequences
        # Divide the angle frequency to fit more rotation into the embedding space.
        max_seq = position_ids.max() + 1

        if max_seq > self.original_max_seq_len:
            scale = max_seq / self.original_max_seq_len
            inv_freq = inv_freq / scale

        # Compute theta = position * frequency
        # Flatten position_ids for batch processing
        flatten_position_ids = position_ids.reshape(-1).float()
        # Element-wise outer product: [seq_len] x [dim/2] => [seq_len, dim/2]
        freqs = flatten_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)
        # Reshape to include batch dimension
        freqs = freqs.reshape(bsz, seq_len, -1)

        # Now create interleaved pattern
        emb = torch.cat([freqs, freqs], dim=-1)

        # Compute cos and sin
        cos = torch.cos(emb) * self.attn_scaling
        sin = torch.sin(emb) * self.attn_scaling

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将隐藏维度拆分为两部分，再通过维度互换与取反操作实现输入数据的旋转变换。
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_position_embd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to query and key tensors in attention mechanisms.


    旋转位置嵌入会向query向量和key向量中融入和位置相关的旋转变换，让 Transformer 模型无需额外的显式位置编码，就能高效完成位置信息的编码工作
    Args:
        q (torch.Tensor): Query tensor with shape [batch_size, num_heads, seq_len, head_dim].
        k (torch.Tensor): Key tensor with shape [batch_size, num_heads, seq_len, head_dim].
        cos (torch.Tensor): Precomputed cosine positional embeddings with shape [batch_size, seq_len, head_dim].
        sin (torch.Tensor): Precomputed sine positional embeddings with shape [batch_size, seq_len, head_dim].
        unsqueeze_dim (int, optional): Dimension index to unsqueeze `cos` and `sin` to enable broadcasting.
                                      Defaults to 1 (typically the heads dimension).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors (`q_embed`, `k_embed`),
                                           each with the same shape as the input tensors.

    How it works:
        - `cos` and `sin` tensors are unsqueezed at `unsqueeze_dim` to broadcast across attention heads.
        - Rotary embeddings apply a complex number rotation in the embedding space using:
            rotated = (original * cos) + (rotate_half(original) * sin)
        - `rotate_half` performs a specific half-dimension rotation on the input tensor.
        - This operation encodes relative position information in q and k without adding explicit positional vectors.

    Example:
        q_embed, k_embed = apply_rotary_pos_embd(q, k, cos, sin)
    """
    # We need to make sure cos and sin can be properly broadcast
    # to the shape of q and k by adding the heads dimension
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)  # [batch_size, 1, seq_len, head_dim]

    # Apply complex multiplication:
    # (q * cos) + (rotate_half(q) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
