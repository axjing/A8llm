import math
from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from src.models.config import LLMConfig
from src.models.position_embedding import apply_rotary_position_embd


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype))


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        of (`int`): The number of output features.
        ix (`int`): The number of input features.
    """

    def __init__(self, ix: int, of: int) -> None:
        super().__init__()

        self.ix = ix
        self.of = of
        self.weight = nn.Parameter(torch.empty(ix, of))
        self.bias = nn.Parameter(torch.zeros(of))
        nn.init.normal_(self.weight, std=0.02)

    def __repr__(self) -> str:
        return "Conv1D(ix={ix}, of={of})".format(**self.__dict__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.of,)
        x = torch.addmm(self.bias, x.contiguous().view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


def scaled_dot_product_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # (如有必要)向下转换回V的数据类型(如果处于混合精度模式)——否则不执行任何操作
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


class LayerNorm(nn.Module):
    """LayerNorm 层"""

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = 1e-5,
        bias: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        self.device = device

        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)

        return o


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Normalizes the input across the last dimension using RMS normalization,
    which scales the input without subtracting the mean. Commonly used as a
    lighter alternative to LayerNorm in transformer models.

    Args:
        cfg: A configuration object containing:
            - lm_hidden_dim (int): The dimensionality of the model hidden states.
            - lm_rms_eps (float): A small constant to avoid division by zero.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.n_embd))
        self.eps = cfg.layer_norm_epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, lm_hidden_dim).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
        """

        # 计算均方根的倒数：对张量逐元素求平方，在隐层维度lm_hidden_dim上求取平均值。
        irms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x * irms * self.weight
        return x


class CausalSelfAttention(nn.Module):
    def __init__(
        self, cfg: LLMConfig, is_cross_attention: bool = False, ctrl_flash: bool = True
    ):
        super().__init__()
        self.is_cross_attention = is_cross_attention

        self.config = cfg
        self.n_embd = cfg.n_embd
        self.n_head = cfg.n_heads
        self.head_dim = self.n_embd // self.n_head
        self.split_size = self.n_embd
        if self.head_dim * self.n_head != self.n_embd:
            raise ValueError(
                f"`n_embd` must be divisible by n_head (got `n_embd`:{self.n_embd} and `n_head`:{self.n_head})"
            )

        if self.is_cross_attention:
            self.c_attn = Conv1D(self.n_embd, 2 * self.n_embd)
            self.q_attn = Conv1D(self.n_embd, self.n_embd)
        else:
            self.c_attn = Conv1D(self.n_embd, 3 * self.n_embd)
        print(f"c_attn {self.c_attn}")
        self.c_proj = Conv1D(self.n_embd, self.n_embd)
        print(f"c_proj {self.c_proj}")
        # regularization
        self.attn_pdrop = cfg.attn_pdrop
        self.resid_pdrop = cfg.resid_pdrop
        self.attn_dropout = nn.Dropout(self.attn_pdrop)
        self.resid_dropout = nn.Dropout(self.resid_pdrop)

        self.ctrl_flash = ctrl_flash
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash or not self.ctrl_flash:
            print(
                "WARNING: scaled dot product attention not available, using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(cfg.n_positions, cfg.n_positions)).view(
                    1, 1, cfg.n_positions, cfg.n_positions
                ),
            )

        # attn scale factor
        self.scaling = self.head_dim ** (-0.5)

    def selfattention_v0(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, q_seq_len, _ = q.size()
        _, _, k_seq_len, _ = k.size()
        scaling = k.size(-1) ** (-0.5)

        # (bsz,n_head,q_seq_len,n_embd)@(bsz,n_head,n_embd,k_seq_len) -> (bsz,n_head,q_seq_len,k_seq_len)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            # ``bias`` is a lazily registered causal-mask buffer, so the torch
            # stubs type it ``Tensor | Module``; narrow before indexing.
            bias = cast(torch.Tensor, self.bias)
            attn_weights = attn_weights.masked_fill(
                bias[:, :, :q_seq_len, :k_seq_len] == 0, -float("inf")
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # (bsz,n_head,q_seq_len,k_seq_len)@(bsz,n_head,v_seq_len,n_embd) -> (bsz,n_head,q_seq_len,n_embd)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len, n_embd = x.size()  # batch_size, seq_len, n_embd

        # 为该批次注意力头计算q,k,v，并将注意力头数n_head的维度前移
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # (bsz,seq_len,n_embd)
        q = q.view(bsz, seq_len, self.n_head, n_embd // self.n_head).transpose(
            1, 2
        )  # (bsz,n_head,seq_len,head_dim)
        k = k.view(bsz, seq_len, self.n_head, n_embd // self.n_head).transpose(
            1, 2
        )  # (bsz,n_head,seq_len,head_dim)
        v = v.view(bsz, seq_len, self.n_head, n_embd // self.n_head).transpose(
            1, 2
        )  # (bsz,n_head,seq_len,head_dim)

        if self.flash and self.ctrl_flash:
            attn_output = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_pdrop if self.training else 0,
                is_causal=True,
            )
            attn_weights = None
            # print(f"attn_output {attn_output.size()}")
        else:
            attn_output, attn_weights = self.selfattention_v0(q, k, v)

        # print(f"attn_output {attn_output.transpose(1,2).size()}")
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, n_embd)
        )
        # print(f"attn_output1 {attn_output.size()}")
        # output projection
        y = self.c_proj(attn_output)
        y = self.resid_dropout(y)

        return y, attn_weights


class GroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA) as used in some transformer-based language models.

    GQA reduces computation by using fewer key-value heads than query heads,
    grouping multiple query heads to share the same key-value heads.

    Args:
        cfg: Configuration object containing:
            - lm_n_heads (int): Number of query heads.
            - lm_n_kv_heads (int): Number of key-value heads.
            - lm_hidden_dim (int): Hidden embedding dimension.
            - lm_dropout (float): Dropout rate.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        # ``__post_init__`` in LLMConfig fills n_kv_heads in when unset.
        assert cfg.n_kv_heads is not None, "n_kv_heads must be set"
        self.n_head = cfg.n_heads
        self.n_kv_head = cfg.n_kv_heads
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

        assert self.n_head % self.n_kv_head == 0, (
            "n_heads must be divisible by n_kv_heads"
        )
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by num_heads"

        self.n_kv_groups = self.n_head // self.n_kv_head
        self.head_dim = self.n_embd // self.n_head

        self.q_attn = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.k_attn = nn.Linear(self.n_embd, self.head_dim * self.n_kv_head, bias=False)
        self.v_attn = nn.Linear(self.n_embd, self.head_dim * self.n_kv_head, bias=False)

        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

        self.sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.sdpa:
            print(
                "WARNING: scaled dot product attention not available,using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # self.register_buffer("bias", torch.tril(torch.ones(cfg.n_positions, cfg.n_positions))
            #                             .view(1, 1, cfg.n_positions, cfg.n_positions))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        block_kv_cache: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for grouped query attention.

        Args:
            x (Tensor): Input tensor of shape (B, seq_len, C), where
                        B = batch size,
                        seq_len = current sequence length,
                        C = embedding dimension.
            cos (Tensor): Rotary embedding cosines, shape compatible with q and k.
            sin (Tensor): Rotary embedding sines, shape compatible with q and k.
            attention_mask (Tensor, optional): Attention mask tensor of shape (B, total_kv_length),
                                               with 1 for tokens to attend to and 0 for padding.
            block_kv_cache (dict, optional): Cache dict with 'key' and 'value' tensors for autoregressive decoding.

        Returns:
            tuple[Tensor, dict]:
                - Output tensor after attention and projection, shape (B, seq_len, C).
                - Updated block_kv_cache dict for caching key-value states.
        """
        bsz, seq_len, n_embd = (
            x.size()
        )  # seq_len is the sequence length of the current input x

        q_cur = (
            self.q_attn(x)
            .view(bsz, seq_len, self.n_head, n_embd // self.n_head)
            .transpose(1, 2)
        )  # (bsz,n_head,seq_len,head_dim)
        k_cur = (
            self.k_attn(x)
            .view(bsz, seq_len, self.n_kv_head, n_embd // self.n_head)
            .transpose(1, 2)
        )  # (bsz,n_head,seq_len,head_dim)
        v_cur = (
            self.v_attn(x)
            .view(bsz, seq_len, self.n_kv_head, n_embd // self.n_head)
            .transpose(1, 2)
        )  # (bsz,n_head,seq_len,head_dim)

        # Apply rotary embeddings to the current q and k
        q_rotated, k_rotated = apply_rotary_position_embd(q_cur, k_cur, cos, sin)

        # check if we can use cached keys and values
        if block_kv_cache is not None and block_kv_cache["key"] is not None:
            # Concatenate with cached K,V
            # k_rotated and v are for the new token(s)
            k = block_kv_cache["key"]
            v = block_kv_cache["value"]
            k = torch.cat([k, k_rotated], dim=2)
            v = torch.cat([v, v_cur], dim=2)
            block_kv_cache["key"] = k
            block_kv_cache["value"] = v
        else:
            # No cache,this is the first pass(prefill)
            k = k_rotated
            v = v_cur
            block_kv_cache = {"key": k, "value": v}
        # Repeat K, V for Grouped Query Attention
        k_exp = k.repeat_interleave(
            self.n_kv_groups, dim=1
        )  # (bsz, n_head, T_kv, head_dim)
        v_exp = v.repeat_interleave(
            self.n_kv_groups, dim=1
        )  # (bsz, n_heads, T_kv, head_dim)
        T_kv = k_exp.size(2)  # Total sequence length of keys/values

        # Prepare attention mask for SDPA or manual path
        # attention_mask is (B, T_kv_total_length), 1 for attend, 0 for pad
        additive_attn_mask = None
        if attention_mask is not None:
            # The current `attention_mask` parameter is assumed to be `[B, total_sequence_length_kv]`
            # Let's make it `[B, 1, 1, T_kv]` for SDPA.
            mask_for_keys = attention_mask[
                :, :T_kv
            ]  # Ensure mask matches key length [B, T_kv]
            additive_attn_mask = (
                1.0 - mask_for_keys.unsqueeze(1).unsqueeze(2).float()
            ) * torch.finfo(q_rotated.dtype).min
            # This additive_attn_mask shape is [B, 1, 1, T_kv]

        if self.sdpa and x.device.type != "mps":
            # During decode, no additional masking needed as [1, T_kv] is naturally causal
            is_causal = seq_len == T_kv and seq_len > 1
            y = torch.nn.functional.scaled_dot_product_attention(
                q_rotated,
                k_exp,
                v_exp,
                attn_mask=additive_attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Manual attention implementation
            attn = torch.matmul(q_rotated, k_exp.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )  # (bsz, n_head, seq_len, T_kv)
            # During decode: no additional masking needed as [1, T_kv] is naturally causal
            if seq_len == T_kv and seq_len > 1:
                causal_mask_val = torch.tril(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
                ).view(1, 1, seq_len, seq_len)
                attn = attn.masked_fill(~causal_mask_val, float("-inf"))

            if additive_attn_mask is not None:  # Additive padding mask
                # additive_attn_mask is [B,1,1,T_kv], needs to be broadcast to [B, n_heads, seq_len, T_kv]
                attn = attn + additive_attn_mask

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v_exp

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, n_embd)
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y, block_kv_cache


class LlamaMLP(nn.Module):
    """
    ref:https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L160
    Implements the feed-forward network (MLP) block used in transformer-based language models.

    This MLP uses a gated activation mechanism where two separate linear projections
    are applied to the input: one passed through an activation function (gate_proj),
    and the other as is (up_proj). Their element-wise product is then projected back
    to the embedding dimension (down_proj).

    Args:
        cfg: Configuration object containing:
            - lm_hidden_dim (int): The embedding dimension size.
            - lm_inter_dim (int): The intermediate dimension size for the MLP.

    Attributes:
        activation_fn (Callable): The activation function used (SiLU).
        gate_proj (nn.Linear): Linear projection for gating pathway.
        up_proj (nn.Linear): Linear projection for upscaling pathway.
        down_proj (nn.Linear): Linear projection for downscaling back to embedding dim.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.n_embd = cfg.n_embd
        # ``__post_init__`` in LLMConfig fills n_intermediate in when unset.
        assert cfg.n_intermediate is not None, "n_intermediate must be set"
        self.n_inner = cfg.n_intermediate

        self.activation_fn = F.silu  # cfg.activation_function
        self.gate_proj = nn.Linear(self.n_embd, self.n_inner, bias=False)
        self.up_proj = nn.Linear(self.n_embd, self.n_inner, bias=False)
        self.down_proj = nn.Linear(self.n_inner, self.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the gated MLP block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embd_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, embd_dim),
                    after gated MLP transformation.
        """
        gate = self.activation_fn(self.gate_proj(x))
        x = self.up_proj(x)
        x = self.down_proj(gate * x)

        return x


class MLP(nn.Module):
    """MLP 层"""

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()

        self.c_fc = Conv1D(cfg.n_embd, cfg.n_embd * 4)
        self.c_proj = Conv1D(cfg.n_embd * 4, cfg.n_embd)
        self.activate_fn = nn.GELU()
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.activate_fn(x)
        x = self.c_proj(x)
        x = self.resid_dropout(x)
        return x
