"""
用于高效推理模型的引擎。

所有操作均围绕token序列展开：
- 用户可向引擎发送token序列
- 引擎返回下一个token

说明：
- 引擎不涉及任何分词处理，仅处理纯tokenID序列。

整体设计尽可能追求高效。
"""

import ast
import inspect
import re
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch
import torch.nn.functional as F

from xlm.models.config import LLMConfig
from xlm.models.gpt import GPT
from xlm.trainer.distributed import autodetect_device_type, compute_init


# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def _null_context() -> Iterator[None]:
    # kept for API compatibility if needed elsewhere
    yield


def _safe_eval_math(expr: str) -> int | float | None:
    """Safely evaluate simple math expressions using AST.

    Supported operators: +, -, *, /, % and unary +/-. Power (**) and any calls
    or names are disallowed.
    """
    try:
        node = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def _check_and_eval(n: ast.AST) -> int | float | None:
        if isinstance(n, ast.Expression):
            return _check_and_eval(n.body)
        if isinstance(n, ast.BinOp):
            left = _check_and_eval(n.left)
            right = _check_and_eval(n.right)
            if left is None or right is None:
                return None
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.Mod):
                return left % right
            # disallow Pow, FloorDiv, Bitwise ops
            return None
        if isinstance(n, ast.UnaryOp):
            operand = _check_and_eval(n.operand)
            if operand is None:
                return None
            if isinstance(n.op, ast.UAdd):
                return +operand
            if isinstance(n.op, ast.USub):
                return -operand
            return None
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            return None
        # For Python <3.8 compatibility, Num
        if isinstance(n, ast.Num):
            return cast(int | float, n.n)
        # All other node types disallowed
        return None

    try:
        result = _check_and_eval(node)
    except ZeroDivisionError:
        # "1/0", "1%0" etc. are parseable but not evaluable.
        return None
    except OverflowError:
        # e.g. huge constants during arithmetic.
        return None
    return result


_STR_COUNT_RE = re.compile(r"^\s*(['\"])(.*)\1\.count\(\s*(['\"])(.*)\3\s*\)\s*$")


def use_calculator(expr: str) -> int | float | None:
    """Evaluate restricted expressions: simple math or '<str>'.count('<sub>')"""
    if not isinstance(expr, str):
        return None
    # Remove commas from numbers like '1,000'
    expr = expr.replace(",", "")

    # Quick reject dangerous tokens
    low = expr.lower()
    for bad in [
        "__",
        "import",
        "exec",
        "eval",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "getattr",
        "setattr",
    ]:
        if bad in low:
            return None

    # Math expression path: allow digits, operators and parentheses
    if re.fullmatch(r"[0-9+\-*/ %.()]+", expr):
        if "**" in expr:
            return None
        return _safe_eval_math(expr)

    # String.count() path
    m = _STR_COUNT_RE.match(expr)
    if m:
        hay = m.group(2)
        needle = m.group(4)
        try:
            return hay.count(needle)
        except (TypeError, ValueError):
            return None

    return None


# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.batch_size: int = batch_size
        self.max_seq_len: int = seq_len
        self.n_layers: int = num_layers
        self.n_heads: int = num_heads
        self.head_dim: int = head_dim
        # Lazy allocation for cache tensors to avoid huge upfront allocations.
        # `seq_len` is treated as a maximum hint; actual buffers are allocated
        # only when needed via `_ensure_capacity` or during prefill.
        self._device: torch.device = device
        self._dtype: torch.dtype = dtype
        self._alloc_len: int = 0  # current allocated time dimension
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens: torch.Tensor = torch.zeros(
            batch_size, dtype=torch.int32, device=device
        )
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding: torch.Tensor | None = None

        # Heuristic: allocate immediately in the common prefill case when
        # batch_size==1 and seq_len is small, otherwise postpone allocation.
        try:
            hint = int(seq_len)
        except (TypeError, ValueError):
            hint = 0
        if batch_size == 1 and hint > 0 and hint <= 4096:
            self._ensure_capacity(hint)

    def reset(self) -> None:
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def _ensure_capacity(self, required_len: int) -> None:
        """Ensure k_cache/v_cache have capacity for `required_len` sequence length.

        If current allocation is smaller, allocate new tensors with the same
        device/dtype and copy existing contents (if any).
        """
        if required_len <= self._alloc_len:
            return
        # clamp to max_seq_len if provided (>0)
        target = required_len
        if (
            hasattr(self, "max_seq_len")
            and self.max_seq_len is not None
            and self.max_seq_len > 0
        ):
            target = min(target, self.max_seq_len)

        new_shape = (
            self.n_layers,
            self.batch_size,
            target,
            self.n_heads,
            self.head_dim,
        )
        # allocate new buffers
        new_k = torch.empty(new_shape, device=self._device, dtype=self._dtype)
        new_v = torch.empty(new_shape, device=self._device, dtype=self._dtype)
        # initialize to zero for safety
        new_k.zero_()
        new_v.zero_()
        # copy old data if present
        if self.k_cache is not None and self.v_cache is not None:
            old_len = self._alloc_len
            new_k[:, :, :old_len, :, :].copy_(self.k_cache)
            new_v[:, :, :old_len, :, :].copy_(self.v_cache)

        self.k_cache = new_k
        self.v_cache = new_v
        self._alloc_len = target

    def get_pos(self) -> int:
        """Get current position (assumes all batch elements at same position)."""
        return int(self.cache_seqlens[0].item())

    def get_layer_cache(
        self, layer_idx: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return (k_cache, v_cache) views for a specific layer."""
        if self.k_cache is None or self.v_cache is None:
            return None, None
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens: int) -> None:
        """Advance the cache position by num_tokens."""
        # ensure we have room for the advance
        max_pos = int(self.cache_seqlens.max().item() + num_tokens)
        if max_pos > self._alloc_len:
            self._ensure_capacity(max_pos)
        self.cache_seqlens += num_tokens

    def prefill(self, other: "KVCache") -> None:
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert (
            self.n_layers == other.n_layers
            and self.n_heads == other.n_heads
            and self.head_dim == other.head_dim
        )
        other_pos = other.get_pos()
        # ensure capacity for other_pos
        if other_pos > 0:
            self._ensure_capacity(other_pos)
            # copy the underlying tensors
            if self.k_cache is not None and other.k_cache is not None:
                self.k_cache[:, :, :other_pos, :, :].copy_(
                    other.k_cache[:, :, :other_pos, :, :]
                )
            if self.v_cache is not None and other.v_cache is not None:
                self.v_cache[:, :, :other_pos, :, :].copy_(
                    other.v_cache[:, :, :other_pos, :, :]
                )
            self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(
                self.batch_size, -1, -1
            ).clone()


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(
    logits: torch.Tensor,
    rng: torch.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


# -----------------------------------------------------------------------------


class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens: list[int] | None = None) -> None:
        self.current_tokens: list[int] = (
            current_tokens or []
        )  # Current token sequence for this row
        self.forced_tokens: deque[int] = deque()  # Queue of tokens to force inject
        self.in_python_block: bool = False  # Whether we are inside a python block
        self.python_expr_tokens: list[
            int
        ] = []  # Tokens of the current python expression
        self.completed: bool = False  # Whether this row has completed generation


class Engine:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        # model/tokenizer are intentionally Any: the engine dispatches dynamically on
        # forward(config) signatures shared by GPT/LlamaTransformer/HF models and mocks.
        self.model = model
        self.tokenizer = tokenizer  # needed for tool use

    def _get_device(self) -> torch.device:
        """Return the device the model lives on."""
        get_device = getattr(self.model, "get_device", None)
        if callable(get_device):
            device = get_device()
            if isinstance(device, torch.device):
                return device
        param: torch.Tensor | None = next(self.model.parameters(), None)
        if param is not None:
            return param.device
        raise RuntimeError("Unable to determine the model's device")

    def _special_tokens(self) -> dict[str, Any]:
        """Resolve the special tokens used by the tool-use state machine."""
        get_special: Callable[[str], Any] = lambda s: self.tokenizer.encode_special(s)
        return {
            "python_start": get_special("<|python_start|>"),
            "python_end": get_special("<|python_end|>"),
            "output_start": get_special("<|output_start|>"),
            "output_end": get_special("<|output_end|>"),
            "assistant_end": get_special("<|assistant_end|>"),
            "bos": self.tokenizer.get_bos_token_id(),
        }

    def _supports_fa3_cache(self) -> bool:
        """True if the model exposes the FA3-style KVCache forward interface."""
        try:
            if "kv_cache" not in inspect.signature(self.model.forward).parameters:
                return False
            cfg = self.model.config
            return all(
                hasattr(cfg, attr) for attr in ("n_kv_head", "n_layer", "sequence_len")
            )
        except (AttributeError, TypeError, ValueError):
            return False

    def _process_rows(
        self,
        row_states: list[RowState],
        sampled_tokens: list[int],
        special: dict[str, Any],
    ) -> tuple[list[int], list[int]]:
        """Choose the next token for each row and update the tool-use state.

        Args:
            row_states (list[RowState]): Per-row generation state.
            sampled_tokens (list[int]): Token sampled by the model for each row.
            special (dict): Resolved special token ids.

        Returns:
            tuple[list[int], list[int]]: The token column (one token per row) and the
            per-row mask (1 = sampled, 0 = forced).
        """
        token_column = []
        token_masks = []
        for i, state in enumerate(row_states):
            is_forced = len(state.forced_tokens) > 0
            token_masks.append(0 if is_forced else 1)
            next_token = (
                state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
            )
            token_column.append(next_token)
            state.current_tokens.append(next_token)
            # On <|assistant_end|> or <|bos|>, mark the row as completed
            if next_token == special["assistant_end"] or next_token == special["bos"]:
                state.completed = True
            # Handle tool logic
            if next_token == special["python_start"]:
                state.in_python_block = True
                state.python_expr_tokens = []
            elif next_token == special["python_end"] and state.in_python_block:
                state.in_python_block = False
                if state.python_expr_tokens:
                    expr = self.tokenizer.decode(state.python_expr_tokens)
                    result = use_calculator(expr)
                    if result is not None:
                        result_tokens = self.tokenizer.encode(str(result))
                        state.forced_tokens.append(special["output_start"])
                        state.forced_tokens.extend(result_tokens)
                        state.forced_tokens.append(special["output_end"])
                state.python_expr_tokens = []
            elif state.in_python_block:
                state.python_expr_tokens.append(next_token)
        return token_column, token_masks

    @torch.inference_mode()
    def generate(
        self,
        tokens: list[int],
        num_samples: int = 1,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Generate `num_samples` sequences from a shared prompt, one token at a time.

        Uses the model's FA3-style KV cache when available (single prefill, then
        cache replication); otherwise falls back to recomputing the full context,
        which works for models without a native KV cache (e.g. GPT).
        """
        assert isinstance(tokens, list) and tokens and isinstance(tokens[0], int), (
            "expecting list of ints"
        )
        device = self._get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        special = self._special_tokens()

        if self._supports_fa3_cache():
            yield from self._generate_fa3(
                tokens,
                num_samples,
                max_tokens,
                temperature,
                top_k,
                rng,
                device,
                special,
            )
        else:
            yield from self._generate_recompute(
                tokens,
                num_samples,
                max_tokens,
                temperature,
                top_k,
                rng,
                device,
                special,
            )

    def _generate_fa3(
        self,
        tokens: list[int],
        num_samples: int,
        max_tokens: int | None,
        temperature: float,
        top_k: int | None,
        rng: torch.Generator,
        device: torch.device,
        special: dict[str, Any],
    ) -> Iterator[tuple[list[int], list[int]]]:
        """KV-cache generation using the FA3-style KVCache (flash_attn_with_kvcache) interface."""
        # NOTE: cuda -> bfloat16, everything else -> float32. See the previous comment:
        # this repo-wide assumption should eventually be replaced by explicit dtype tracking.
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        m = self.model.config
        kv_model_kwargs = {
            "num_heads": m.n_kv_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer,
        }

        # 1) Run a batch-1 prefill of the prompt tokens
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (
            (len(tokens) + max_tokens)
            if max_tokens is not None
            else self.model.config.sequence_len
        )
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill  # no need to keep this memory around

        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()
            token_column, token_masks = self._process_rows(
                row_states, sampled_tokens, special
            )

            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for the next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(
                1
            )
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[
                :, -1, :
            ]  # (B, vocab_size)

    def _generate_recompute(
        self,
        tokens: list[int],
        num_samples: int,
        max_tokens: int | None,
        temperature: float,
        top_k: int | None,
        rng: torch.Generator,
        device: torch.device,
        special: dict[str, Any],
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Generation fallback for models without a native KV cache (recomputes the full context each step)."""
        n_positions = getattr(self.model.config, "n_positions", None)
        with torch.inference_mode():
            context = (
                torch.tensor([tokens], dtype=torch.long, device=device)
                .expand(num_samples, -1)
                .clone()
            )
            logits, _ = self.model.forward(context)
            logits = logits[:, -1, :]  # (B, vocab_size)

            row_states = [RowState(tokens.copy()) for _ in range(num_samples)]
            num_generated = 0
            while True:
                # Stop condition: we've reached max tokens
                if max_tokens is not None and num_generated >= max_tokens:
                    break
                # Stop condition: all rows are completed
                if all(state.completed for state in row_states):
                    break

                next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()
                token_column, token_masks = self._process_rows(
                    row_states, sampled_tokens, special
                )

                yield token_column, token_masks
                num_generated += 1

                # Append the selected tokens and recompute over the (possibly truncated) context.
                new_ids = torch.tensor(
                    token_column, dtype=torch.long, device=device
                ).unsqueeze(1)
                context = torch.cat([context, new_ids], dim=1)
                if n_positions is not None and context.size(1) > n_positions:
                    context = context[:, -n_positions:]
                logits, _ = self.model.forward(context)
                logits = logits[:, -1, :]  # (B, vocab_size)

    def generate_batch(
        self, tokens: list[int], num_samples: int = 1, **kwargs: Any
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    Smoke test: make sure Engine.generate is equivalent to a naive greedy
    autoregressive decoding loop on a small GPT model.
    """

    class MockTokenizer:
        def __init__(self) -> None:
            self._special = {
                "<|python_start|>": 10001,
                "<|python_end|>": 10002,
                "<|output_start|>": 10003,
                "<|output_end|>": 10004,
                "<|assistant_end|>": 10005,
            }

        def encode_special(self, s: str) -> int:
            return self._special[s]

        def get_bos_token_id(self) -> int:
            return 2

        def encode(self, text: str, prepend: int | None = None) -> list[int]:
            ids = [ord(c) % 256 for c in str(text)]
            if prepend is not None:
                ids = [prepend] + ids
            return ids

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(int(t) % 256) for t in tokens)

    device_type = autodetect_device_type()
    device = compute_init(device_type)[-1]
    cfg = LLMConfig(
        vocab_size=256,
        n_embd=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        n_positions=128,
        bias=True,
    )
    model = GPT(cfg).to(device).eval()
    tokenizer = MockTokenizer()

    prompt_tokens = tokenizer.encode("hello world")
    max_tokens: int = 32
    kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": 0.0}

    engine = Engine(model, tokenizer)
    generated = []
    for token_column, token_masks in engine.generate(
        prompt_tokens, num_samples=1, **kwargs
    ):
        generated.append(token_column[0])

    reference = []
    context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(max_tokens):
            logits, _ = model(context)
            ref_token = int(logits[0, -1].argmax().item())
            reference.append(ref_token)
            if ref_token == tokenizer.get_bos_token_id():
                break
            context = torch.cat(
                [context, torch.tensor([[ref_token]], dtype=torch.long, device=device)],
                dim=1,
            )

    print(f"Engine:    {tokenizer.decode(generated)}")
    print(f"Reference: {tokenizer.decode(reference)}")
    assert generated == reference, f"Mismatch: {generated} != {reference}"
    print("Match: True")
