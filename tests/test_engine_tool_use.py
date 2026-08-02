"""CPU tests for Engine.generate_batch at temperature=0 (greedy) and tool-use state machine."""

import torch

from src.engine.engine_inference import Engine, KVCache
from src.models.config import LLMConfig
from src.models.gpt import GPT


class MockTokenizer:
    def __init__(self) -> None:
        self._special_map = {
            "<|python_start|>": 10001,
            "<|python_end|>": 10002,
            "<|output_start|>": 10003,
            "<|output_end|>": 10004,
            "<|assistant_end|>": 10005,
        }
        self._bos = 2

    def encode_special(self, s: str) -> int:
        return self._special_map[s]

    def get_bos_token_id(self) -> int:
        return self._bos

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        ids = [ord(c) % 256 for c in str(text)]
        if prepend is not None:
            ids = [prepend] + ids
        return ids

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(int(t) % 256) for t in tokens)


class ScriptedConfig:
    def __init__(
        self,
        n_kv_head: int = 2,
        n_embd: int = 16,
        n_head: int = 2,
        n_layer: int = 3,
        sequence_len: int = 128,
    ) -> None:
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.sequence_len = sequence_len


class ScriptedModel:
    """Model whose greedy argmax emits a fixed token script (independent of input)."""

    def __init__(
        self, device: torch.device, script: list[int], vocab_size: int = 10006
    ) -> None:
        self.device = device
        self.config = ScriptedConfig()
        self.vocab_size = vocab_size
        self._script = script
        self._call_count = 0

    def get_device(self) -> torch.device:
        return self.device

    def forward(
        self, ids: torch.Tensor, kv_cache: KVCache | None = None
    ) -> torch.Tensor:
        if kv_cache is not None:
            pos = kv_cache.get_pos()
            kv_cache._ensure_capacity(pos + ids.size(1))
            kv_cache.cache_seqlens += ids.size(1)
        token = self._script[min(self._call_count, len(self._script) - 1)]
        self._call_count += 1
        logits = torch.full(
            (ids.size(0), ids.size(1), self.vocab_size), 0.0, device=self.device
        )
        logits[:, :, token] = 100.0
        return logits


def test_generate_batch_tool_use_calculator_at_temperature_zero() -> None:
    """Greedy generation runs the calculator and injects forced <|output_*|> tokens."""
    device = torch.device("cpu")
    tokenizer = MockTokenizer()
    model = ScriptedModel(device, script=[10001, 54, 42, 55, 10002, 10005])
    engine = Engine(model, tokenizer)

    prompt = [1, 2, 3, 4]
    results, masks = engine.generate_batch(
        prompt, num_samples=1, max_tokens=16, temperature=0.0
    )

    expected = prompt + [10001, 54, 42, 55, 10002, 10003, 52, 50, 10004]
    assert results[0] == expected
    # Forced tokens (output_start / result digits / output_end) must be marked with mask 0
    assert masks[0][len(prompt) :] == [1, 1, 1, 1, 1, 0, 0, 0, 0]
    # assistant_end (10005) must not appear in the returned sequence
    assert 10005 not in results[0]


def test_generate_batch_temperature_zero_matches_naive_greedy_gpt() -> None:
    """generate_batch (recompute path) with temperature=0 matches a naive greedy loop on GPT."""
    device = torch.device("cpu")
    tokenizer = MockTokenizer()
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
    engine = Engine(model, tokenizer)

    prompt = tokenizer.encode("hello world")
    max_tokens = 16
    results, _ = engine.generate_batch(
        prompt, num_samples=1, max_tokens=max_tokens, temperature=0.0
    )

    reference = prompt.copy()
    context = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(max_tokens):
            logits, _ = model(context)
            ref_token = int(logits[0, -1].argmax().item())
            reference.append(ref_token)
            context = torch.cat(
                [context, torch.tensor([[ref_token]], dtype=torch.long, device=device)],
                dim=1,
            )

    assert results[0] == reference
