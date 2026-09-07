import pytest
import torch

from xlm.engine.engine_inference import Engine, KVCache


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
        # simple, deterministic tokenization for test purposes
        return [ord(c) % 256 for c in str(text)]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(int(t) % 256) for t in tokens)


class MockConfig:
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


class MockModel:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.config = MockConfig()
        self.vocab_size = 256

    def get_device(self) -> torch.device:
        return self.device

    def forward(
        self, ids: torch.Tensor, kv_cache: KVCache | None = None
    ) -> torch.Tensor:
        bsz = ids.size(0)
        seq_len = ids.size(1)
        # if KV cache provided, write "fake" k/v entries and advance the position
        if kv_cache is not None:
            pos = kv_cache.get_pos()
            kv_cache._ensure_capacity(pos + seq_len)
            k_cache = kv_cache.k_cache
            v_cache = kv_cache.v_cache
            assert k_cache is not None and v_cache is not None
            # fill with deterministic values so we can assert
            with torch.no_grad():
                for l in range(kv_cache.n_layers):
                    k_cache[l, :, pos : pos + seq_len, :, :].copy_(
                        torch.full(
                            (
                                kv_cache.batch_size,
                                seq_len,
                                kv_cache.n_heads,
                                kv_cache.head_dim,
                            ),
                            float(l + 1),
                            device=k_cache.device,
                            dtype=k_cache.dtype,
                        )
                    )
                    v_cache[l, :, pos : pos + seq_len, :, :].copy_(
                        torch.full(
                            (
                                kv_cache.batch_size,
                                seq_len,
                                kv_cache.n_heads,
                                kv_cache.head_dim,
                            ),
                            float(l + 10),
                            device=v_cache.device,
                            dtype=v_cache.dtype,
                        )
                    )
            kv_cache.cache_seqlens += seq_len

        # Return fake logits
        return torch.randn(bsz, seq_len, self.vocab_size, device=self.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for this test")
def test_kvcache_prefill_and_prefill_copy_gpu() -> None:
    device = torch.device("cuda")
    dtype = torch.float32

    # create source cache (batch=1) and fill it with known values
    xlm = KVCache(
        batch_size=1,
        num_heads=2,
        seq_len=8,
        head_dim=4,
        num_layers=2,
        device=device,
        dtype=dtype,
    )
    xlm._ensure_capacity(5)
    assert xlm.k_cache is not None and xlm.v_cache is not None
    xlm.k_cache.fill_(3.14)
    xlm.v_cache.fill_(2.71)
    xlm.cache_seqlens.fill_(5)

    # destination cache (batch >1)
    dst = KVCache(
        batch_size=4,
        num_heads=2,
        seq_len=16,
        head_dim=4,
        num_layers=2,
        device=device,
        dtype=dtype,
    )
    dst.prefill(xlm)
    assert dst.k_cache is not None and dst.v_cache is not None

    # after prefill, positions should be copied and values present
    assert int(dst.get_pos()) == 5
    # check k/v values copied for first positions
    assert torch.allclose(
        dst.k_cache[:, :, :5, :, :], torch.full_like(dst.k_cache[:, :, :5, :, :], 3.14)
    )
    assert torch.allclose(
        dst.v_cache[:, :, :5, :, :], torch.full_like(dst.v_cache[:, :, :5, :, :], 2.71)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for this test")
def test_engine_generate_integration_gpu() -> None:
    device = torch.device("cuda")
    tokenizer = MockTokenizer()
    model = MockModel(device)
    engine = Engine(model, tokenizer)

    prompt = [1, 2, 3, 4]
    # run generation: 2 samples, deterministic (temperature=0)
    gen = engine.generate(
        prompt, num_samples=2, max_tokens=3, temperature=0.0, top_k=None, seed=123
    )

    count = 0
    for token_column, token_masks in gen:
        # token_column length equals batch size
        assert len(token_column) == 2
        assert all(isinstance(t, int) for t in token_column)
        count += 1
    assert count == 3
