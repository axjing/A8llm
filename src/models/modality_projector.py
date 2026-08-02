import torch
from torch import nn

from src.models.config import VLMConfig


class ModalityProjector(nn.Module):
    def __init__(self, cfg: VLMConfig) -> None:
        super().__init__()

        self.input_dim = cfg.vit_n_embd * (cfg.mp_pixel_shuffle_factor**2)
        self.output_dim = cfg.n_embd

        self.scale_factor = cfg.mp_pixel_shuffle_factor

        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        # https://github.com/huggingface/smollm/blob/main/vision/m4/models/vllama3/modeling_vllama3.py#L1281
        bsz, seq_len, n_embd = x.size()
        seq_root = int(seq_len**0.5)
        assert seq_root**2 == seq_len, (
            "Sequence length must be a perfect square for pixel shuffle"
        )
        assert seq_root % self.scale_factor == 0, (
            "Sequence root must be divisible by scale factor"
        )

        height = width = seq_root

        x = x.view(bsz, height, width, n_embd)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor

        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, n_embd)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(bsz, h_out * w_out, n_embd * self.scale_factor**2)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixel_shuffle(x)
        x = self.proj(x)
        return x
