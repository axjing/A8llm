import inspect
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from xlm.models.config import LLMConfig
from xlm.models.layers import MLP, CausalSelfAttention, LayerNorm
from xlm.models.transformer_utils import (
    build_muon_optimizer,
    count_scaling_params,
    estimate_flops,
    init_weights,
    scale_residual_projections,
)

if TYPE_CHECKING:
    from xlm.trainer.optim import MuonAdamW


class Block(nn.Module):
    """Block 层"""

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, bias=cfg.bias)

        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln_1(x)
        x, _ = self.attn(x)
        x = x + residual

        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class GPT(nn.Module):
    """GPT 模型"""

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__()
        self.config = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embedding
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)  # position embedding
        self.drop = nn.Dropout(cfg.attn_pdrop)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias)
        # self.transformer=nn.ModuleDict(dict(
        #     wte=self.wte,
        #     wpe=self.wpe,
        #     drop=self.drop,
        #     h=self.h,
        #     ln_f=self.ln_f,
        # ))

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # self.transformer.wte.weight=self.lm_head.weight
        self.wte.weight = self.lm_head.weight

        # init all weights using shared utility
        self.apply(lambda m: init_weights(m, rms_norm=False))
        # 按照GPT-2论文所述，对残差投影应用特殊的缩放初始化
        scale_residual_projections(self, n_layers=cfg.n_layers, suffix="c_proj.weight")

        # report number of parameters
        print(f"Number of parameters:  {self.get_num_params() / 1e6:.2f}M")

    def init_weights(self) -> None:
        """重新初始化全部权重（含残差投影缩放）。

        与 ``__init__`` 中的初始化逻辑保持一致，供检查点重建等场景调用。
        """
        self.apply(lambda m: init_weights(m, rms_norm=False))
        scale_residual_projections(
            self, n_layers=self.config.n_layers, suffix="c_proj.weight"
        )

    def get_device(self) -> torch.device:
        """Return the device the model parameters live on."""
        return next(self.parameters()).device

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        返回模型中的参数数量。
        Args:
            non_embedding (bool, optional): 对于非嵌入计数(默认情况)，位置嵌入会被减去。. Defaults to True.原本token embedding也会被减去，但由于参数共享，这些参数实际上被用作最后一层的权重，因此我们将它们包含在内。
        """

        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # n_params-=self.transformer.wpe.weight.numel()
            n_params -= self.wpe.weight.numel()
        return n_params

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device

        _, seq_len = idx.size()

        assert seq_len <= self.config.n_positions, (
            f"序列长度 {seq_len}超过模型支持的最长上下文长度 {self.config.n_positions}"
        )

        positions = torch.arange(
            0, seq_len, dtype=torch.long, device=device
        )  # shape(seq_len)

        # forward pass

        # token_embd=self.transformer.wte(idx) # token embedding shape(bsz,seq_len,n_embd)
        # position_embd=self.transformer.wpe(positions) # position embedding shape(seq_len,n_embd)

        # x=self.transformer.drop(token_embd+position_embd)

        # for block in self.transformer.h:
        #     x=block(x)
        # x=self.transformer.ln_f(x)

        inputs_embeds = self.wte(idx)  # token embedding shape(bsz,seq_len,n_embd)
        position_embd = self.wpe(positions)  # position embedding shape(seq_len,n_embd)
        x = self.drop(inputs_embeds + position_embd)

        for block in self.h:
            x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            # 如果tragets非空，计算loss
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(
                x[:, [-1], :]
            )  # # note：使用列表 [-1] 来保留sequence维度
            loss = None

        return logits, loss

    def crop_n_position(self, n_position: int) -> None:
        """
        # 必要时通过模型调整上下文长度
        # 例如，我们可能加载了GPT2预训练模型检查点(块大小为1024)
        # 但希望在一些更小、更简单的模型中使用更小的块大小

        Args:
            n_position (int): _description_
        """
        assert n_position <= self.config.n_positions, (
            f"n_position {n_position}超过模型支持的最长上下文长度 {self.config.n_positions}"
        )

        self.config.n_positions = n_position

        self.wpe.weight = nn.Parameter(self.wpe.weight[:n_position])

        for block in self.h:
            if hasattr(block.attn, "bias"):
                # ``bias`` is a lazily registered causal-mask buffer, so the
                # torch stubs type it ``Tensor | Module``; narrow explicitly.
                bias = cast(torch.Tensor, block.attn.bias)
                block.attn.bias = bias[:, :, :n_position, :n_position]

    @classmethod
    def from_pretrained(
        cls, model_type: str, override_args: dict[str, Any] | None = None
    ) -> "GPT":
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        # Config kwargs mix str/bool/int fields coming from the model-type table
        # plus a user dropout override, so the builder is a heterogeneous dict.
        config_args: dict[str, Any] = {
            "gpt2": {"n_layers": 12, "n_heads": 12, "n_embd": 768},  # 124M params
            "gpt2-medium": {
                "n_layers": 24,
                "n_heads": 16,
                "n_embd": 1024,
            },  # 350M params
            "gpt2-large": {
                "n_layers": 36,
                "n_heads": 20,
                "n_embd": 1280,
            },  # 774M params
            "gpt2-xl": {"n_layers": 48, "n_heads": 25, "n_embd": 1600},  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["n_positions"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = LLMConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [
            k for k in sd if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = [
            k
            for k in sd_hf
            if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
        ]  # ignore these buffers
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), (
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 仅保留需要梯度更新的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 创建优化组。任何二维参数都将进行权重衰减，否则不进行。即，矩阵乘法和嵌入中的所有权重张量会衰减，所有偏置和层归一化不会。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # 创建AdamW优化器，如果融合版本可用则使用融合版本
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """以A100 bfloat16峰值FLOPS为单位估算模型的FLOPS利用率(model flops utilization,MFU)"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = (
            cfg.n_layers,
            cfg.n_heads,
            cfg.n_embd // cfg.n_heads,
            cfg.n_positions,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def num_scaling_params(self) -> dict[str, int]:
        """Return parameter counts by category for scaling laws analysis."""
        cfg = self.config
        return count_scaling_params(
            n_layers=cfg.n_layers,
            n_embd=cfg.n_embd,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads,
            n_intermediate=(
                cfg.n_intermediate if cfg.n_intermediate is not None else 4 * cfg.n_embd
            ),
            vocab_size=cfg.vocab_size,
            n_positions=cfg.n_positions,
            bias=cfg.bias,
        )

    def estimate_flops(self) -> int:
        """Estimate FLOPs per training iteration (forward + backward)."""
        N = self.get_num_params()
        T = self.config.n_positions
        return estimate_flops(N, T)

    def setup_optimizer(
        self,
        unembedding_lr: float = 0.008,
        embedding_lr: float = 0.3,
        scalar_lr: float = 0.5,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.28,
    ) -> "MuonAdamW":
        """Create a combined MuonAdamW optimizer for the model."""
        return build_muon_optimizer(
            self,
            embed_lr=embedding_lr,
            head_lr=unembedding_lr,
            scalar_lr=scalar_lr,
            matrix_lr=matrix_lr,
            weight_decay=weight_decay,
        )

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        取一个索引的条件序列idx（形状为(bsz,seq_len)的LongTensor），并将该序列补全max_new_tokens次，每次都将预测结果反馈到模型中。大多数情况下，你可能需要确保为此处于model.eval()操作模式。
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文变得太长，我们必须将其裁剪到块大小。
            idx_cond = (
                idx
                if idx.size(1) <= self.config.n_positions
                else idx[:, -self.config.n_positions :]
            )
            # 向前传递模型以获取序列中该索引的logits输出
            logits, _ = self(idx_cond)
            # 在最后一步提取logits输出，并按期望的temperature进行缩放
            logits = logits[:, -1, :] / temperature
            # 可选地裁剪logits输出，仅保留top_k个选项
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # 对logits输出应用softmax，将其转换为归一化后的概率分布
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将采样的索引添加到运行序列中并继续生成
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
