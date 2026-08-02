"""
Train LLM model. From root directory of the project, run as:

python -m src.train_llm

or distributed as:

torchrun --nproc_per_node=8 -m src.train_llm

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m src.train_llm --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import contextlib
import gc
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Any, cast

import swanlab
import torch
import torch.distributed as dist
from torch import nn

from src.common.file_os import get_base_dir
from src.common.logger import DummySwanLab, print0, print_banner
from src.common.tokenizer import get_token_bytes, get_tokenizer
from src.data.text_pretrain_loader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from src.engine.engine_inference import Engine
from src.engine.utils_checkpoints import load_checkpoint, save_checkpoint
from src.eval import CoreResults, evaluate_core
from src.evaluator.eval_loss import evaluate_bpb
from src.models.config import LLMConfig
from src.models.flash_attention import HAS_FA3, USE_FA3
from src.models.gpt import GPT
from src.models.layers import Linear
from src.trainer.distributed import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    autodetect_device_type,
    compute_init,
    destory_ddp_process_group,
    get_peak_flops,
    is_ddp_initialized,
    is_master,
)
from src.trainer.optim import MuonAdamW
from src.trainer.train_fp8 import (
    Float8LinearConfig,
    convert_to_float8_training,
)

print_banner()


# -----------------------------------------------------------------------------
# Training configuration


@dataclass
class TrainConfig:
    # Logging
    run: str = "dummy"
    # Runtime
    device_type: str = ""
    # FP8 training
    fp8: bool = False
    fp8_recipe: str = "tensorwise"
    # Model architecture
    depth: int = 20
    aspect_ratio: int = 64
    head_dim: int = 128
    max_seq_len: int = 2048
    window_pattern: str = "SSSL"
    # Training horizon
    num_iterations: int = -1
    target_flops: float = -1.0
    target_param_data_ratio: float = 12
    # Optimization
    device_batch_size: int = 32
    total_batch_size: int = -1
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    weight_decay: float = 0.28
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    resume_from_step: int = -1
    # Evaluation
    eval_every: int = 250
    eval_tokens: int = 80 * 524288
    core_metric_every: int = 2000
    core_metric_max_per_task: int = 500
    sample_every: int = 2000
    save_every: int = -1
    # Output
    model_tag: str | None = None

    def to_json(self, file_path: str | None = None, indent: int = 4) -> str:
        data_dict = asdict(self)
        json_str = json.dumps(data_dict, ensure_ascii=False, indent=indent)
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=indent)
            print0(f">>> Config saved: {file_path}")
        return json_str


# -----------------------------------------------------------------------------
# Helpers


def get_lr_multiplier(
    step: int,
    num_iterations: int,
    warmup_steps: int,
    warmdown_ratio: float,
    final_lr_frac: float,
) -> float:
    """Cosine LR schedule with linear warmup and warmdown."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    warmdown_start = int(num_iterations * (1 - warmdown_ratio))
    if step >= warmdown_start:
        progress = (step - warmdown_start) / (num_iterations - warmdown_start)
        return 1.0 - progress * (1.0 - final_lr_frac)
    return 1.0


def get_muon_momentum(step: int, num_iterations: int, warmdown_ratio: float) -> float:
    """Muon momentum schedule: linear ramp to 0.97, then warmdown to 0.90."""
    warmdown_start = int(num_iterations * (1 - warmdown_ratio))
    if step < 400:
        frac = step / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif step >= warmdown_start:
        progress = (step - warmdown_start) / (num_iterations - warmdown_start)
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97


def get_weight_decay(
    step: int, num_iterations: int, weight_decay_scaled: float
) -> float:
    """Weight decay schedule: cosine decay to zero over training."""
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * step / num_iterations))


def build_model_meta(depth: int, vocab_size: int, args: TrainConfig) -> GPT:
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = LLMConfig(
        n_positions=args.max_seq_len,
        vocab_size=vocab_size,
        n_layers=depth,
        n_heads=num_heads,
        n_kv_heads=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta


def get_scaling_params(m: GPT) -> int:
    """Get parameter count used for scaling laws (transformer matrices + lm_head)."""
    params_counts = m.num_scaling_params()
    return params_counts["transformer_matrices"] + params_counts["lm_head"]


def compute_training_horizon(
    args: TrainConfig, model: GPT, d12_ref: GPT, num_flops_per_token: int
) -> tuple[int, int]:
    """Compute num_iterations and total_batch_size based on scaling laws."""
    num_scaling_params = get_scaling_params(model)
    target_tokens = int(args.target_param_data_ratio * num_scaling_params)

    D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)
    B_REF = 2**19  # ~524,288 tokens

    total_batch_size = args.total_batch_size
    if total_batch_size == -1:
        batch_size_ratio = target_tokens / D_REF
        predicted_batch_size = B_REF * batch_size_ratio**0.383
        total_batch_size = 2 ** round(math.log2(predicted_batch_size))
        print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

    if args.num_iterations > 0:
        num_iterations = args.num_iterations
        print0(f"Using user-provided number of iterations: {num_iterations:,}")
    elif args.target_flops > 0:
        num_iterations = round(
            args.target_flops / (num_flops_per_token * total_batch_size)
        )
        print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
    else:
        num_iterations = target_tokens // total_batch_size
        print0(
            f"Calculated number of iterations from param:data ratio: {num_iterations:,}"
        )

    return num_iterations, total_batch_size


def compute_lr_and_wd_scales(
    args: TrainConfig, total_batch_size: int, target_tokens: int, d12_ref: GPT
) -> tuple[float, float]:
    """Compute learning rate and weight decay scaling factors."""
    B_REF = 2**19
    D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)

    batch_lr_scale = 1.0
    batch_ratio = total_batch_size / B_REF
    if batch_ratio != 1.0:
        batch_lr_scale = batch_ratio**0.5
        print0(
            f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})"
        )

    weight_decay_scaled = (
        args.weight_decay
        * math.sqrt(total_batch_size / B_REF)
        * (D_REF / target_tokens)
    )
    if weight_decay_scaled != args.weight_decay:
        print0(
            f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f}"
        )

    return batch_lr_scale, weight_decay_scaled


# -----------------------------------------------------------------------------
# FP8 helpers


def convert_model_to_fp8(model: GPT, recipe: str) -> None:
    """Convert Linear layers to Float8Linear for FP8 training."""

    def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        return min(mod.in_features, mod.out_features) >= 128

    fp8_config = Float8LinearConfig.from_recipe_name(recipe)
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    convert_to_float8_training(
        model, config=fp8_config, module_filter_fn=fp8_module_filter
    )
    num_fp8 = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
    num_skipped = num_linear - num_fp8
    print0(
        f"FP8 training enabled ({recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped}"
    )


@contextlib.contextmanager
def disable_fp8(model: GPT) -> Iterator[None]:
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation."""

    fp8_locations: list[tuple[nn.Module, str, nn.Linear]] = []
    for name, module in model.named_modules():
        if "Float8" in type(module).__name__:
            if "." in name:
                parent_name, attr_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, cast(nn.Linear, module)))

    if not fp8_locations:
        yield
        return

    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)


# -----------------------------------------------------------------------------
# Dataloader setup


def get_dataloaders(
    tokenizer: Any,
    args: TrainConfig,
    device: torch.device,
    resume_state_dict: dict[str, int] | None = None,
) -> tuple[
    Iterator[tuple[torch.Tensor, torch.Tensor, dict[str, int]]],
    Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
    torch.Tensor,
    torch.Tensor,
    dict[str, int],
]:
    """Create train/val dataloaders."""
    dataloader_resume_state_dict = None if not resume_state_dict else resume_state_dict
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        split="train",
        device=device.type,
        resume_state_dict=dataloader_resume_state_dict,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        split="val",
        device=device.type,
    )
    x, y, dataloader_state_dict = next(train_loader)
    return train_loader, build_val_loader, x, y, dataloader_state_dict


# -----------------------------------------------------------------------------
# Model & optimizer setup


def build_model_and_optimizer(
    args: TrainConfig,
    vocab_size: int,
    device: torch.device,
    ddp_rank: int,
    checkpoint_dir: str,
) -> tuple[
    GPT,
    GPT,
    MuonAdamW,
    torch.amp.GradScaler | None,
    int,
    int,
    int,
    float,
    float,
    int,
    dict[str, Any],
    dict[str, Any] | None,
]:
    """Build model, optionally resume, set up FP8, compile, and create optimizer."""
    # Build model
    model: GPT = build_model_meta(args.depth, vocab_size, args)
    model_config = model.config
    model_config_kwargs = asdict(model_config)
    print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
    model.to_empty(device=device)
    model.init_weights()

    # Resume if requested
    resuming = args.resume_from_step != -1
    model_data: dict[str, torch.Tensor] | None = None
    optimizer_data: dict[str, Any] | None = None
    meta_data: dict[str, Any] | None = None
    if resuming:
        print0(f"Resuming optimization from step {args.resume_from_step}")
        model_data, optimizer_data, meta_data = load_checkpoint(
            checkpoint_dir,
            args.resume_from_step,
            device,
            load_optimizer=True,
            rank=ddp_rank,
        )
        model.load_state_dict(model_data, strict=True, assign=True)
        del model_data

    # FP8 conversion
    device_type = device.type
    if args.fp8:
        if device_type != "cuda":
            print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
        else:
            convert_model_to_fp8(model, args.fp8_recipe)

    # Compile
    orig_model = model
    model = cast(GPT, torch.compile(model, dynamic=False))

    # Scaling laws & batch size / LR / WD computation
    param_counts = model.num_scaling_params()
    print0("Parameter counts:")
    for key, value in param_counts.items():
        print0(f"{key:24s}: {value:,}")
    num_flops_per_token = model.estimate_flops()
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    d12_ref = build_model_meta(12, vocab_size, args)
    num_iterations, total_batch_size = compute_training_horizon(
        args,
        model,
        d12_ref,
        num_flops_per_token,
    )

    num_scaling_params = get_scaling_params(model)
    target_tokens = int(args.target_param_data_ratio * num_scaling_params)
    batch_lr_scale, weight_decay_scaled = compute_lr_and_wd_scales(
        args,
        total_batch_size,
        target_tokens,
        d12_ref,
    )

    # Optimizer
    optimizer = model.setup_optimizer(
        unembedding_lr=args.unembedding_lr * batch_lr_scale,
        embedding_lr=args.embedding_lr * batch_lr_scale,
        scalar_lr=args.scalar_lr * batch_lr_scale,
        matrix_lr=args.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
    )

    if resuming and optimizer_data is not None:
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data

    # GradScaler
    scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
    if scaler is not None:
        print0("GradScaler enabled for fp16 training")

    return (
        model,
        orig_model,
        optimizer,
        scaler,
        num_flops_per_token,
        num_iterations,
        total_batch_size,
        batch_lr_scale,
        weight_decay_scaled,
        target_tokens,
        model_config_kwargs,
        meta_data,
    )


# -----------------------------------------------------------------------------
# Evaluation & sampling


def run_evaluation(
    model: GPT,
    val_loader: Iterator[tuple[torch.Tensor, torch.Tensor]],
    eval_steps: int,
    token_bytes: torch.Tensor,
    step: int,
    swanlab_run: Any,
    total_training_time: float,
    flops_so_far: float,
    min_val_bpb: float,
) -> tuple[float, float]:
    """Run validation bpb evaluation."""
    model.eval()
    with disable_fp8(model):
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
    min_val_bpb = min(min_val_bpb, val_bpb)
    swanlab_run.log(
        {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        }
    )
    model.train()
    return val_bpb, min_val_bpb


def run_core_evaluation(
    orig_model: GPT,
    tokenizer: Any,
    device: torch.device,
    step: int,
    swanlab_run: Any,
    total_training_time: float,
    flops_so_far: float,
    max_per_task: int,
) -> CoreResults:
    """Run CORE metric evaluation."""
    orig_model.eval()
    with disable_fp8(orig_model):
        results = evaluate_core(
            orig_model, tokenizer, device, max_per_task=max_per_task
        )
    print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
    swanlab_run.log(
        {
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        }
    )
    orig_model.train()
    return results


def run_sampling(
    orig_model: GPT, tokenizer: Any, device: torch.device, step: int
) -> None:
    """Sample text from the model for debugging."""
    orig_model.eval()
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    engine = Engine(orig_model, tokenizer)
    for prompt in prompts:
        tokens = cast(list[int], tokenizer(prompt, prepend="<|bos|>"))
        with disable_fp8(orig_model):
            sample, _ = engine.generate_batch(
                tokens, num_samples=1, max_tokens=16, temperature=0
            )
        print0(tokenizer.decode(sample[0]))
    orig_model.train()


# -----------------------------------------------------------------------------
# Checkpointing


def maybe_save_checkpoint(
    checkpoint_dir: str,
    step: int,
    num_iterations: int,
    args: TrainConfig,
    orig_model: GPT,
    optimizer: MuonAdamW,
    model_config_kwargs: dict[str, Any],
    user_config: dict[str, Any],
    device_batch_size: int,
    max_seq_len: int,
    total_batch_size: int,
    dataloader_state_dict: dict[str, int],
    min_val_bpb: float,
    smooth_train_loss: float,
    total_training_time: float,
    val_bpb: float | None,
    ddp_rank: int,
) -> None:
    """Save checkpoint if conditions are met."""
    last_step = step >= num_iterations - 1
    should_save = last_step or (
        step > 0
        and step != args.resume_from_step
        and args.save_every > 0
        and step % args.save_every == 0
    )
    if not should_save:
        return

    save_checkpoint(
        checkpoint_dir,
        step,
        orig_model.state_dict(),
        optimizer.state_dict(),
        {
            "step": step,
            "val_bpb": val_bpb,
            "model_config": model_config_kwargs,
            "user_config": user_config,
            "device_batch_size": device_batch_size,
            "max_seq_len": max_seq_len,
            "total_batch_size": total_batch_size,
            "dataloader_state_dict": dataloader_state_dict,
            "loop_state": {
                "min_val_bpb": min_val_bpb,
                "smooth_train_loss": smooth_train_loss,
                "total_training_time": total_training_time,
            },
        },
        rank=ddp_rank,
    )


# -----------------------------------------------------------------------------
# Training step


def train_one_step(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    train_loader: Iterator[tuple[torch.Tensor, torch.Tensor, dict[str, int]]],
    grad_accum_steps: int,
    scaler: torch.amp.GradScaler | None,
    optimizer: MuonAdamW,
    step: int,
    num_iterations: int,
    warmup_steps: int,
    warmdown_ratio: float,
    final_lr_frac: float,
    weight_decay_scaled: float,
    synchronize: Callable[..., None],
) -> tuple[float, float, torch.Tensor, torch.Tensor, dict[str, int], float]:
    """Execute a single training step with gradient accumulation."""
    synchronize()
    t0 = time.time()
    train_loss: torch.Tensor | None = None

    for micro_step in range(grad_accum_steps):
        _, loss = model(x, y)
        assert loss is not None
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

    # Optimizer step with dynamic schedules
    lrm = get_lr_multiplier(
        step, num_iterations, warmup_steps, warmdown_ratio, final_lr_frac
    )
    muon_momentum = get_muon_momentum(step, num_iterations, warmdown_ratio)
    muon_weight_decay = get_weight_decay(step, num_iterations, weight_decay_scaled)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)

    assert train_loss is not None
    train_loss_f = float(train_loss.item())
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    return train_loss_f, dt, x, y, dataloader_state_dict, lrm


# -----------------------------------------------------------------------------
# Main training function


def train(args: TrainConfig) -> None:
    """Main training entry point."""
    # Distributed & device init
    device_type = (
        autodetect_device_type() if args.device_type == "" else args.device_type
    )
    _, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    synchronize: Callable[..., None] = (
        torch.cuda.synchronize if device_type == "cuda" else (lambda: None)
    )
    get_max_memory: Callable[..., int] = (
        torch.cuda.max_memory_allocated if device_type == "cuda" else (lambda: 0)
    )

    if device_type == "cuda":
        gpu_device_name = torch.cuda.get_device_name(0)
        gpu_peak_flops = get_peak_flops(gpu_device_name)
        print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    else:
        gpu_peak_flops = float("inf")
    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

    # SwanLab logging
    use_dummy_swanlab = args.run == "dummy" or not master_process
    user_config = asdict(args)
    # swanlab_run: SwanLab run or DummySwanLab (untyped third-party / local stub)
    swanlab_run: Any = (
        DummySwanLab()
        if use_dummy_swanlab
        else swanlab.init(
            project="nanochat",
            name=args.run,
            config=user_config,
        )
    )

    # Flash Attention status
    using_fa3 = USE_FA3
    if using_fa3:
        print0(
            "Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome."
        )
    else:
        print0("!" * 80)
        if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
            print0(
                f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback"
            )
        else:
            print0(
                "WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback"
            )
        print0("WARNING: Training will be less efficient without FA3")
        if args.window_pattern != "L":
            print0(
                f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible."
            )
            print0(
                "WARNING: Recommend using --window-pattern L for full context attention."
            )
        print0("!" * 80)

    # Tokenizer
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device.type)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

    # Checkpoint directory
    base_dir = get_base_dir()
    output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
    checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)

    # Build model & optimizer
    (
        model,
        orig_model,
        optimizer,
        scaler,
        num_flops_per_token,
        num_iterations,
        total_batch_size,
        _batch_lr_scale,
        weight_decay_scaled,
        _target_tokens,
        model_config_kwargs,
        resume_meta,
    ) = build_model_and_optimizer(args, vocab_size, device, ddp_rank, checkpoint_dir)

    # Dataloaders
    resume_state = resume_meta["dataloader_state_dict"] if resume_meta else None
    train_loader, build_val_loader, x, y, dataloader_state_dict = get_dataloaders(
        tokenizer,
        args,
        device,
        resume_state_dict=resume_state,
    )

    # Restore loop state if resuming
    min_val_bpb = float("inf")
    smooth_train_loss = 0.0
    total_training_time = 0.0
    if resume_meta and "loop_state" in resume_meta:
        ls: dict[str, Any] = resume_meta["loop_state"]
        min_val_bpb = ls.get("min_val_bpb", float("inf"))
        smooth_train_loss = ls.get("smooth_train_loss", 0.0)
        total_training_time = ls.get("total_training_time", 0.0)

    # Gradient accumulation
    grad_accum_steps = max(
        1,
        total_batch_size
        // (args.device_batch_size * args.max_seq_len * ddp_world_size),
    )
    actual_batch_size = (
        args.device_batch_size * args.max_seq_len * grad_accum_steps * ddp_world_size
    )
    print0(f"Gradient accumulation steps: {grad_accum_steps}")
    print0(f"Actual total batch size: {actual_batch_size:,} tokens")

    # weight_decay_scaled already returned by build_model_and_optimizer

    # Training loop
    step = 0
    if args.resume_from_step != -1:
        step = args.resume_from_step + 1

    val_bpb: float | None = None
    flops_so_far = 0

    while step < num_iterations:
        last_step = step >= num_iterations - 1

        # Periodic evaluation
        if args.eval_every > 0 and (
            last_step or (step > 0 and step % args.eval_every == 0)
        ):
            val_loader = build_val_loader()
            eval_steps = max(
                1,
                args.eval_tokens
                // (args.device_batch_size * args.max_seq_len * ddp_world_size),
            )
            val_bpb, min_val_bpb = run_evaluation(
                model,
                val_loader,
                eval_steps,
                token_bytes,
                step,
                swanlab_run,
                total_training_time,
                flops_so_far,
                min_val_bpb,
            )

        # CORE metric
        if args.core_metric_every > 0 and (
            last_step or (step > 0 and step % args.core_metric_every == 0)
        ):
            run_core_evaluation(
                orig_model,
                tokenizer,
                device,
                step,
                swanlab_run,
                total_training_time,
                flops_so_far,
                args.core_metric_max_per_task,
            )

        # Sampling
        if (
            args.sample_every > 0
            and master_process
            and (last_step or (step > 0 and step % args.sample_every == 0))
        ):
            run_sampling(orig_model, tokenizer, device, step)

        # Checkpoint
        maybe_save_checkpoint(
            checkpoint_dir,
            step,
            num_iterations,
            args,
            orig_model,
            optimizer,
            model_config_kwargs,
            user_config,
            args.device_batch_size,
            args.max_seq_len,
            total_batch_size,
            dataloader_state_dict,
            min_val_bpb,
            smooth_train_loss,
            total_training_time,
            val_bpb,
            ddp_rank,
        )

        if last_step:
            break

        # Training step
        train_loss_f, dt, x, y, dataloader_state_dict, lrm = train_one_step(
            model,
            x,
            y,
            train_loader,
            grad_accum_steps,
            scaler,
            optimizer,
            step,
            num_iterations,
            args.warmup_steps,
            args.warmdown_ratio,
            args.final_lr_frac,
            weight_decay_scaled,
            synchronize,
        )

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * step / num_iterations
        tok_per_sec = int(total_batch_size / dt)
        flops_per_sec = num_flops_per_token * total_batch_size / dt
        mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
        flops_so_far += num_flops_per_token * total_batch_size

        if step > 10:
            total_training_time += dt

        steps_done = step - 10
        eta_str = ""
        if steps_done > 0:
            avg_time_per_step = total_training_time / steps_done
            remaining_steps = num_iterations - step
            eta_seconds = remaining_steps * avg_time_per_step
            eta_str = f" | eta: {eta_seconds / 60:.1f}m"

        epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
        print0(
            f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) "
            f"| loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} "
            f"| dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} "
            f"| bf16_mfu: {mfu:.2f} | epoch: {epoch} "
            f"| total time: {total_training_time / 60:.2f}m{eta_str}"
        )

        if step % 100 == 0:
            swanlab_run.log(
                {
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "train/loss": debiased_smooth_loss,
                    "train/lrm": lrm,
                    "train/dt": dt,
                    "train/tok_per_sec": tok_per_sec,
                    "train/mfu": mfu,
                    "train/epoch": epoch,
                }
            )

        step += 1

        # Garbage collection management
        if step == 1 or (step == 0 and args.resume_from_step != -1):
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()

    # Final stats
    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time / 60:.2f}m")
    if val_bpb is not None:
        print0(f"Minimum validation bpb: {min_val_bpb:.6f}")


# -----------------------------------------------------------------------------
# CLI & entry point


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Pretrain base model")
    parser.add_argument("--run", type=str, default="dummy")
    parser.add_argument("--device-type", type=str, default="")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument(
        "--fp8-recipe",
        type=str,
        default="tensorwise",
        choices=["rowwise", "tensorwise"],
    )
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--window-pattern", type=str, default="SSSL")
    parser.add_argument("--num-iterations", type=int, default=-1)
    parser.add_argument("--target-flops", type=float, default=-1.0)
    parser.add_argument("--target-param-data-ratio", type=float, default=12)
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--total-batch-size", type=int, default=-1)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.008)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--scalar-lr", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    parser.add_argument("--final-lr-frac", type=float, default=0.05)
    parser.add_argument("--resume-from-step", type=int, default=-1)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-tokens", type=int, default=80 * 524288)
    parser.add_argument("--core-metric-every", type=int, default=2000)
    parser.add_argument("--core-metric-max-per-task", type=int, default=500)
    parser.add_argument("--sample-every", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=-1)
    parser.add_argument("--model-tag", type=str, default=None)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    args = parse_args()
    if is_master():
        print(">>> ---------- Train Config ---------- <<<")
        print(args.to_json())
    train(args)
    if is_ddp_initialized():
        destory_ddp_process_group()


if __name__ == "__main__":
    main()
