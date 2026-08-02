import os

# 关闭 **Hugging Face Tokenizers 分词器的多线程并行**，避免多进程/多线程场景下出现警告、死锁或性能异常。
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 开启 PyTorch CUDA **可扩展显存分段分配**，优化GPU显存管理，减少显存碎片、降低OOM（显存溢出）概率，提升显存利用率。
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import warnings

warnings.filterwarnings("ignore", message=".*Length o IterableDataset.*")

import argparse
import contextlib
import json
import math
import random
import re
import subprocess
import time
from collections.abc import Iterator
from dataclasses import asdict
from statistics import mean
from typing import Any, cast

import numpy as np
import swanlab
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
from datasets import (
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
    load_from_disk,
)
from datasets import config as datasets_config

# 解决 Python PIL/Pillow 库打开部分 PNG 图片时报 Decompressed data too large（解压数据过大） 的报错，是处理超大文本块 PNG 的通用修复方案
from PIL import PngImagePlugin

from src.common.logger import logger
from src.data.data_utils import synchronized_dataloader_step
from src.data.datasets_advanced import ConstantLengthDataset
from src.data.processors import get_image_processor, get_tokenizer
from src.data.vqa_datasets import VQACollator, VQADataset
from src.models.config import VLMConfig
from src.models.vision_language_model import VisionLanguageModel
from src.trainer.config import TrainConfig
from src.trainer.distributed import (
    destory_ddp_process_group,
    distributed_gather,
    distributed_mean_scalar,
    get_rank,
    get_world_size,
    init_dist,
    is_ddp_initialized,
    is_master,
    wrap_model,
)

PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 * 1024)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_run_name(train_cfg: TrainConfig, vlm_cfg: VLMConfig) -> str:
    batch_size = f"bsz{int(train_cfg.batch_size * get_world_size() * train_cfg.gradient_accumulation_steps)}"
    max_training_steps = f"{train_cfg.max_training_steps}"
    learning_rate = f"lr_version_{train_cfg.lr_language_backbone}-language_{train_cfg.lr_language_backbone}-{train_cfg.lr_mp}"

    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}_{vlm_cfg.max_img_size}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"

    return f"VLM_{vit}_{mp}_{llm}_{num_gpus}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"


def get_dataloaders(
    train_cfg: TrainConfig, vlm_cfg: VLMConfig
) -> tuple[
    DataLoader[Any],
    DataLoader[Any],
    Iterator[dict[str, Any]],
    Iterator[dict[str, Any]],
]:
    if is_master():
        logger.info(f"Getting dataloaders from {train_cfg.train_dataset_path}")

    # Create datasets
    if is_master():
        logger.info("Create datasets")
    image_processor = get_image_processor(
        vlm_cfg.max_img_size, vlm_cfg.image_size, vlm_cfg.resize_to_max_side_len
    )

    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )

    dataset_name_to_load: list[str] = list(train_cfg.train_dataset_name)
    if "shards" in dataset_name_to_load:
        if is_master():
            logger.info("Loading shards...")
        total_shards = 56
        dataset_name_to_load = [
            f"{train_cfg.train_dataset_path}/shards_{i}" for i in range(total_shards)
        ]

    if "all" in dataset_name_to_load:
        dataset_name_to_load = get_dataset_config_names(train_cfg.train_dataset_path)

    # Load and combine all training datasets
    if is_master():
        logger.info("Load and combine all training datasets...")
    combined_train_data = []
    for dataset_name in dataset_name_to_load:
        if is_master():
            logger.info(f">>>Loading dataset: {train_cfg.train_dataset_path} ...")
            logger.info(f">>>Loading dataset: {dataset_name} ...")

        if "shard_" in dataset_name:
            try:
                train_dateset = load_from_disk(dataset_name)
                combined_train_data.append(train_dateset)
                continue
            except Exception as e:  # noqa: BLE001  (datasets lib raises version-dependent exception types; resilience required across data sources)
                if is_master():
                    logger.warning(
                        f'Failed to load dataset shard: "{dataset_name}" from "{train_cfg.train_dataset_path}". Error:{e}'
                    )
                continue

        try:
            train_dataset = load_dataset(
                train_cfg.train_dataset_path,
                dataset_name,
                streaming=train_cfg.stream_dataset,
                on_bad_files="warn",
            )["train"]
            if is_master():
                logger.info(
                    f"datasets 全局缓存根目录：{datasets_config.HF_DATASETS_CACHE}"
                )
                logger.info(f"数据集详情信息：\n{train_dataset.info}")
            # print(f"本地数据文件路径：{train_dataset._data_files}")
            if train_cfg.stream_dataset:
                next(iter(train_dataset))  # Check if the dataset is loaded correctly
            else:
                train_dataset[0]  # Check if the dataset is loaded correctly

            combined_train_data.append(train_dataset)
        except Exception as e:  # noqa: BLE001  (datasets lib raises version-dependent exception types; resilience required across data sources)
            if is_master():
                logger.warning(
                    f"Failed to load dataset config '{dataset_name}' from '{train_cfg.train_dataset_path}'. Error: {e}"
                )
            continue

    if not combined_train_data:
        raise ValueError(
            f"No valid datasets were loaded. Please check your dataset path and configurations:\ndata_path:{train_cfg.train_dataset_path};\ndata_name:{train_cfg.train_dataset_name}"
        )

    train_dataset = concatenate_datasets(combined_train_data)

    if not train_cfg.stream_dataset:
        if is_master():
            logger.info(
                "# Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets  "
            )
        train_dataset = train_dataset.shuffle(seed=0)

    if is_ddp_initialized():  # We need to shard the dataset in DDP since we are using an iterable dataset instead of the distributed sampler
        train_dataset = train_dataset.shard(
            num_shards=get_world_size(), index=get_rank()
        )

    if is_master():
        logger.info(
            "# train_dataset = train_dataset.shuffle(buffer_size=10000, seed=0) # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets"
        )
    val_size = int(train_cfg.val_size / get_world_size())
    if is_master():
        logger.info(f"Val size per GPU: {val_size}")

    if train_cfg.stream_dataset:
        val_ds = train_dataset.take(val_size)
        train_dataset = train_dataset.skip(val_size)
    else:
        val_ds = train_dataset.select(range(val_size))
        train_dataset = train_dataset.select(range(val_size, len(train_dataset)))

    train_dataset = VQADataset(
        train_dataset,
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        train_cfg.relevance_min_rating,
        train_cfg.image_correspondence_min_rating,
        train_cfg.visual_dependency_min_rating,
        train_cfg.formatting_min_rating,
    )
    val_dataset = VQADataset(
        val_ds,
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        train_cfg.relevance_min_rating,
        train_cfg.image_correspondence_min_rating,
        train_cfg.visual_dependency_min_rating,
        train_cfg.formatting_min_rating,
    )

    train_dataset_packed = ConstantLengthDataset(
        train_dataset,
        infinite=False,
        max_sample_length=train_cfg.max_sample_length,
        seq_length=vlm_cfg.n_positions,
        num_of_sequences=train_cfg.batch_size * 2,
        queue_size=2,
        max_images_per_example=train_cfg.max_images_per_example,
        max_images_per_knapsack=train_cfg.max_images_per_knapsack,
    )

    val_dataset_packed = ConstantLengthDataset(
        val_dataset,
        infinite=False,
        max_sample_length=train_cfg.max_sample_length,
        seq_length=vlm_cfg.n_positions,
        num_of_sequences=train_cfg.batch_size * 2,
        queue_size=2,
        max_images_per_example=train_cfg.max_images_per_example,
        max_images_per_knapsack=train_cfg.max_images_per_knapsack,
    )

    # Create collators
    vqa_collator = VQACollator(tokenizer, vlm_cfg.n_positions)

    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloaders

    train_loader = DataLoader(
        train_dataset_packed,
        batch_size=train_cfg.batch_size,  # =per device BS in DDP
        collate_fn=vqa_collator,
        num_workers=3,
        # timeout=600,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset_packed,
        batch_size=train_cfg.batch_size,
        collate_fn=vqa_collator,
        num_workers=1,
        # timeout=600,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Warmup dataloaders to kickstart worker processes
    if is_master():
        logger.info("Warming up dataloaders...")
    iter_train_loader = iter(train_loader)
    iter_val_loader = iter(val_loader)
    try:
        next(iter_train_loader)
        if is_master():
            logger.info(">>> Train loader warmup complete...")
    except (StopIteration, RuntimeError, ValueError) as e:
        raise ValueError(f"Error warmup train loader: {e}")

    try:
        next(iter_val_loader)
        if is_master():
            logger.info(">>> Val loader warmup complete...")
    except (StopIteration, RuntimeError, ValueError) as e:
        raise ValueError(f"Error warmup val loader: {e}")

    return train_loader, val_loader, iter_train_loader, iter_val_loader


def get_learn_rate(it: int, max_lr: float, max_steps: int) -> float:
    """
    # Cosine learning rate schedule with warmup (from Karpathy)
    # https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L353
    """
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def train(train_cfg: TrainConfig, vlm_cfg: VLMConfig) -> None:

    train_loader, val_loader, iter_train_loader, iter_val_loader = get_dataloaders(
        train_cfg, vlm_cfg
    )

    if is_ddp_initialized():
        if is_master():
            logger.info(
                f"Rank {get_rank()} Waiting for all workers to get dataloaders..."
            )
            logger.info("Waiting for all workers to get dataloaders...")
        dist.barrier(device_ids=int(os.environ["LOCAL_RANK"]))
        if is_master():
            logger.info("All workers have gotten dataloaders.")

    run_name = get_run_name(train_cfg, vlm_cfg)
    if train_cfg.log_wandb and is_master():
        swanlab.init(
            entity=train_cfg.wandb_entity,
            project="VLM",
            config={"VLMConfig": asdict(vlm_cfg), "TrainConfig": asdict(train_cfg)},
            name=run_name,
        )
        # Define a custom x-axis for lmms-eval metrics
        lmms_eval_step = "<lmms-eval-step>"
        # run.define_metric(name="lmms_eval/*", step_metric=lmms_eval_step) # TODO 如何使用swanlab替换wandb

    if is_master():
        logger.info(">>> # Initialize model ...")
    # model: wrapped by torch.compile / DDP at runtime, hence dynamic typing
    model: Any
    if train_cfg.resume_from_vlm_checkpoint:
        if is_master():
            logger.info(f"Resuming from VLM checkpoint: {vlm_cfg.vlm_checkpoint_path}")
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(
            vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights
        )

    if is_master():
        logger.info(
            f"VLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
        logger.info(
            f"Training summary {'(global)' if is_ddp_initialized() else ''}: {-1 * get_world_size()} samples, batch size {int(train_cfg.batch_size * get_world_size() * train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_ddp_initialized() else ''}"
        )
        if is_ddp_initialized():
            logger.info(
                f"Training summary per GPU: batch size {train_loader.batch_size}"
            )
        logger.info(
            f"Validation summary{' (global)' if is_ddp_initialized() else ''}: {-1 * get_world_size()} samples, batch size {int(train_cfg.batch_size * get_world_size() * train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_ddp_initialized() else ''}"
        )
        if is_ddp_initialized():
            logger.info(
                f"Validation summary per GPU: batch size {val_loader.batch_size}"
            )

    pstr2 = """
    # Define optimizer groups
    # Since we have pretrained vision and language backbones, but a newly initialized modality projection layer, it doesn't make sense to train them with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, but finetuning them with a lower learning rate makes the training as a whole easier
    """
    if is_master():
        logger.info(f">>> {pstr2} <<<")

    param_groups: list[dict[str, Any]] = []
    if train_cfg.lr_mp > 0:
        param_groups.append(
            {"params": list(model.MP.parameters()), "lr": train_cfg.lr_mp}
        )
    else:
        for p in list(model.MP.parameters()):
            p.requires_grad = False
    if train_cfg.lr_vision_backbone > 0:
        param_groups.append(
            {
                "params": list(model.vision_encoder.parameters()),
                "lr": train_cfg.lr_vision_backbone,
            }
        )
    else:
        for p in list(model.vision_encoder.parameters()):
            p.requires_grad = False
    if train_cfg.lr_language_backbone > 0:
        param_groups.append(
            {
                "params": list(model.decoder.parameters()),
                "lr": train_cfg.lr_language_backbone,
            }
        )
    else:
        for p in list(model.decoder.parameters()):
            p.requires_grad = False

    optimizer = optim.AdamW(param_groups)
    all_params = [p for group in optimizer.param_groups for p in group["params"]]

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    if device.type == "mps":  # 当MPS不支持某算子时,自动切换到 CPU 执行，避免报错。
        if hasattr(torch.backends.mps, "enable_fallback_to_cpu"):
            torch.backends.mps.enable_fallback_to_cpu = True
        else:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        torch.mps.empty_cache()

    if is_master():
        logger.info(f">>> Using device: {device}")
    model.to(device)

    if train_cfg.compile:
        model = torch.compile(model)
    if is_ddp_initialized():
        if is_master():
            logger.info("Wrapping model for DDP")
        model = wrap_model(model)
        if is_master():
            logger.info("Model wrapped for DDP")

    epoch_times: list[float] = []
    best_val_loss = float("inf")
    best_model_path: str | None = None
    logged_eval_steps: set[int] = set()
    global_step = 0
    epoch = 0

    # Training stats accumulators
    accumulated_stats: dict[str, list[float]] = {
        "tokens_per_second": [],
        "data_load_time": [],
        "fw_bw_time": [],
        "post_process_time": [],
        "images_per_sample": [],
    }

    while global_step < train_cfg.max_training_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss: float = 0
        total_tokens_processed: float = 0
        optimizer.zero_grad()
        data_load_start = time.time()

        if is_master():
            logger.info("Starting training loop")
        for i, batch in enumerate(
            synchronized_dataloader_step(iter_train_loader, is_ddp_initialized())
        ):
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0
            batch_start_time = time.time()
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            data_load_time = time.time() - data_load_start

            # When using DDP with gradient accumulation,
            # skip gradient synchronization on intermediate steps to save time.
            # Gradients only need to be synced at the end of each accumulation cycle.
            if (
                is_ddp_initialized()
                and train_cfg.gradient_accumulation_steps > 1
                and not is_update_step
            ):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            fw_bw_start = time.time()
            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16
                if device.type in ["cuda", "cpu"]
                else torch.float16,
            )
            with autocast_context, context:
                _, loss = model(
                    input_ids, images, attention_mask=attention_mask, targets=labels
                )

            if train_cfg.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.gradient_accumulation_steps

            loss.backward()

            fw_bw_time = time.time() - fw_bw_start
            post_process_start = time.time()
            if is_update_step:
                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_params, max_norm=train_cfg.max_grad_norm
                    )

                param_group_idx = 0
                if is_master():
                    logger.info("# multimodel project layer optimizer update")
                if train_cfg.lr_mp > 0:
                    adj_lr_mp = get_learn_rate(
                        global_step, train_cfg.lr_mp, train_cfg.max_training_steps
                    )
                    optimizer.param_groups[param_group_idx]["lr"] = adj_lr_mp
                    param_group_idx += 1
                if is_master():
                    logger.info("# vision model optimizer update")
                if train_cfg.lr_vision_backbone > 0:
                    adj_lr_vision_backbone = get_learn_rate(
                        global_step,
                        train_cfg.lr_vision_backbone,
                        train_cfg.max_training_steps,
                    )
                    optimizer.param_groups[param_group_idx]["lr"] = (
                        adj_lr_vision_backbone
                    )
                    param_group_idx += 1

                if is_master():
                    logger.info("# language model optimizer update")
                if train_cfg.lr_language_backbone > 0:
                    adj_lr_language_backbone = get_learn_rate(
                        global_step,
                        train_cfg.lr_language_backbone,
                        train_cfg.max_training_steps,
                    )
                    optimizer.param_groups[param_group_idx]["lr"] = (
                        adj_lr_language_backbone
                    )

                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item()
            if train_cfg.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss

            num_tokens = float(
                torch.sum(attention_mask).item()
            )  # Sum of attention mask gives number of tokens
            total_tokens_processed += num_tokens
            post_process_time = time.time() - post_process_start

            images_per_sample = [float(len(image_pack)) for image_pack in images]

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = (
                get_world_size() * num_tokens / batch_duration
            )  # Multiply by world size to get global tokens/s

            # Accumulate training stats
            accumulated_stats["tokens_per_second"].append(tokens_per_second)
            accumulated_stats["data_load_time"].append(data_load_time)
            accumulated_stats["fw_bw_time"].append(fw_bw_time)
            accumulated_stats["post_process_time"].append(post_process_time)
            accumulated_stats["images_per_sample"].extend(images_per_sample)

            if (
                train_cfg.eval_in_epochs
                and global_step % train_cfg.eval_interval == 0
                and is_update_step
            ):
                if is_master():
                    logger.info("Starting evaluation")
                model.eval()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    total_val_loss = 0
                    val_batches = 0
                    for batch in synchronized_dataloader_step(
                        iter_val_loader, is_ddp_initialized()
                    ):
                        if val_batches > 64:
                            if is_master():
                                logger.info(f"Evaluated {val_batches} batches")
                            break
                        images = batch["images"]
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        with autocast_context:
                            _, loss = model(
                                input_ids,
                                images,
                                attention_mask=attention_mask,
                                targets=labels,
                            )

                        total_val_loss += loss.item()
                        val_batches += 1

                    iter_val_loader = iter(val_loader)
                    avg_val_loss = (
                        total_val_loss / val_batches if val_batches > 0 else 0
                    )
                    avg_val_loss = (
                        mean(cast(list[float], distributed_gather(avg_val_loss)))
                        if is_ddp_initialized()
                        else avg_val_loss
                    )

                    checkpoint_path_step = ""
                    if is_master():
                        # Save a checkpoint for this evaluation step
                        checkpoint_path_step = os.path.join(
                            vlm_cfg.vlm_checkpoint_path, run_name, f"step_{global_step}"
                        )
                        save_model = (
                            model.module if is_ddp_initialized() else model
                        )  # unwrap the model for saving if DDP
                        save_model.save_pretrained(save_directory=checkpoint_path_step)

                        if (
                            train_cfg.use_lmms_eval
                            and global_step % (train_cfg.eval_interval * 2) == 0
                        ):
                            # Submit evaluation job
                            cmd = f"sbatch eval.slurm {checkpoint_path_step} {global_step} {run_name} {train_cfg.lmms_eval_limit} {train_cfg.lmms_eval_tasks} {train_cfg.lmms_eval_batch_size}"
                            if is_master():
                                logger.info(f"Submitting evaluation job: {cmd}")
                            subprocess.run(cmd, shell=True, check=False)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        if is_master():
                            best_model_path = checkpoint_path_step

                    if is_master():
                        logger.info(
                            f"Step: {global_step}, Val Loss: {avg_val_loss:.4f}, Tokens/s: {tokens_per_second:.2f}"
                        )
                        if train_cfg.log_wandb:
                            swanlab.log({"val_loss": avg_val_loss}, step=global_step)

                model.train()

            # Log training stats every N steps (ALL RANKS must participate in collective ops) 每N步记录一次训练数据（**所有进程**均需参与集合运算）
            if (
                global_step % train_cfg.stats_log_interval == 0
                and len(accumulated_stats["tokens_per_second"]) > 0
                and is_update_step
            ):
                # ALL RANKS: Perform collective operations for training stats
                stats: dict[str, Any] = {}
                for key in [
                    "tokens_per_second",
                    "data_load_time",
                    "fw_bw_time",
                    "post_process_time",
                    "images_per_sample",
                ]:
                    if is_ddp_initialized():
                        all_values = cast(
                            list[list[float]],
                            distributed_gather(accumulated_stats[key]),
                        )
                        all_values_flat = [
                            item for sublist in all_values for item in sublist
                        ]  # Flatten list of lists
                        stats[f"avg_{key}"] = mean(all_values_flat)
                    else:
                        stats[f"avg_{key}"] = mean(accumulated_stats[key])

                for key in [
                    "data_load_time",
                    "fw_bw_time",
                    "post_process_time",
                    "images_per_sample",
                ]:
                    if is_ddp_initialized():
                        all_values = cast(
                            list[list[float]],
                            distributed_gather(accumulated_stats[key]),
                        )
                        all_values_flat = [
                            item for sublist in all_values for item in sublist
                        ]
                        stats[f"max_{key}"] = max(all_values_flat)
                    else:
                        stats[f"max_{key}"] = max(accumulated_stats[key])

                if is_ddp_initialized():
                    all_images_values = cast(
                        list[list[float]],
                        distributed_gather(accumulated_stats["images_per_sample"]),
                    )
                    all_images_flat = [
                        item for sublist in all_images_values for item in sublist
                    ]
                    stats["min_images_per_sample"] = min(all_images_flat)
                else:
                    stats["min_images_per_sample"] = min(
                        accumulated_stats["images_per_sample"]
                    )

                # MASTER ONLY: Log to wandb
                if train_cfg.log_wandb and is_master():
                    swanlab.log(
                        {
                            **{
                                f"training_stats/{key}": value
                                for key, value in stats.items()
                            },
                        },
                        step=global_step,
                    )

                    # Check for and log new lmms-eval results
                    eval_results_dir = os.path.join("eval_results", run_name)
                    if os.path.exists(eval_results_dir):
                        logged_results_count = 0
                        for result_file in os.listdir(eval_results_dir):
                            # Match only files like "step_1234.json" (no extra text)
                            match = re.fullmatch(r"step_(\d+)\.json", result_file)
                            if not match:
                                continue  # skip if the filename has extra text like taskname

                            try:
                                step = int(match.group(1))
                                if step not in logged_eval_steps:
                                    with open(
                                        os.path.join(eval_results_dir, result_file), "r"
                                    ) as f:
                                        eval_data = json.load(f)

                                    lmms_results = eval_data.get("results", {})
                                    if lmms_results:
                                        metrics = {
                                            f"lmms_eval/{key}": value
                                            for key, value in lmms_results.items()
                                        }
                                        metrics[lmms_eval_step] = eval_data[
                                            "global_step"
                                        ]
                                        if logged_results_count > 0 and is_master():
                                            logger.info(
                                                f"Logging more than one lmms-eval result for step {global_step}, try to avoid this."
                                            )
                                        swanlab.log(
                                            metrics,
                                            step=global_step + logged_results_count,
                                        )
                                        logged_results_count += 1
                                        if is_master():
                                            logger.info(
                                                f"Logged lmms-eval results from step {eval_data['global_step']}"
                                            )

                                    logged_eval_steps.add(step)
                            except (ValueError, KeyError, json.JSONDecodeError) as e:
                                if is_master():
                                    logger.warning(
                                        f"Could not process eval result file {result_file}. Error: {e}"
                                    )
                                continue

                # ALL RANKS: Reset accumulators
                for key in accumulated_stats:
                    accumulated_stats[key] = []

            # Log batch loss
            if is_update_step:
                # ALL RANKS: gather loss from all ranks if DDP
                if is_ddp_initialized():
                    batch_loss_gathered = distributed_mean_scalar(batch_loss)
                else:
                    batch_loss_gathered = batch_loss

                # MASTER ONLY: Log to wandb
                if train_cfg.log_wandb and is_master():
                    swanlab.log(
                        {
                            "batch_loss": batch_loss_gathered,
                            **(
                                {"grad_norm": grad_norm}
                                if train_cfg.max_grad_norm is not None
                                else {}
                            ),
                        },
                        step=global_step,
                    )

            if is_update_step:
                global_step += 1
                if global_step >= train_cfg.max_training_steps:
                    break
            data_load_start = time.time()

        iter_train_loader = iter(train_loader)
        avg_train_loss = total_train_loss / i
        # gather average batch loss from all ranks if DDP
        avg_train_loss = (
            mean(cast(list[float], distributed_gather(avg_train_loss)))
            if is_ddp_initialized()
            else avg_train_loss
        )

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # gather and sum total_tokens_processed across all ranks if DDP
        total_tokens_processed = (
            sum(cast(list[float], distributed_gather(total_tokens_processed)))
            if is_ddp_initialized()
            else total_tokens_processed
        )
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            if train_cfg.log_wandb:
                swanlab.log(
                    {
                        "epoch_loss": avg_train_loss,
                        "epoch_duration": epoch_duration,
                        "epoch_tokens_per_second": epoch_tokens_per_second,
                    }
                )

            logger.info(
                f"Epoch: {epoch}, Step: {global_step}/{train_cfg.max_training_steps}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}"
            )

    # Summary Statistics
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        batch_size = int(
            train_cfg.batch_size
            * get_world_size()
            * train_cfg.gradient_accumulation_steps
        )
        total_samples_processed = batch_size * global_step
        avg_time_per_sample = total_training_time / total_samples_processed
        logger.info(f"Average time per epoch: {avg_epoch_time:.2f}s")
        logger.info(f"Average time per sample: {avg_time_per_sample:.4f}s")

        # Push the best model to the hub (Please set your user name in the config!)
        if vlm_cfg.hf_repo_name is not None and best_model_path:
            if is_master():
                logger.info(
                    f"Training complete. Pushing best model from {best_model_path} to Hugging Face Hub..."
                )
            hf_model = VisionLanguageModel.from_pretrained(best_model_path)
            hf_model.push_to_hub(vlm_cfg.hf_repo_name)

        if train_cfg.log_wandb:
            swanlab.log(
                {
                    "avg_epoch_time": avg_epoch_time,
                    "avg_time_per_sample": avg_time_per_sample,
                }
            )
            swanlab.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vlm_config",
        type=str,
        help="Path to a VLMConfig JSON file (overridden by explicit CLI flags)",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        help="Path to a TrainConfig JSON file (overridden by explicit CLI flags)",
    )
    parser.add_argument(
        "--lr_mp", type=float, help="Learning rate for the mapping network"
    )
    parser.add_argument(
        "--lr_vision_backbone", type=float, help="Learning rate for the vision backbone"
    )
    parser.add_argument(
        "--lr_language_backbone",
        type=float,
        help="Learning rate for the language backbone",
    )
    parser.add_argument(
        "--vlm_checkpoint_path",
        type=str,
        help="Path to the VLM checkpoint for loading or saving",
    )
    parser.add_argument(
        "--compile", type=bool, help="Use torch.compile to optimize the model"
    )
    parser.add_argument("--log_wandb", type=bool, help="Log to wandb")
    parser.add_argument(
        "--resume_from_vlm_checkpoint",
        type=bool,
        default=False,
        help="Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)",
    )
    parser.add_argument(
        "--no_log_wandb", action="store_true", help="Do not log to wandb"
    )
    parser.add_argument("--train_dataset_path", type=str, help="Train dataset path")
    parser.add_argument(
        "--relevance_min_rating",
        type=int,
        help="Minimum relevance rating of images per sample",
    )
    parser.add_argument(
        "--image_correspondence_min_rating",
        type=int,
        help="Minimum image correspondence rating of images per sample",
    )
    parser.add_argument(
        "--visual_dependency_min_rating",
        type=int,
        help="Minimum visual dependency rating of images per sample",
    )
    parser.add_argument(
        "--formatting_min_rating",
        type=int,
        help="Minimum formatting rating of images per sample",
    )

    args = parser.parse_args()

    if args.vlm_config is not None:
        vlm_cfg = VLMConfig.from_json(args.vlm_config)
    else:
        vlm_cfg = VLMConfig()
    if args.train_config is not None:
        train_cfg = TrainConfig.from_json(args.train_config)
    else:
        train_cfg = TrainConfig()

    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_vision_backbone is not None:
        train_cfg.lr_vision_backbone = args.lr_vision_backbone
    if args.lr_language_backbone is not None:
        train_cfg.lr_language_backbone = args.lr_language_backbone
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.no_log_wandb is True:
        train_cfg.log_wandb = False
    if args.train_dataset_path is not None:
        train_cfg.train_dataset_path = args.train_dataset_path
    if args.relevance_min_rating is not None:
        train_cfg.relevance_min_rating = args.relevance_min_rating
    if args.image_correspondence_min_rating is not None:
        train_cfg.image_correspondence_min_rating = args.image_correspondence_min_rating
    if args.visual_dependency_min_rating is not None:
        train_cfg.visual_dependency_min_rating = args.visual_dependency_min_rating
    if args.formatting_min_rating is not None:
        train_cfg.formatting_min_rating = args.formatting_min_rating

    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
        train_cfg.resume_from_vlm_checkpoint = True
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()

    if is_master():
        logger.info(">>> ---------- VLM Config ---------- <<<")
        logger.info(vlm_cfg.to_json())
        logger.info(">>> ---------- Train Config ---------- <<<")
        logger.info(train_cfg.to_json())

    train(train_cfg, vlm_cfg)

    if is_ddp_initialized():
        destory_ddp_process_group()


if __name__ == "__main__":
    main()
