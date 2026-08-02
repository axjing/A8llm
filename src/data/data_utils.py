from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.distributed as dist


def _is_batch_valid(batch: dict[str, Any]) -> bool:
    """
    Check if a batch is valid for training/evaluation.
    A valid batch must have input_ids and at least one image.
    """
    if not batch:
        return False
    # The collator can return a batch with empty lists
    if len(batch["input_ids"]) == 0:
        return False

    if len(batch["images"]) == 0:
        return False

    # `images` is a list of lists of tensors. Check that at least one image is not None.
    # During training, not having images creates gradients computed without all model parameters.
    # This creates deadlocks in DDP.
    return len([img for sublist in batch["images"] for img in sublist]) != 0


def synchronized_dataloader_step(
    train_loader: Iterable[dict[str, Any]], is_dist: bool
) -> Iterator[dict[str, Any]]:
    """
    Create a synchronized iterator that handles uneven data distribution in DDP.
    All ranks will stop when the first rank runs out of data.
    This happens because when packing a presharded dataset, a rank might have less groups than the others.
    It also handles cases where a collator returns an empty/invalid batch on some ranks,
    by ensuring all ranks skip the invalid batch and attempt to fetch a new one.
    """
    if not is_dist:
        # For single GPU, we don't need synchronization, just filter invalid batches.
        for batch in train_loader:
            if _is_batch_valid(batch):
                yield batch
        return

    # For DDP, we need synchronization.
    if isinstance(train_loader, Iterator):
        train_iter = train_loader
    else:
        train_iter = iter(train_loader)

    while True:
        valid_batch: dict[str, Any] | None = None
        has_data: torch.Tensor | None = None
        try:
            while True:
                candidate = next(train_iter)
                if _is_batch_valid(candidate):
                    valid_batch = candidate
                    break
            has_data = torch.tensor(1, device=torch.cuda.current_device())
        except StopIteration:
            valid_batch = None
            has_data = torch.tensor(0, device=torch.cuda.current_device())

        assert has_data is not None
        # We synchronize across all ranks. If any rank is out of data, all ranks stop.
        dist.all_reduce(has_data, op=dist.ReduceOp.MIN)

        if has_data.item() == 0:
            # At least one rank is out of data. All ranks should stop.
            break
        assert valid_batch is not None
        yield valid_batch
    return None
