#!/bin/bash

# Train nanoVLM-220M (RTX 5060 Ti 16G single-GPU) or a DDP/Slurm variant.
# Single GPU:
#   bash scripts/train_vlm.sh
# Slurm / multi-GPU:
#   srun torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE \
#       --nnodes=$SLURM_NNODES \
#       --rdzv_id=$SLURM_JOB_ID \
#       --rdzv_backend=c10d \
#       --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#       -m src.train_vlm --vlm_config configs/vlm_rtx5060ti.json

python -m src.train_vlm \
    --vlm_config configs/vlm_rtx5060ti.json \
    --train_config configs/train_rtx5060ti.json
