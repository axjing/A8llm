# nanoVLM-220M 设计：RTX 5060 Ti 16G 单卡可训练

## 1. 目标与约束

- 硬件：NVIDIA GeForce RTX 5060 Ti 16G（单卡）。
- 显存预算：模型权重 + 梯度 + 优化器状态 + 激活值，总占用需显著低于 16 GB，留出评估与 CUDA context 余量。
- 复用本项目现有架构：`VisionLanguageModel`（`src/models/vision_language_model.py`）=
  `ViT`（`src/models/vision_transformer.py`）+ `ModalityProjector`（`src/models/modality_projector.py`）
  + `LlamaTransformer`（`src/models/llama.py`），不新增模块。
- 骨干权重通过现有 `from_pretrained` 流程加载。

## 2. 方案概览（实测参数量）

| 组件 | 配置 | 参数量 |
|---|---|---|
| 视觉编码器 | `google/siglip2-base-patch16-256` | 86.4M |
| 语言骨干 | `HuggingFaceTB/SmolLM2-135M-Instruct` | 134.6M |
| 投影层 | Linear(768×16 → 576) | 7.1M |
| **总计** | | **228M** |

> 参数量在项目代码上直接构造模型实测得到（未下载权重）。
> 参考：项目 `MODEL_CARD_TEMPLATE` 中描述的 nanoVLM-222M（SigLIP-B/16 + SmolLM2-135M），本设计即为其正确落地版本。

## 3. 关键配置（VLMConfig）

```python
vit_model_type        = "google/siglip2-base-patch16-256"
image_size            = 256          # 256 分辨率 → 256 patch → pixel-shuffle×4 → 16 图像 token
patch_size            = 16
vit_n_embd            = 768
vit_n_intermediate    = 3072
vit_n_heads           = 12
vit_n_layers          = 12

lm_model_type         = "HuggingFaceTB/SmolLM2-135M-Instruct"
lm_tokenizer          = "HuggingFaceTB/SmolLM2-135M-Instruct"
n_positions           = 2048         # SmolLM2-135M 原生上下文
n_embd                = 576
n_layers              = 30
n_heads               = 9
n_kv_heads            = 3
n_intermediate        = 1536
vocab_size            = 49152 + len(vlm_extra_tokens)

mp_pixel_shuffle_factor = 4
mp_image_token_length   = 16          # 256 patch / 16 = 16
vlm_load_backbone_weights = True
```

**为什么 image_size 必须用 256 而不是 512：**

- 512 分辨率 → 32×32=1024 个 patch，ViT 自注意力矩阵 1024×1024，是单卡训练的主要计算与显存负担。
- 256 分辨率 → 16×16=256 个 patch，pixel-shuffle×4 后仅 **16 个图像 token**，ViT 注意力成本降低约 16×。
- SigLIP2 官方训练即使用 256 分辨率（patch grid 16×16），是模型的原生最优分辨率。

**为什么语言骨干用 135M 而非默认的 360M：**

- 460M 总参 + 512 分辨率在 16G 卡上可运行但激活值吃紧、训练速度慢。
- 135M（228M 总参）保留完整对话能力，且 `from_pretrained` 会从 HF config 自动套用 576/30/9/3/2048 等架构参数。

## 4. 显存预算（估算）

| 项目 | 估算 |
|---|---|
| 权重 (fp32) | 228M × 4B ≈ 0.9 GB |
| 梯度 (fp32) | ≈ 0.9 GB |
| AdamW 状态 (fp32, m+v) | ≈ 1.8 GB |
| 激活值 (batch 1, seq 2048, bf16) | ≈ 2–4 GB |
| **合计** | **≈ 8 GB** |
| 16 GB 余量 | ~8 GB（可升 batch 2 + 梯度累积 8） |

训练采用 `torch.autocast(bf16)`（`src/train_vlm.py:504`），保持现有 fp32 主权重 + bf16 计算模式。

## 5. 训练配置（TrainConfig）

```python
batch_size                  = 1
gradient_accumulation_steps = 16        # 等效全局 batch 16
max_sample_length           = 2048
max_training_steps          = 15000
lr_mp                       = 5e-4      # 投影层新权重，学习率最高
lr_vision_backbone          = 5e-5
lr_language_backbone        = 5e-5
max_grad_norm               = 1.0
max_images_per_example      = 4
max_images_per_knapsack     = 8
use_lmms_eval               = False     # 本地单卡，无 Slurm
eval_interval               = 500
```

启动入口应为：

```bash
python -m src.train_vlm \
  --lr_mp 5e-4 --lr_vision_backbone 5e-5 --lr_language_backbone 5e-5 \
  --no_log_wandb --train_dataset_path <dataset> --compile False
```

## 6. 落地前置问题（设计阶段不修复，仅记录）

1. **`VLMConfig.__post_init__` 无条件覆盖**（`src/models/config.py:242-267`）会把
   `n_embd/n_layers/n_heads/n_kv_heads/n_intermediate/n_positions/vocab_size` 覆盖回 360M 默认值。
   语言骨干尺寸最终由 `from_pretrained` 从 HF config 覆盖，因此 135M 设计可运行；但
   `image_size=256` 等字段需确保不被覆盖，落地时需给 `__post_init__` 增加"仅在未显式指定时覆盖"逻辑。
2. **`train_vlm.py::main()` 不支持 JSON/CLI 加载配置**，仅构造 `VLMConfig()`。落地需扩展
   `--config <path>` 参数，并新增 `configs/vlm_rtx5060ti.json`。
3. **`scripts/train_vlm.sh:8`** 引用不存在的 `../src/train.py`，应改为 `python -m src.train_vlm`。
4. **`src/train_vlm.py:653`** `sbatch eval.slurm` 依赖 Slurm 环境，单卡本地训练应设 `use_lmms_eval=False`。
5. **`src/models/vision_transformer.py:153-159`** `ViTBlock.forward` 残差写法错误
   （pre-norm 应为 `x = x + attn(ln1(x))`），加载 SigLIP 权重后输出损坏，训练前需修复。
6. **`src/models/llama.py:433`** `loaded_keys.add()` 缩进错误，仅影响多分片 checkpoint
   （SmolLM2-1.7B/3B）；135M 为单文件，本设计不受影响。

## 7. 扩展方向（可选）

- 视觉更细：升级 `image_size=512` 需把 `mp_image_token_length` 改为 64，并接受约 4× 的 ViT 计算开销。
- 更强语言：升级 `SmolLM2-360M-Instruct`（总参 460M），需配合 256 分辨率并降低 batch。
- 显存进一步压缩：冻结视觉骨干（`lr_vision_backbone=0`）可将训练预算降至 ~5 GB。
