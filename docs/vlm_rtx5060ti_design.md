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

启动入口（配置已落地为 JSON，见 `configs/vlm_rtx5060ti.json` / `configs/train_rtx5060ti.json`）：

```bash
python -m src.train_vlm --vlm_config configs/vlm_rtx5060ti.json --train_config configs/train_rtx5060ti.json
```

> 注意：不要传 `--compile False` —— `argparse` 的 `type=bool` 会把字符串 `"False"`
> 解析为 `True`；依赖 train config 的 `compile=false` 即可。

**图像缩放参数（落地时新增，必填）：**

- `max_img_size=256` + `resize_to_max_side_len=false`：任意图像缩放至长边 ≤256 →
  单个 256×256 块 → 每图恰 16 个图像 token，与 `mp_image_token_length=16` 匹配。
- 若缺省（默认 `2048/True`），`DynamicResize` 会把任意图像强制放大到长边 2048 再切块，
  单图最多产生约 1040 个图像 token，击穿 `n_positions=2048` 的打包预算。

## 6. 落地前置问题（均已在提交 3216c96 修复）

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

## 8. 训练与数据准备

### 8.1 环境

- 目标机器需能访问 HuggingFace Hub：首次运行自动下载 SmolLM2-135M-Instruct
  权重与 tokenizer、SigLIP2 权重（`from_pretrained`）及训练数据集。
- 依赖：`uv sync`（`torchvision` 已入依赖）。
- 单卡直接运行（无 `RANK`/`WORLD_SIZE` 环境变量则自动跳过 DDP，`src/train_vlm.py:994`）。
- 显存预算约 8 GB，16G 卡余量充足（可升 batch 2）。

### 8.2 启动训练

```bash
python -m src.train_vlm \
  --vlm_config configs/vlm_rtx5060ti.json \
  --train_config configs/train_rtx5060ti.json
```

- 等效全局 batch = `batch_size(1) × gradient_accumulation_steps(16) = 16`。
- 每 500 步自动 eval 并存档到 `checkpoints/<run_name>/step_<N>/`（safetensors）。
- 换数据集用 `--train_dataset_path <path>`；其余超参数经 `--train_config` JSON 覆盖。
- 训练 loss 在每个优化步打印到控制台（`Step: N, Loss: ...`），并同步记录到 swanlab。

### 8.2.1 swanlab 实验记录

- `train_rtx5060ti.json` 中 `log_wandb=true`、`wandb_entity="anxiangjing"`（改成你的 swanlab 用户名）。
- 首次使用需登录：`uv run swanlab login`（或设置环境变量 `SWANLAB_API_KEY`）。
- 记录内容：`batch_loss`、`grad_norm`、`val_loss`、`epoch_loss`、吞吐与数据加载统计；
  实验名自动生成（含模型/GPU/batch/步数等，`get_run_name`）。
- 关闭记录：`--no_log_wandb` 或把 JSON 中 `log_wandb` 改回 `false`。

### 8.3 断点续训

```bash
python -m src.train_vlm \
  --vlm_config configs/vlm_rtx5060ti.json \
  --train_config configs/train_rtx5060ti.json \
  --resume_from_vlm_checkpoint True \
  --vlm_checkpoint_path checkpoints/<run_name>
```

续训会自动置 `vlm_load_backbone_weights=False`（`src/train_vlm.py:989-992`）。

### 8.4 数据格式要求

训练数据为 HF `datasets` 格式，每个样本包含：

| 字段 | 类型 | 说明 |
|---|---|---|
| `images` | PIL / 图像列表 | 每个样本的关联图像 |
| `texts` | `{"user": str, "assistant": str}` 列表 | 图文对话对 |
| `relevance_ratings` 等 | 可选 int | 低于 `train_cfg` 对应 `*_min_rating=1` 的样本被过滤 |

处理流程（`src/data/vqa_datasets.py`）：

1. 图像经 `DynamicResize`（长边 ≤256）→ 单 256×256 块 → 16 个图像 token。
2. 文本经 `lm_chat_template`（im_start 格式）套用为对话。
3. `ConstantLengthDataset` 将样本打包至 `seq_length=2048`（`src/train_vlm.py:226`）。
4. `VQACollator` 左填充至 `max_sample_length`，标签 padding 用 `-100`。

默认数据集 `HuggingFaceM4/FineVision` + `sharegpt4v(coco)`（`stream_dataset=False`
全量落盘）；数据下载可在有网环境单独完成，再通过 `--train_dataset_path` 指定本地路径。
