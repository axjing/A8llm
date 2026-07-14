# XlmPrimer - 从第一性原理构建的轻量级 GPT 与视觉语言模型实现

基于第一性原理设计的轻量级 GPT 语言模型和视觉语言模型 (VLM) 实现，
参考 [nanoGPT](https://github.com/karpathy/nanoGPT) 和 [nanochat](https://github.com/karpathy/nanochat) 的最佳实践。

## 特性

- **简洁高效** — 代码简洁易懂，支持 GPT-2 到中等规模的训练
- **双模型家族** — LLM（GPT-2 风格）和 VLM（ViT + Llama 风格语言模型）
- **现代特性** — Flash Attention (FA3)、RoPE、分组查询注意力 (GQA)、KV Cache 推理
- **混合精度** — FP8 训练支持（PyTorch Float8Linear）
- **分布式训练** — `torchrun` / DDP 多 GPU 训练，Scaling Law 自动计算最优参数
- **完整工具链** — Tokenizer 训练、预训练、SFT、推理、评估
- **类型安全** — 完整的类型注解，Google-style 文档字符串
- **HuggingFace 兼容** — 可加载 HF 权重 (SmolLM2, GPT-2, SigLIP)

## 项目结构

```
XlmPrimer/
├── src/
│   ├── models/                    # 模型定义
│   │   ├── config.py              #   LLMConfig / VLMConfig (dataclass)
│   │   ├── gpt.py                 #   GPT-2 风格解码器 (Block, GPT)
│   │   ├── layers.py              #   基础模块 (Linear, Conv1D, LayerNorm, RMSNorm, ...)
│   │   ├── language_model.py      #   LlamaTransformer (RoPE, RMSNorm, GQA, KV Cache)
│   │   ├── flash_attention.py     #   Flash Attention 3 (Hopper GPUs)
│   │   ├── position_embedding.py  #   RoPE 位置编码
│   │   ├── vision_transformer.py  #   ViT (SigLIP 风格视觉编码器)
│   │   ├── vision_language_model.py # VisionLanguageModel (ViT + Llama + Projector)
│   │   └── modality_projector.py  #   跨模态投影 (pixel shuffle)
│   ├── trainer/                   # 训练工具
│   │   ├── config.py              #   训练配置
│   │   ├── distributed.py         #   DDP 分布式训练
│   │   ├── optim.py               #   优化器配置
│   │   ├── train_fp8.py           #   FP8 训练
│   │   ├── train_tokenizer.py     #   Tokenizer 训练
│   │   └── eval_tokenizer.py      #   Tokenizer 评估
│   ├── data/                      # 数据处理
│   │   ├── get_datasets.py        #   HuggingFace 数据集下载
│   │   ├── text_pretrain_loader.py #   文本预训练数据加载
│   │   ├── vqa_datasets.py        #   VQA 数据集
│   │   ├── processors.py          #   Tokenizer / Processor 工具
│   │   └── ...
│   ├── engine/                    # 推理引擎
│   │   ├── engine_inference.py    #   Engine (KV Cache 推理, tool-use 状态机)
│   │   └── utils_checkpoints.py   #   检查点工具
│   ├── evaluator/                 # 评估
│   │   ├── eval_core.py           #   CORE 指标评估
│   │   └── eval_loss.py           #   基于 Loss 的评估
│   ├── common/                    # 通用工具
│   │   ├── execution.py           #   执行工具
│   │   ├── file_os.py             #   文件操作
│   │   ├── logger.py              #   日志 (SwanLab)
│   │   └── tokenizer.py           #   Tokenizer 工具
│   ├── train_llm.py               # LLM 预训练入口 (Muon + AdamW 混合优化)
│   ├── train_vlm.py               # VLM 训练入口
│   ├── eval.py                    # 评估入口
│   └── report.py                  # 训练报告生成
├── configs/                       # 配置文件
│   └── gpt2.json                  #   GPT-2 配置示例
├── scripts/                       # 训练脚本
│   ├── train_gpt.sh               #   训练启动脚本 (Slurm + torchrun)
│   └── train_vlm.sh               #   VLM 训练启动脚本
├── tests/                         # 单元测试
│   ├── test_modeling.py
│   ├── test_language_model.py
│   ├── test_layers.py
│   ├── test_engine_kvcache.py
│   ├── test_tokenizer.py
│   └── common.py
└── examples/
    └── basic_usage.py             # 基础使用示例
```

## 安装

```bash
# 使用 uv (推荐)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync

# 或手动安装
pip install torch torchvision torchaudio
pip install tiktoken tokenizers
pip install transformers  # 可选，用于加载预训练模型
pip install datasets      # 可选，用于数据集加载
pip install ruff mypy pytest  # 可选，用于开发
```

## 快速开始

### 创建模型

```python
from src.models.config import LLMConfig, get_llm_config
from src.models.gpt import GPT

# 方式 1: 使用预定义配置
config = get_llm_config("small")  # tiny / small / base / medium / large / xl

# 方式 2: 自定义配置
config = LLMConfig(
    vocab_size=32768,
    n_positions=1024,
    n_embd=768,
    n_layers=12,
    n_heads=12,
    dropout=0.1,
)

# 创建模型
model = GPT(config)
print(f"模型参数: {model.get_num_params()/1e6:.2f}M")

# 前向传播
import torch
input_ids = torch.randint(0, config.vocab_size, (2, 32))
logits, loss = model(input_ids)
print(f"输出形状: {logits.shape}")  # [2, 32, vocab_size]
```

### 加载预训练权重

```python
from src.models.llama import LlamaTransformer

# 从 HuggingFace Hub 加载
model = LlamaTransformer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
print(f"模型参数: {model.get_num_params()/1e6:.2f}M")

# 从本地检查点加载
config = LLMConfig.from_json("configs/gpt2.json")
model = GPT(config)
checkpoint = torch.load("checkpoint.pt", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
```

### 训练

```bash
# 单命令启动完整训练流程 (Tokenizer → 预训练 → 评估)
bash scripts/train_gpt.sh

# 或手动分步执行
# 1. 下载数据集
python -m src.data.get_datasets -n 170

# 2. 训练 Tokenizer
python -m src.trainer.train_tokenizer

# 3. 预训练 (8 GPU)
torchrun --standalone --nproc_per_node=8 -m src.train -- --depth=24 --device-batch-size=16 --fp8

# 4. 评估
torchrun --standalone --nproc_per_node=8 -m src.eval -- --device-batch-size=16
```

### 推理 (KV Cache)

```python
from src.engine.engine_inference import Engine
from src.models.gpt import GPT
from src.models.config import LLMConfig

# 加载模型
config = LLMConfig()
model = GPT(config)

# 创建推理引擎 (带 KV Cache)
engine = Engine(model)

# 生成文本
tokens = engine.generate(input_ids, max_new_tokens=50, temperature=0.8)
```

### 视觉语言模型 (VLM)

```python
from src.models.config import VLMConfig
from src.models.vision_language_model import VisionLanguageModel

# VLM: SigLIP ViT + SmolLM2 + ModalityProjector
model = VisionLanguageModel.from_pretrained()
print(f"VLM 参数: {model.get_num_params()/1e6:.2f}M")

# 图文生成
output = model.generate(input_ids, pixel_values, max_new_tokens=50)
```

## 核心模块

### 模型配置

`LLMConfig` / `VLMConfig` 遵循 HuggingFace Transformers 模式，支持 `from_pretrained()`、
`from_json()`、`save()`、`update_from_hf_config()` 等方法。

### GPT 模型

GPT-2 风格因果语言模型，架构：`Embedding → Block×N → LayerNorm → LMHead`
其中 `Block = LayerNorm → CausalSelfAttention → LayerNorm → MLP`。
支持绝对位置编码、窗口注意力模式 (`window_pattern`)。

### Language Model (Llama 风格)

`LlamaTransformer` 包含 RoPE 位置编码、RMSNorm、分组查询注意力 (GQA)、KV Cache，
支持从 HuggingFace Hub 加载 SmolLM2 权重。

### 训练

`src/train_llm.py` 支持：DDP 分布式、cosine LR 调度、warmup/warmdown、
Chinchilla Scaling Law 自动计算最优参数、FP8 训练、检查点保存/恢复、
训练过程评估和采样。

### VLM 训练

`src/train_vlm.py` 支持：图文联合训练、梯度累积、DDP、lmms-eval 集成评估。

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 或运行特定测试
python -m pytest tests/test_modeling.py -v
python -m pytest tests/test_engine_kvcache.py -v
```

## 开发

```bash
# 代码检查
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# 类型检查
uv run mypy src/ tests/ --strict

# 遵循 PEP 8 编码规范、Google-style 文档字符串、完整类型注解
```

## 贡献

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'feat(models): add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

本项目参考了以下优秀开源项目：

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy 的极简 GPT 实现
- [nanochat](https://github.com/karpathy/nanochat) — 轻量级聊天模型框架
- [HuggingFace Transformers](https://github.com/huggingface/transformers) — 优秀的 NLP 库

---

**XlmPrimer** — 让每个人都能理解和使用的 GPT 与 VLM 实现！
