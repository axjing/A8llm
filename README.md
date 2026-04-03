# A8llm - 基于第一性原理的轻量级GPT实现

一个基于第一性原理设计的轻量级GPT语言模型实现，参考nanoGPT和nanochat的最佳实践。

## 🚀 特性

- **简洁高效**: 基于第一性原理设计，代码简洁易懂
- **模块化架构**: 高度模块化的组件设计，易于扩展和维护
- **现代化实现**: 支持Flash Attention、混合精度训练等现代特性
- **完整工具链**: 提供训练、推理、评估等完整工具
- **类型安全**: 完整的类型注解，提高代码可靠性
- **全面测试**: 完整的单元测试和集成测试

## 📁 项目结构

```
A8llm/
├── src/                          # 源代码
│   ├── configuration_model.py   # 模型配置
│   ├── model.py                 # GPT模型实现
│   ├── tokenizer.py             # 分词器模块
│   ├── trainer.py               # 训练器
│   ├── inference.py             # 推理工具
│   ├── test_model.py            # 测试套件
│   └── common.py                # 通用工具
├── examples/                     # 使用示例
│   └── basic_usage.py           # 基础使用示例
├── 3rdparty/                    # 第三方参考实现
│   ├── nanochat/                # nanochat参考
│   └── nanoGPT/                 # nanoGPT参考
└── README.md                    # 项目文档
```

## 🛠️ 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio

# 分词器支持
pip install tiktoken tokenizers

# 加载预训练模型（可选）
pip install transformers

# 开发依赖
pip install pytest black isort mypy
```

## 🚀 快速开始

### 基础使用

```python
from src.configuration_model import GPTConfig
from src.model import GPT

# 创建模型配置
config = GPTConfig(
    vocab_size=1000,
    n_positions=128,
    n_embd=64,
    n_layer=4,
    n_head=4
)

# 创建模型
model = GPT(config)
print(f"模型参数: {model.get_num_params()/1e3:.1f}K")

# 前向传播
input_ids = torch.randint(0, 1000, (2, 32))
logits, loss = model(input_ids)
print(f"输出形状: {logits.shape}")
```

### 文本生成

```python
from src.inference import TextGenerator
from src.tokenizer import get_tokenizer

# 创建生成器
tokenizer = get_tokenizer("tiktoken")
generator = TextGenerator(model, tokenizer)

# 生成文本
prompt = "今天天气很好，"
generated = generator.generate(
    prompt,
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.9
)
print(f"生成结果: {generated}")
```

### 训练模型

```python
from src.trainer import Trainer
from torch.utils.data import DataLoader

# 创建训练器
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    learning_rate=3e-4,
    weight_decay=0.1
)

# 开始训练
trainer.train(max_epochs=10)
```

## 📚 核心模块

### 1. 模型配置 (configuration_model.py)

```python
config = GPTConfig(
    vocab_size=50304,      # 词汇表大小
    n_positions=1024,      # 最大序列长度
    n_embd=768,            # 隐藏层维度
    n_layer=12,             # Transformer层数
    n_head=12,             # 注意力头数
    dropout=0.1,           # Dropout率
    bias=False             # 是否使用偏置
)
```

### 2. GPT模型 (model.py)

- **LayerNorm**: 带可选偏置的层归一化
- **Linear**: 支持混合精度的线性层
- **CausalSelfAttention**: 因果自注意力（支持Flash Attention）
- **MLP**: 多层感知机
- **Block**: Transformer块
- **GPT**: 完整的GPT语言模型

### 3. 分词器 (tokenizer.py)

支持多种分词器后端：

- **TiktokenTokenizer**: 基于tiktoken的高效分词器
- **HFTokenizerWrapper**: HuggingFace分词器包装器
- **TokenizerFactory**: 分词器工厂类

### 4. 训练器 (trainer.py)

提供完整的训练功能：

- 梯度裁剪和混合精度训练
- 学习率调度和早停
- 模型检查点和恢复
- 训练进度监控

### 5. 推理工具 (inference.py)

- **TextGenerator**: 文本生成器
- **ChatBot**: 聊天机器人
- 支持多种生成策略（top-k, top-p, 温度采样）

## 🧪 测试

运行完整测试套件：

```bash
cd src
python test_model.py
```

或者使用pytest：

```bash
pytest src/test_model.py -v
```

## 🔧 开发指南

### 代码规范

- 遵循PEP 8编码规范
- 使用类型注解提高代码可靠性
- 编写详细的文档字符串
- 保持函数单一职责原则

### 扩展模型

要添加新的模型组件：

1. 在`model.py`中定义新的模块类
2. 添加相应的单元测试
3. 更新模型配置类（如需要）
4. 更新文档

### 性能优化

- 使用Flash Attention加速注意力计算
- 启用混合精度训练减少内存占用
- 使用梯度检查点节省显存
- 优化数据加载器提高训练速度

## 📊 性能基准

| 模型规模 | 参数数量 | 训练速度 | 内存占用 |
|---------|---------|---------|---------|
| Small   | 124M    | 快      | 低      |
| Medium  | 350M    | 中等    | 中等    |
| Large   | 774M    | 慢      | 高      |

## 🤝 贡献指南

我们欢迎各种形式的贡献！请参考以下步骤：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 🙏 致谢

本项目参考了以下优秀开源项目：

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy的极简GPT实现
- [nanochat](https://github.com/yourusername/nanochat) - 轻量级聊天模型框架
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - 优秀的NLP库

## 📞 联系我们

- 项目主页: [GitHub Repository]
- 问题反馈: [GitHub Issues]
- 邮箱: <your-email@example.com>

---

**A8llm** - 让每个人都能理解和使用的GPT实现！
