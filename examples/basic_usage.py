"""
基础使用示例 - 展示如何训练和使用GPT模型
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.configuration_model import GPTConfig
from src.model import GPT
from src.trainer import Trainer
from src.inference import TextGenerator, ChatBot
from common.tokenizer import get_tokenizer


def create_sample_data(vocab_size: int = 1000, seq_len: int = 32, num_samples: int = 1000):
    """创建示例训练数据"""
    # 生成随机token序列
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # 目标序列是输入序列向右移动一位
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    targets[:, -1] = 0  # 最后一个token的目标设为0
    
    return TensorDataset(input_ids, targets)


def example_training():
    """训练示例"""
    print("=== GPT模型训练示例 ===")
    
    # 创建配置
    config = GPTConfig(
        vocab_size=1000,
        n_positions=128,
        n_embd=64,
        n_layer=4,
        n_head=4,
        dropout=0.1,
    )
    
    # 创建模型
    model = GPT(config)
    print(f"模型参数: {model.get_num_params()/1e3:.1f}K")
    
    # 创建示例数据
    dataset = create_sample_data(vocab_size=1000, seq_len=32, num_samples=1000)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        learning_rate=1e-3,
        weight_decay=0.1,
        grad_clip=1.0,
        device="cpu",  # 使用CPU进行演示
    )
    
    # 训练少量步骤进行演示
    print("开始训练...")
    trainer.train(max_epochs=2, eval_interval=50, checkpoint_interval=100)
    
    print("训练完成！")
    return model


def example_inference(model: GPT):
    """推理示例"""
    print("\n=== GPT模型推理示例 ===")
    
    # 创建模拟分词器
    class MockTokenizer:
        def encode(self, text):
            # 简单映射：字符到token ID
            return [ord(c) for c in text if ord(c) < 1000]
        
        def decode(self, ids):
            return ''.join(chr(i) for i in ids if i < 1000)
        
        def get_vocab_size(self):
            return 1000
        
        def get_bos_token_id(self):
            return 1
    
    tokenizer = MockTokenizer()
    
    # 创建文本生成器
    generator = TextGenerator(model, tokenizer, device="cpu")
    
    # 生成文本
    prompt = "Hello"
    generated = generator.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        do_sample=True
    )
    
    print(f"提示: {prompt}")
    print(f"生成: {generated}")
    
    return generator


def example_chatbot(model: GPT):
    """聊天机器人示例"""
    print("\n=== 聊天机器人示例 ===")
    
    # 创建模拟分词器
    class MockTokenizer:
        def encode(self, text):
            return [ord(c) for c in text if ord(c) < 1000]
        
        def decode(self, ids):
            return ''.join(chr(i) for i in ids if i < 1000)
        
        def get_vocab_size(self):
            return 1000
        
        def get_bos_token_id(self):
            return 1
        
        def encode_special(self, token):
            return 1  # 简单映射
    
    tokenizer = MockTokenizer()
    
    # 创建聊天机器人
    chatbot = ChatBot(
        model=model,
        tokenizer=tokenizer,
        system_prompt="你是一个有帮助的AI助手。",
        device="cpu"
    )
    
    # 测试对话
    test_messages = [
        "你好，请介绍一下你自己。",
        "Python是什么？",
        "谢谢你的帮助！"
    ]
    
    for message in test_messages:
        print(f"用户: {message}")
        response = chatbot.chat(message, max_new_tokens=30)
        print(f"AI: {response}")
        print("-" * 40)
    
    return chatbot


def example_pretrained_loading():
    """加载预训练模型示例"""
    print("\n=== 加载预训练模型示例 ===")
    
    try:
        # 尝试加载GPT-2模型（需要安装transformers）
        model = GPT.from_pretrained("gpt2")
        print("✅ 成功加载GPT-2模型")
        print(f"模型参数: {model.get_num_params()/1e6:.2f}M")
        
        return model
    except ImportError:
        print("❌ 需要安装transformers库: pip install transformers")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
    
    return None


def main():
    """主函数"""
    print("GPT模型示例程序")
    print("=" * 50)
    
    # 示例1: 训练模型
    model = example_training()
    
    # 示例2: 文本生成
    generator = example_inference(model)
    
    # 示例3: 聊天机器人
    chatbot = example_chatbot(model)
    
    # 示例4: 加载预训练模型
    pretrained_model = example_pretrained_loading()
    
    print("\n" + "=" * 50)
    print("所有示例执行完成！")
    
    # 保存模型示例
    if model:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': model.config.to_dict(),
        }
        torch.save(checkpoint, 'example_model.pth')
        print("✅ 模型已保存到: example_model.pth")


if __name__ == "__main__":
    main()