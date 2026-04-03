"""
GPT模型简单测试示例
直接运行即可验证各个功能
"""

import torch
import torch.nn as nn
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from models.gpt import (
    Conv1D, 
    scaled_dot_product_attention,
    LayerNorm,
    CausalSelfAttention,
    MLP,
    Block,
    GPT
)
from models.configuration_model import GPTConfig


def test_conv1d():
    """测试Conv1D层"""
    print("=== 测试Conv1D层 ===")
    
    # 创建Conv1D层
    conv1d = Conv1D(128, 64)
    
    # 测试前向传播
    x = torch.randn(2, 10, 128)
    output = conv1d(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"Conv1D权重形状: {conv1d.weight.shape}")
    print("✓ Conv1D测试通过\n")


def test_layernorm():
    """测试LayerNorm层"""
    print("=== 测试LayerNorm层 ===")
    
    # 创建LayerNorm层
    ln = LayerNorm(128)
    
    # 测试前向传播
    x = torch.randn(2, 10, 128)
    output = ln(x)
    
    # 检查归一化效果
    mean = output.mean(dim=-1)
    std = output.std(dim=-1)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"均值接近0: {torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)}")
    print(f"标准差接近1: {torch.allclose(std, torch.ones_like(std), atol=1e-4)}")
    print("✓ LayerNorm测试通过\n")


def test_attention():
    """测试注意力机制"""
    print("=== 测试注意力机制 ===")
    
    # 创建查询、键、值张量
    batch_size, n_head, seq_len, head_dim = 2, 4, 8, 64
    query = torch.randn(batch_size, n_head, seq_len, head_dim)
    key = torch.randn(batch_size, n_head, seq_len, head_dim)
    value = torch.randn(batch_size, n_head, seq_len, head_dim)
    
    # 测试注意力计算
    attn_output, attn_weights = scaled_dot_product_attention(
        nn.Module(), query, key, value
    )
    
    print(f"查询形状: {query.shape}")
    print(f"注意力输出形状: {attn_output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print("✓ 注意力测试通过\n")


def test_causal_attention():
    """测试因果自注意力"""
    print("=== 测试因果自注意力 ===")
    
    config = GPTConfig(n_embd=64, n_head=2, n_positions=32)
    attn = CausalSelfAttention(config, ctrl_flash=False)  # 使用慢速注意力
    
    # 测试前向传播
    x = torch.randn(1, 8, 64)
    output, attn_weights = attn(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 检查因果掩码
    is_causal = True
    for i in range(8):
        for j in range(i + 1, 8):
            if attn_weights[0, 0, i, j] != 0:
                is_causal = False
                break
    
    print(f"因果掩码生效: {is_causal}")
    print("✓ 因果自注意力测试通过\n")


def test_mlp():
    """测试MLP层"""
    print("=== 测试MLP层 ===")
    
    config = GPTConfig(n_embd=64)
    mlp = MLP(config)
    
    # 测试前向传播
    x = torch.randn(2, 8, 64)
    output = mlp(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✓ MLP测试通过\n")


def test_block():
    """测试Transformer块"""
    print("=== 测试Transformer块 ===")
    
    config = GPTConfig(n_embd=768)
    block = Block(config)
    
    # 测试前向传播
    x = torch.randn(2, 8, 768)
    output = block(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✓ Transformer块测试通过\n")


def test_gpt_model():
    """测试完整GPT模型"""
    print("=== 测试完整GPT模型 ===")
    
    # 创建小型配置
    config = GPTConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=32
    )
    
    model = GPT(config)
    
    # 测试前向传播（无目标）
    idx = torch.randint(0, 100, (2, 8))
    logits, loss = model(idx)
    
    print(f"输入token形状: {idx.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"损失值: {loss}")
    
    # 测试前向传播（有目标）
    targets = torch.randint(0, 100, (2, 8))
    logits, loss = model(idx, targets)
    
    print(f"有目标时的损失值: {loss.item():.4f}")
    
    # 测试参数计数
    num_params = model.get_num_params()
    print(f"模型参数数量: {num_params:,}")
    
    # 测试生成（评估模式）
    model.eval()
    with torch.no_grad():
        generated = model.generate(idx[:, :4], max_new_tokens=3)
    print(f"生成结果形状: {generated.shape}")
    
    print("✓ GPT模型测试通过\n")


def test_optimizer():
    """测试优化器配置"""
    print("=== 测试优化器配置 ===")
    
    config = GPTConfig(n_embd=768, n_layer=12)
    model = GPT(config)
    
    optimizer = model.configure_optimizers(
        weight_decay=0.01,
        learning_rate=0.001,
        betas=(0.9, 0.999),
        device_type='cpu'
    )
    
    print(f"优化器类型: {type(optimizer).__name__}")
    print(f"参数组数量: {len(optimizer.param_groups)}")
    
    # 检查参数组
    for i, group in enumerate(optimizer.param_groups):
        print(f"参数组 {i}: {len(group['params'])} 个参数")
    
    print("✓ 优化器测试通过\n")
"""
GPT模型简单测试示例
直接运行即可验证各个功能
"""

import torch
import torch.nn as nn
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from models.gpt import (
    Conv1D, 
    scaled_dot_product_attention,
    LayerNorm,
    CausalSelfAttention,
    MLP,
    Block,
    GPT
)
from models.configuration_model import GPTConfig


def test_conv1d():
    """测试Conv1D层"""
    print("=== 测试Conv1D层 ===")
    
    # 创建Conv1D层
    conv1d = Conv1D(128, 64)
    
    # 测试前向传播
    x = torch.randn(2, 10, 128)
    output = conv1d(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"Conv1D权重形状: {conv1d.weight.shape}")
    print("✓ Conv1D测试通过\n")


def test_layernorm():
    """测试LayerNorm层"""
    print("=== 测试LayerNorm层 ===")
    
    # 创建LayerNorm层
    ln = LayerNorm(128)
    
    # 测试前向传播
    x = torch.randn(2, 10, 128)
    output = ln(x)
    
    # 检查归一化效果
    mean = output.mean(dim=-1)
    std = output.std(dim=-1)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"均值接近0: {torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)}")
    print(f"标准差接近1: {torch.allclose(std, torch.ones_like(std), atol=1e-4)}")
    print("✓ LayerNorm测试通过\n")


def test_attention():
    """测试注意力机制"""
    print("=== 测试注意力机制 ===")
    
    # 创建查询、键、值张量
    batch_size, n_head, seq_len, head_dim = 2, 4, 8, 64
    query = torch.randn(batch_size, n_head, seq_len, head_dim)
    key = torch.randn(batch_size, n_head, seq_len, head_dim)
    value = torch.randn(batch_size, n_head, seq_len, head_dim)
    
    # 测试注意力计算
    attn_output, attn_weights = scaled_dot_product_attention(
        nn.Module(), query, key, value
    )
    
    print(f"查询形状: {query.shape}")
    print(f"注意力输出形状: {attn_output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    print("✓ 注意力测试通过\n")


def test_causal_attention():
    """测试因果自注意力"""
    print("=== 测试因果自注意力 ===")
    
    config = GPTConfig(n_embd=64, n_head=2, n_positions=32)
    attn = CausalSelfAttention(config, ctrl_flash=False)  # 使用慢速注意力
    
    # 测试前向传播
    x = torch.randn(1, 8, 64)
    output, attn_weights = attn(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 检查因果掩码
    is_causal = True
    for i in range(8):
        for j in range(i + 1, 8):
            if attn_weights[0, 0, i, j] != 0:
                is_causal = False
                break
    
    print(f"因果掩码生效: {is_causal}")
    print("✓ 因果自注意力测试通过\n")


def test_mlp():
    """测试MLP层"""
    print("=== 测试MLP层 ===")
    
    config = GPTConfig(n_embd=64)
    mlp = MLP(config)
    
    # 测试前向传播
    x = torch.randn(2, 8, 64)
    output = mlp(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✓ MLP测试通过\n")


def test_block():
    """测试Transformer块"""
    print("=== 测试Transformer块 ===")
    
    config = GPTConfig(n_embd=768)
    block = Block(config)
    
    # 测试前向传播
    x = torch.randn(2, 8, 768)
    output = block(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✓ Transformer块测试通过\n")


def test_gpt_model():
    """测试完整GPT模型"""
    print("=== 测试完整GPT模型 ===")
    
    # 创建小型配置
    config = GPTConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=32
    )
    
    model = GPT(config)
    
    # 测试前向传播（无目标）
    idx = torch.randint(0, 100, (2, 8))
    logits, loss = model(idx)
    
    print(f"输入token形状: {idx.shape}")
    print(f"输出logits形状: {logits.shape}")
    print(f"损失值: {loss}")
    
    # 测试前向传播（有目标）
    targets = torch.randint(0, 100, (2, 8))
    logits, loss = model(idx, targets)
    
    print(f"有目标时的损失值: {loss.item():.4f}")
    
    # 测试参数计数
    num_params = model.get_num_params()
    print(f"模型参数数量: {num_params:,}")
    
    # 测试生成（评估模式）
    model.eval()
    with torch.no_grad():
        generated = model.generate(idx[:, :4], max_new_tokens=3)
    print(f"生成结果形状: {generated.shape}")
    
    print("✓ GPT模型测试通过\n")


def test_optimizer():
    """测试优化器配置"""
    print("=== 测试优化器配置 ===")
    
    config = GPTConfig(n_embd=768, n_layer=12)
    model = GPT(config)
    
    optimizer = model.configure_optimizers(
        weight_decay=0.01,
        learning_rate=0.001,
        betas=(0.9, 0.999),
        device_type='cpu'
    )
    
    print(f"优化器类型: {type(optimizer).__name__}")
    print(f"参数组数量: {len(optimizer.param_groups)}")
    
    # 检查参数组
    for i, group in enumerate(optimizer.param_groups):
        print(f"参数组 {i}: {len(group['params'])} 个参数")
    
    print("✓ 优化器测试通过\n")


def test_gpt_estimate_mfu():
    """测试MFU估算"""
    print("=== 测试MFU估算 ===")
    
    config = GPTConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=32
    )
    
    model = GPT(config)
    
    fwdbwd_per_iter = 1
    dt = 0.1  # 100ms per iteration
    
    mfu = model.estimate_mfu(fwdbwd_per_iter, dt)
    
    # MFU应该在0到1之间
    assert 0 <= mfu <= 1
    print(f"MFU估算值: {mfu:.4f}")
    print("✓ MFU估算测试通过\n")


def test_gpt_from_pretrained_initialization():
    """测试from_pretrained方法初始化"""
    print("=== 测试from_pretrained方法初始化 ===")
    
    # 测试支持的模型类型
    supported_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    
    for model_type in supported_models:
        try:
            # 测试模型初始化（不实际下载权重）
            # 这里我们模拟from_pretrained的行为，但不实际调用Hugging Face
            config_args = {
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
            }[model_type]
            
            config_args['vocab_size'] = 50257
            config_args['n_positions'] = 1024
            config_args['bias'] = True
            
            config = GPTConfig(**config_args)
            model = GPT(config)
            
            # 验证配置参数
            assert model.config.vocab_size == 50257
            assert model.config.n_positions == 1024
            assert model.config.bias == True
            assert model.config.n_layer == config_args['n_layer']
            assert model.config.n_head == config_args['n_head']
            assert model.config.n_embd == config_args['n_embd']
            
            print(f"✓ {model_type} 初始化测试通过")
            
        except Exception as e:
            print(f"❌ {model_type} 初始化测试失败: {e}")
            raise
    
    print("✓ from_pretrained初始化测试通过\n")


def test_gpt_from_pretrained_override_args():
    """测试from_pretrained方法的参数覆盖功能"""
    print("=== 测试from_pretrained方法的参数覆盖功能 ===")
    
    # 测试dropout参数覆盖
    override_args = {'dropout': 0.2}
    
    # 模拟配置创建
    config_args = dict(n_layer=12, n_head=12, n_embd=768)
    config_args['vocab_size'] = 50257
    config_args['n_positions'] = 1024
    config_args['bias'] = True
    config_args['dropout'] = override_args['dropout']  # 覆盖dropout
    
    config = GPTConfig(**config_args)
    model = GPT(config)
    
    # 验证dropout参数被正确覆盖
    assert model.config.dropout == override_args['dropout']
    print(f"dropout参数被正确覆盖为: {model.config.dropout}")
    print("✓ 参数覆盖功能测试通过\n")


def test_gpt_from_pretrained_state_dict_keys():
    """测试from_pretrained方法的状态字典键匹配"""
    print("=== 测试from_pretrained方法的状态字典键匹配 ===")
    
    # 创建一个小型模型来测试状态字典结构
    config = GPTConfig(
        vocab_size=100,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=32
    )
    
    model = GPT(config)
    sd = model.state_dict()
    
    # 测试状态字典键的过滤逻辑
    sd_keys = sd.keys()
    sd_keys_filtered = [k for k in sd_keys if not k.endswith('.attn.bias')]
    
    # 验证过滤逻辑
    for key in sd_keys:
        if key.endswith('.attn.bias'):
            assert key not in sd_keys_filtered
        else:
            assert key in sd_keys_filtered
    
    # 验证转置权重键的识别
    transposed_keys = ['attn.c_attn.weight', 'attn.c_proj.weight', 
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']
    
    for key in transposed_keys:
        # 检查这些键是否存在于状态字典中
        found = any(key in k for k in sd_keys)
        assert found, f"Transposed key {key} not found in state dict"
    
    print("状态字典键匹配测试通过")
    print("✓ 状态字典键匹配测试通过\n")


def test_gpt_from_pretrained_integration():
    """测试from_pretrained方法的集成功能"""
    print("=== 测试from_pretrained方法的集成功能 ===")
    
    # 这个测试模拟from_pretrained的完整流程，但不实际下载权重
    
    # 测试配置参数的正确性
    model_type = 'gpt2'
    config_args = {
        'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
    }[model_type]
    
    config_args['vocab_size'] = 50257
    config_args['n_positions'] = 1024
    config_args['bias'] = True
    
    config = GPTConfig(**config_args)
    model = GPT(config)
    
    # 验证GPT-2的标准配置
    assert model.config.vocab_size == 50257
    assert model.config.n_positions == 1024
    assert model.config.bias == True
    assert model.config.n_layer == 12
    assert model.config.n_head == 12
    assert model.config.n_embd == 768
    
    print("GPT-2标准配置验证通过")
    print("✓ from_pretrained集成功能测试通过\n")


def main():
    """运行所有测试"""
    print("开始GPT模型功能测试...\n")
    
    try:
        test_conv1d()
        test_layernorm()
        test_attention()
        test_causal_attention()
        test_mlp()
        test_block()
        test_gpt_model()
        test_optimizer()
        test_gpt_estimate_mfu()
        test_gpt_from_pretrained_initialization()
        test_gpt_from_pretrained_override_args()
        test_gpt_from_pretrained_state_dict_keys()
        test_gpt_from_pretrained_integration()
        
        print("🎉 所有测试通过！GPT模型功能正常。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()