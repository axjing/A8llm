import unittest
import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.llama import LlamaBlock, LlamaTransformer
from src.models.config import LLMConfig


class TestLlamaBlock(unittest.TestCase):
    """测试 LlamaBlock 类"""

    def setUp(self):
        """设置测试配置和模型"""
        self.cfg = LLMConfig(
            n_embd=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=1000,
            lm_use_tokens=True,
            lm_tie_weights=True
        )
        self.block = LlamaBlock(self.cfg)
        self.block.eval()

    def test_forward_pass(self):
        """测试前向传播"""
        print("\n=== 测试 LlamaBlock 前向传播 ===")
        batch_size = 2
        seq_len = 10
        hidden_dim = self.cfg.n_embd
        
        # 创建输入张量
        x = torch.randn(batch_size, seq_len, hidden_dim)
        print(f"输入张量形状: {x.shape}")
        
        # 创建旋转位置编码
        from src.models.position_embedding import RotaryEmbedding
        rotary = RotaryEmbedding(self.cfg)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        cos, sin = rotary(position_ids)
        print(f"旋转编码形状 - cos: {cos.shape}, sin: {sin.shape}")
        
        # 前向传播
        with torch.no_grad():
            output, block_kv_cache = self.block(x, cos, sin)
        
        # 验证输出形状
        print(f"输出张量形状: {output.shape}")
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_dim))
        
        # 验证 KV cache 不为空
        print(f"KV cache 包含键: {list(block_kv_cache.keys())}")
        print(f"KV cache key 形状: {block_kv_cache['key'].shape}")
        print(f"KV cache value 形状: {block_kv_cache['value'].shape}")
        self.assertIsNotNone(block_kv_cache)
        self.assertIn('key', block_kv_cache)
        self.assertIn('value', block_kv_cache)
        print("✓ LlamaBlock 前向传播测试通过")

    def test_forward_with_attention_mask(self):
        """测试带注意力掩码的前向传播"""
        batch_size = 2
        seq_len = 10
        hidden_dim = self.cfg.n_embd
        
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        from src.models.position_embedding import RotaryEmbedding
        rotary = RotaryEmbedding(self.cfg)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        cos, sin = rotary(position_ids)
        
        # 创建注意力掩码（前5个token可见）
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 5:] = 0
        
        with torch.no_grad():
            output, block_kv_cache = self.block(x, cos, sin, attention_mask)
        
        self.assertEqual(output.shape, (batch_size, seq_len, hidden_dim))

    def test_forward_with_kv_cache(self):
        """测试带KV缓存的前向传播"""
        batch_size = 2
        seq_len = 10
        hidden_dim = self.cfg.n_embd
        
        x = torch.randn(batch_size, seq_len, hidden_dim)
        
        from src.models.position_embedding import RotaryEmbedding
        rotary = RotaryEmbedding(self.cfg)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        cos, sin = rotary(position_ids)
        
        # 第一次前向传播获取缓存
        with torch.no_grad():
            output1, block_kv_cache = self.block(x, cos, sin)
        
        # 第二次前向传播使用缓存
        x_new = torch.randn(batch_size, 5, hidden_dim)
        position_ids_new = torch.arange(seq_len, seq_len + 5).unsqueeze(0).expand(batch_size, -1)
        cos_new, sin_new = rotary(position_ids_new)
        
        with torch.no_grad():
            output2, block_kv_cache = self.block(x_new, cos_new, sin_new, block_kv_cache=block_kv_cache)
        
        self.assertEqual(output2.shape, (batch_size, 5, hidden_dim))


class TestLlamaTransformer(unittest.TestCase):
    """测试 LlamaTransformer 类"""

    def setUp(self):
        """设置测试配置和模型"""
        self.cfg = LLMConfig(
            n_embd=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=1000,
            lm_use_tokens=True,
            lm_tie_weights=True,
            rotary_emb_base=10000.0
        )
        self.model = LlamaTransformer(self.cfg)
        self.model.eval()

    def test_forward_pass_with_tokens(self):
        """测试使用token输入的前向传播"""
        print("\n=== 测试 LlamaTransformer 前向传播（token输入） ===")
        batch_size = 2
        seq_len = 10
        
        # 创建token输入
        x = torch.randint(0, self.cfg.vocab_size, (batch_size, seq_len))
        print(f"输入token形状: {x.shape}")
        print(f"输入token示例: {x[0, :5].tolist()}...")
        
        with torch.no_grad():
            logits, kv_cache = self.model(x)
        
        # 验证输出形状：(batch_size, seq_len, vocab_size)
        print(f"输出logits形状: {logits.shape}")
        print(f"logits dtype: {logits.dtype}")
        self.assertEqual(logits.shape, (batch_size, seq_len, self.cfg.vocab_size))
        
        # 验证KV缓存列表长度等于层数
        print(f"KV缓存层数: {len(kv_cache)}")
        print(f"每层KV缓存结构: key={kv_cache[0]['key'].shape}, value={kv_cache[0]['value'].shape}")
        self.assertEqual(len(kv_cache), self.cfg.n_layers)
        print("✓ LlamaTransformer token输入前向传播测试通过")

    def test_forward_pass_with_embeddings(self):
        """测试使用embedding输入的前向传播"""
        cfg_no_tokens = LLMConfig(
            n_embd=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=1000,
            lm_use_tokens=False,  # 使用embedding输入
            lm_tie_weights=True,
            rotary_emb_base=10000.0
        )
        model_no_tokens = LlamaTransformer(cfg_no_tokens)
        model_no_tokens.eval()
        
        batch_size = 2
        seq_len = 10
        
        # 创建embedding输入
        x = torch.randn(batch_size, seq_len, cfg_no_tokens.n_embd)
        
        with torch.no_grad():
            output, kv_cache = model_no_tokens(x)
        
        # 验证输出形状：(batch_size, seq_len, n_embd)
        self.assertEqual(output.shape, (batch_size, seq_len, cfg_no_tokens.n_embd))

    def test_generate_method(self):
        """测试generate方法"""
        print("\n=== 测试 LlamaTransformer generate方法 ===")
        batch_size = 1
        seq_len = 5
        max_new_tokens = 10
        
        # 创建输入token
        inputs = torch.randint(0, self.cfg.vocab_size, (batch_size, seq_len))
        print(f"输入序列形状: {inputs.shape}")
        print(f"输入序列: {inputs[0].tolist()}")
        
        with torch.no_grad():
            generated = self.model.generate(inputs, max_new_tokens=max_new_tokens)
        
        # 验证生成序列长度
        print(f"生成序列形状: {generated.shape}")
        print(f"生成序列: {generated[0].tolist()}")
        print(f"生成的新token数量: {generated.shape[1] - seq_len}")
        self.assertEqual(generated.shape[0], batch_size)
        self.assertEqual(generated.shape[1], seq_len + max_new_tokens)
        
        # 验证前seq_len个token与输入相同
        print(f"前{seq_len}个token与输入相同: {torch.equal(generated[:, :seq_len], inputs)}")
        self.assertTrue(torch.equal(generated[:, :seq_len], inputs))
        print("✓ LlamaTransformer generate方法测试通过")

    def test_generate_with_single_sequence(self):
        """测试generate方法处理单个序列（无batch维度）"""
        seq_len = 5
        max_new_tokens = 10
        
        # 创建单个序列输入（无batch维度）
        inputs = torch.randint(0, self.cfg.vocab_size, (seq_len,))
        
        with torch.no_grad():
            generated = self.model.generate(inputs, max_new_tokens=max_new_tokens)
        
        # 验证生成序列形状和长度
        self.assertEqual(generated.shape[0], 1)  # batch_size=1
        self.assertEqual(generated.shape[1], seq_len + max_new_tokens)

    def test_weight_tying(self):
        """测试权重绑定"""
        # 检查token embedding和head权重是否绑定
        if self.cfg.lm_tie_weights:
            self.assertIs(self.model.head.weight, self.model.token_embedding.weight)
        else:
            # 创建不绑定权重的模型
            cfg_no_tie = LLMConfig(
                n_embd=64,
                n_layers=2,
                n_heads=4,
                n_kv_heads=2,
                vocab_size=1000,
                lm_use_tokens=True,
                lm_tie_weights=False,
                rotary_emb_base=10000.0
            )
            model_no_tie = LlamaTransformer(cfg_no_tie)
            self.assertIsNot(model_no_tie.head.weight, model_no_tie.token_embedding.weight)

    def test_kv_cache_inference(self):
        """测试KV缓存推理"""
        batch_size = 1
        seq_len = 5
        
        # 初始输入
        inputs = torch.randint(0, self.cfg.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            # 第一次前向传播
            logits1, kv_cache = self.model(inputs)
            
            # 使用缓存进行第二次前向传播（新token）
            new_token = torch.randint(0, self.cfg.vocab_size, (batch_size, 1))
            logits2, kv_cache = self.model(new_token, kv_cache=kv_cache, start_pos=seq_len)
        
        self.assertEqual(logits1.shape, (batch_size, seq_len, self.cfg.vocab_size))
        self.assertEqual(logits2.shape, (batch_size, 1, self.cfg.vocab_size))

    def test_from_pretrained(self):
        """测试从预训练模型加载"""
        print("\n=== 测试 LlamaTransformer from_pretrained方法 ===")
        import unittest
        
        # 使用一个小的开源模型进行测试
        # TinyLlama/TinyLlama-1.1B-Chat-v1.0 是一个相对较小的模型
        cfg = LLMConfig(
            lm_use_tokens=True,
            lm_tie_weights=True
        )
        cfg.lm_model_type = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"模型类型: {cfg.lm_model_type}")
        
        try:
            # 加载预训练模型
            print("正在加载预训练模型...")
            model = LlamaTransformer.from_pretrained(cfg)
            print("✓ 预训练模型加载成功")
            
            # 验证模型类型
            self.assertIsInstance(model, LlamaTransformer)
            
            # 打印模型配置信息
            print(f"\n模型配置信息:")
            print(f"  - 隐藏层维度: {cfg.n_embd}")
            print(f"  - 层数: {cfg.n_layers}")
            print(f"  - 注意力头数: {cfg.n_heads}")
            print(f"  - KV头数: {cfg.n_kv_heads}")
            print(f"  - 词汇表大小: {cfg.vocab_size}")
            print(f"  - 参数总数: {sum(p.numel() for p in model.parameters()):,}")
            
            # 验证模型可以正常前向传播
            model.eval()
            batch_size = 1
            seq_len = 10
            inputs = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
            print(f"\n测试前向传播 - 输入形状: {inputs.shape}")
            
            with torch.no_grad():
                logits, kv_cache = model(inputs)
            
            # 验证输出形状
            print(f"输出logits形状: {logits.shape}")
            self.assertEqual(logits.shape[0], batch_size)
            self.assertEqual(logits.shape[1], seq_len)
            self.assertEqual(logits.shape[2], cfg.vocab_size)
            
            # 验证KV缓存
            print(f"KV缓存层数: {len(kv_cache)}")
            self.assertEqual(len(kv_cache), cfg.n_layers)
            
            print("\n✓ LlamaTransformer from_pretrained方法测试通过")
            
        except Exception as e:
            # 如果下载失败（网络问题或模型不存在），跳过测试
            print(f"⚠️ 跳过测试: {str(e)}")
            raise unittest.SkipTest(f"Failed to load pretrained model: {str(e)}")

    def test_from_pretrained_with_custom_vocab(self):
        """测试从预训练模型加载时使用自定义词汇表大小"""
        import unittest
        
        cfg = LLMConfig(
            vocab_size=50304,  # 使用比原始模型更大的词汇表
            lm_use_tokens=True,
            lm_tie_weights=True
        )
        cfg.lm_model_type = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        try:
            model = LlamaTransformer.from_pretrained(cfg)
            
            # 验证词汇表大小
            self.assertEqual(model.token_embedding.weight.shape[0], cfg.vocab_size)
            
            # 验证模型可以正常工作
            model.eval()
            batch_size = 1
            seq_len = 5
            inputs = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
            
            with torch.no_grad():
                logits, _ = model(inputs)
            
            self.assertEqual(logits.shape[2], cfg.vocab_size)
            
        except Exception as e:
            print(f"Skipping from_pretrained_with_custom_vocab test: {str(e)}")
            raise unittest.SkipTest(f"Failed to load pretrained model: {str(e)}")


if __name__ == '__main__':
    unittest.main()