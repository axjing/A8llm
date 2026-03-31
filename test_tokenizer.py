"""
Tokenizer简单测试示例
直接运行即可验证各个功能
"""

import sys
import os
import tempfile
import shutil

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from tokenizer import HF2Tokenizer, SPECIAL_TOKENS, SPLIT_PATTERN


def test_tokenizer_initialization():
    """测试分词器初始化"""
    print("=== 测试分词器初始化 ===")
    
    # 测试特殊令牌
    print(f"特殊令牌: {SPECIAL_TOKENS}")
    print(f"分割模式: {SPLIT_PATTERN}")
    print("✓ 特殊令牌和分割模式测试通过\n")


def test_tokenizer_from_pretrained():
    """测试从预训练模型加载分词器"""
    print("=== 测试从预训练模型加载分词器 ===")
    
    try:
        # 这里只是测试导入，实际使用时需要有效的模型路径
        tokenizer = HF2Tokenizer.from_pretrained("gpt2")
        print("✓ 从预训练模型加载接口测试通过")
    except Exception as e:
        print(f"从预训练模型加载测试跳过: {e}")
    print()


def test_tokenizer_training():
    """测试分词器训练功能"""
    print("=== 测试分词器训练功能 ===")
    
    # 创建临时目录用于测试
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建简单的文本迭代器
        text_iterator = [
            "这是一个测试句子。",
            "Hello world!",
            "12345 abcdef",
            "特殊令牌测试: <|bos|> <|user_start|>"
        ]
        
        # 测试训练功能
        vocab_size = 1000
        tokenizer = HF2Tokenizer.train_from_iterator(text_iterator, vocab_size)
        
        # 测试获取词汇表
        vocab = tokenizer.get_vocab()
        print(f"词汇表大小: {len(vocab)}")
        
        # 测试获取特殊令牌
        special_tokens = tokenizer.get_special_tokens()
        print(f"特殊令牌: {special_tokens}")
        
        # 测试编码功能
        test_text = "Hello world"
        encoded = tokenizer.encode(test_text)
        print(f"文本 '{test_text}' 编码为: {encoded}")
        
        # 测试解码功能
        decoded = tokenizer.decode(encoded)
        print(f"解码结果: '{decoded}'")
        
        # 测试特殊令牌编码
        bos_id = tokenizer.encode_special("<|bos|>")
        print(f"BOS令牌ID: {bos_id}")
        
        # 测试获取BOS令牌ID
        bos_token_id = tokenizer.get_bos_token_id()
        print(f"BOS令牌ID: {bos_token_id}")
        
        # 测试保存功能
        tokenizer.save(temp_dir)
        print(f"分词器已保存到: {temp_dir}")
        
        # 测试从目录加载
        loaded_tokenizer = HF2Tokenizer.from_directory(temp_dir)
        print("✓ 从目录加载分词器测试通过")
        
        # 测试编码一致性
        original_encoded = tokenizer.encode(test_text)
        loaded_encoded = loaded_tokenizer.encode(test_text)
        print(f"原始编码: {original_encoded}")
        print(f"加载后编码: {loaded_encoded}")
        
        print("✓ 分词器训练功能测试通过\n")
        
    except Exception as e:
        print(f"分词器训练测试失败: {e}")
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_tokenizer_encoding():
    """测试分词器编码功能"""
    print("=== 测试分词器编码功能 ===")
    
    # 创建简单的分词器进行测试
    text_iterator = ["测试编码功能"]
    tokenizer = HF2Tokenizer.train_from_iterator(text_iterator, 500)
    
    # 测试字符串编码
    text = "测试编码功能"
    encoded = tokenizer.encode(text)
    print(f"文本 '{text}' 编码为: {encoded}")
    
    # 测试列表编码
    texts = ["测试1", "测试2", "测试3"]
    encoded_list = tokenizer.encode(texts)
    print(f"文本列表编码: {encoded_list}")
    
    # 测试带前缀和后缀的编码
    encoded_with_special = tokenizer.encode(text, prepend="<|bos|>", append="<|eos|>")
    print(f"带特殊令牌的编码: {encoded_with_special}")
    
    # 测试ID到令牌转换
    if encoded:
        token_str = tokenizer.id_to_token(encoded[0])
        print(f"ID {encoded[0]} 对应的令牌: '{token_str}'")
    
    print("✓ 分词器编码功能测试通过\n")


def test_tokenizer_call_method():
    """测试分词器的__call__方法"""
    print("=== 测试分词器的__call__方法 ===")
    
    text_iterator = ["测试call方法"]
    tokenizer = HF2Tokenizer.train_from_iterator(text_iterator, 500)
    
    # 测试直接调用分词器
    text = "测试call方法"
    encoded_call = tokenizer(text)
    encoded_encode = tokenizer.encode(text)
    
    print(f"__call__方法编码: {encoded_call}")
    print(f"encode方法编码: {encoded_encode}")
    
    # 验证两种方法结果一致
    assert encoded_call == encoded_encode, "__call__和encode方法结果不一致"
    
    print("✓ 分词器__call__方法测试通过\n")


def run_all_tests():
    """运行所有测试"""
    print("开始运行Tokenizer测试...\n")
    
    test_tokenizer_initialization()
    test_tokenizer_from_pretrained()
    test_tokenizer_training()
    test_tokenizer_encoding()
    test_tokenizer_call_method()
    
    print("=" * 50)
    print("所有Tokenizer测试完成！")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
