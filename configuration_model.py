from dataclasses import dataclass

@dataclass
class GPTConfig:
    
    # attribute_map:dict={
    #     "hidden_size": "n_embd",
    #     "max_position_embeddings": "n_positions",
    #     "num_attention_heads": "n_head",
    #     "num_hidden_layers": "n_layer",
    # }
    
    # model scale parameters
    n_positions:int=1024 # 最大上下文长度
    vocab_size:int=50304 # GPT-2的词汇表大小为50257，为提高效率，填充到最接近的64的倍数。
    n_embd:int=768 # word embedding/hidden 维度
    n_layer:int=12 # Transformer块 层数
    n_head:int=12 # 多头注意力头数，n_embd // n_head= 64，每个头的维度为64
    
    n_inner :int|None=4*n_embd # 前馈网络中间层维度
    
    # activation & normalization parameters
    activation_function:str="relu"
    dropout:float=0.0 # dropout 率，用于防止过拟合
    resid_pdrop:float=0.1 # 残差链接层 dropout 率，用于防止过拟合
    embd_pdrop:float=0.1 # embedding 层 dropout 率，用于防止过拟合
    attn_pdrop:float=0.1 # 多头注意力层 dropout 率，用于防止过拟合
    layer_norm_epsilon:float=1e-5 # LayerNorm 中的 epsilon 值，用于防止除0错误
    bias:bool=False # True：线性层和 LayerNorm 中存在偏置，就像 GPT-2 一样。False：性能稍好且速度更快
    
    # generation & pool
    summary_type="cls_index" # 取<CLS> 位置做句子、分类表示
    summary_use_proj:bool=True # 是否对池化结果做线性投影
    summary_activation:str|None=None # 池化结果的激活函数，默认使用 None
    summary_proj_to_labels:bool=True # 是否对池化结果做线性投影到标签维度
    summary_first_dropout:float=0.1 # 池化结果的 dropout 率，用于防止过拟合
    
    
    # attention & inference 优化 parameters
    use_cache:bool=True
    scale_attn_weights: bool = True # 注意力分数按头维度的根号缩放
    scale_attn_by_inverse_layer_idx:bool=True # 是否按层索引的倒数缩放注意力权重
    reorder_and_upcast_attn=False # 不重排、提升精度计算注意力    
    
    # special tokens
    bos_token_id=5026 # 开始符 token id，用于表示序列的开始
    eos_token_id=5026 # 介绍符 token id,用于表示序列的结束,默认与开始符相同
    pad_token_id=None # 填充 token id，用于填充到最大上下文长度
    
    # 结构开关
    add_cross_attention:bool=False # 是否添加跨注意力机制
    tie_word_embeddings=True # 是否将 word embedding 层和 output 层的权重共享(GPT经典做法，为了省参数量)

