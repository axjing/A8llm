import math
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.configuration_model import GPTConfig


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        of (`int`): The number of output features.
        ix (`int`): The number of input features.
    """

    def __init__(self, ix, of):
        super().__init__()
        
        self.ix = ix
        self.of = of
        self.weight = nn.Parameter(torch.empty(ix, of))
        self.bias = nn.Parameter(torch.zeros(of))
        nn.init.normal_(self.weight, std=0.02)

    def __repr__(self) -> str:
        return "Conv1D(ix={ix}, of={of})".format(**self.__dict__)

    def forward(self, x:torch.Tensor):
        size_out = x.size()[:-1] + (self.of,)
        x = torch.addmm(self.bias, x.contiguous().view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


def scaled_dot_product_attention(module: nn.Module,
                                 query: torch.Tensor,
                                 key: torch.Tensor,
                                 value: torch.Tensor,
                                 attention_mask: torch.Tensor|None=None,scaling=None,
                                 dropout:float=0.0,**kwargs):
    if scaling is None:
        scaling=query.size(-1)**-0.5
        
    attn_weights=torch.matmul(query,key.transpose(-1,-2))*scaling
    
    if attention_mask is not None:
        attn_weights=attn_weights+attention_mask
        
    attn_weights=nn.functional.softmax(attn_weights,dim=-1)
    
    # (如有必要)向下转换回V的数据类型(如果处于混合精度模式)——否则不执行任何操作
    attn_weights=attn_weights.type(value.dtype)
    attn_weights=nn.functional.dropout(attn_weights,p=dropout,training=module.training)
    
    attn_output=torch.matmul(attn_weights,value)
    attn_output=attn_output.transpose(1,2)
    
    return attn_output,attn_weights


class LayerNorm(nn.Module):
    """LayerNorm 层"""
    def __init__(self,normalized_shape,eps: float = 1e-5, bias: bool = True,  device=None,):
        super().__init__()
        self.normalized_shape=normalized_shape
        self.eps=eps
        self.bias=bias
        self.device=device
        
        self.weight=nn.Parameter(torch.ones(normalized_shape))
        self.bias=nn.Parameter(torch.zeros(normalized_shape)) if bias else None
    
    def forward(self,x):
        o=F.layer_norm(x,self.weight.shape, self.weight, self.bias, self.eps)
        
        return o
        
class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig,is_cross_attention:bool=False,ctrl_flash:bool=True):
        super().__init__()
        self.is_cross_attention=is_cross_attention
        
        self.config=config
        self.n_embd=config.n_embd
        self.n_head=config.n_head
        self.head_dim=self.n_embd//self.n_head
        self.split_size = self.n_embd
        if self.head_dim*self.n_head!=self.n_embd:
            raise ValueError(f"`n_embd` must be divisible by n_head (got `n_embd`:{self.n_embd} and `n_head`:{self.n_head})")
        
        if self.is_cross_attention:
            self.c_attn=Conv1D(self.n_embd,2*self.n_embd)
            self.q_attn=Conv1D(self.n_embd,self.n_embd)
        else:
            self.c_attn=Conv1D(self.n_embd,3*self.n_embd)    
        print(f"c_attn {self.c_attn}")
        self.c_proj=Conv1D(self.n_embd,self.n_embd)
        print(f"c_proj {self.c_proj}")
        # regularization
        self.attn_pdrop=config.attn_pdrop
        self.resid_pdrop=config.resid_pdrop
        self.attn_dropout = nn.Dropout(self.attn_pdrop)
        self.resid_dropout = nn.Dropout(self.resid_pdrop)
        
        
        self.ctrl_flash=ctrl_flash
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if  not self.flash or not self.ctrl_flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions))
                                        .view(1, 1, config.n_positions, config.n_positions))
            
        # attn scale factor
        self.scaling =  self.head_dim**(-0.5)
        
    def selfattention_v0(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,attention_mask: torch.Tensor|None=None,scaling=None,dropout:float=0.0,**kwargs):
        bsz,n_head,q_seq_len,n_embd=q.size()
        _,_,k_seq_len,_=k.size()
        scaling=k.size(-1)**(-0.5)
        
        # (bsz,n_head,q_seq_len,n_embd)@(bsz,n_head,n_embd,k_seq_len) -> (bsz,n_head,q_seq_len,k_seq_len)
        attn_weights=torch.matmul(q,k.transpose(-1,-2))*scaling
        
        if attention_mask is not None:
            attn_weights=attn_weights+attention_mask
        else:
            attn_weights=attn_weights.masked_fill(self.bias[:,:,:q_seq_len,:k_seq_len]==0,-float('inf'))
            
        attn_weights=nn.functional.softmax(attn_weights,dim=-1)
        attn_weights=self.attn_dropout(attn_weights)
        
        # (bsz,n_head,q_seq_len,k_seq_len)@(bsz,n_head,v_seq_len,n_embd) -> (bsz,n_head,q_seq_len,n_embd)
        attn_output=torch.matmul(attn_weights,v)
        return attn_output,attn_weights
        
    def forward(self,x):
        bsz,seq_len,n_embd=x.size() # batch_size, seq_len, n_embd
        
        # 为该批次注意力头计算q,k,v，并将注意力头数n_head的维度前移
        q,k,v=self.c_attn(x).split(self.n_embd,dim=2) # (bsz,seq_len,n_embd)
        q=q.view(bsz,seq_len,self.n_head,n_embd//self.n_head).transpose(1,2) # (bsz,n_head,seq_len,head_dim)
        k=k.view(bsz,seq_len,self.n_head,n_embd//self.n_head).transpose(1,2) # (bsz,n_head,seq_len,head_dim)
        v=v.view(bsz,seq_len,self.n_head,n_embd//self.n_head).transpose(1,2) # (bsz,n_head,seq_len,head_dim)
        
        if self.flash and self.ctrl_flash:
            attn_output=nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.attn_pdrop if self.training else 0,is_causal=True)
            attn_weights=None
            # print(f"attn_output {attn_output.size()}")
        else:
            attn_output,attn_weights=self.selfattention_v0(q,k,v)
        
        # print(f"attn_output {attn_output.transpose(1,2).size()}")
        attn_output=attn_output.transpose(1,2).contiguous().view(bsz,seq_len,n_embd)
        # print(f"attn_output1 {attn_output.size()}")
        # output projection
        y=self.c_proj(attn_output)
        y=self.resid_dropout(y)
        
        
        
        return y,attn_weights

class MLP(nn.Module):
    """MLP 层"""
    def __init__(self,config: GPTConfig):
        super().__init__()
        
        self.c_fc=Conv1D(config.n_embd,config.n_embd*4)
        self.c_proj=Conv1D(config.n_embd*4,config.n_embd)
        self.activate_fn=nn.GELU()
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
    def forward(self,x):
        x=self.c_fc(x)
        x=self.activate_fn(x)
        x=self.c_proj(x)
        x=self.resid_dropout(x)
        return x
    
class Block(nn.Module):
    """Block 层"""
    def __init__(self,config: GPTConfig):
        super().__init__()
        self.ln_1=LayerNorm(config.n_embd,bias=config.bias)
        self.attn=CausalSelfAttention(config)
        self.ln_2=LayerNorm(config.n_embd,bias=config.bias)
        
        
        self.mlp=MLP(config)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        residual=x
        x=self.ln_1(x)
        x,_=self.attn(x)
        x=x+residual
        
        residual=x
        x=self.ln_2(x)
        x=self.mlp(x)
        x=x+residual
        
        return x
        
    
class GPT(nn.Module):
    """GPT 模型"""
    def __init__(self,config: GPTConfig):
        super().__init__()
        self.config=config
        
        self.wte=nn.Embedding(config.vocab_size,config.n_embd) # token embedding
        self.wpe=nn.Embedding(config.n_positions,config.n_embd) # position embedding
        self.drop=nn.Dropout(config.attn_pdrop)
        self.h=nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f=LayerNorm(config.n_embd,bias=config.bias)
        # self.transformer=nn.ModuleDict(dict(
        #     wte=self.wte,
        #     wpe=self.wpe,
        #     drop=self.drop,
        #     h=self.h,
        #     ln_f=self.ln_f,
        # ))
        
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        # self.transformer.wte.weight=self.lm_head.weight
        self.wte.weight=self.lm_head.weight
        
        # init all weights
        self.apply(self._init_weights)
        # 按照GPT-2论文所述，对残差投影应用特殊的缩放初始化
        for pn,p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p,mean=0.0,std=0.02*(2*config.n_layer)**(-0.5))
                
        # report number of parameters
        print(f"Number of parameters:  {self.get_num_params()/1e6 :.2f}M")
        
    def get_num_params(self,non_embedding=True):
        """
        返回模型中的参数数量。
        Args:
            non_embedding (bool, optional): 对于非嵌入计数(默认情况)，位置嵌入会被减去。. Defaults to True.原本token embedding也会被减去，但由于参数共享，这些参数实际上被用作最后一层的权重，因此我们将它们包含在内。
        """
        
        n_params=sum(p.numel() for p in self.parameters())
        if non_embedding:
            # n_params-=self.transformer.wpe.weight.numel()
            n_params-=self.wpe.weight.numel()
        return n_params
    
    def _init_weights(self,module: nn.Module):
        """
        初始化模型参数

        Args:
            module (nn.Module): _description_
        """
        if isinstance(module,nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx:torch.Tensor,targets=None):
        device=idx.device
        
        bsz,seq_len=idx.size()
        
        assert seq_len <=self.config.n_positions, f"序列长度 {seq_len}超过模型支持的最长上下文长度 {self.config.n_positions}"
        
        ipositions=torch.arange(0,seq_len,dtype=torch.long,device=device) # shape(seq_len)
        
        # forward pass
        
        # token_embd=self.transformer.wte(idx) # token embedding shape(bsz,seq_len,n_embd)
        # position_embd=self.transformer.wpe(ipositions) # position embedding shape(seq_len,n_embd)
        
        # x=self.transformer.drop(token_embd+position_embd)
        
        # for block in self.transformer.h:
        #     x=block(x)
        # x=self.transformer.ln_f(x)
        
        inputs_embeds=self.wte(idx) # token embedding shape(bsz,seq_len,n_embd)
        position_embd=self.wpe(ipositions) # position embedding shape(seq_len,n_embd)
        x=self.drop(inputs_embeds+position_embd)
        
        for block in self.h:
            x=block(x)
        x=self.ln_f(x)
    
        
        if targets is not None:
            # 如果tragets非空，计算loss
            logits=self.lm_head(x)
            loss=nn.functional.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
        else:
            logits=self.lm_head(x[:,[-1],:]) # # note：使用列表 [-1] 来保留sequence维度
            loss=None
            
        return logits,loss
    
    def crop_n_position(self,n_position:int):
        """
        # 必要时通过模型调整上下文长度
        # 例如，我们可能加载了GPT2预训练模型检查点(块大小为1024)
        # 但希望在一些更小、更简单的模型中使用更小的块大小

        Args:
            n_position (int): _description_
        """
        assert n_position<=self.config.n_positions, f"n_position {n_position}超过模型支持的最长上下文长度 {self.config.n_positions}"
        
        self.config.n_positions=n_position
        
        self.wpe.weight=nn.Parameter(self.wpe.weight[:n_position])
        
        for block in self.h:
            if hasattr(block.attn,"bias"):
                block.attn.bias=block.attn.bias[:,:,:n_position,:n_position]
                
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['n_positions'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 仅保留需要梯度更新的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 创建优化组。任何二维参数都将进行权重衰减，否则不进行。即，矩阵乘法和嵌入中的所有权重张量会衰减，所有偏置和层归一化不会。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # 创建AdamW优化器，如果融合版本可用则使用融合版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ 以A100 bfloat16峰值FLOPS为单位估算模型的FLOPS利用率(model flops utilization,MFU)
        
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.n_positions
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        取一个索引的条件序列idx（形状为(bsz,seq_len)的LongTensor），并将该序列补全max_new_tokens次，每次都将预测结果反馈到模型中。大多数情况下，你可能需要确保为此处于model.eval()操作模式。
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文变得太长，我们必须将其裁剪到块大小。
            idx_cond = idx if idx.size(1) <= self.config.n_positions else idx[:, -self.config.n_positions:]
            # 向前传递模型以获取序列中该索引的logits输出
            logits, _ = self(idx_cond)
            # 在最后一步提取logits输出，并按期望的temperature进行缩放
            logits = logits[:, -1, :] / temperature
            # 可选地裁剪logits输出，仅保留top_k个选项
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 对logits输出应用softmax，将其转换为归一化后的概率分布
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将采样的索引添加到运行序列中并继续生成
            idx = torch.cat((idx, idx_next), dim=1)

        return idx