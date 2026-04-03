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