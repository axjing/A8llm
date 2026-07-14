import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.config import VLMConfig

class ViTPatchEmbeddings(nn.Module):
    # Ref:[SigLIP](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L245)
    def __init__(self,cfg:VLMConfig):
        super().__init__()
        self.n_embd=cfg.vit_n_embd
        self.image_size=cfg.image_size
        self.patch_size=cfg.patch_size
        self.n_patch=(self.image_size//self.patch_size)**2
        self.n_channel=cfg.n_channels
        
        self.cls_flag=cfg.vit_cls_flag
        
        # conv layer to extract the patches
        self.conv=nn.Conv2d(
            in_channels=self.n_channel,
            out_channels=self.n_embd,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        
        if self.cls_flag:
            self.cls_token=nn.Parameter(torch.zeros(1,1,self.n_embd))
            self.position_embedding=nn.Parameter(torch.rand(1,self.n_patch+1,self.n_embd))
        else:
            self.position_embedding=nn.Parameter(torch.rand(1,self.n_patch,self.n_embd))
    
    def forward(self,x):
        x=self.conv(x) # 提取 patches
        x=x.flatten(2) # 将图像块展平为一维数据
        x=x.transpose(1,2) # transpose to (bs,n_patch,n_embd)
        
        # Add CLS token (according to original ViT Paper) and position embeddings
        if self.cls_flag:
            cls_token=self.cls_token.expand(x.shape[0],-1,-1)
            x=torch.cat((cls_token,x),dim=1)
        # print(f"self.position_embedding:{self.position_embedding.shape}")
        # print(x.shape)
        x=x+self.position_embedding
        
        return x
        

class ViTMultiHeadAttention(nn.Module):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L381
    # https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
    def __init__(self,cfg: VLMConfig):
        super().__init__()
        
        self.n_head=cfg.vit_n_heads
        self.n_embd=cfg.vit_n_embd
        assert self.n_embd % self.n_head==0, f"embd dim must be divisible by n_head,current:n_embed % n_head={self.n_embd}%{self.n_head}"
        
        self.head_dim=self.n_embd//self.n_head
        
        self.droput =cfg.vit_pdropout
        
        # Combined projected for all heads
        self.c_attn=nn.Linear(self.n_embd,3*self.n_embd,bias=True)
        self.c_proj=nn.Linear(self.n_embd,self.n_embd,bias=True)
        
        # Dropout layers
        self.attn_dropout=nn.Dropout(self.droput)
        self.resid_dropout=nn.Dropout(self.droput)
        
        # Use scaled dot product attention if available
        self.sdpa=hasattr(torch.nn.functional,'scaled_dot_product_attention')
        if not self.sdpa:
            print("Warning: scaled dot product attention not available. Using standard attention in ViT.")
            
    def forward(self,x:torch.Tensor):
        bsz,seq_len,n_embd=x.size()
        
        q,k,v=self.c_attn(x).split(n_embd,dim=2)
        
        # (bsz,seq_len,n_embed) -> (bsz,seq_len,n_head,head_dim) -> (bsz,n_head,seq_len,head_dim)
        q=q.view(bsz,seq_len,self.n_head,self.head_dim).transpose(1,2)
        k=k.view(bsz,seq_len,self.n_head,self.head_dim).transpose(1,2)
        v=v.view(bsz,seq_len,self.n_head,self.head_dim).transpose(1,2)
        
        if self.sdpa:
            y=torch.nn.functional.scaled_dot_product_attention(
                q,k,v,
                attn_mask=None,
                dropout_p=self.droput if self.training else 0.,
                is_causal=False # ViT attention is bidirectional
            )
        else:
            # (bsz,n_head,seq_len,head_dim) @ (bsz,n_head,head_dim,seq_len)->(bsz,n_head,seq_len,seq_len)
            attn =(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
            attn = F.softmax(attn,dim=-1)
            attn=self.attn_dropout(attn)
            y=attn@v # (bsz,n_head,seq_len,seq_len) @ (bsz,n_head,seq_len,n_embed) -> (bsz,n_head,seq_len,head_dim)
            
        # transpose back from (bsz,n_head,seq_len,n_embed) to (bsz,seq_len,n_head * n_embd) and combine all heads to [bsz,seq_len,n_embd]
        y=y.transpose(1,2).contiguous().view(bsz,seq_len,n_embd)
        
        y=self.c_proj(y)
        y=self.resid_dropout(y)
        
        return y
    
    
class ViTMLP(nn.Module):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L453
    def __init__(self,cfg: VLMConfig):
        super().__init__()
        self.activate_fn=nn.GELU(approximate='tanh')
        self.fc1=nn.Linear(cfg.vit_n_embd,cfg.vit_n_intermediate)
        self.fc2=nn.Linear(cfg.vit_n_intermediate,cfg.vit_n_embd)
        self.dropout=nn.Dropout(cfg.vit_pdropout)
        
    def forward(self,x: torch.Tensor):
        x=self.fc1(x)
        x=self.activate_fn(x)
        x=self.fc2(x)
        x=self.dropout(x)
        return x
    
class ViTBlock(nn.Module):
    def __init__(self, cfg: VLMConfig) -> None:
        super().__init__()
        
        self.ln1=nn.LayerNorm(cfg.vit_n_embd,eps=cfg.vit_layernorm_eps)
        self.attn=ViTMultiHeadAttention(cfg)
        self.ln2=nn.LayerNorm(cfg.vit_n_embd,eps=cfg.vit_layernorm_eps)
        self.mlp=ViTMLP(cfg)
        
    def forward(self,x: torch.Tensor):
        x=self.ln1(x)
        x=x+self.attn(x)
        x=self.ln2(x)
        x=x+self.mlp(x)
        
        return x
    
class ViT(nn.Module):
    def __init__(self,cfg: VLMConfig):
        super().__init__()
        
        self.cfg=cfg

        self.patch_embedding=ViTPatchEmbeddings(self.cfg)
        
        self.cls_flag=self.cfg.vit_cls_flag
        self.dropout=nn.Dropout(self.cfg.vit_pdropout)
        self.blocks=nn.ModuleList([ViTBlock(self.cfg) for _ in range(self.cfg.vit_n_layers)])
        self.layer_norm = nn.LayerNorm(self.cfg.vit_n_embd, eps=self.cfg.vit_layernorm_eps)
        
        self.apply(self.__init_weights)
        
    def __init_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self,x: torch.Tensor):
        x=self.patch_embedding(x)
        x=self.dropout(x)
        for block in self.blocks:
            x=block(x)
        if self.cls_flag:
            x=self.layer_norm(x[:,0])
        else:
            x=self.layer_norm(x)
        
        return x
    
    @classmethod
    def from_pretrained(cls,cfg: VLMConfig):
        from transformers import Siglip2VisionConfig
        from huggingface_hub import hf_hub_download
        import safetensors
        
        hf_config=Siglip2VisionConfig.from_pretrained(cfg.vit_model_type,filename="model.safetensors")

        cfg.attn_pdrop=hf_config.attention_dropout
        cfg.n_embd=hf_config.hidden_size
        cfg.image_size=hf_config.image_size
        cfg.n_intermediate=hf_config.intermediate_size
        cfg.layer_norm_eps=hf_config.layer_norm_eps
        cfg.n_heads=hf_config.num_attention_heads
        cfg.n_layers=hf_config.num_hidden_layers
        cfg.patch_size=hf_config.patch_size
        cfg.activation_function=hf_config.hidden_act
        
        model=cls(cfg)
        safetensors_file=hf_hub_download(
            repo_id=cfg.vit_model_type,
            filename="model.safetensors",
            # endpoint="https://modelscope.cn/hub"
            )
        
        sd=model.state_dict()
        mapping = {
            'vision_model.embeddings.patch_embedding.weight': 'patch_embedding.conv.weight',
            'vision_model.embeddings.patch_embedding.bias': 'patch_embedding.conv.bias',
            'vision_model.embeddings.position_embedding.weight': 'patch_embedding.position_embedding',
            'vision_model.post_layernorm.weight': 'layer_norm.weight',
            'vision_model.post_layernorm.bias': 'layer_norm.bias',
        }
        for i in range(cfg.n_layers):
            # Layer norms
            mapping[f'vision_model.encoder.layers.{i}.layer_norm1.weight'] = f'blocks.{i}.ln1.weight'
            mapping[f'vision_model.encoder.layers.{i}.layer_norm1.bias'] = f'blocks.{i}.ln1.bias'
            mapping[f'vision_model.encoder.layers.{i}.layer_norm2.weight'] = f'blocks.{i}.ln2.weight'
            mapping[f'vision_model.encoder.layers.{i}.layer_norm2.bias'] = f'blocks.{i}.ln2.bias'
            
            # MLP
            mapping[f'vision_model.encoder.layers.{i}.mlp.fc1.weight'] = f'blocks.{i}.mlp.fc1.weight'
            mapping[f'vision_model.encoder.layers.{i}.mlp.fc1.bias'] = f'blocks.{i}.mlp.fc1.bias'
            mapping[f'vision_model.encoder.layers.{i}.mlp.fc2.weight'] = f'blocks.{i}.mlp.fc2.weight'
            mapping[f'vision_model.encoder.layers.{i}.mlp.fc2.bias'] = f'blocks.{i}.mlp.fc2.bias'
            
            # Output projection
            mapping[f'vision_model.encoder.layers.{i}.self_attn.out_proj.weight'] = f'blocks.{i}.attn.c_proj.weight'
            mapping[f'vision_model.encoder.layers.{i}.self_attn.out_proj.bias'] = f'blocks.{i}.attn.c_proj.bias'
        with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)
                    if tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)
                    else:
                        if 'position_embedding' in hf_key:
                            sd[our_key].copy_(tensor.unsqueeze(0))
                        else:
                            print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")
            
            # Manually handle QKV concatenation since our implementation combines Q, K, V into one
            for i in range(model.cfg.n_layers):
                q_weight = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.q_proj.weight')
                k_weight = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.k_proj.weight')
                v_weight = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.v_proj.weight')
                qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
                sd[f'blocks.{i}.attn.c_attn.weight'].copy_(qkv_weight)
                
                q_bias = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.q_proj.bias')
                k_bias = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.k_proj.bias')
                v_bias = f.get_tensor(f'vision_model.encoder.layers.{i}.self_attn.v_proj.bias')
                
                qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
                sd[f'blocks.{i}.attn.c_attn.bias'].copy_(qkv_bias)
        
        model.load_state_dict(sd)
        print(f"Successfully loaded {cfg.vit_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model
        
    
if __name__=="__main__":
    print("...")
    cfg=VLMConfig()
    vit_multi_attn=ViTMultiHeadAttention(cfg)
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    # vit_multi_attn=vit_multi_attn.to(device)
    # x=torch.rand(8,128,cfg.n_embd).to(device)
    # y=vit_multi_attn(x)
    # print(y.shape)
    

    sigclip_config=VLMConfig()
    vit=ViT(sigclip_config)
    vit=vit.from_pretrained(sigclip_config).to(device=device)
    image_rand=torch.rand((1,3,384,384)).to(device)
    
    import matplotlib.pyplot as plt
    plt.imshow(image_rand[0,...].detach().cpu().numpy().transpose(1,2,0))
    plt.show()
    print(image_rand.shape)
    o=vit(image_rand)
    print(o.shape)
    
    
    
    