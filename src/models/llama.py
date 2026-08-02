import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import LLMConfig
from src.models.layers import RMSNorm, LlamaMLP, GroupedQueryAttention
from src.models.position_embedding import RotaryEmbedding
from src.models.transformer_utils import (
    init_weights,
    count_scaling_params, estimate_flops,
    build_muon_optimizer,
)

class LlamaBlock(nn.Module):
    # https://github.com/meta-llama/llama3/blob/main/llama/model.py#L222
    def __init__(self, cfg:LLMConfig) -> None:
        super().__init__()
        
        self.mlp=LlamaMLP(cfg)
        self.attn=GroupedQueryAttention(cfg)
        self.norm1=RMSNorm(cfg)
        self.norm2=RMSNorm(cfg)
        
    def forward(self,x:torch.Tensor,cos:torch.Tensor,sin: torch.Tensor, attention_mask: torch.Tensor|None=None, block_kv_cache: dict|None=None):
        """
        Forward pass of the Transformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            cos (Tensor): Cosine positional embeddings for rotary embedding, shape
                matching sequence length and head dimension.
            sin (Tensor): Sine positional embeddings for rotary embedding, same shape as cos.
            attention_mask (Tensor, optional): Attention mask of shape (batch_size, total_kv_length),
                with 1 indicating tokens to attend to and 0 for padding tokens.
            block_kv_cache (dict, optional): Key-value cache dict for cached keys and values
                during decoding. If None, no cache is used.

        Returns:
            Tuple[Tensor, dict]: Output tensor after the block (same shape as input),
                and the updated key-value cache dictionary.
        """
        res = x
        x=self.norm1(x)
        x,block_kv_cache=self.attn(x,cos,sin,attention_mask,block_kv_cache)
        x=res+x
        
        res=x
        x=self.norm2(x)
        x=self.mlp(x)
        x=res+x
        
        return x,block_kv_cache

class LlamaTransformer(nn.Module):
    # https://github.com/meta-llama/llama3/blob/main/llama/model.py#L251
    def __init__(self,cfg: LLMConfig):
        super().__init__()
        
        self.lm_use_tokens=cfg.lm_use_tokens
        self.lm_tie_weights=cfg.lm_tie_weights
        
        self.token_embedding=nn.Embedding(cfg.vocab_size,cfg.n_embd)
        
        self.rotary_embd=RotaryEmbedding(cfg)
        
        self.blocks=nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layers)])
        
        self.norm=RMSNorm(cfg)
        
        self.head=nn.Linear(cfg.n_embd,cfg.vocab_size,bias=False)
        
        if self.lm_tie_weights:
            self.head.weight=self.token_embedding.weight
        
        self.apply(self._init_weights)
        
        self.cfg=cfg

    def get_device(self):
        """Return the device the model parameters live on."""
        return next(self.parameters()).device
    
    def _init_weights(self, module):
        """Initialize model weights using the shared utility."""
        init_weights(module, rms_norm=True)
    def forward(self,x: torch.Tensor,attention_mask: torch.Tensor|None=None,kv_cache:list[dict]|None=None,start_pos:int=0):
        """
        Performs a forward pass through the language model.

        Args:
            x (Tensor): Input tensor. If `lm_use_tokens` is True, this should be
                token indices with shape (batch_size, sequence_length).
                If False, it should be embeddings of shape (batch_size, sequence_length, hidden_dim).
            attention_mask (Tensor, optional): Mask tensor for attention to
                specify which tokens to attend to, typically of shape
                (batch_size, sequence_length). Default is None.
            kv_cache (list[dict], optional): List of key-value caches for each transformer
                block to enable efficient autoregressive decoding.
                If None, no cache is used and new ones are created. Default is None.
            start_pos (int, optional): The starting position index for the current input
                sequence. Used to compute rotary positional embeddings correctly,
                especially for cached sequences during generation. Default is 0.

        Returns:
            Tuple:
                - Tensor: Output logits with shape (batch_size, sequence_length, vocab_size)
                if `lm_use_tokens` is True, otherwise the hidden state embeddings
                (batch_size, sequence_length, hidden_dim).
                - list: Updated list of key-value caches, one for each transformer block,
                useful for autoregressive decoding and incremental generation.

        Behavior:
            - If `lm_use_tokens` is True, the input token indices are first embedded.
            - Rotary positional embeddings are generated for the current input positions,
            which are passed along to each transformer block.
            - For each transformer block, the input is processed along with
            rotary embeddings, attention mask, and optional cached key-values.
            - After processing all blocks, a final RMS normalization is applied.
            - If tokens are used, the normalized hidden states are projected to logits
            over the vocabulary
            - The method returns the logits or embeddings along with the updated
            cache for efficient decoding.
        """
        
        if self.lm_use_tokens:
            # Check if input is already embedded (float type) or token indices (integer type)
            if x.dtype in (torch.float16, torch.float32, torch.float64):
                # Input is already embedded, skip token embedding
                print("Input is already embedded (float type), skipping token_embedding")
            elif x.dtype in (torch.int32, torch.int64, torch.long, torch.int):
                # Input is token indices, apply token embedding
                x = self.token_embedding(x)
            else:
                # Unknown type, raise error with helpful message
                raise ValueError(f"Unsupported input dtype: {x.dtype}. Expected integer (token indices) or float (embeddings).")
        
        bsz,seq_len_crr,n_embed=x.size()
        
        # Create position_ids for the current sequence based on start_pos
        current_position_ids=torch.arange(start_pos,start_pos+seq_len_crr,device=x.device).unsqueeze(0).expand(bsz,-1)
        
        # Get rotary position embeddings for current tokens
        cos,sin=self.rotary_embd(current_position_ids)
        
        # Initialize new KV cache if none provided
        if kv_cache is None:
            kv_cache=[None] *len(self.blocks)
            
        for i,block in enumerate(self.blocks):
            x,kv_cache[i]=block(x,cos,sin,attention_mask,kv_cache[i])
        x=self.norm(x)
        
        # Compute logits if we are using tokens, otherwise stay in the embedding space
        if self.lm_use_tokens:
            x=self.head(x)
        
        return x,kv_cache
    
    @torch.inference_mode()
    def generate(self,inputs:torch.Tensor,max_new_tokens:int=20):
        """
        Generate tokens autoregressively from a given input sequence.

        Args:
            inputs (torch.Tensor): Input tensor containing token indices or embeddings.
                Shape: (batch_size, sequence_length) or (sequence_length,) for a single sequence.
            max_new_tokens (int): Number of new tokens to generate after the input sequence.

        Returns:
            torch.Tensor: The generated sequence, including the original inputs and newly generated tokens.
                Shape: (batch_size, sequence_length + max_new_tokens)
        """
        
        if inputs.dim()==1:
            inputs=inputs.unsqueeze(0)
        generated_outputs=inputs.clone()
        prompt_output,kv_cache_list=self.forward(generated_outputs,attention_mask=None,kv_cache=None,start_pos=0)
        
        last_output=prompt_output[:,-1,:]
        
        # Decode Phase with KV cache
        for i in range(max_new_tokens):
            if self.lm_use_tokens:
                # Now the model outputs logits
                next_output=torch.argmax(last_output,dim=-1,keepdim=True)
            else:
                # Now the model outputs embeddings
                next_output=last_output.unsqueeze(1)
            
            generated_outputs=torch.cat([generated_outputs,next_output],dim=1)
            
            # The token being,processed is `next_token`, Its position is `generated_ouputs.size(1)-1`
            current_tokens_start_pos=generated_outputs.size(1)-1
            
            
            # 到达max_new_tokens
            if i==max_new_tokens-1:
                break
            
            decode_step_outputs,kv_cache_list=self.forward(next_output,attention_mask=None,kv_cache=kv_cache_list,start_pos=current_tokens_start_pos)
            
            last_output=decode_step_outputs[:,-1,:]
            
        return generated_outputs

    # ------------------------------------------------------------------
    # Shared utilities (delegated to transformer_utils)

    def num_scaling_params(self) -> dict[str, int]:
        """Return parameter counts by category for scaling laws analysis."""
        cfg = self.cfg
        return count_scaling_params(
            n_layers=cfg.n_layers,
            n_embd=cfg.n_embd,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads if cfg.n_kv_heads is not None else cfg.n_heads,
            n_intermediate=cfg.n_intermediate,
            vocab_size=cfg.vocab_size,
            n_positions=cfg.n_positions,
            bias=cfg.bias,
        )

    def estimate_flops(self) -> int:
        """Estimate FLOPs per training iteration (forward + backward)."""
        N = sum(p.numel() for p in self.parameters())
        T = self.cfg.n_positions
        return estimate_flops(N, T)

    def setup_optimizer(self, unembedding_lr: float = 0.008, embedding_lr: float = 0.3,
                        scalar_lr: float = 0.5, matrix_lr: float = 0.02,
                        weight_decay: float = 0.28):
        """Create a combined MuonAdamW optimizer for the model."""
        return build_muon_optimizer(
            self,
            embed_lr=embedding_lr,
            head_lr=unembedding_lr,
            scalar_lr=scalar_lr,
            matrix_lr=matrix_lr,
            weight_decay=weight_decay,
        )

    @classmethod
    def from_pretrained(cls,cfg: LLMConfig):
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        import safetensors
        import torch.nn.init as init
        import json
        from huggingface_hub.utils import EntryNotFoundError
        
        # Load the HuggingFace config
        hf_config=AutoConfig.from_pretrained(cfg.lm_model_type)
        
        # Store original HF vocab size before we modify it
        original_vocab_size=hf_config.vocab_size
        print(f"Original vocabulary size from pretrained model: {original_vocab_size}")
        
        
        
        # We're keeping our own vocab size in cfg, but checking it's larger than original
        if hasattr(cfg,'vocab_size'):
            if cfg.vocab_size < original_vocab_size:
                raise ValueError(f"Config vocab size ({cfg.vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})")
            print(f"Using vocabulary size: {cfg.vocab_size}")
            
        else:
            # If net specified,use the original
            cfg.vocab_size=original_vocab_size
            print(f"Using original vocabulary size: {cfg.vocab_size}")
            
        
        cfg.n_embd=hf_config.hidden_size
        cfg.n_intermediate=hf_config.intermediate_size
        cfg.layer_norm_epsilon=hf_config.rms_norm_eps
        # Handle different naming conventions for RoPE theta across versions
        cfg.rotary_emb_base=getattr(hf_config, 'rope_theta', getattr(hf_config, 'rope_scaling', 10000.0))
        if isinstance(cfg.rotary_emb_base, dict):
            cfg.rotary_emb_base = cfg.rotary_emb_base.get('factor', 10000.0)
        cfg.n_positions=hf_config.max_position_embeddings
        
        cfg.n_heads=hf_config.num_attention_heads
        cfg.n_kv_heads=hf_config.num_key_value_heads
        cfg.dropout=hf_config.attention_dropout
        cfg.n_layers=hf_config.num_hidden_layers

        # Create our model with potentially larger vocabulary
        model=cls(cfg)
        
        try:
            index_path=hf_hub_download(repo_id=cfg.lm_model_type,filename="model.safetensors.index.json")
            
            with open(index_path,'r') as f:
                index=json.load(f)
                
            # Get unique filenames form weight map
            safetensors_filenames=sorted(list(set(index['weight_map'].values())))
            
            # Download all the sharded files
            safetensors_files=[hf_hub_download(repo_id=cfg.lm_model_type,filename=fn) for fn in safetensors_filenames]
            
        except EntryNotFoundError:
            safetensors_files=[hf_hub_download(repo_id=cfg.lm_model_type,filename='model.safetensors')]
        
        sd=model.state_dict()
        
        mapping={
            "model.embed_tokens.weight":"token_embedding.weight",
            "model.norm.weight":"norm.weight"
        }
        
        for i in range(cfg.n_layers):
            layer_prefix=f'model.layers.{i}.'
            block_prefix=f'blocks.{i}.'
            
            mapping.update({
                f'{layer_prefix}self_attn.q_proj.weight':f'{block_prefix}attn.q_attn.weight',
                f'{layer_prefix}self_attn.k_proj.weight':f'{block_prefix}attn.k_attn.weight',
                f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_attn.weight",
                f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.c_proj.weight",
                f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",
                f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",
                f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight"
            })
            
        # Special handling for token embeddings with extended vocabulary
        has_extended_embedding=False
        loaded_keys=set()
        
        for safetensors_file in safetensors_files:
            with safetensors.safe_open(filename=safetensors_file,framework='pt',device='cpu') as f:
                for hf_key,our_key in mapping.items():
                    if our_key in loaded_keys:
                        continue
                    
                    if hf_key in f.keys() and our_key in sd:
                        tensor=f.get_tensor(hf_key)
                        
                        # Special handling for token embeddings if vocab sizes differ
                        if hf_key=='model.embed_tokens.weight' and tensor.shape[0] != sd[our_key].shape[0]:
                            has_extended_embedding=True
                            print(f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}")
                        
                            # Copy existing embeddings to the beginning of our larger embedding matrix
                            sd[our_key][:tensor.shape[0]].copy_(tensor)
                            
                            # Initialize the new embeddings using the same approach as the original model
                            std = 0.02  # Common value, but you might want to adjust based on model
                            init.normal_(sd[our_key][tensor.shape[0]:], mean=0.0, std=std)
                            
                            print(f"Initialized {sd[our_key].shape[0] - tensor.shape[0]} new token embeddings")
                            sd['head.weight'].copy_(sd[our_key])  # Update the head weights as well
                        elif tensor.shape==sd[our_key].shape:
                            sd[our_key].copy_(tensor)
                        
                        else:
                            print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")
                    
                    loaded_keys.add(our_key)
        for hf_key, our_key in mapping.items():
            if our_key not in loaded_keys:
                if our_key in sd:
                    print(f"Warning: Key {our_key} not found in any safetensors file (HF key: {hf_key})")

        # Load the state dict
        model.load_state_dict(sd)
        
        # Handle output projection / language modeling head
        if has_extended_embedding and hasattr(model, 'head') and 'head.weight' in sd:
            # If we have a separate output projection layer and extended the vocab
            # we should handle it similarly to the input embeddings
            lm_head_loaded = False
            for safetensors_file in safetensors_files:
                with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                    if 'lm_head.weight' in f.keys():
                        lm_head = f.get_tensor('lm_head.weight')
                        if lm_head.shape[0] != sd['head.weight'].shape[0]:
                            print(f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}")
                            # Copy existing weights
                            sd['head.weight'][:lm_head.shape[0]].copy_(lm_head)
                            # Initialize new weights
                            std = 0.02
                            init.normal_(sd['head.weight'][lm_head.shape[0]:], mean=0.0, std=std)
                            # Load updated weights
                            model.load_state_dict(sd)
                        lm_head_loaded = True
                        break
        
        # Handle weight tying (if needed)
        if cfg.lm_tie_weights and hasattr(model, 'head') and hasattr(model, 'token_embedding'):
            model.head.weight = model.token_embedding.weight
            # print("Tied token embedding and LM head weights")
        
        print(f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model