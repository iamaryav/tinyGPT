"""
Qwen2 model
Features:
- untied weights for token and lm_head
- rotary embeddings
- Multi Query Attention (MQA) - Faster Inference
- KV cache
- RMSNorm
"""
import math
import inspect
from functools import partial
from typing import Any, Optional, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.helpers.muon import Muon

@dataclass
class Qwen2Config():

    # For PC & local testing
    # vocab_size: int = 1024
    # hidden_size: int = 64
    # intermediate_size: int = 320 
    # num_hidden_layers: int = 2
    # num_attention_heads: int = 4
    # num_key_value_heads: int = 2 
    # max_seq_len: int = 1024

    # Actual Config 
    # vocab_size: int = 151936
    vocab_size: int = 1024
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2 
    intermediate_size: int = 8960 # 5 times of hidden size
    max_seq_len: int = 1024
    # max_seq_len: int = 32678

    # making false to match nanochat gpt py
    bias: bool = False # True
    layer_types: list = ("full_attention") # for _ in range(num_hidden_layers)]

    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight= nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class RotaryEmbedding(nn.Module):
    """
    Positional Embedding using Rotary Embedding.
    It gets added while causal attention calculation.
    Rotation matrix is used to do rotation in 2D plane
    theta = (p / (base ** (2i/d)))
    theta is in radian
    Rotatory Position embeddings: reltive position between tokens
    """
    def __init__(self, config, device=None):
        super().__init__()
        self.dim= config.hidden_size // config.num_attention_heads # we rotate hidden dim of a head
        self.base = 1000000.0
        self.rope_scaling = False
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        if self.rope_scaling and self.rope_scaling["type"] == "linear":
            inv_freq /= self.rope_scaling["factor"]
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    def forward(self, x, position_ids):
        # inv_freq.shape # (dim/2)
        # position_ids.shape # (batch, seq_len)
        inv_freq_expanded = self.inv_freq[None, :, None].float() # (1, dim/2, 1)
        pos_expanded = position_ids[:, None, :].float() # (batch, 1, seq_len)
        freqs = (inv_freq_expanded @ pos_expanded).transpose(1, 2) # (1, dim/2, 1) @ (B, 1, seq_len)
        # duplicating so we can match the dimension of head_dim
        # x, y co-ordinates
        emb = torch.cat((freqs, freqs), dim=-1) # (batch, seq_len, dim)
        # cosine and sine value for all the (x, y) pair
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos, sin

def rotate_half(x):
    # x: (Batch, seq_len)
    # (x, y), (x, -y)
    # other way to split half
    x1 = x[..., :x.shape[-1] // 2] # x
    x2 = x[..., x.shape[-1] // 2 :] # -y
    return torch.cat((-x2, x1), dim= -1)

def apply_rotary_pos_emb(q, k, cos, sin, positions_ids=None, unsqueeze_dim=1):

    """ 
    cos0, sin0 xcos0 - ysin0, xsin0 + ycos0
    q: (batch, num_heads, seq_len, hidden_dim)
    k: (batch, num_key_value_heads, seq_len, hidden_dim)
    sin/cos: (batch, hidden_dim, seq_len)
    """
    cos = cos.unsqueeze(unsqueeze_dim)  #(batch, 1, seq_len, head_dime)
    sin = sin.unsqueeze(unsqueeze_dim)  #same
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def repeat_kv(hidden_states: torch.tensor, n_rep: int) -> torch.tensor:
    """
    GQA has less key value heads so kv heads get shared in multiple query
    repeat the hidden_states to match the query dimensions i.e.
    (batch, num_key_value_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim) 
    """
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1)

def loss_function(logits: Optional[torch.LongTensor],
                 labels: Optional[torch.LongTensor], 
                 vocab_size: int, 
                 **kwargs,):
    B, T, C = logits.shape
    logits_flat = logits.view(-1, C)
    labels_flat = labels.view(-1)
    loss = F.cross_entropy(logits_flat, labels_flat)
    return loss

class Cache:
    """
    Minimaslistic Key/Value cache for autoregressive decoding.
    cache layer that grows dynamically as more tokens are generated
    It stores the ky and value states as tensors of shape [batch_size, num_heads, seq_len, head_dim]
    """

    def __init__(self):
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
    
    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value states for a specific layer

        Append new key/Value to this layer's cache.
        Returns the full key/value for that layer.

        Args:
            key_states: [batch_size, num_heads, seq_len, head_dim]
            value_states: [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Layer index to update

        Returns:
            Update (key, values) tuple for this layer

        """
        # Extend cache if this layer doesn't exist
        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx + 1):
                self.key_cache.append(None)
                self.value_cache.append(None)

        # Concatenate with existing cache if present
        if self.key_cache[layer_idx] is not None:
            keys = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            values = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        else:
            keys = key_states
            values = value_states
        
        self.key_cache[layer_idx] = keys
        self.value_cache[layer_idx] = values

        return keys, values
    
    def get(self, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached key-value pair for a specific layer
        """
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        return None

    def get_seq_len(self, layer_idx: int=0) -> int:
        """
        Get current seq_len from cache
        """
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].size(2)
        return 0

    def reset(self):
        """
        Clear the key and value cache 
        """
        self.key_cache.clear()
        self.value_cache.clear()


def create_causal_mask(batch_size, query_length, kv_length, dtype=torch.float32, device=None):
    mask = torch.full((batch_size, 1, query_length, kv_length), 0.0, dtype=dtype, device=device)
    mask = torch.triu(torch.full_like(mask, float("-inf")), diagonal=1)
    return mask


class CausalAttention(nn.Module):
    """
    Multi Head Attention 
    Attention(Q, K, V) 
    attention(Q, K, V) = softmax((Q @ K.T)/(dK ** 0.5)) @ V
    GQA - Grouped query attentions: group multiple query for one key and value pair
    self.kqv_proj = nn.Linear(config.num_hidden_layers, (config.num_attention_heads + 2 * config.num_key_value_heads) * config.num_hidden_layer, config.bias)
    Because this is GQA and query dimensions are diff than key and value dimensions
    GQA is used means each key value pair is used in multiple query

    """
    def __init__(self, config, layer_idx):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % config.num_key_value_heads == 0
        self.layer_idx = layer_idx
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.causal = True
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, config.bias)
    
    def forward(self, x: torch.Tensor,
                pos_embds: tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                past_key_value: Optional[Cache]=None,
                use_cache: Optional[bool]= False,
                cache_position: Optional[torch.LongTensor]=None,
                positions_ids: Optional[torch.LongTensor]=None,):
        
        batch, seq_len = x.shape[:-1]
        dropout= 0.0 if not self.training else self.attention_dropout

        # input projections and reshape to implement MHA, GQA
        query_states = self.q_proj(x).view(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2) # (batch, num_attenion_heads, seq_len, head_dim)
        key_states = self.k_proj(x).view(batch, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2) # (batch, num_Key_value_heads, seq_len, head_dim)
        value_states = self.v_proj(x).view(batch, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2) # (batch, num_Key_value_heads, seq_len, head_dim)

        # Rotary positional embeddings
        cos, sin = pos_embds
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache implemenation
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups) # (batch, num_heads, seq_len, head_dim)
        value_states = repeat_kv(value_states, self.num_key_value_groups) # same

        attention_weight = query_states @ key_states.transpose(-1, -2) * self.scaling # (batc, num_attention_heads, seq_len, seq_len)
        if attention_mask is not None:
            causal_mask = attention_mask[:,:,:,:key_states.shape[-2]] # to match the shape of key_states
            # print(f"attention mask device: {attention_weight.device} causal mask device: {causal_mask.device}")
            attention_weight = attention_weight + causal_mask
        
        attention_weight = F.softmax(attention_weight, dim=-1) # add dtype if needed
        attention_weight = F.dropout(attention_weight, p=dropout, training=self.training)
        attn_output = attention_weight @ value_states # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1) # (batch, seq_len, hidden_dim)
        attn_output = self.o_proj(attn_output)# (Batch, seq_len, hidden_dim)

        # implement sliding attention

        return attn_output, attention_weight

class MLP(nn.Module):
    """
    This is also called FeedForwardNetwork
    Gated MLP (SwiGLU-style) in Qwen2
    generally Gated MLP helps model to converge faster, and more expressive, dynamic feature selection, stable training
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        y = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return y

class DecoderLayer(nn.Module):
    """
    Transformer decoder Layer
    normalization -> MLA calcualtion -> normalization -> MLP
        # data -> after MLP
        # residual connection/pathways
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CausalAttention(config=config, layer_idx=layer_idx)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        self.attention_type = config.layer_types[layer_idx]
        # config.layer= "full_attention" 

    def forward(self, 
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor]=None,
                position_ids: Optional[torch.LongTensor]=None,
                past_key_values: Optional[Cache]=None,
                use_cache: Optional[bool]=False,
                cache_positions: Optional[torch.LongTensor]=None,
                position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]]=None,
                **kwargs,):
        residual = x
        x = self.input_layernorm(x)
        # self attention
        x, _ = self.self_attn(x, 
                              pos_embds=position_embeddings,
                              attention_mask= attention_mask, 
                              positions_ids=position_ids, 
                              past_key_value=past_key_values,
                              use_cache=use_cache,
                              cache_position=cache_positions,)
        x = residual + x 

        # fully connected
        residual = x
        x = self.post_attention_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x

class Qwen2Model(nn.Module):
    """
    multi head attention - kqv
    Feed forward - MLP
    Linear layer
    final softmax
    output prob
    lm_head weight tied to embed tokens to improve space
    self.lm_head.weight = self.model.embed_tokens.weight # weights tying
    cache setup
    position ids setup
    Attention mask setup
    position embedding
    Decoder layer
    Final NOrmalization
    Retrun output

    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # model wrapper
        self.model = nn.Module()
        self.vocab_size = config.vocab_size
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.config.layer_types = ["full_attention"] * config.num_hidden_layers
        self.model.layers = nn.ModuleList([DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        # RMS Norm and Layer Norm understand both
        self.model.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.model.rotary_emb = RotaryEmbedding(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init() 

    def post_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: Optional[torch.LongTensor]=None,
                labels: Optional[torch.LongTensor]=None,
                use_cache: Optional[bool] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor]=None,
                inputs_embeds: Optional[torch.FloatTensor]= None,
                logits_to_Keep: Union[int, torch.Tensor]= 0,
                **kwargs
                ):
        
        assert input_ids is not None, "Input must not be empty"
        # input_ids - (batch, seq_len)
        inputs_embeds= self.model.embed_tokens(input_ids) # (batch, seq_len, hidden_dim)
        assert input_ids.max() < self.config.vocab_size, " Token Id out of vocab range"

        if use_cache and past_key_values is None:
            past_key_values = Cache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_len() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):

            mask_kwargs = {
                "batch_size": inputs_embeds.size(0),
                "query_length": inputs_embeds.size(1),
                "kv_length": inputs_embeds.size(1),
                "device": inputs_embeds.device
            }

            # create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs)
            }

        hidden_states = inputs_embeds
        # create position embeddings
        cos, sin = self.model.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.model.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
                **kwargs,
            )
        hidden_states = self.model.norm(hidden_states)

        slice_indices = slice(-logits_to_Keep, None) if isinstance(logits_to_Keep, int) else logits_to_Keep
        # scale up the output to the vocab size
        logits = self.lm_head(hidden_states[:,slice_indices,:])

        loss = None
        if labels is not None:
            loss = loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        
        return logits, loss
    
    def get_num_params(self, non_embedding=True):
        num_params = sum(p.numel() for p in self.parameters())
        return num_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # estimate number of flops per iteration
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size//cfg.num_attention_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) 
        flops_promised = 3e12 # nvidia 1650
        mfu = flops_achieved / flops_promised
        return mfu

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.hidden_size
        # split the parameters in the three group
        # removing all the 1D - layernorm and biases that will not be used in muon
        matrix_Params = list(param for param in self.model.layers.parameters() if param.dim() >= 2)
        embedding_params=  list(self.model.embed_tokens.parameters())
        lm_head_params = list(self.lm_head.parameters())
        print(f"self parameters: {len(list(self.parameters()))}, matrix_params: {len(matrix_Params)}, embeddings_params: {len(embedding_params)} lm_head_params: {len(lm_head_params)}")
        # assert len(list(self.parameters())) == len(matrix_Params) + len(embedding_params) + len(lm_head_params) + 5
        assert len(list(self.parameters())) == len(matrix_Params) + len(embedding_params) + len(lm_head_params) + 57
        # AdamW optimizer is used for the embedding and lm_head
        # this below line scales the LR for the AdamW parameters by 1/(model ** 0.5)
        dmodel_lr_scale = (model_dim / 64) ** -0.5
        print(f"Scaling the RL for the AdamW parameters ∝1/√({model_dim}/64) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale)
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamFactory = partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamFactory(adam_groups, **adamw_kwargs)
        # Create muon optimizer
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = Muon
        muon_optimizer = MuonFactory(matrix_Params, **muon_kwargs)
        # combine the optimizer
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        return optimizers
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int= 50, top_k=None):
        
        assert input_ids is not None and input_ids.numel() > 0, "Input token must contain some value"
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        # cache initialization

        past_key_value = Cache() 
        use_cache = True
        temp = 1.0 
        cur_input_ids = input_ids

        for _ in range(max_new_tokens):
            # forward pass on input tokens
            past_seen_tokens = past_key_value.get_seq_len() if past_key_value.get_seq_len() > 0 else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + cur_input_ids.shape[1],
                device=device
            )
            logits, _ = self.forward(cur_input_ids,
                                        use_cache=use_cache,
                                        past_key_values=past_key_value,
                                        cache_position=cache_position
                                        )
            # get the last token
            next_token_logits = logits[:, -1, :]

            if temp != 1.0:
                next_token_logits = next_token_logits / temp
            
            # get the probablity from last hidden dimensions
            probs = F.softmax(next_token_logits, dim=-1)
            # predict from top k probablities
            if top_k is not None:
                topk_probs, topk_idx  = torch.topk(probs, top_k, dim=-1)
                next_tokens = topk_idx.gather(-1, torch.multinomial(topk_probs, num_samples=1))
            else:
                next_tokens = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_tokens], dim=1)

            cur_input_ids = next_tokens

            # add code for eos token id code to break the loop
            eos_token_id = 151645
            if (next_tokens == eos_token_id).any():
                break

        return input_ids

def test_with_random_tokens_and_weights():

    # Inference test weight actual architecture
    batch_size = 1
    seq_len = 16
    config = Qwen2Config()
    model = Qwen2Model(config)
    model.eval()
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=20, top_k=50)
    print(f"input: {input_ids}")
    print(f"--- output ---")
    print(output_ids)
    print(f"---- new tokens ---")
    print(output_ids[:,input_ids.shape[1]:])
    print(output_ids.shape)
    print(f"new generated token: {output_ids.shape[1] - seq_len}")

if __name__ == "__main__":
    test_with_random_tokens_and_weights()
    # config = Qwen2Config()
    # m= Qwen2Model(config)
    # m.setup_optimizers()
    