import math
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# model architecture on high level
# then piece by piece
"""
Requirements
Linux
pytorch
cuda
Download model weights
download project there

"""

class Qwen2Config():
    # Define config like Qwen model did
    # nice way to define config though
    
    vocab_size: int = 151936
    hidden_size: int = 1536
    intermediate_size: int = 8960
    hidden_act: str = "silu"
    max_positional_embeddings: int = 32678
    num_attention_heads: int = 12
    # if less Then GQA(Grouped Query Attention) else Standard attention
    num_key_value_heads: int = 2 
    num_hidden_layers: int = 28
    rms_norm_eps: int = 1e-06
    rope_theta: float = 1000000.0
    bias: bool = True
    layer_types: list = ["full_attention" for _ in range(num_hidden_layers)]
    use_sliding_window: bool = False
    sliding_window: int = 4096 
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
    def __init__(self, config, device=None):
        super().__init__()
        # form of positional embeddings
        # absolute pos embed addded added in start
        # rotaryembeddings applied in query and key in attention block
        # Rotatory Position embeddings: reltive position between tokens
        # Rotate the query/key vectors
        # Rotation matrix is used to do rotation in 2D plane
        # theta is in radian
        # theta = (p / (base ** (2i/d)))
        self.dim= config.hidden_size # we rotate hidden dim
        self.base = 1000000.0
        self.rope_sacling = False
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        # rope scaling if provided
        if self.rope_sacling and self.rope_scaling["type"] == "linear":
            inv_freq /= self.rope_scaling["factor"]
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # inv_freq.shape # (dim/2)
        # position_ids.shape # (batch, seq_len)
        # (1, dim/2, 1)
        inv_freq_expanded = self.inv_freq[None, :, None] # (1, dim/2, 1)
        # (batch, 1, seq_len)
        pos_expanded = position_ids[:, None, :].float() # (batch, 1, seq_len)
        # to get rotation angle
        # 
        freqs = inv_freq_expanded @ pos_expanded # (1, dim/2, 1) @ (B, 1, seq_len)
        freqs = freqs.transpose(1, 2) # (B, dim/2, seq_len)
        # duplicating so we can match the dimension and use same theta with
        # x, y co-ordinates
        emb = torch.cat((freqs, freqs), dim=1) # (batch, dim, seq_len)

        # cosine and sine value for all the radians
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
def apply_rotary_pos_emb(q, k, cos, sin, positions_ids=None, unsqueeze_dim=1):
    """ 
    cos0, sin0
    xcos0 - ysin0, xsin0 + ycos0
    
    """
    # TODO move this method outside
    def rotate_half(x):
        # x: (Batch, seq_len)
        # (x, y), (x, -y)
        # x1 = x[..., ::2] # first half of x
        # x2 = x[..., 1::2] # second half of x
        # return torch.stack((-x2, x1), dim=-1).reshape_as(x)
        # other way to split half
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim= -1)
    cos = cos.unsqueeze(unsqueeze_dim) # (batch, 1, head_dim, seq_len)
    sin = sin.unsqueeze(unsqueeze_dim) 
    # q: (batch, num_heads, seq_len, hidden_dim)
    # sin/cos: (batch, 1, hidden_dim, seq_len)
    # q @ cos = 
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
    # hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, seq_len, head_dim)
    # return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)
    return hidden_states.repeat_interleave(n_rep, dim=1)

class Cache():
    """
    cache layer that grows dynamically as more tokens are generated
    It stores the ky and value states as tensors of shape [batch_size, num_heads, seq_len, head_dim]

    """
    def __init__(self, config):
        self.config = config

    def allocate(self, batch_size, num_heads, max_seq_len, head_dim, device, dtype):
        self.keys = torch.empty(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.values = torch.empty(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.cur_pos = 0

    # TODO: remove if not needed
    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
    
    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        update the key, value inplace and return the udpated key and value
        """
        # old and inefficient way
        # if self.keys is None:
        #     self.lazy_initialization(key_states)
        
        # self.keys = torch.cat([self.keys, key_states], dim=-2)
        # self.values = torch.cat([self.values, value_states], dim=-2)
        # return self.keys, self.values

        seq_len = key_states.size(2)
        self.keys[:,:,self.cur_pos:self.cur_pos + seq_len] = key_states
        self.values[:,:,self.cur_pos:self.cur_pos + seq_len] = value_states
        self.cur_pos += seq_len
        return self.keys[:,:,:self.cur_pos], self.values[:,:,:self.cur_pos]

# def create_casual_mask(batch_size, query_length, kv_length, dtype=torch.float32):
#     mask = torch.zeros((batch_size, 1, query_length, kv_length), dtype=dtype)
#     mask[:,:,:,:] = float("-inf")
#     mask = torch.triu(mask, diagonal=1)
#     return mask

def create_casual_mask(batch_size, query_length, kv_length, dtype=torch.float32, device=None):
    mask = torch.full((batch_size, 1, query_length, kv_length), 0.0, dtype=dtype, device=device)
    mask = torch.triu(torch.full_like(mask, float("-inf")), diagonal=1)
    return mask




class CasualAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # this is single piece of attention head
        # Attention(Q, K, V) 
        # definition
        # num_heads 12
        # kqv does magic in hidden dimensions
        # dimensions of kqv
        # num_hidden_layers // num_attention_heads 
        # 1536 // 12 = 128
        # then later concatenated to same again
        # what if it was single attention head
        # 28 Layers/blocks -> each layer has 12 heads
        # multi head to after concatenation it should
        # sum up to hidden dimensions
        # formula and their ranges remember
        # few of the activation- softmax, Relu, leaky-relu, gelu, silu, sigmoid, tanh, softplus
        assert config.hidden_size % config.num_key_value_heads == 0
        assert config.num_attention_heads % config.num_key_value_heads == 0
        self.layer_idx = layer_idx
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.casual = True
        # hidden_size = num_attention_heads * head_dim
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        # Flash Attention - ? 
        # GQA - Grouped query attentions: group multiple query for one key and value pair
        # what's wrong if i want to define in to one
        # self.kqv_proj = nn.Linear(config.num_hidden_layers, (config.num_attention_heads + 2 * config.num_key_value_heads) * config.num_hidden_layer, config.bias)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, config.bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, config.bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, config.bias)
        # mix the attention from each head to learn from each other
        # kind of like packing bag
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, config.bias)
        # where is your dropouts
        # define mask
        # sliding window: fix previous element visit
        # we can define behaviour of each layer differently that's nice
        # self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        # what about masking?
        # self.register_buffer('tril', torch.tril(torch.ones(config.num_hidden_layer, config.num_hidden_layer)))
        # i think i understand rotary embeddings now :D
    
    def forward(self, x: torch.Tensor,
                pos_embds: tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                past_key_value: Optional[Cache]=None,
                use_cache: Optional[bool]= False,
                cache_position: Optional[torch.LongTensor]=None,
                positions_ids: Optional[torch.LongTensor]=None,
                dropout: float= 0.0,
                **kwargs,):
        # casual attention forward pass
        """
        # plain casual attention
        attention(Q, K, V) = softmax((Q @ K.T)/(dK ** 0.5)) @ V
        
        """
        # 
        # multihead attention
        # hidden_size * hidden_size
        # reshuffle the tensor for num_heads 
        # x: (B, seq_len, hidden_size)
        # we're not taking last dimension 
        # because this is GQA and query dimensions are diff than key and value dimensions
        # GQA is used means each key value pair is used in multiple query
        batch, seq_len = x.shape[:-1]
        # model suggested this one to use
        # batch, seq_len = x.size()
        # hidden_size = num_attention_heads * head_dim
        # hidden_shape = (batch, seq_len, self.config.num_attention_heads, self.head_dim) 
        # input projections and reshape to implement MLA, GQA
        query_states = self.q_proj(x).view(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2) # (batch, num_attenion_heads, seq_len, head_dim)
        key_states = self.k_proj(x).view(batch, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2) # (batch, num_Key_value_heads, seq_len, head_dim)
        value_states = self.v_proj(x).view(batch, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2) # (batch, num_Key_value_heads, seq_len, head_dim)

        # Rotary positional embeddings
        cos, sin = pos_embds
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache implemenation
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups) # (batch, num_heads, seq_len, head_dim)
        value_states = repeat_kv(value_states, self.num_key_value_groups) # same

        attention_weight = query_states @ key_states.transpose(-1, -2) * self.scaling # (batc, num_attention_heads, seq_len, seq_len)
        if attention_mask is not None:
            casual_mask = attention_mask[:,:,:,:key_states.shape[-2]] # to match the shape of key_states
            attention_weight = attention_weight + casual_mask
        
        attention_weight = F.softmax(attention_weight, dim=-1) # add dtype if needed
        attention_weight = F.dropout(attention_weight, p=dropout, training=self.training)
        attn_output = attention_weight @ value_states # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
        # contiguous nearby space in memory
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1) # (batch, seq_len, hidden_dim)
        attn_output = self.o_proj(attn_output)# (Batch, seq_len, hidden_dim)
        # attn? kv cache, rotary embedding, sliding attention
        # implement sliding attention

        return attn_output, attention_weight


        # B, T, C = x.shape
        # # x: (Batch, seq_len, hidden_size)
        # # q: (hidden_size, hidden_size)
        # q = self.q_proj(x) # (Batch, seq_len, hidden_size)
        # k = self.k_proj(x) # (Batch, seq_len, hidden_size)
        # # q @ k # (batch, seq_len, seq_len)
        # attn = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
        # ones = torch.tril(torch.ones(T, T))
        # attn_scores = attn.masked_fill(ones == 0, float("-inf"))
        # attn_weights = F.softmax(attn, dim=-1)
        # v = self.v_proj(x)
        # y = attn @ v # (batch, seq_len, hidden_size)
        # y = self.o_proj(y) # (B, T, C)
        # return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Normal MLP
        # Gated MLP (SwiGLU-style) in Qwen2
        # generally Gated MLP helps model to converge faster
        # more expressive, dynamic feature selection, stable training
        self.config = config
        self.hidden_size = config.hidden_size
        # intermediate size is 5 times of hidden size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # define silu here
        self.act_fn = nn.SiLU()
        # dropouts
    def forward(self, x):
        hidden_shape = x.shape
        y = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return y

class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CasualAttention(config=config, layer_idx=layer_idx)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)
        # attention type flag, probe more on this
        self.attention_type = config.layer_types[layer_idx]

    def forward(self, 
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor]=None,
                position_ids: Optional[torch.LongTensor]=None,
                past_key_values: Optional[Cache]=None,
                use_cache: Optional[bool]=False,
                cache_positions: Optional[torch.LongTensor]=None,
                position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]]=None,
                **kwargs,):
        # decoder layer
        # normalization -> MLA calcualtion -> normalization -> MLP
        # data -> after MLP
        # residual connection/pathways
        residual = x
        x = self.input_layernorm(x)
        # self attention
        x, _ = self.self_attn(x, 
                                      pos_embds=position_embeddings,
                                      attention_mask= attention_mask, 
                                      position_ids=position_ids, 
                                      past_key_value=past_key_values,
                                      use_cache=use_cache,
                                      cache_position=cache_positions,
                                      **kwargs,)
        x = residual + x 

        # fully connected
        residual = x
        x = self.post_attention_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x

        



class Qwen2Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        # define the model architecture
        # token embeddings
        self.config = config
        # model wrapper
        self.model = nn.Module()
        # self.padding_idx = config.pad_token_id # what is this?
        self.vocab_size = config.vocab_size

        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # multi head attention - kqv
        # Feed forward - MLP
        # Linear layer
        # final softmax
        # output prob
        self.model.layers = nn.ModuleList([DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        # RMS Norm and Layer Norm understand both
        self.model.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # this decides th position emb
        self.model.rotary_emb = RotaryEmbedding(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # lm_head weight tied to embed tokens to improve space
        self.lm_head.weight = self.model.embed_tokens.weight # weights tying

        # checkpoint and has sliding layers later as i progress and get more understanding
        # self.gradient_checkpointing = False
        # self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        # Initialize weights and apply final processing
        # steps before training 
        self.post_init() # understand this as well

    def post_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: Optional[torch.LongTensor]=None,
                use_cache: Optional[bool] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[Cache] = None,
                cache_position: Optional[torch.LongTensor]=None,
                inputs_embeds: Optional[torch.FloatTensor]= None,
                labels: Optional[torch.FloatTensor]= None,
                logits_to_Keep: Union[int, torch.Tensor]= 0,
                **kwargs
                ):
        
        # embedding lookup
        # input ids are input token
        assert input_ids is not None, "Input must not be empty"
        inputs_embeds= self.model.embed_tokens(input_ids) # (batch, seq_len, hidden_dim)

        # checking for cache
        if use_cache and past_key_values is None:
            past_key_values = Cache(config=self.config)

        # skipping cache for now 
        # if cache_position is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position = torch.arange(
        #         past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        #     )
        
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)
        if position_ids is None:
            seq_len = inputs_embeds.size(1)
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)


        if not isinstance(casual_mask_mapping := attention_mask, dict):

            # attention mask arguements
            mask_kwargs = {
                "batch_size": inputs_embeds.size(0),
                "query_length": inputs_embeds.size(1),
                "kv_length": inputs_embeds.size(1)
            }

            # create the masks
            casual_mask_mapping = {
                "full_attention": create_casual_mask(**mask_kwargs)
            }

            # TODO implment this once the architecture start working
            # if self.has_sliding_layers:
            #     casual_mask_mapping["sliding_attention"] = create_sliding_window_casual_mask(**mask_kwargs)
        
        hidden_states = inputs_embeds
        # create position embeddings
        cos, sin = self.model.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.model.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=casual_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
                **kwargs,
            )
        hidden_states = self.model.norm(hidden_states)
        # return hidden_states, past_key_values if use_cache else None

        slice_indices = slice(-logits_to_Keep, None) if isinstance(logits_to_Keep, int) else logits_to_Keep
        # scale up the output to the vocab size
        logits = self.lm_head(hidden_states[:,slice_indices,:])

        # calculate loss only while training
        # TODO define loss function
        # for testing purpose commented this out
        loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        
        return logits, loss

        # cache setup
        # position ids setup
        # Attention mask setup
        # position embedding
        # Decoder layer
        # Final NOrmalization
        # Retrun output

    @classmethod
    def from_pretrained(cls):
        # load the model to see what model architecture I wrote is correct or not
        """
        Loads qwen 2 weight into our hand written architecture
        """
        # model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        # loading hugginface model
        from transformers import AutoConfig, AutoModelForCausalLM
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = "./qwen"
        # hf_config = AutoConfig.form_pretrained(model_path)
        config = Qwen2Config()
        hf_model = AutoModelForCausalLM.from_pretrained(model_path, config=config)
        hf_state = hf_model.state_dict()

        # creating instance of our model
        model = Qwen2Model()

        missing, unexpected = model.load_state_dict(hf_state, strict=False)
        print("Missing Keys: ", missing)
        print("Unexpected keys: ", unexpected)
        return model.to(device)
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int= 50, top_k=None):
        # do some inference with your loaded model
        # model class will take care of both caching and position embedding
        # get the logits
        # apply the softmax
        # get the output from the sample distribution
        # keep doing for number of token required
        assert input_ids is not None and input_ids.numel() > 0, "Input token must contain some value"
        device = next(self.parameters()).device

        # Path to the model
        # config = Qwen2Config()
        # model = Qwen2Model(config)
        temp = 1.0 

        for _ in range(max_new_tokens):
            # forward pass on input tokens
            logits, loss = self.forward(input_ids)
            # get the last token
            next_token_logits = logits[:, -1, :]

            # sampling
            # temp to scale up 
            # basically to more focus on high probablity tokens results
            # means more repititive result
            # add kv cache and other things once you successfully load the model

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

            # add code for eos token id code to break the loop
            eos_token_id = 151645
            if (next_tokens == eos_token_id).any():
                break

        return input_ids

if __name__ == "__main__":
    prompt = "what is capital of India?"
    model_path = "./qwen"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input = tokenizer(prompt, return_tensors="pt").to(device)

    config = Qwen2Config()
    model = Qwen2Model.from_pretrained()
    output_tokens = model.generate(input["input_ids"])
    output =  tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("Generate text from existing weights: \n")
    print(output)