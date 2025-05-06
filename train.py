
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------

@dataclass
class GPTConfig:
    # 50000 BPE merges + 256 bytes token + 1 <|endoftext|> token
    vocab_size: int = 50257  # Number of tokens recognized by GPT
    block_size: int = 1024 # Context length, Maximum sequence length
    n_embd: int = 768 # Feature vector for each token, Embedding dimensions
    n_layer: int = 12 # Number of transformer blocks used
    n_head: int =  12 # number of heads in each transformer block

class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Mulit-Head attention
        assert config.n_embed % config.n_head == 0
        # self.key = nn.Linear(config.n_embd, config.n_head, bias=False)
        # self.query = nn.Linear(config.n_embd, config.n_head, bias=False)
        # self.key = nn.Linear(config.n_embd, config.n_head, bias=False)
        # instead of creatig key, query, value
        # creating a c_attn and making it with bigger dimensions
        # and using it wisely to behave like it key, query, value
        # Every token in sequence will these three vectors
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        # Not bias, It is a mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
        
        def forward(self, x):
            B, T, C = x.size() # Batch size, sequence size, embedding dimensions
            qkv = self.c_attn(x)

            # q @ k
            # nh * ns = n_embd
            # we are seperating the embedding dimensions in to multiple heads and size
            # Instead of creating three diff linear layer converting everything in a single layer
            # and viewing as a q, k, v like three feature vector 
            # It makes calculation faster compare to having three vector
            # 
            q, k, v = qkv.split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            # Manaully calculating attention
            # how interesting they find each other
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
            # Normalizes the attention
            att = F.softmax(att, dim=-1) 
            # Way to do weighted sum
            y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
            # using pytoruch deafult
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
            # In the form of original shape
            y = y.transpose(1, 2).contiguous().view(B, T, C) # same like concating the result
            # output projection
            y = self.c_proj(y)
            return y


        



class MLP(nn.Module):
    """
    Model thinks here so for thinking increase the embedding dim
    then decrease
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc= nn.Linear(config.n_embd, 4 * config.n_embd)
        # Using GELU to avoid dead RELU neuron problem
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """
    Attention is communication mechanism from past tokens
    This is way of aggregating information from past tokens
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd) # Normalize the input
        self.atten = CasualSelfAttention(config) # Attention block
        self.ln_2 = nn.LayerNorm(config.n_embd) # Normalize the weight
        # Below model is to let model think what they learn
        self.mlp = MLP(config) # Feed Forward network

    def forward(self, x):
        x = x + self.atten(self.ln_1(x)) # Reduce
        x = x + self.mlp(self.ln_2(x)) # Map
        return x



class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Feature of tokens
            wpe = nn.Embedding(config.block_size, config.n_embd), # Feautre of positions
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Transformer block
            ln_f = nn.LayerNorm(config.n_embd) # Normalize the weights
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Project the hidden layer weight to vocab size (vocab_size * vocab_size)
    
    def forward(self, idx, target=None):
        # idx will be shape of (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T}, is greater than vocab size{self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        tok_emb = self.transformer.wte(pos) # (T, n_embd)
        pos_emb = self.transformer.wpe(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        # Forwarding through transformer blocks
        # Wondering through transformer forests :)
        for block in self.transformer.h:
            x = block(x)
            
        # applying final layer norm
        x = self.transformer.ln_f(x)
        # Forwarding through final linear layer
        # to get the dimension same as vocab size
        logits = self.lm_head(x) # (B, T, vocab_size)
        # probablity
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss
            






    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
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

# ------------------------------------------------------
model = GPT.from_pretrained("gpt2")
print("Reached here...")












