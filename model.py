# Implementing gpt implementation from scratch


# Transformer architechture
# output embedding + positional embedding
# -> Layer Norm -> Multi Headed Attention block -> Layer Norm -> Feed Forward 
# -> Linear -> Softmax

# Predict/Generate
# File create model architecture
# train with smaller dataset tiny shakespear .py
# keep developing and testing

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------

@dataclass
class GPTConfig:
    # Keep the number in power of two or divisible by two
    # because most computation happens in power of two (0, 1)
    vocab_size = 50304 # GPT-2: vocab size 50257 # Number of token recognized by GPT
    # Maximum number of token a model can take at a single time
    # context length, Maximum sequence length
    block_size = 1024 
    # Feature vector for each token, embedding dimensions, what information that token holds
    n_embd = 768 
    # No of transformer block | Number of layers
    # Number of residual layer
    n_layer = 12
    n_head = 12 # number of heads in each transformer block
    dropout = 0.0
    bias = True # True: bias in Linears and LayerNorms, Like GPT-2. False: a bit better and faster

# # number of heads in each transformer block follow the gpt2 naming convention
# Create this model class
class CasualSelfAttention(nn.Module):
    """
    # TODO: add comment
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # self.key = nn.Linear(config.n_embd, config.n_head, bias=False)
        # self.query = nn.Linear(config.n_embd, config.n_head, bias=False)
        # self.key = nn.Linear(config.n_embd, config.n_head, bias=False)
        # instead of creatig key, query, value
        # creating a c_attn and making it with bigger dimensions
        # and using it wisely to behave like it key, query, value
        # Every token in sequence will these three vectors
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias= config.bias)
        self.c_proj= nn.Linear(config.n_embd, config.n_embd, bias= config.bias)
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.dropout = config.dropout
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resd_dropout = nn.Dropout(config.dropout)

        # not bias it is a mask
        # this helps us to take average from previous tokens
        # in other word learn from past models
        # pytorch internal attention is more efficient in GPU due
        # to less operations
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("using slow attention, for flash attention use pytorch >=2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
                            
        def forward(self, x):
            # Dimension of data
            # Actual training tokens
            hs = config.n_head
            B, T, C = x.shape
            # this contains query, key and value all three
            # we are using single c_attn to have all three by 
            # having three times of n_embd dimensions
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)

            # nh * hs = n_embd
            # we are seperating the embedding dimensions in to multiple heads and size
            # Instead of creating three diff linear layer converting everything in a single layer
            # and viewing as a q, k, v like three feature vector 
            # It makes calculation faster compare to having three vector
            # head size = n_embd // n_head
            ns = C // self.n_head
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)


            if self.flash:
                # pytorch Default calculatioin of self attention
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)
            else:
                # (query @ key) / dk ** 0.5
                # implementation of attention manually
                # how interesting they find each other
                attn = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))) # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
                attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
                attn = F.softmax(attn, dim=-1)
                attn = self.attn_dropout(attn)
                y = attn @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
            # we need to transpose then change the view
            # concatinating the output and making it as same dimension as
            # B, T, C
            y = y.transpose(1, 2).contiguous().view(B, T, C)

            # output projection
            y = self.resid_dropout(self.c_proj(y))
            return y
            

class MLP(nn.Module):
    """
    Feed forward Network present in Transformer decoder block
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    # The transformer block
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Attention block
        # Holy grail of Transformer architechture
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        # Normalize the embedding dimensions
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # this is where model learns or thinks on its learning
        self.mlp = MLP(config) # Feed forward network
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        


class GPT(nn.Module):
    """
    GPT implementation
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        # output embedding
        # positional embedding
        # Attention block
        # lm_head
        self.transformer = nn.ModuleDict(dict(
            # Contains Feature of tokens
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embeddings
            # Contains feature of position 
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embeddings
            # Multiple transformer blocks called layer here
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Normalize weights 
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        # Projection from hidden layer weight(n_embd) to vocab size
        # this will help to calculate probablity
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # TODO this line of code migh create an issue be aware
        self.transformer.wte.weight = self.lm_head.weight

        # initialize all weights 
        self.apply(self._init_weights)
        # as per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def forward(self, x):
        # will code forward pass at last
        pass




# ---------------------
# for now to run model class testing purpose
# will remove later
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(text[:100])





