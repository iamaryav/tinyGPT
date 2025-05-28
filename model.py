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
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------

@dataclass
class GPTConfig:
    vocab_size = 50257 # Number of token recognized by GPT
    # Maximum number of token a model can take at a single time
    # context length, Maximum sequence length
    block_size = 1024 
    # Feature vector for each token, embedding dimensions, what information that token holds
    n_embd = 768 
    # No of transformer block
    n_layer = 12
    n_head = 12 # number of heads in each transformer block

# # number of heads in each transformer block follow the gpt2 naming convention
# Create this model class
class CasualSelfAttention(nn.Module):
    """
    # TODO: add comment
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n
        # self.key = nn.Linear(config.n_embd, config.n_head, bias=False)
        # self.query = nn.Linear(config.n_embd, config.n_head, bias=False)
        # self.key = nn.Linear(config.n_embd, config.n_head, bias=False)
        # instead of creatig key, query, value
        # creating a c_attn and making it with bigger dimensions
        # and using it wisely to behave like it key, query, value
        # Every token in sequence will these three vectors
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj= nn.Linear(config.n_embd, config.n_embd)
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        # not bias it is a mask
        # this helps us to take average from previous tokens
        # in other word learn from past models 


class MLP(nn.Module):
    """
    # TODO: add comment
    """
    pass

class Block(nn.Module):
    """
    # TODO: add comment
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Attention block
        # Holy grail of Transformer architechture
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.atten = CasualSelfAttention(config)
        # Normalize the embedding dimensions
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # this is where model learns or thinks on its learning
        self.mlp = MLP(config) # Feed forward network
    
    def forward(self, x):
        pass
        


class GPT(nn.Module):
    """
    # TODO: add comment
    """

    def __init__(self, config):
        super().__init__()
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

    def forward(self, x):
        # will code forward pass at last
        pass




# ---------------------
# for now to run model class testing purpose
# will remove later
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(text[:100])





