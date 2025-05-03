
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50
    block_size: int = 1024 # Context length
    n_embd: int = 768 # Feature vector for token in context length
    n_layer: int = 12 # Number of blocks used
    n_head: int =  12 # number of heads in multi head attention

class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # Mulit-Head attention
        assert config.n_embed % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_head, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_head, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_head, bias=False)

        



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
        self.atten = CasualSelfAtention(config) # Attention block
        self.ln_2 = nn.LayerNorm(config.n_embd) # Normalize the weight
        # Below model is to let model think what they learn
        self.mlp = MLP(config) # Feed Forward network

    def forward(self, x):
        x = x + self.atten(self.ln_1(x)) # Reduce
        x = x + self.mlp(self.ln_2(x)) # Map



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
    
    def forward(self, idx):
        pass
