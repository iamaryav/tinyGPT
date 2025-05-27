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

# follow the gpt2 naming convention
# Create this model class
# 

class GPT(nn.Module):

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
        pass







# ---------------------
# for now to run model class testing purpose
# will remove later
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(text[:100])





