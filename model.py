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
    pass

# follow the gpt2 naming convention
# Create this model class
# 

class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        # output embedding
        # positional embedding
        # Attention block
        # lm_head
        





# ---------------------
# for now to run model class testing purpose
# will remove later
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(text[:100])





