# Implementing gpt implementation from scratch


# Transformer architechture
# output embedding + positional embedding
# -> Layer Norm -> Multi Headed Attention block -> Layer Norm -> Feed Forward 
# -> Linear -> Softmax

# Predict/Generate

import torch
import torch.nn as nn
import torch.functional as F


