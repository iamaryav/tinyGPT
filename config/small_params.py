import torch

# param to run on my pc
# python -m scripts.train_qwen config/small_params.py
num_iterations = 2000 
eval_every = 500
log_interval = 10
hidden_size: int = 256 
intermediate_size: int = hidden_size * 3 
num_hidden_layers: int = 2
num_attention_heads: int = 4
num_key_value_heads: int = 2 
dtype = torch.float16
matrix_lr = 0.005 # learning rate for the matrix parameters (Muon)