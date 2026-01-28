# param to run on my pc
# python -m scripts.train_qwen config/small_params.py
hidden_size: int = 256 
intermediate_size: int = hidden_size * 3 
num_hidden_layers: int = 2
num_attention_heads: int = 4
num_key_value_heads: int = 2 
