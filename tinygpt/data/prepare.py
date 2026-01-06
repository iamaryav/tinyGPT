# download the tiny shakespeare dataset 
# enode those data
# and then create and save those data in file for training purposes
import os
import requests
import numpy as np
from transformers import AutoTokenizer

vocab_size = 1024 # depends on the model architecture
input_file_path = os.path.join(os.path.dirname(__file__), "tiny_shakespeare.txt")
print(input_file_path)
if not os.path.exists(input_file_path):
    print(f"File doesn't exist in local downloading from web...")
    data_url = "https://raw.githubusercontent.com/iamaryav/tinyGPT/refs/heads/main/data/input.txt"
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()

n = len(data)
print(f"Number of character in file {n}")

# training data
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]
print(f"Length of training data: {len(train_data)} and validation data: {len(val_data)}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B", use_fast=True)

# to convert big tokenizer to small tokenizer
new_tokenizer = tokenizer.train_new_from_iterator(train_data, vocab_size=vocab_size)
new_tokenizer.save_pretrained("./qwen-small-tokenizer")

tokenizer = AutoTokenizer.from_pretrained("./qwen-small-tokenizer")

# train_tokens = tokenizer.tokenize(train_data, add_special_tokens=False)
# val_tokens = tokenizer.tokenize(val_data, add_special_tokens=False)

train_ids = tokenizer(train_data, add_special_tokens=False)["input_ids"]
val_ids = tokenizer(val_data, add_special_tokens=False)["input_ids"]

print(f"training data has {len(train_ids)} tokens")
print(f"val data has {len(val_ids)} tokens")

# export to bin files

train_ids = np.array(train_ids, dtype=np.uint32)
val_ids = np.array(val_ids, dtype=np.uint32)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

print(f"train and val bin file created successfully")