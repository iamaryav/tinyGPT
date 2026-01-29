#!/bin/bash

# install uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# create a local virtual env .venv (if it doesn't exist)
[-d ".venv"] || uv venv

# install the dependencies
uv sync

# activate the virtual env
source .venv/bin/activate

# download the datasets
# change dataset for different runs
python -m tinygpt.data.prepare_shakespeare

# dataset size - 1GB
# python -m tinygpt.data.prepare_story  

# dataset size - 40GB
# python -m tinygpt.data.prepare_openweb


# comment this and uncomment others to start different types of training runs
# to start the qwen2 training
python -m scripts.train_qwen

# to start the gpt2 training 
# python -m scripts.train_gpt2

# ---------------------------------------------------
# run training with different params sizes, and config

# run with small params for pc
# python -m scripts.train_qwen config/small_params.py

# run with qwen actual param size
# python -m scripts.train_qwen config/qwen_params.py

# run with gpt2 params
# python -m scripts.train_qwen config/gpt2_params.py
