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
# python -m tinygpt.data.prepare_openweb
# python -m tinygpt.data.prepare_story


# run the training scripts
python -m scripts.train_qwen

# run with small params for pc
# python -m scripts.train_qwen config/small_params.py

# run with qwen_params
# python -m scripts.train_qwen config/qwen_params.py

# run with gpt2 params
# python -m scripts.train_qwen config/gpt2_params.py
