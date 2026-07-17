# tinyGPT

A small GPT training project inspired by karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and OpenAI's GPT-2. It focuses on pre-training and includes four architectures. For an explanation of the concepts behind GPT architecture and training, see the [Build GPT from scratch](https://aryav.bearblog.dev/build-gpt-from-scratch/) blog post.

## Architectures

- Bigram
- GPT
- GPT-2
- Qwen2.5-style Transformer

## Requirements

tinyGPT requires Python 3.10 or later. The default Qwen2.5 training configuration uses CUDA; multi-GPU training additionally requires a CUDA/NCCL-enabled PyTorch installation.

## Installation

Install the project dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

The default launcher, `bash ./runs/run.sh`, creates a virtual environment, installs dependencies, prepares Tiny Shakespeare, and starts single-node Qwen training. Edit it when switching datasets or training commands.

## Dataset preparation

Prepare a dataset before training. These scripts write tokenized `train.bin` and `val.bin` files to `tinygpt/data/`:

```bash
# Small local dataset used by the default run script
python -m tinygpt.data.prepare_shakespeare

# Alternative datasets
python -m tinygpt.data.prepare_story
python -m tinygpt.data.prepare_openweb
```

## Training

### Qwen2.5

Train the default Qwen2.5-style model:

```bash
python -m scripts.train_qwen
```

Use a supplied configuration file to change model size:

```bash
python -m scripts.train_qwen config/small_params.py
python -m scripts.train_qwen config/qwen_params.py
python -m scripts.train_qwen config/gpt2_params.py
```

The training scripts also accept `--key=value` overrides, for example `--compile=False`. Qwen training uses gradient accumulation, validation-based checkpoints, optional `torch.compile`, checkpoint resume (`--init_from=resume`), and optional offline Weights & Biases logging (`wandb_log=True`; install `wandb` separately to enable it).

### GPT-2

```bash
python -m scripts.train_gpt2
```

### GPT

```bash
python -m tinygpt.gpt
```

### Bigram

```bash
python -m tinygpt.bigram
```

## Distributed training

For DDP on four GPUs on one node, run:

```bash
torchrun --standalone --nproc_per_node=4 -m scripts.train_qwen
```

For multi-GPU GPT-2 training, replace `scripts.train_qwen` with `scripts.train_gpt2` in the command above.

## Inference and checkpoints

Once training is complete, interact with the Qwen model:

```bash
python -m tinygpt.out
```

Qwen checkpoints are saved to `out/ckpt.pt`. The interactive script loads that location and expects the checkpoint to match its Qwen model configuration.

## Qwen2.5 architecture

The Qwen2.5-style configuration uses a 12-layer, decoder-only Transformer with a 768-dimensional hidden state, 12 attention heads, 2 key/value heads (grouped-query attention), a 1,024-token context window, RMSNorm, rotary positional embeddings (RoPE), KV caching for efficient autoregressive inference, and an untied language-model head.

Training supports multi-GPU Distributed Data Parallel (DDP) through `torchrun`; each rank runs on its local CUDA device, synchronizes gradients, and rank 0 handles logging and checkpoints. Forward passes use automatic mixed precision, preferring `bfloat16` when supported and falling back to `float16`. Optimisation is split by parameter type: AdamW updates token embeddings and the LM head, while Muon updates the two-dimensional matrix parameters in Transformer layers using momentum with Newton--Schulz orthogonalization.

## Example output

```text
CAMILLO:
There is a sickness
Which puts some of us in distemper, but
I cannot name the like.

POLIXENES:
No longer.

POLIXENES:
A sickness caught of me, and yet you;
I mean better.


POLIXENES:
A caught of me, and haveAs heavens caught
So young met thus; no, to mean better:
Which 'twere my daughter,
You never spoke what did become you'll procure that honesty
 life born.
```
