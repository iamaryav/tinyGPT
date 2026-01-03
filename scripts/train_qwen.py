"""
normal training run - python3 train.py
Training run example:
$ python3 train.py --batch_size=8 --compile=False

To run ddp with 1 node and 4 gpus
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes example
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py

- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""
import os
import time
import numpy as np
import torch
import pickle
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer
# from torch.cuda.amp import GradScaler
from models.transformers.qwen2_model import Qwen2Config, Qwen2Model
# from models.helpers.muon import Muon
from models.helpers.muon import Muon
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

# --------------------------------------------------------------------------------------
# User defined constants
num_iterations = 2000 # 5000 # 600 # 1000 # 4000 # 8000
eval_every = 500 
# eval_every = 10 
log_interval = 1
vocab_size: int = 1024
# hidden_size: int = 256 # 256 # 64
# intermediate_size: int = hidden_size * 5 # five times as per qwen 2.5
# num_hidden_layers: int = 2
# num_attention_heads: int = 4
# num_key_value_heads: int = 2 
hidden_size: int = 1536 # 256 # 64
intermediate_size: int = hidden_size * 5 # five times as per qwen 2.5
num_hidden_layers: int = 28
num_attention_heads: int = 12 
num_key_value_heads: int = 2 
max_seq_len: int = 1024
block_size = 1024 # 64 # seq_len, max_context_length, max_seq_len
embedding_lr = 0.002 # 0.2 # learning rate for embedding params (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding params (Adam)
weight_decay = 0.0 # weight decay for the embedding and unembedding params (Adam)
matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
learning_rate= 6e-4
compile = True # False
batch_size = 4
dataset = "shakespeare"
device= "cuda"
# skip because my pc graphics doesn't support bfloat16 by default 
# dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 # torch.float32
dtype = torch.float16
print(f"Device for auotcast: {device}, and dtype: {dtype}")
autocast_ctx = torch.amp.autocast(device_type=device, dtype=dtype)
# output
init_from = "scratch" # "scratch" and "resume"
always_save_checkpoint = True # if True always save checkpoint after each eval
wandb_log = False
model_tag = "" # 
best_val_loss = 1e9
output_dirname = "out"
# --------------------------------------------------------------------------------------
# all global variable present in this script
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
# print(f"config_keys: {config_keys}")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "models", "helpers", "configurator.py")
print(f"config path: {CONFIG_PATH}")
exec(open(CONFIG_PATH).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # useful for logging
# --------------------------------------------------------------------------------------
# I/O setup
# torchrun --nproc_per_node=1 train.py
backend = "nccl" # "gloo"
ddp = int(os.environ.get("RANK", -1)) != -1 # is thi ddp run?
print(f"ddp run: {ddp}")
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    node_rank = int(os.environ["NODE_RANK"])
    device = f"cuda:{ddp_local_rank}"
    master_process = ddp_rank == 0 # master process will do logging, checkpionting etc
    seed_offset = ddp_rank # each process gets different seed
    # print(f"ddp_rank: {ddp_rank}")
    # assert grad_accum_steps % ddp_world_size == 0
    # grad_accum_steps //= ddp_world_size
else:
    # if not ddp then we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(output_dirname, exist_ok=True)

# --------------------------------------------------------------------------------------
# logging
run = "dummy"
wandb_project = "owt"
if wandb_log:
    import wandb
    wandb_run = wandb.init(project=wandb_project, name=run, config=user_config, mode="offline")

# --------------------------------------------------------------------------------------
# I want 32 sample/batch per optimizer update
# to mimic bigger batch size -- if you want 32, and your gpu supports 8
# then you can have 4 forward and backward pass for one gradient update
total_batch_size =  1024 # 4096 # 2048 # 8192 # 16384 # 32768 # 65336
batch_size = 1 # 2 # device_batch_size # no of sequence per GPU
# ddp_world_size = 1 # number of GPUs
tokens_per_fwdbwd = batch_size * block_size * ddp_world_size
# print(f"tokens_per_fwdbwd: {tokens_per_fwdbwd}, total_batch_size: {total_batch_size}")
assert total_batch_size % tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // tokens_per_fwdbwd

print(f"tokens per iteration will be: {tokens_per_fwdbwd:,} and grad_accum_steps: {grad_accum_steps}")
print(f"total batch size with grad accum steps: {grad_accum_steps * batch_size}")

# --------------------------------------------------------------------------------------
# Tiny data loader
device_type = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = os.path.join('data', dataset)
def get_batch(split: str):
    # np memmap is more space efficient than normal file loading
    # it only loads the file in RAM when needed
    if split == "train":
        # print(f" path for traind data: {os.path.join(data_dir, 'train.bin')}")
        # print(f"files exists: {os.path.exists(data_dir+ '/train.bin')}")
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint32, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint32, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i+block_size + 1]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    # print(f"train data size: {x.size(), y.size()}")
    return x, y

meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# --------------------------------------------------------------------------------------
# Initialize the Model
model_args = dict(num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, hidden_size=hidden_size, intermediate_size=intermediate_size, max_seq_len=max_seq_len)
if init_from == "scratch":
    print(f"Intializing a new model from scratch...")
    config = Qwen2Config(**model_args)
    model = Qwen2Model(config)
elif init_from == "resume":
    print(f"Resuming training from {output_dirname}")
    ckpt_path = os.path.join(output_dirname, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    # we need to force these attributes to be equal otherwise we can't resume training
    # the rest of attributes (e.g. dropout) can stay as desired from command line
    for k in ["num_hidden_layers", "num_attention_heads", "num_key_value_heads", "hidden_size", "intermediate_size", "max_seq_len"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    qwen_config = Qwen2Config(**model_args)
    model = Qwen2Model(qwen_config)
    state_dict = checkpoint["model"]
    # fix the keys of prefix
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    step = checkpoint["step"]
    best_val_loss = checkpoint["best_val_loss"]
# crop down block size if needed refer nanoGPT
model.to(device)
print(f"config of the model: {model.config}")

# --------------------------------------------------------------------------------------
# optimizer Initialization
# optimizer
# weight = weight - learning_rate * gradient
# forward pass = calculate the loss and prediction
# backward pass = compute the gradient, how much each parameter contributed to the error
# optimizer - uses gradients to update the weights in smart way
# optimizers = model.setup_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimzer = optimizers

if init_from == "resume":
    opt_state_dicts = checkpoint["optimizers"]
    for opt, state_dict in zip(optimizers, opt_state_dicts):
        opt.load_state_dict(state_dict)

checkpoint = None # free up memory


# --------------------------------------------------------------------------------------
# Set up hyperparameter Initialization
# learning rate decay
warmup_ratio = 0.0 # ratio iteration for LR warmup
warmdown_ratio = 0.2 # ration of iterations of LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmdown_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# --------------------------------------------------------------------------------------

# compile the model
if compile:
    print("compiling the model... takes some time")
    unoptimized_model = model
    model = torch.compile(model)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# --------------------------------------------------------------------------------------
# Loss calculation, tracking and visualization
val_losses = []
iterations = []

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_every)
        for k in range(eval_every):
            X, Y = get_batch(split)
            with autocast_ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
# Real time plot for loss visualization
def plot_losses(save_path='training_progress.png'):
    """Plot training and validtion losses and save to file"""
    plt.figure(figsize=(10, 6))
    # plt.plot(iterations, train_losses, label='Train Loss', marker='o', linewidth=2, markersize=4)
    plt.plot(iterations, val_losses, label='Val Loss',)# linewidth=2, markersize=4)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Val Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    # print(f"Loss plot saved to {save_path}")

# --------------------------------------------------------------------------------------
# fix
# loss is getting to NaN after 1500 iterations
# something related to learing rate I guess

X, Y = get_batch('train')
raw_model = model.module if ddp else model # unwrap DDP container if needed
local_step = 0
running_mfu = -1.0
# scaler = GradScaler()

# training run
for step in range(num_iterations + 1):
    last_step = step == num_iterations

    # evaluate the loss on eval_every step
    if last_step or (step % eval_every == 0 and master_process):
        losses = estimate_loss()
        print(f"step {step}/{num_iterations}: train loss: {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # print(f"evaluating training loss...")
        val_losses.append(losses["val"].item())
        iterations.append(step)
        if wandb_log:
            wandb_run.log({
                "step": step,
                "train/loss": losses["train"],
                "val/loss": losses["val"],
                "lr": learning_rate,
                # "mfu": running_mfu * 100,
            })
        plot_losses()
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if step > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizers": [opt.state_dict() for opt in optimizers],
                    # "optimizers": optimizers,
                    "model_args": model_args,
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "user_config" : user_config,
                }
                print(f"saving checkpoint to {output_dirname}")
                torch.save(checkpoint, os.path.join(output_dirname, "ckpt.pt"))

    # -----------------------------------------------------------------------------------
    # Single training step
    # evaluate the gradient
    # set the learning rate for this iteration
    t0 = time.time()
    # single batch combined with several mini batches
    for micro_step in range(grad_accum_steps):
        # print(f"inside combined microstep loss ddp: {ddp}")
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with autocast_ctx:
            # print(f"inside autocast")
            logits, loss = model(X, Y)
            # print(f"loss calculated: {loss}")
            loss = loss / grad_accum_steps
            # print(f"loss calculated: {loss}")
        # print(f"outside autocast")
        loss.backward()
        # scaler.scale(loss).backward()
        # print(f"get the train batch")
        X, Y = get_batch("train")
        # print(f"I have the train batch")
    
    if grad_clip > 0.0:
        # print(f"inside gradient clipping")
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

    # print(f"before optimizers")
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimzer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    # print(f"after optimizers")

    # -----------------------------------------------------------------------------------
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if step % log_interval == 0 and master_process:
        lossf = loss.item() * grad_accum_steps
        if step >= 5:
            mfu = raw_model.estimate_mfu(batch_size * grad_accum_steps, dt)
            # print(f"mu")
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"steps {step}; loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
    # local_step += 1

    # -----------------------------------------------------------------------------------

print(f"Training complete.")

# --------------------------------------------------------------------------------------
# Save final plot
plt.figure(figsize=(12, 7))
plt.plot(iterations, val_losses, label='Val Loss',) 
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Val Loss', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_loss_final.png', dpi=300, bbox_inches='tight')
plt.close()
print("Final loss plot saved as 'training_loss_final.png'")

# --------------------------------------------------------------------------------------
# Generate sampele from model
print("# -----------------------------------------------------------------------------")
print(f"generating the model output...")
tokenizer = AutoTokenizer.from_pretrained("./data/shakespeare/qwen-small-tokenizer")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(context, max_new_tokens=500, top_k=50)[0]))
print("# -----------------------------------------------------------------------------")

# --------------------------------------------------------------------------------------
# cleanup
if wandb_log:
    wandb_run.finish()

if ddp:
    destroy_process_group()