"""
this script will generate the output from trained models
i.e saved in out directory as checkpoint ckpt.pt
"""
import torch
import tiktoken
from tinygpt.model_qwen import Qwen2Config, Qwen2Model

device = "cuda" if torch.cuda.is_available() else "cpu"

# Build model exactly like training
config = Qwen2Config()
model = Qwen2Model(config).to(device)

# Load checkpoint
state = torch.load("out/ckpt.pt", map_location=device)
state = state["model"]

# remove DDP/compile preficx _orign_mod
new_state = {}
for k, v in state.items():
    new_state[k.replace("_orig_mod.","")] = v

model.load_state_dict(new_state, strict=True)
model.eval()

# Tokenizer
enc = tiktoken.get_encoding("gpt2")   # or your custom vocab

# Chat loop
while True:
    prompt = input("prompt>>> ")
    if not prompt:
        continue

    # encode
    ids = enc.encode(prompt)
    context = torch.tensor([ids], dtype=torch.long, device=device)

    # generate
    with torch.no_grad():
        out = model.generate(context, max_new_tokens=200, top_k=50)

    tokens = out[0].tolist()
    print(enc.decode(tokens[len(ids):]))

