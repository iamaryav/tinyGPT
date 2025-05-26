# Implementing gpt implementation from scratch


# Transformer architechture
# output embedding + positional embedding
# -> Layer Norm -> Multi Headed Attention block -> Layer Norm -> Feed Forward 
# -> Linear -> Softmax

# Predict/Generate


B, T, C = 4, 8, 16
x = torch.randn(B, T, C)
embd_d = 0
dropout = 0.0

class Head(nn.Module):
    def __init__(self, head_size):
        # key, query, value
        self.key = nn.Linear(embd_d, head_size, bias=False) # y = x wT + b
        self.query = nn.Linear(embd_d, head_size, bias=False) 
        self.value = nn.Linear(embd_d, head_size, bias=False) 
        # create mask that will help in aggregating or calculating the average 
        # all past tokens
        self.register_buffer = ("tril", torch.tril(torch.ones(block_size, block_size)))
        # to introduce regrularization in neural network
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # This block is just to implement the actual attention formula
        # 
        # x = B, T, C
        k = self.key(x) # B, T, hs 
        q = self.query(x) # B, T, hs
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** 0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        # Deactivate next tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei) # (B, T, T)
        wei = self.dropout(wei) # (B, T, T)
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Heads(head_size) for _ in range(num_heads)]
        self.proj = nn.Linear(head_size * num_heads, embd_d)
        self.dropouts = nn.Dropout(dropout)

    def forward(self, x):
        # Concatinating all the heads into one
        out = torch.cat([h(x) for h in self.heads])
        # converting output to normal dimensions and applying dropout
        out = self.dropouts(self.proj(out))
        return out


class FeedForward(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        # Linear layer
        nn.Linear(embd_d, 4 * embd_d), # y = x(wT) + b, T -> transpose of matrix
        # Introduces non-linearirty and this is the most important 
        # things in Neural networks
        nn.ReLU(), # F(x) = max(0, x)
        nn.Linear(4 * embd_d, embd_d),
        # Regularization technique drops diff neurons in diff training cycles
        nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    





class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.multi_head = MultiHeadAttention(num_heads, head_size)
        self.feed_forward = FeedForward()
        # Layer Norm normalizes the data with 0 mean and 1 SD
        self.ln_1 = nn.LayerNorm(embd_d)
        self.ln_2 = nn.LayerNorm(embd_d)

    def forward(self, x):
        # also residual connection technique is used here as mentioned in transformer block paper 
        x = x + self.multi_head(self.ln_1(x))# where model learns stuff main attention mechanism
        x = x + self.feed_forward(self.ln_2(x)) # where model thinks

        return x
        


