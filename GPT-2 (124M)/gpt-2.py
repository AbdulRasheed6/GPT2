from dataclasses  import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
import math



# =========================================================
# Device
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Download Tiny Shakespeare (Colab-safe)
# =========================================================
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "/content/tiny_shakespeare.txt"

if not os.path.exists(DATA_PATH):
    print("Downloading Tiny Shakespeare dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")

# =========================================================
# nanoGPT-style Character Encoding
# =========================================================
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)   # (N,)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("Vocab size:", vocab_size)


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init()
        assert config.n_embed % config.n_head ==0
        #key, query, value projection for all heads, but in a batch
        self.c_attn= nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj= nn.Linear(config.n_embed, config.n_embed)

        # regularisation
        self.n_head= config.n_head
        self.n_embed= config.n_embed
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size, config.block_size))


    def forward(self, x):
        B, T, C= x.size()
        # batch size, sequence length, embedding dimensionality(n_embed) 
        #  calculate query, key values for all heads in batch
        # nh is  "number of heads", hs is "head_size" and C (number of channels)= nh * hs
        # eg. in gpt-2 (124M), n_head=12, hs=64, so nh*hs=C=768 
        qkv= self.c_attn(x)
        q, k, v= qkv.split(self.n_embed, dim=2)
        k= k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)     
        q=q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)     
        v=v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)     
        att= (q @ k.transpose(-2,-1)* (1/math.sqrt(k.size(-1)))) # (B, nh, T, T)
        att= att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att= F.softmax(att, dim=-1)
        y= att @ v #(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y= y.transpose(1,2).contiguous().view(B, T, C)
        y= self.c_proj(y)
        return y

    


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc= nn.Linear(config.n_embed, 4* config.n_embed)
        self.gelu= nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x= self.c_fc(x)
        x= self.gelu(x)
        x= self.c_proj(x)
        return x



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1= nn.LayerNorm(config.n_embed)
        self.attn= CasualSelfAttention(config)
        self.ln_2= nn.LayerNorm(config.n_embed)
        self.mlp= MLP(config)

    def forward(self, x):
        x= x + self.attn(self.ln_1(x))
        x= x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int=1024 #max sequence length
    vocab_size: int=50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int=12
    n_head: int=12
    n_embed: int=768 #embedding dimension


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config

        self.transformer= nn.ModuleDict(dict(
            wte= nn.Embedding(config.vocab_size, config.n_embed),
            wpe= nn.Embedding(config.block_size, config.n_embed),
            h  = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f= nn.LayerNorm(config.n_embed),
        ))
        self.lm_head= nn.Linear(config.n_embed, config.vocab_size, bias=False)


    def forward(self, input_ids):
        # input_ids is of shape (B,T)
        
        B, T= input_ids.shape

        assert T<= self.config.block_size
        # forward the token and position embeddings
        pos= torch.arange(0, T, dtype=torch.long, device=input_ids.device) # shape(T)
        pos_emb= self.transformer.wpe(pos) # postion embeddings of shape(T, n_embed)
        tok_emb= self.transformer.wte(input_ids) # token embeddings of shape(B, T, n_embed)
        x= tok_emb + pos_emb

        # forward the blocks of the transformer

        for block in self.transformer.h:
            x= block(x)

        # forward the final layer_norm
        x= self.transformer.ln_f(x)
        logits= self.lm_head(x)
        return logits


# Training and  Generation Example 
# =========================================================
# Training Setup
# =========================================================
config = GPTConfig(
    n_embed=384,
    n_layers=6,
    n_head=6,
    vocab_size=vocab_size,
    seq_len=128
)

model = GPT2(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

def get_batch(batch_size=16):
    data = train_data
    ix = torch.randint(len(data) - config.seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+config.seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+config.seq_len+1] for i in ix]).to(device)
    return x, y


# =========================================================
# Training Loop
# =========================================================
model.train()
for step in range(2000):
    xb, yb = get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")


# =========================================================
# Text Generation
# =========================================================
@torch.no_grad()
def generate(prompt, max_new_tokens=300):
    model.eval()
    input_ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        input_ids = input_ids[:, -config.seq_len:]
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1], dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

    return decode(input_ids[0].tolist())


print("\n================= GENERATED SHAKESPEARE =================\n")
print(generate("ROMEO:"))

