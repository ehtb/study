"""
GPT language model — shared between train.py and full_gpt.py.

    from utils.model import GPT

    model = GPT(
        vocab_size=65,
        block_size=32,
        n_embd=64,
        n_heads=4,
        n_layers=4,
        dropout=0.2,
    )
    model.to(device)

All config is passed explicitly; there are no module-level globals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # register_buffer so the mask moves with model.to(device)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        head_size = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * head_size**-0.5    # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v                                       # (B, T, head_size)


class _MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads   = nn.ModuleList([_Head(n_embd, head_size, block_size, dropout) for _ in range(n_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)    # (B, T, n_embd)
        return self.dropout(self.proj(out))                     # (B, T, n_embd)


class _FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        self.sa  = _MultiHeadAttention(n_embd, n_heads, block_size, dropout)
        self.ff  = _FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # communicate: each token gathers context
        x = x + self.ff(self.ln2(x))   # compute: each token processes what it gathered
        return x


class GPT(nn.Module):
    """GPT language model — the architecture from notebooks/transformer.ipynb.

    Parameters
    ----------
    vocab_size : int    number of distinct tokens
    block_size : int    maximum context length (also sets the causal mask size)
    n_embd     : int    embedding / residual stream width
    n_heads    : int    parallel attention heads per block
    n_layers   : int    transformer blocks stacked
    dropout    : float  fraction of activations zeroed during training
    """

    def __init__(self, vocab_size, block_size, n_embd, n_heads, n_layers, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[_Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                                    # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb                                                  # (B, T, n_embd)
        x = self.blocks(x)                                                     # (B, T, n_embd)
        x = self.ln_f(x)                                                       # (B, T, n_embd)
        logits = self.lm_head(x)                                               # (B, T, vocab_size)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
