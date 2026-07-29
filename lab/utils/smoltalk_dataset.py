"""
SmolTalk dataset utilities for GPT training experiments.
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk

Covers: loading, formatting, vocabulary building, splitting, and batching.
Import this module in any demo that trains on conversational data.
"""

import torch
from datasets import load_dataset


def load_smoltalk(subset_size=None, hf_split="train", seed=42):
    """Load the smol-smoltalk dataset, optionally capped at subset_size conversations.

    subset_size: int or None. None loads the full split (~460K train rows).
    hf_split: "train" or "test"
    Returns a HuggingFace Dataset object.

    Cached at $HF_DATASETS_CACHE (default ~/.cache/huggingface/datasets).
    Override by setting that env var before running.
    """
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=hf_split)
    if subset_size is not None:
        ds = ds.shuffle(seed=seed).select(range(min(subset_size, len(ds))))
    return ds


def format_conversation(messages):
    """Render one conversation's message list as a flat string with role markers.

    Format: <|role|>\\ncontent\\n — each turn separated by a newline.
    An <|end|> sentinel closes the conversation.
    """
    parts = []
    for msg in messages:
        parts.append(f"<|{msg['role']}|>\n{msg['content']}")
    return "\n".join(parts) + "\n<|end|>\n"


def dataset_to_text(ds):
    """Flatten all conversations in a Dataset into one big string."""
    return "\n".join(format_conversation(row["messages"]) for row in ds)


def build_vocab(text):
    """Build a character-level vocabulary from a text string.

    Returns (stoi, itos, vocab_size):
      stoi: dict mapping char → int
      itos: dict mapping int → char
      vocab_size: number of unique characters
    """
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, len(chars)


def split_data(data, val_fraction=0.1):
    """Split a 1-D LongTensor of token IDs into (train, val) tensors."""
    n = int((1 - val_fraction) * len(data))
    return data[:n], data[n:]


def get_batch(train_data, val_data, split, block_size, batch_size, device="cpu"):
    """Sample a random batch of (inputs, targets) for next-token prediction.

    Returns x, y both shaped (batch_size, block_size).
    y is x shifted right by one position — the standard LM target.
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


def prepare(subset_size=None, val_fraction=0.1, seed=42, tokenizer="char"):
    """End-to-end helper: load → format → encode → split.

    tokenizer: "char" (default) or "tiktoken".
      "char"    — character-level vocabulary built from the loaded text.
                  vocab_size varies per subset (~150–250 chars).
      "tiktoken" — GPT-2 BPE encoding (requires: pip install tiktoken).
                  vocab_size is fixed at 50257 regardless of the data,
                  so the embedding tables are much larger (~12M extra params).
                  Better tokenization quality; slower on CPU for small runs.

    Returns (train_data, val_data, encode_fn, decode_fn, vocab_size).
      encode_fn(text: str) -> list[int]
      decode_fn(ids: list[int]) -> str
    Use get_batch() to sample from train_data / val_data.
    """
    print(f"Loading SmolTalk (subset_size={subset_size}, tokenizer={tokenizer})...")
    ds = load_smoltalk(subset_size=subset_size, seed=seed)
    print(f"Loaded {len(ds)} conversations.")

    text = dataset_to_text(ds)
    print(f"Total characters: {len(text):,}")

    if tokenizer == "tiktoken":
        try:
            import tiktoken
        except ImportError:
            raise ImportError("tiktoken is not installed. Run: pip install tiktoken")
        enc = tiktoken.get_encoding("gpt2")
        encode_fn = enc.encode
        decode_fn = lambda ids: enc.decode(ids)
        vocab_size = enc.n_vocab
        tokens = torch.tensor(enc.encode(text), dtype=torch.long)
    else:
        stoi, itos, vocab_size = build_vocab(text)
        encode_fn = lambda t: [stoi[c] for c in t]
        decode_fn = lambda ids: "".join(itos[i] for i in ids)
        tokens = torch.tensor(encode_fn(text), dtype=torch.long)

    print(f"Vocab size: {vocab_size}  |  Total tokens: {len(tokens):,}")
    train_data, val_data = split_data(tokens, val_fraction=val_fraction)
    print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

    return train_data, val_data, encode_fn, decode_fn, vocab_size
