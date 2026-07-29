"""
Train the GPT from notebooks/scaling.ipynb on tiny Shakespeare.

This is the notebook's training loop in script form: same model, same four steps,
same checkpointing — but with argparse so hyperparameters come from the command line
and training survives SSH disconnects via screen/tmux.

Usage
-----
    # Fresh run (small config, ~1 min on CPU)
    python train.py

    # Big config — needs GPU; ~30 min on Colab T4
    python train.py --block-size 256 --n-embd 384 --n-heads 6 --n-layers 6 --steps 50000

    # Longer run of an existing session
    python train.py --resume <id> --steps 10000

    # Explicit device
    python train.py --device cuda

Each run prints an ID at startup. Pass it to --resume to continue training in a new
session. The checkpoint stores weights, optimizer state, hyperparameters, and step
count so training resumes smoothly with no loss spike.

Checkpoints are saved to lab/checkpoints/ (gitignored).
"""

import argparse
import uuid
from pathlib import Path

import torch

from utils.model import GPT
from utils.checkpoints import save, load

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train GPT on tiny Shakespeare",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--resume", metavar="ID", help="continue from checkpoint with this run ID")
parser.add_argument("--steps",      type=int,   default=5000,  help="training steps (added on top of resumed step)")
# Architectural params use None as default so resume mismatches can be detected.
parser.add_argument("--block-size", type=int,   default=None, metavar="N",
                    help="context length in tokens (default: 32, loaded from checkpoint on resume)")
parser.add_argument("--n-embd",     type=int,   default=None, metavar="N",
                    help="embedding dimension (default: 64, loaded from checkpoint on resume)")
parser.add_argument("--n-heads",    type=int,   default=None, metavar="N",
                    help="attention heads per block (default: 4, loaded from checkpoint on resume)")
parser.add_argument("--n-layers",   type=int,   default=None, metavar="N",
                    help="transformer blocks stacked (default: 4, loaded from checkpoint on resume)")
# Non-architectural params — safe to change on resume.
parser.add_argument("--batch-size", type=int,   default=64,    help="sequences per gradient step")
parser.add_argument("--dropout",    type=float, default=0.2,   help="dropout rate during training")
parser.add_argument("--lr",         type=float, default=3e-4,  help="AdamW learning rate")
parser.add_argument("--eval-every", type=int,   default=500,   help="steps between loss estimates")
parser.add_argument("--device",     default="auto",            help="cpu | cuda | auto")
cli = parser.parse_args()

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if cli.device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif cli.device == "cuda" and not torch.cuda.is_available():
    print("Warning: --device cuda specified but CUDA is not available; falling back to cpu")
    device = "cpu"
else:
    device = cli.device

# ---------------------------------------------------------------------------
# Run ID and checkpoint path
# ---------------------------------------------------------------------------

if cli.resume:
    run_id = cli.resume
    RESUME = True
else:
    run_id = uuid.uuid4().hex[:8]
    RESUME = False

CHECKPOINT_PATH = f"checkpoints/gpt_train_v1_{run_id}.pt"

# ---------------------------------------------------------------------------
# Hyperparameters
# Architectural params (block_size, n_embd, n_heads, n_layers) are locked once
# a run starts — they define the model shape. On resume they are always loaded
# from the checkpoint; passing them via CLI on a resume is ignored with a warning.
# Non-architectural params (batch_size, dropout, lr) can be changed on resume.
# ---------------------------------------------------------------------------

_ARCH_DEFAULTS = dict(block_size=32, n_embd=64, n_heads=4, n_layers=4)

if RESUME:
    raw = torch.load(CHECKPOINT_PATH, weights_only=True)
    saved_meta = raw.get("meta", {})

    block_size = saved_meta.get("block_size", _ARCH_DEFAULTS["block_size"])
    n_embd     = saved_meta.get("n_embd",     _ARCH_DEFAULTS["n_embd"])
    n_heads    = saved_meta.get("n_heads",     _ARCH_DEFAULTS["n_heads"])
    n_layers   = saved_meta.get("n_layers",    _ARCH_DEFAULTS["n_layers"])

    for name, saved_val in [("block-size", block_size), ("n-embd", n_embd),
                             ("n-heads", n_heads), ("n-layers", n_layers)]:
        cli_val = getattr(cli, name.replace("-", "_"))
        if cli_val is not None and cli_val != saved_val:
            print(f"Warning: --{name} {cli_val} ignored on resume; "
                  f"using {saved_val} from checkpoint (changing it would cause a shape mismatch)")

    if "train_loss" in saved_meta:
        prev_step = raw.get("step", 0)
        print(f"\nPrevious run ended at step {prev_step}  |  "
              f"train loss {saved_meta['train_loss']:.4f}  |  "
              f"val loss {saved_meta['val_loss']:.4f}")
else:
    block_size = cli.block_size or _ARCH_DEFAULTS["block_size"]
    n_embd     = cli.n_embd    or _ARCH_DEFAULTS["n_embd"]
    n_heads    = cli.n_heads   or _ARCH_DEFAULTS["n_heads"]
    n_layers   = cli.n_layers  or _ARCH_DEFAULTS["n_layers"]

batch_size = cli.batch_size
dropout    = cli.dropout
lr         = cli.lr

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

if n_embd % n_heads != 0:
    print(f"Warning: n_embd ({n_embd}) is not divisible by n_heads ({n_heads}); "
          f"head_size will be floored to {n_embd // n_heads}, losing {n_embd % n_heads} dims per head")

if device == "cpu" and (block_size > 128 or n_embd > 256 or n_layers > 6):
    print(f"Warning: large config (block_size={block_size}, n_embd={n_embd}, n_layers={n_layers}) "
          f"on CPU will be slow — consider --device cuda or a smaller config")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Tiny Shakespeare corpus not found at {DATA_PATH}.\n"
        f"Expected it at data/tinyshakespeare.txt — download with:\n"
        f"  curl -o data/tinyshakespeare.txt https://raw.githubusercontent.com/"
        f"karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    )

text = DATA_PATH.read_text()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i    : i + block_size    ] for i in ix])
    y = torch.stack([d[i + 1: i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, n_batches=40):
    was_training = model.training
    model.eval()
    results = {}
    for split in ("train", "val"):
        losses = torch.zeros(n_batches)
        for i in range(n_batches):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        results[split] = losses.mean().item()
    if was_training:
        model.train()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

torch.manual_seed(1337)

model     = GPT(vocab_size, block_size, n_embd, n_heads, n_layers, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
start_step = 0

if RESUME:
    start_step, _ = load(CHECKPOINT_PATH, model, optimizer)

end_step = start_step + cli.steps

print(f"\nRun ID   : {run_id}")
print(f"Device   : {device}")
print(f"Params   : {sum(p.numel() for p in model.parameters()):,}")
print(f"Config   : block_size={block_size}  n_embd={n_embd}  "
      f"n_heads={n_heads}  n_layers={n_layers}")
print(f"Steps    : {start_step} → {end_step}\n")

model.train()
for step in range(start_step, end_step):
    x, y = get_batch("train")

    logits, loss = model(x, y)           # 1 + 2. predict and score
    optimizer.zero_grad(set_to_none=True)
    loss.backward()                      # 3. assign blame
    optimizer.step()                     # 4. nudge

    if step % cli.eval_every == 0 or step == end_step - 1:
        losses = estimate_loss(model)
        print(f"step {step:>6d}  |  train {losses['train']:.4f}  |  val {losses['val']:.4f}")
        save(CHECKPOINT_PATH, model, optimizer, step=step, meta={
            "block_size": block_size,
            "n_embd":     n_embd,
            "n_heads":    n_heads,
            "n_layers":   n_layers,
            "batch_size": batch_size,
            "dropout":    dropout,
            "lr":         lr,
            "train_loss": round(losses["train"], 4),
            "val_loss":   round(losses["val"],   4),
        })
        model.train()

# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

model.eval()
seed = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(seed, max_new_tokens=300)
print("\n--- Sample (temperature 1.0) ---")
print(decode(generated[0].tolist()))

print(f"\nTo continue: python train.py --resume {run_id}")
