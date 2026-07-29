"""
GPT trained on smol-smoltalk conversational data, with production training optimizations.

gpt_train_v1.py is the verbatim notebook loop — same code, just a script.
This script adds the two optimizations every real training run uses:

  1. Cosine LR schedule with linear warmup
     The learning rate ramps up linearly for --warmup-steps, then decays
     following a cosine curve down to --min-lr. Flat LR converges to a
     worse minimum; cosine decay reliably shaves ~0.05–0.10 off val loss
     for the same step budget.

  2. Gradient clipping (--clip-grad, default 1.0)
     Clips the global gradient norm before the optimizer step. Prevents
     the occasional large gradient spike from corrupting weights — critical
     with deep models and long sequences.

Both work on CPU and GPU. The tokenizer is always tiktoken (GPT-2 BPE,
~50k vocab), which is the right choice for conversational data.

Usage
-----
    python gpt_train_v2.py                                   # fresh run
    python gpt_train_v2.py --steps 10000                     # longer run
    python gpt_train_v2.py --resume a3f1c2b4                 # continue from checkpoint
    python gpt_train_v2.py --resume a3f1c2b4 --steps 10000
    python gpt_train_v2.py --device cuda                     # explicit GPU
"""

import sys
import math
import uuid
import argparse

import torch

from utils.model import GPT
from utils.smoltalk_dataset import prepare, get_batch
from utils.checkpoints import save, load

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train GPT on smol-smoltalk (tiktoken, with cosine LR + grad clipping)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--resume",      metavar="ID", help="continue from checkpoint with this run ID")
parser.add_argument("--steps",       type=int,   default=5000,
                    help="training steps (added on top of resumed step)")
parser.add_argument("--subset-size", type=int,   default=500,
                    help="number of smol-smoltalk conversations to train on")
# Architectural params use None as default so resume mismatches can be detected.
parser.add_argument("--block-size",  type=int,   default=None, metavar="N",
                    help="context length in tokens (default: 64, loaded from checkpoint on resume)")
parser.add_argument("--n-embd",      type=int,   default=None, metavar="N",
                    help="embedding dimension (default: 128, loaded from checkpoint on resume)")
parser.add_argument("--n-heads",     type=int,   default=None, metavar="N",
                    help="attention heads per block (default: 4, loaded from checkpoint on resume)")
parser.add_argument("--n-layers",    type=int,   default=None, metavar="N",
                    help="transformer blocks stacked (default: 4, loaded from checkpoint on resume)")
# Non-architectural params — safe to change on resume.
parser.add_argument("--batch-size",  type=int,   default=32,   help="sequences per gradient step")
parser.add_argument("--dropout",     type=float, default=0.2,  help="dropout rate during training")
parser.add_argument("--lr",          type=float, default=3e-4, help="peak AdamW learning rate")
parser.add_argument("--min-lr",      type=float, default=None,
                    help="minimum LR after cosine decay (default: lr / 10)")
parser.add_argument("--warmup-steps", type=int,  default=100,
                    help="steps over which LR ramps linearly from 0 to --lr")
parser.add_argument("--clip-grad",   type=float, default=1.0,
                    help="gradient norm clipping threshold (0 = disabled)")
parser.add_argument("--eval-every",  type=int,   default=200,  help="steps between loss estimates")
parser.add_argument("--device",      default="auto",           help="cpu | cuda | auto")
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

CHECKPOINT_PATH = f"checkpoints/gpt_train_v2_{run_id}.pt"

# The data seed is randomised per fresh run and saved in the checkpoint.
# On resume it is restored, guaranteeing the same subset → same vocab →
# same model shape → weights load without mismatches.
data_seed = None  # resolved below

# ---------------------------------------------------------------------------
# Hyperparameters
# Architectural params (block_size, n_embd, n_heads, n_layers) are locked once
# a run starts — they define the model shape. On resume they are always loaded
# from the checkpoint; passing them via CLI on a resume is ignored with a warning.
# Non-architectural params (batch_size, dropout, lr, etc.) can be changed on resume.
# ---------------------------------------------------------------------------

_ARCH_DEFAULTS = dict(block_size=64, n_embd=128, n_heads=4, n_layers=4)

if RESUME:
    raw = torch.load(CHECKPOINT_PATH, weights_only=True)
    saved_meta = raw.get("meta", {})
    data_seed = saved_meta["data_seed"]

    saved_tokenizer = saved_meta.get("tokenizer", "tiktoken")
    if saved_tokenizer != "tiktoken":
        print(f"Error: checkpoint was trained with tokenizer '{saved_tokenizer}'; "
              f"gpt_train_v2.py only supports tiktoken. Start a fresh run.")
        sys.exit(1)

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
        print(f"\nPrevious run ended at step {raw.get('step', '?')}  |  "
              f"train loss {saved_meta['train_loss']:.4f}  |  val loss {saved_meta['val_loss']:.4f}")
else:
    block_size = cli.block_size or _ARCH_DEFAULTS["block_size"]
    n_embd     = cli.n_embd    or _ARCH_DEFAULTS["n_embd"]
    n_heads    = cli.n_heads   or _ARCH_DEFAULTS["n_heads"]
    n_layers   = cli.n_layers  or _ARCH_DEFAULTS["n_layers"]
    data_seed  = int(torch.randint(0, 2**31, (1,)).item())

batch_size  = cli.batch_size
dropout     = cli.dropout
lr          = cli.lr
min_lr      = cli.min_lr if cli.min_lr is not None else lr / 10

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

if n_embd % n_heads != 0:
    print(f"Warning: n_embd ({n_embd}) is not divisible by n_heads ({n_heads}); "
          f"head_size will be floored to {n_embd // n_heads}, losing {n_embd % n_heads} dims per head")

if cli.steps < 2000:
    print(f"Warning: tiktoken has a ~50k-token vocab so the embedding and output layers are large. "
          f"{cli.steps} steps is likely too few for meaningful convergence — consider --steps 5000+")

if cli.warmup_steps >= cli.steps:
    print(f"Warning: --warmup-steps ({cli.warmup_steps}) >= --steps ({cli.steps}); "
          f"the LR will never reach the cosine decay phase")

if device == "cpu" and (block_size > 128 or n_embd > 256 or n_layers > 6):
    print(f"Warning: large config (block_size={block_size}, n_embd={n_embd}, n_layers={n_layers}) "
          f"on CPU will be slow — consider --device cuda or a smaller config")

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step, end_step):
    """Cosine decay with linear warmup.

    Ramps from 0 to `lr` over `warmup_steps`, then decays via cosine to
    `min_lr` by `end_step`. The decay is smooth and has zero derivative at
    both endpoints, which avoids the abrupt loss spikes that come with
    step-wise or linear decay.
    """
    if step < cli.warmup_steps:
        return lr * (step + 1) / cli.warmup_steps
    progress = (step - cli.warmup_steps) / max(1, end_step - cli.warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _batch(split):
    return get_batch(train_data, val_data, split, block_size, batch_size, device=device)


@torch.no_grad()
def estimate_loss(model, n_batches=50):
    model.eval()
    results = {}
    for split in ("train", "val"):
        losses = torch.zeros(n_batches)
        for i in range(n_batches):
            x, y = _batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        results[split] = losses.mean().item()
    model.train()
    return results


def train(model, optimizer, start_step, end_step):
    for step in range(start_step, end_step):
        # Cosine LR schedule: update before the forward pass so step 0 uses
        # the warmup LR rather than the full lr (avoids a large first step).
        lr_now = get_lr(step, end_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        x, y = _batch("train")
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping: rescale the entire gradient vector so its L2 norm
        # is at most clip_grad. Prevents a bad batch from sending weights far
        # off course. Disabled when clip_grad == 0.
        if cli.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cli.clip_grad)

        optimizer.step()

        if step % cli.eval_every == 0 or step == end_step - 1:
            losses = estimate_loss(model)
            print(f"step {step:>5d}  lr {lr_now:.2e}  |  "
                  f"train loss {losses['train']:.4f}  |  val loss {losses['val']:.4f}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

train_data, val_data, encode_fn, decode_fn, vocab_size = prepare(
    subset_size=cli.subset_size, seed=data_seed, tokenizer="tiktoken"
)

torch.manual_seed(1337)

model     = GPT(vocab_size, block_size, n_embd, n_heads, n_layers, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
start_step = 0

if RESUME:
    start_step, _ = load(CHECKPOINT_PATH, model, optimizer)

end_step = start_step + cli.steps

print(f"\nRun ID      : {run_id}")
print(f"Device      : {device}")
print(f"Params      : {sum(p.numel() for p in model.parameters()):,}")
print(f"Config      : block_size={block_size}  n_embd={n_embd}  n_heads={n_heads}  n_layers={n_layers}")
print(f"Data        : {cli.subset_size} conversations  tokenizer=tiktoken")
print(f"Steps       : {start_step} → {end_step}")
print(f"LR schedule : warmup {cli.warmup_steps} steps, cosine decay {lr:.2e} → {min_lr:.2e}")
print(f"Grad clip   : {cli.clip_grad if cli.clip_grad > 0 else 'disabled'}\n")

train(model, optimizer, start_step, end_step)

final_losses = estimate_loss(model)
save(CHECKPOINT_PATH, model, optimizer, step=end_step, meta={
    "data_seed":    data_seed,
    "tokenizer":    "tiktoken",
    "block_size":   block_size,
    "n_embd":       n_embd,
    "n_heads":      n_heads,
    "n_layers":     n_layers,
    "batch_size":   batch_size,
    "dropout":      dropout,
    "lr":           lr,
    "min_lr":       min_lr,
    "warmup_steps": cli.warmup_steps,
    "clip_grad":    cli.clip_grad,
    "train_loss":   round(final_losses["train"], 4),
    "val_loss":     round(final_losses["val"],   4),
})

model.eval()
seed_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(seed_ids, max_new_tokens=200)
print("\n--- Sample generation ---")
print(decode_fn(generated[0].tolist()))

print(f"\nFinal loss  |  train: {final_losses['train']:.4f}  |  val: {final_losses['val']:.4f}")
print(f"To continue training: python gpt_train_v2.py --resume {run_id}")
