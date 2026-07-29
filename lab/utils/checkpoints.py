"""
Save and load model checkpoints.

A checkpoint has two layers:
  model weights   — the learned parameters (always saved)
  optimizer state — AdamW tracks a running mean and variance for every
                    parameter (the m and v momentum buffers). Omitting
                    these when resuming resets the adaptive learning rate,
                    causing a loss spike for the first few hundred steps.

Use save()/load() with an optimizer to resume training.
Use save()/load() without one for inference or weight sharing.
"""

import torch
from pathlib import Path


def save(path, model, optimizer=None, step=None, meta=None):
    """Write a checkpoint to disk.

    path: file path, e.g. "checkpoints/full_gpt.pt"
    model: any nn.Module
    optimizer: pass the live optimizer to make the checkpoint resumable
    step: training step count to store alongside the weights
    meta: dict of arbitrary serializable values (e.g. {"data_seed": 42})
    """
    checkpoint = {"model": model.state_dict()}
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if step is not None:
        checkpoint["step"] = step
    if meta is not None:
        checkpoint["meta"] = meta
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved checkpoint → {path}  (step {step})")


def load(path, model, optimizer=None):
    """Read a checkpoint from disk and restore model (and optionally optimizer) state.

    Returns (step, meta): the saved step count (or 0) and meta dict (or {}).

    Always pass the same optimizer you will train with so its momentum
    buffers are restored. Omit it when loading for inference only.
    """
    # weights_only=True refuses to unpickle arbitrary Python objects —
    # safe default when loading files from untrusted sources.
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint.get("step", 0)
    meta = checkpoint.get("meta", {})
    print(f"Loaded checkpoint ← {path}  (step {step})")
    return step, meta
