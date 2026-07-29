# Lab

Standalone scripts that apply the concepts from the notebooks to real data and real tooling. Each script is self-contained and runnable from the command line — no notebook required.

Unlike the notebooks, which build understanding step by step, the lab scripts are meant to be run, modified, and experimented with.

## Structure

```
lab/
├── gpt_train_v1.py          # The notebook training loop as a script (Shakespeare, char tokenizer)
├── gpt_train_v2.py          # Adds cosine LR schedule + gradient clipping (smol-smoltalk, tiktoken)
└── utils/
    ├── model.py              # Shared GPT model (imported by both scripts)
    ├── smoltalk_dataset.py   # Dataset loading, formatting, tokenising, batching
    └── checkpoints.py        # Save and load model checkpoints
```

`utils/` contains reusable modules shared across scripts. Import them with `from utils.module import ...`.

## Scripts

### gpt_train_v1.py

The training loop from `notebooks/transformer.ipynb` and `notebooks/scaling.ipynb` packaged as a script — same model, same four steps, same checkpoint logic. Trains on tiny Shakespeare with a character tokenizer.

```bash
cd lab

uv run python gpt_train_v1.py                                          # fresh run, small config (CPU)
uv run python gpt_train_v1.py --block-size 256 --n-embd 384 \
    --n-heads 6 --n-layers 6 --steps 50000 --device cuda           # big config (GPU)
uv run python gpt_train_v1.py --resume <id>                            # continue a previous run
uv run python gpt_train_v1.py --resume <id> --steps 10000
```

### gpt_train_v2.py

Adds two standard training optimizations on top of `gpt_train_v1.py`:

- **Cosine LR schedule with linear warmup** (`--warmup-steps`, `--min-lr`) — ramps the learning rate up, then decays it smoothly to a floor. Reliably improves final val loss vs. a flat LR for the same step budget.
- **Gradient clipping** (`--clip-grad`, default 1.0) — rescales the gradient vector so a single bad batch can't corrupt weights.

Trains on [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) conversational data with the tiktoken BPE tokenizer.

```bash
cd lab

uv run python gpt_train_v2.py                                          # fresh run
uv run python gpt_train_v2.py --steps 10000                            # longer run
uv run python gpt_train_v2.py --resume <id>                            # continue a previous run
uv run python gpt_train_v2.py --warmup-steps 200 --clip-grad 0.5      # tune the optimizations
```

Each run prints a short ID at startup. Use it with `--resume` to continue training in a new session — the checkpoint stores weights, optimizer state, hyperparameters, and step count so training resumes smoothly with no loss spike.

Checkpoints are saved to `lab/checkpoints/` (gitignored).

## Running

All scripts are run from the `lab/` directory so that `utils/` is on the Python path:

```bash
cd lab && uv run python <script>.py
```
