A collection of Jupyter notebooks exploring how language models work — from the ground up. Each notebook is self-contained with explanations, code, and runnable examples.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [transformer/notebook.ipynb](transformer/notebook.ipynb) | **Building a Transformer from Scratch** — step-by-step from a bigram model to a full GPT architecture in 6 incremental steps, training on tiny Shakespeare |

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

```bash
uv sync
```

This installs all dependencies from `pyproject.toml` into a `.venv` automatically.

## Run

```bash
uv run jupyter notebook
```

Then open any notebook and run all cells.
