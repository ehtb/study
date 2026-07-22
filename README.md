# Transformer Notebooks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A collection of Jupyter notebooks exploring how language models work — from the ground up. Each notebook is self-contained with explanations, code, and runnable examples.

## The journey

The notebooks form one continuous story: how do you get from basic calculus to a chat model? Each one builds on the last, replacing exactly one piece of hand-waving with something built from scratch.

1. **[Neural network](notebooks/neural_net.ipynb)** — the foundation. Build an autograd engine around a single number, derive backpropagation, and train a hand-written MLP to separate two rings of points. Everything after this is the same four-step loop — predict, score, assign blame, nudge — applied to bigger models.
2. **[Embeddings](notebooks/embeddings.ipynb)** — the first language model. Scalars become tensors (verified to be the same autograd engine), and characters become learned vectors via an embedding table. An MLP over 8 characters of Shakespeare beats the bigram baseline, then hits a wall: its fixed context can't scale. That wall is why attention exists.
3. **[Transformer](notebooks/transformer.ipynb)** — the answer to the wall. Starting from a bigram baseline, invent attention one step at a time — uniform averaging, learned queries/keys/values, multiple heads, feed-forward, residuals — and stack it into a full GPT that writes recognizable Shakespeare.
4. **[Tokenizer](notebooks/tokenizer.ipynb)** — from characters to real tokens. The previous notebooks used a 65-character vocabulary; real models use learned subword vocabularies of 32k+. Build Byte-Pair Encoding from scratch in ~40 lines of pure Python, and see why LLMs struggle with spelling, arithmetic, and non-English text.
5. **[Fine-tuning](notebooks/finetuning.ipynb)** — from predictor to assistant. Take a real pretrained model (TinyLlama 1.1B, with exactly the architecture and tokenizer built above) and shape its behavior with LoRA, supervised fine-tuning, and DPO preference optimization — on a MacBook.

By the end, a prompt's full path is covered: bytes → BPE tokens (4) → embedding rows (2) → attention and next-token prediction (3), trained by backpropagation (1) and aligned into an assistant (5).

## Notebooks

| Notebook                                                   | Description                                                                                                                                                                                                             |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [notebooks/neural_net.ipynb](notebooks/neural_net.ipynb)   | **Building a Neural Network from Scratch** — from derivatives to a working multi-layer perceptron with automatic differentiation, backpropagation, and training optimizations, using only pure Python                   |
| [notebooks/embeddings.ipynb](notebooks/embeddings.ipynb)   | **Language Modeling with Embeddings — an MLP** — the bridge from scalars to tensors and from characters to learned vectors; a Bengio-2003-style MLP language model on tiny Shakespeare, every parameter managed by hand |
| [notebooks/transformer.ipynb](notebooks/transformer.ipynb) | **Building a Transformer from Scratch** — step-by-step from a bigram model to a full GPT architecture in 6 incremental steps, training on tiny Shakespeare                                                              |
| [notebooks/tokenizer.ipynb](notebooks/tokenizer.ipynb)     | **Building a BPE Tokenizer from Scratch** — Byte-Pair Encoding in pure Python: bytes, pair counting, merges, encode/decode, and the token-level quirks that explain everyday LLM weirdness                              |
| [notebooks/finetuning.ipynb](notebooks/finetuning.ipynb)   | **Fine-Tuning with LoRA and DPO** — adapt a pretrained model (TinyLlama 1.1B) using parameter-efficient finetuning and preference optimization, runnable on a MacBook                                                   |

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

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Inspired by Andrej Karpathy's educational approach to neural networks and the broader ML community's commitment to accessible education.
