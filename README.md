# Building Transformers from First Principles

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A learning journey through the mathematics and engineering of modern language models. Every component — from backpropagation to attention to tokenization — built from scratch, explained from the ground up, and demonstrated with working code.

## The journey

From basic calculus to a working chat model, one concept at a time. Each notebook replaces one piece of abstraction with something concrete you can understand, modify, and rebuild yourself.

### 1. [Neural Networks](notebooks/neural_net.ipynb) — The Foundation

Build an autograd engine around a single number, derive backpropagation from the chain rule, and train a hand-written MLP to classify two rings of points. Everything after this is the same four-step loop — **predict, score, assign blame, nudge** — applied to bigger models.

**You'll learn:** Why gradients accumulate with `+=`, how ReLU acts as a gradient gate, why learning rates decay, and what makes a layer "dead." The `Value` class here is PyTorch's tensor in miniature.

### 2. [Embeddings](notebooks/embeddings.ipynb) — The First Language Model

Scalars become tensors (verified to be the same autograd engine on the exact `(a+b)*b` example), and characters become learned vectors via an embedding table. An MLP trained on 8 characters of Shakespeare beats the bigram baseline, then hits a wall: its fixed context cannot scale.

**You'll learn:** Why raw token IDs can't be fed to networks, why one-hot encoding is correct but a dead end, the `one_hot(i) @ W == W[i]` identity, why untrained loss equals `-log(1/vocab_size)`, and the three structural reasons this MLP architecture fails to scale (no weight sharing, parameters grow with context, fixed slot weighting ignores content). **That wall is why attention exists.**

### 3. [Transformer](notebooks/transformer.ipynb) — The Answer to the Wall

Starting from a bigram baseline, invent attention one step at a time — uniform averaging, learned queries/keys/values, multiple heads, feed-forward layers, residual connections — and stack them into a full GPT that writes recognizable Shakespeare.

**You'll learn:** How attention computes data-dependent weights, why Q/K dot products measure relevance, what multiple heads buy you, why residuals and LayerNorm matter, and how the whole architecture composes into something that can model long-range dependencies.

### 4. [Tokenizer](notebooks/tokenizer.ipynb) — From Characters to Real Tokens

The previous notebooks used a 65-character vocabulary; real models use learned subword vocabularies of 32k+. Build Byte-Pair Encoding from scratch in ~40 lines of pure Python.

**You'll learn:** Why character-level tokenization doesn't scale, how BPE discovers common subwords through greedy pair merging, why LLMs struggle with spelling and arithmetic, and why non-English text costs more tokens (and thus more money).

### 5. [Fine-tuning](notebooks/finetuning.ipynb) — From Predictor to Assistant

Take a real pretrained model (TinyLlama 1.1B, with exactly the architecture and tokenizer built above) and shape its behavior with LoRA, supervised fine-tuning, and DPO preference optimization — on a MacBook.

**You'll learn:** How LoRA adapts massive models by training only low-rank residuals, how SFT teaches response format, how DPO aligns models to human preferences without reward models, and why pretrained models need instruction-tuning to become assistants.

---

**By the end, a prompt's full path is covered:** bytes → BPE tokens (4) → embedding rows (2) → attention and next-token prediction (3), trained by backpropagation (1), and aligned into an assistant (5).

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
