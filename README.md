# Machine Learning from First Principles

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A hands-on study of machine learning concepts, built from scratch. 

## Notebooks

### Foundational

The prerequisites for every course. Two notebooks that build the shared mechanics from scratch in pure Python, before any domain-specific content appears.

#### 1. [ML Foundations](notebooks/foundational/1_foundations.ipynb) — The Learning Paradigm

Establish the vocabulary and mechanics of machine learning before any neural networks appear. Train a linear model on toy house prices using gradient descent written in pure Python. See overfitting happen in real time with a polynomial model, then measure it with a train/val split. Finish with logistic regression and binary cross-entropy.

**You'll learn:** The four-step training loop (predict, score, assign blame, nudge) that every subsequent notebook reuses. Why gradients are the nudge signal and what their sign and magnitude mean. Why validation loss is the only honest measure of performance. How sigmoid squashes any score to a probability.

#### 2. [Neural Networks](notebooks/foundational/2_neural_net.ipynb) — Autograd and Backpropagation

Build an autograd engine around a single number, derive backpropagation from the chain rule, and train a hand-written MLP to classify two rings of points. Everything after this is the same four-step loop — **predict, score, assign blame, nudge** — applied to bigger models.

**You'll learn:** Why gradients accumulate with `+=`, how ReLU acts as a gradient gate, why learning rates decay, and what makes a layer "dead." The `Value` class here is PyTorch's tensor in miniature.

---

### Course 1 — Building a GPT from Scratch

Build a modern language model, component by component. Each notebook removes one abstraction and replaces it with working code.

#### 1. [Embeddings](notebooks/gpt/1_embeddings.ipynb) — The First Language Model

Scalars become tensors (verified to be the same autograd engine on the exact `(a+b)*b` example), and characters become learned vectors via an embedding table. An MLP trained on 8 characters of Shakespeare beats the bigram baseline, then hits a wall: its fixed context cannot scale.

**You'll learn:** Why raw token IDs can't be fed to networks, why one-hot encoding is correct but a dead end, the `one_hot(i) @ W == W[i]` identity, why untrained loss equals `-log(1/vocab_size)`, and the three structural reasons this MLP architecture fails to scale. **That wall is why attention exists.**

#### 2. [Transformer](notebooks/gpt/2_transformer.ipynb) — Attention from Scratch

Starting from a bigram baseline, invent attention one step at a time — uniform averaging, learned queries/keys/values, multiple heads, feed-forward layers, residual connections — and stack them into a full GPT that writes recognizable Shakespeare.

**You'll learn:** How attention computes data-dependent weights, why Q/K dot products measure relevance, what multiple heads buy you, why residuals and LayerNorm matter, and how the whole architecture composes into something that can model long-range dependencies.

#### 3. [Modern GPT](notebooks/gpt/3_modern_gpt.ipynb) — Six Improvements That Matter

The GPT from notebook 2 is the 2020 baseline. Every production model since — Llama, Mistral, Gemma — applies six improvements. This notebook adds them one at a time: RMSNorm, Flash Attention, Rotary Positional Embeddings (RoPE), Group-Query Attention (GQA), a KV cache for fast generation, and BPE tokenisation. All six run on CPU.

**You'll learn:** Why RMSNorm drops the centering step without losing quality. How Flash Attention achieves O(T) memory instead of O(T²). How RoPE encodes relative position via rotation of Q and K. How GQA reduces the KV cache by sharing key/value heads. How a KV cache turns O(T²) generation into O(T).

#### 4. [Fine-tuning and RL](notebooks/gpt/4_finetuning.ipynb) — From Predictor to Assistant

The GPT from notebooks 2–3 is a next-token predictor. To turn it into a useful assistant, it undergoes fine-tuning — but fine-tuning is not one thing. This notebook walks the full pipeline: Supervised Fine-Tuning (SFT) teaches the model the format of helpful responses through imitation; Reinforcement Learning from Human Feedback (RLHF) then aligns it to human preferences using reward signals rather than labels. Direct Preference Optimization (DPO) is shown as a simpler alternative that skips the reward model entirely.

**You'll learn:** Why pre-training alone produces a knowledgeable but unusable model. How SFT differs from RL — one is pattern matching, the other is learning from feedback. How a reward model is trained and used. Why DPO can replace PPO for preference alignment. How LoRA adapts massive models by training only low-rank residuals — on a MacBook.

---

### Course 2 — Recommendation Engines

Where Course 1 asked "what comes next in a sequence?", Course 2 asks "what does this user want to watch next?" The same embeddings and the same transformer appear — but the problem structure is different. Most of the data is missing. There are no clean input→label pairs. Users never told you what they like; you infer it from what they watched.

#### 1. [The Problem](notebooks/recommendation_engine/1_the_problem.ipynb) — A Matrix Full of Holes

The user-item matrix: rows are users, columns are items, most cells are empty. Establish this as matrix completion — a different problem from classification. Build three baselines from scratch: global mean, item mean, and user+item biases. Introduce RMSE and Precision@K as the two ways to measure a recommender — and show they sometimes disagree.

**You'll learn:** Why rating prediction and ranking quality are different objectives. How user and item biases alone beat naive baselines. Why evaluation on training data is meaningless for recommendation.

#### 2. [Collaborative Filtering](notebooks/recommendation_engine/2_collaborative_filtering.ipynb) — Users Like You

Find users similar to the target user, borrow their ratings. Build user-user and item-item collaborative filtering from scratch using cosine similarity. Show where it works and where it collapses: sparse users share too few items to establish reliable similarity.

**You'll learn:** How cosine similarity measures taste overlap. Why item-item CF is more stable than user-user CF. Why sparsity is the fundamental enemy of neighbourhood methods.

#### 3. [Matrix Factorization](notebooks/recommendation_engine/3_matrix_factorization.ipynb) — Learning Hidden Taste

Every user becomes a short vector. Every item becomes a short vector. The predicted rating is their dot product. Build this in pure Python with SGD — the same four-step training loop from ML Foundations. The embeddings are the same idea as Course 1 notebook 2, now applied to people and films instead of characters.

**You'll learn:** How latent factors emerge from ratings alone — nobody labels the genres, the model discovers them. Why L2 regularisation is essential with many embedding parameters. The cold start problem: new users and items have no vector.

#### 4. [Neural Collaborative Filtering](notebooks/recommendation_engine/4_neural_collaborative_filtering.ipynb) — Beyond Dot Products

Replace the dot product with an MLP. User and item embeddings are concatenated and passed through hidden layers, learning nonlinear interactions the dot product cannot express. Extend to the two-tower architecture — separate user and item networks whose embeddings combine only at the end — which is how YouTube and Pinterest serve recommendations at scale.

**You'll learn:** Why the dot product is a linear bottleneck. How the two-tower design enables precomputed item embeddings and fast nearest-neighbour retrieval. How adding content features (genre, metadata) partially solves cold start.

#### 5. [Sequential Recommendation](notebooks/recommendation_engine/5_sequential_recommendation.ipynb) — What You're In the Mood For

Treat each user's watch history as a sequence ordered by time. Predict the next item given the preceding ones. This is the language model problem from Course 1 with a different vocabulary: items instead of characters, watch history instead of text. Apply the transformer directly.

**You'll learn:** Why a bag-of-ratings misses mood shifts. How the same causal self-attention from Course 1 applies unchanged to item sequences. The difference between Hit@K (ranking) and RMSE (rating prediction) as evaluation targets.

#### 6. [Exploration vs Exploitation](notebooks/recommendation_engine/6_exploration_exploitation.ipynb) — The Bandit Problem

Every recommender faces a bootstrap problem: how do you learn what a new user likes before they've rated anything? How do you surface a new item that has no ratings yet? Build three strategies from scratch — ε-greedy, UCB, and Thompson Sampling — and race them on a simulated engagement environment.

**You'll learn:** Why pure exploitation creates filter bubbles and why pure exploration wastes recommendations. How UCB uses uncertainty as a bonus to direct exploration. How Thompson Sampling frames the problem as Bayesian inference. How the multi-armed bandit is RL with one step — and how it connects to RLHF.

## Lab

Standalone scripts that apply notebook concepts to real-world data — runnable from the command line, no notebook required. See [lab/README.md](lab/README.md) for details.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

```bash
uv sync
```

## Run

```bash
uv run jupyter notebook
```

## License

MIT — see [LICENSE](LICENSE).
