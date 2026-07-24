
# Building Transformers from First Principles

## Project purpose

This is an **educational project** — a personal learning journey through the mathematics and engineering of modern language models. Every component (autograd, embeddings, attention, tokenization, fine-tuning) is built from scratch with detailed explanations.

The goal is **understanding**, not production code. Clarity beats cleverness. Explicit beats implicit. A 10-line pedagogical implementation beats a 1-line library call.

## Structure

```
notebooks/
├── neural_net.ipynb      # 1. Autograd engine, backprop, MLP (pure Python)
├── embeddings.ipynb      # 2. First language model, PyTorch tensors
├── transformer.ipynb     # 3. Attention mechanism, full GPT
├── tokenizer.ipynb       # 4. Byte-Pair Encoding from scratch
└── finetuning.ipynb      # 5. LoRA, SFT, DPO on TinyLlama

input.txt                 # tiny Shakespeare corpus (used by notebooks 2-3)
```

**The notebooks form one continuous narrative.** Each builds on the last, replacing one abstraction with something concrete. Read them in order.

## Core principles

### 1. Everything from scratch, explained

- Build components from first principles, not library calls
- The neural_net notebook uses **pure Python** (no NumPy, no PyTorch) to implement autograd
- Later notebooks use PyTorch, but derive and verify that it's the same math
- When introducing a new concept, show the identity or derivation that motivates it (e.g., `one_hot(i) @ W == W[i]`)

### 2. Pedagogical style

The notebooks follow a consistent teaching pattern:

- **Step-by-step structure** — numbered sections that build progressively
- **Explain the job, then the implementation** — "Job: turn symbols into vectors" before showing embedding lookup
- **Concrete examples with toy sizes** — trace through calculations with small numbers before showing real shapes
- **Verification and sanity checks** — untrained loss should equal `-log(1/vocab_size)`, gradients should match hand calculations
- **"Your turn" sections** — exercises to cement understanding

### 3. Writing style

From the notebooks:
- Short sentences. Direct language.
- Bold for key terms on first use (**embedding**, **softmax**, **attention**)
- Code comments explain *why*, not *what* — only when non-obvious
- Tables for comparisons (e.g., ring classifier vs language model architecture)
- Markdown cells lead with the concept, code cells demonstrate it

Avoid:
- Unnecessary jargon without definition
- Hand-waving over important details
- Emojis (unless explicitly requested)
- Claiming something is "obvious" or "trivial"

## Working with this project

### When adding or modifying notebooks

- **Maintain the narrative arc.** Each notebook should have:
  - Clear learning objectives stated upfront
  - Progressive reveal (simple → complex)
  - Connection to previous notebooks
  - A "what we learned" summary at the end
- **Verify all claims.** If you state a formula or identity, show it's true with code
- **Keep the same training data.** Notebooks 2-3 use `input.txt` (tiny Shakespeare) so losses are directly comparable
- **Reproducibility.** Use fixed random seeds (`torch.manual_seed(1337)` is the convention from Karpathy)

### When explaining concepts

- **Start with the problem**, not the solution (e.g., "Why can't raw token IDs be fed to networks?" before introducing embeddings)
- **Use consistent shapes.** The `(B, T, C)` convention (batch, time, channels) runs through all notebooks
- **Show the same example across multiple implementations** to prove equivalence (like `(a+b)*b` in Value vs PyTorch tensor)
- **Count parameters explicitly.** Show where 337 or 29,915 comes from with the "one weight per input + bias" rule

### Code conventions

- **Explicit over implicit.** Write out the four-step training loop every time (predict, score, zero grad, backward, update) rather than hiding it
- **No premature abstractions.** If a notebook manages parameters manually (`p.data -= lr * p.grad`), keep doing that — don't introduce `nn.Module` until it's pedagogically motivated
- **Meaningful variable names:**
  - `C` = embedding table (established convention from Bengio 2003)
  - `W1, W2, b1, b2` = layer weights and biases
  - `Xtr, Ytr, Xva, Yva` = training/validation inputs/targets
  - `emb, embcat, h, logits` = embedding, concatenated, hidden, output scores
- **Small exploratory cells.** Each cell should demonstrate one idea

### Testing and verification

Every notebook should have:
- **Sanity checks** — untrained loss near theoretical baseline, gradients match expected values
- **Shape annotations** — comments like `# (B, 8, 10)` on every intermediate tensor
- **Reproducible outputs** — fixed seeds, deterministic operations

## Common tasks

### Adding a new notebook

1. Follow the numbered sequence and narrative structure
2. Start with a clear problem statement (what limitation of the previous notebook are we fixing?)
3. Include a "Your turn" section at the end with exercises
4. Update `README.md` to describe the notebook in the journey section
5. Use the existing notebooks as templates for style and structure

### Explaining a concept to the user

- Check which notebooks they've completed (ask if unclear)
- Explain at the appropriate level (pure Python if on notebook 1, PyTorch if beyond)
- Use examples from the notebooks they've seen
- When they summarize from memory, verify against the actual notebook content (like we just did)

### When asked to review or debug

- Read the relevant notebook first to understand context
- Check that shapes match the `(B, T, C)` conventions
- Verify parameter counts with the "one weight per input + bias" rule
- Make sure loss curves start near the theoretical baseline

## What NOT to do

- Don't create documentation files (*.md) unless explicitly requested
- Don't add unnecessary dependencies — keep the environment minimal
- Don't replace from-scratch implementations with library calls (defeats the purpose)
- Don't add type hints or excessive error handling — readability over production rigor
- Don't assume the user knows something because "everyone knows that" — this is a learning project

## File locations

- **Notebooks:** `notebooks/*.ipynb`
- **Training data:** `input.txt` (tiny Shakespeare, ~1.1MB, 65 unique chars)
- **Dependencies:** `pyproject.toml` (managed with `uv`)
- **Project docs:** `README.md`, `CONTRIBUTING.md`, `LICENSE`, `CODE_OF_CONDUCT.md`

## Memory and context

This project uses Claude's file-based memory system. Key things worth remembering:

- **User background** — learning transformers from scratch, working through the notebooks sequentially
- **Learning style** — wants to understand deeply, asks for verification of understanding
- **Current progress** — has studied neural_net and embeddings notebooks, can quiz them on those concepts

When the user asks questions about concepts:
1. Check which notebook covers it
2. Read that notebook if not already in context
3. Answer based on what that notebook actually teaches, not from general knowledge
4. Quiz them to verify understanding (they explicitly requested this pattern)

## References and inspiration

- Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) series
- Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- Bengio et al. 2003, "A Neural Probabilistic Language Model"
- The "attention is all you need" paper (Vaswani et al. 2017)

This project translates those ideas into a self-contained learning path with maximum pedagogical clarity.
