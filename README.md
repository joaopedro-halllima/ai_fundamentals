# Learning AI engineering

A from-scratch deep dive into transformer architecture, built incrementally in PyTorch. No high-level shortcuts: no `nn.Transformer`, no `nn.MultiheadAttention`.

## What's in here as of now

- **`mlp_mnist.py`**: A basic feedforward neural network (MLP), trained on MNIST. Covers the core training loop: forward pass, loss, backpropagation, optimizer step. ~97% test accuracy.
- **`attention.py`**: Self-attention and multi-head attention, implemented from scratch (Query/Key/Value, scaled dot-product attention, multi-head splitting/concatenation), plus a full transformer block (attention, feed-forward, residual connections, layer normalization).
- **`gpt.py`**: A tiny character-level GPT. Custom tokenizer, token and positional embeddings, stacked transformer blocks, a training loop, and autoregressive text generation, trained on Shakespeare's complete works.

### Simpler Ones

- **`context_window`**
- **`tokenizer`**

## Why

Built as part of a self-directed deep dive into AI/ML fundamentals: understanding the actual mechanics behind modern language models (the same core architecture behind GPT and Claude), rather than only using high-level libraries.
