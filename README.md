# Personal training of AI, building a transformer from scratch

A from-scratch deep dive into transformer architecture, built incrementally in PyTorch. No high-level shortcuts: no `nn.Transformer`, no `nn.MultiheadAttention`.

## What's in here

- **`mlp_mnist.py`**: A basic feedforward neural network (MLP), trained on MNIST. Covers the core training loop: forward pass, loss, backpropagation, optimizer step. ~97% test accuracy.
- **`attention.py`**: Self-attention and multi-head attention, implemented from scratch (Query/Key/Value, scaled dot-product attention, multi-head splitting/concatenation), plus a full transformer block (attention, feed-forward, residual connections, layer normalization).
- **`gpt.py`**: A tiny character-level GPT. Custom tokenizer, token and positional embeddings, stacked transformer blocks, a training loop, and autoregressive text generation, trained on Shakespeare's complete works.

## Why

Built as part of a self-directed deep dive into AI/ML fundamentals: understanding the actual mechanics behind modern language models (the same core architecture behind GPT and Claude), rather than only using high-level libraries.

## Results

- MLP: ~97% test accuracy on MNIST
- Tiny GPT: trains successfully (loss drops from ~4.4 to ~0.1), generates text with correct structural patterns (character-name dialogue formatting, word-like fragments), not yet coherent, expected given the small scale (4 transformer blocks, embed_dim=64)

## Stack

Python, PyTorch, torchvision, matplotlib

## Next steps

Scaling up (more layers, larger embeddings, subword tokenization), and moving into an original project, likely something in interpretability, fine-tuning, or evaluation.
