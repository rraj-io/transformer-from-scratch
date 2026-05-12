# Transformer From Scratch (PyTorch)

A clean educational implementation of the Transformer architecture built from scratch in **PyTorch**, designed to help understand the internal mechanics behind modern sequence models such as GPT, BERT, and encoder-decoder Transformers.

This project focuses on implementing the core Transformer building blocks manually rather than relying on high-level framework abstractions, making it ideal for learning, experimentation, and debugging.

---

## Project Goals

- Build a Transformer from first principles using PyTorch
- Understand how embeddings, positional encoding, and attention interact
- Implement core Transformer components manually
- Explore tokenization, masking, and sequence modeling
- Serve as a lightweight educational sandbox for future LLM experimentation

---

## Features

- Custom token embedding layer
- Positional encoding implementation
- Scaled dot-product attention
- Multi-head self-attention
- Causal masking for autoregressive tasks
- Feed-forward neural network blocks
- Layer normalization + residual connections
- Modular Transformer block design
- Test functions for validating architecture components

---

## Repository Structure

```bash
transformer-from-scratch/
│── transformer.py          # Main transformer architecture
│── attention.py            # Self-attention and scaled dot-product attention
│── embed.py                # Token embeddings + positional encoding
│── tests/                  # Unit tests / validation scripts
│── notebooks/              # Experimental notebooks and explanations
│── README.md               # Project documentation
```

---

## Core Concepts Covered

### 1. Embeddings
Tokens are converted into dense vector representations:

```python
embedding = nn.Embedding(vocab_size, d_model)
```

These vectors capture semantic meaning.

---

### 2. Positional Encoding
Since Transformers lack recurrence, positional encodings inject sequence order information.

**Formula:**

```math
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

### 3. Self-Attention
Computes relationships between tokens using:

```math
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
```

This allows each token to dynamically attend to relevant context.

---

### 4. Multi-Head Attention
Instead of one attention mechanism, multiple heads learn different contextual relationships simultaneously.

---

### 5. Masking
Causal masking prevents future token leakage during training for decoder-style generation.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rraj-io/transformer-from-scratch.git
cd transformer-from-scratch
```

Install dependencies:

```bash
pip install torch numpy matplotlib
```

---

## Usage

Run the Transformer:

```bash
python transformer.py
```

Run attention module tests:

```bash
python attention.py
```

---

## Example Workflow

```python
from transformer import Transformer

model = Transformer(
    vocab_size=5000,
    d_model=512,
    num_heads=8,
    num_layers=6
)

output = model(input_tokens)
print(output.shape)
```

---

## Learning Objectives

This project is particularly useful if you want to:

- Understand **Attention Is All You Need**
- Prepare for GPT/LLM development
- Build custom architectures
- Learn PyTorch deeply
- Debug Transformer internals

---

## Future Improvements

- Byte Pair Encoding (BPE) tokenizer
- Training loop on text corpus
- Decoder-only GPT mode
- Encoder-decoder translation mode
- Beam search / sampling
- Visualization of attention maps
- Mixed precision training

---

## References

- Vaswani et al. (2017) — *Attention Is All You Need*
- PyTorch Documentation
- Annotated Transformer
- Karpathy’s nanoGPT concepts

---

## Why This Project Matters

Building a Transformer from scratch is one of the best ways to move from “using AI models” to truly understanding how modern language models work internally.

This repository emphasizes:  
**clarity > abstraction**  
**education > optimization**

---

## Author

**Rohit Raj**  
Focused on AI, generative modeling, scientific ML, and engineering optimization.

GitHub: https://github.com/rraj-io

---

## License

MIT License (recommended)

---

## Acknowledgments

Inspired by the original Transformer paper and the open-source ML education community.
