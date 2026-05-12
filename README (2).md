# Transformer From Scratch (PyTorch)

A clean, educational implementation of the original **encoder–decoder Transformer** architecture from *Attention Is All You Need* (Vaswani et al., 2017), built from the ground up in **PyTorch**.

Every component — embeddings, positional encoding, multi-head attention, residual connections, encoder/decoder stacks, and the projection head — is implemented manually, without relying on high-level framework abstractions like `nn.Transformer`. The goal is to make the internal mechanics of modern sequence models (GPT, BERT, T5, etc.) transparent and easy to follow.

---

## Project Goals

- Implement the Transformer architecture end-to-end, from first principles
- Understand how embeddings, positional encoding, and attention interact
- Build every block (attention, residual, normalization, feed-forward) manually
- Provide a working training loop on a real sequence-to-sequence task
- Serve as a foundation for future LLM and seq2seq experimentation

---

## Repository Structure

```
transformer-from-scratch/
├── model.py          # Full Transformer architecture (encoder + decoder)
├── dataset.py        # Dataset wrapper, tokenization, masking, padding
├── config.py         # Hyperparameters and training configuration
├── train.py          # Training loop, optimizer, loss, checkpointing
├── .gitignore
└── README.md
```

---

## Architecture Overview

`model.py` implements the complete encoder–decoder Transformer described in the paper:

| Component | Class | Role |
|---|---|---|
| Token embeddings | `InputEmbeddings` | Maps token IDs → dense vectors, scaled by √d_model |
| Positional encoding | `PositionalEncoding` | Adds sinusoidal position info to embeddings |
| Layer normalization | `LayerNormalization` | Stabilizes activations across features |
| Feed-forward block | `FeedForwardBlock` | Two-layer MLP with ReLU |
| Multi-head attention | `MultiHeadAttentionBlock` | Scaled dot-product attention across H heads |
| Residual connection | `ResidualConnection` | Pre-norm + residual + dropout wrapper |
| Encoder layer | `EncoderBlock` | Self-attention → feed-forward |
| Decoder layer | `DecoderBlock` | Masked self-attention → cross-attention → feed-forward |
| Output projection | `ProjectionLayer` | Maps decoder output → vocab logits (log-softmax) |
| Top-level model | `Transformer` | Wires encoder, decoder, embeddings, and projection together |
| Factory function | `build_transformer(...)` | Constructs a full model with Xavier-initialized weights |

### Default hyperparameters

| Parameter | Value |
|---|---|
| `d_model` | 512 |
| `N` (encoder/decoder layers) | 6 |
| `h` (attention heads) | 8 |
| `d_ff` (feed-forward dim) | 2048 |
| `dropout` | 0.1 |

These mirror the base configuration from the original paper.

---

## Core Concepts Covered

### 1. Embeddings

Token IDs are projected into a `d_model`-dimensional space and scaled by √d_model (as specified in the paper to balance the magnitude of embeddings and positional encodings):

```
embedding(x) * sqrt(d_model)
```

### 2. Positional Encoding

Sinusoidal positional encodings inject sequence-order information since the Transformer itself has no recurrence or convolution:

$$
PE(pos, 2i)   = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

### 3. Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

The √d_k scaling keeps gradients stable as the head dimension grows.

### 4. Multi-Head Attention

The model runs `h` parallel attention heads, each operating on a `d_k = d_model / h`-dimensional subspace, then concatenates and projects the result. This lets the model attend to different relationship types simultaneously.

### 5. Masking

- **Source mask** — hides padding tokens in the encoder input.
- **Target mask** — combines padding mask with a causal (look-ahead) mask so the decoder can only attend to past positions during training.

### 6. Encoder–Decoder Cross-Attention

In each decoder layer, the second attention block queries the decoder state against the **encoder's output**, allowing the decoder to condition its predictions on the full source sequence.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rraj-io/transformer-from-scratch.git
cd transformer-from-scratch
```

Install dependencies:

```bash
pip install torch tokenizers datasets tqdm
```

---

## Usage

### Configure

Edit `config.py` to set hyperparameters — sequence length, batch size, learning rate, number of epochs, model dimensions, and the dataset / language pair.

### Train

```bash
python train.py
```

This will:
1. Load and tokenize the dataset (`dataset.py`)
2. Build the Transformer via `build_transformer(...)` from `model.py`
3. Run the training loop with the configuration in `config.py`
4. Save checkpoints periodically

### Build the model directly

```python
from model import build_transformer

model = build_transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    src_seq_len=128,
    tgt_seq_len=128,
    d_model=512,
    N=6,
    h=8,
    dropout=0.1,
    d_ff=2048,
)
```

---

## Learning Objectives

This project is useful if you want to:

- Internalize the mechanics of *Attention Is All You Need*
- Move from "using" Transformers to *understanding* them
- Build intuition before diving into GPT/BERT/T5-style variants
- Debug attention, masking, and shape transformations hands-on
- Get comfortable with PyTorch at the `nn.Module` level

---

## Roadmap / Future Improvements

- [ ] Switch to Byte-Pair Encoding (BPE) tokenizer for better vocabulary coverage
- [ ] Add label smoothing and Noam learning-rate schedule
- [ ] Mixed-precision (AMP) training
- [ ] Greedy + beam-search inference script
- [ ] Attention-map visualizations
- [ ] Decoder-only (GPT-style) training mode
- [ ] BLEU evaluation on a translation benchmark
- [ ] Unit tests for each architectural component

---

## References

- Vaswani et al. (2017). *[Attention Is All You Need](https://arxiv.org/abs/1706.03762).*
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)

---

## Why This Project Matters

Reading the Transformer paper is one thing. Implementing it — wiring up the Q/K/V projections, getting the masks right, debugging the cross-attention shapes — is what makes the architecture *click*.

This repository deliberately favors **clarity over abstraction** and **education over optimization**. It's meant to be read.

---

## Author

**Rohit Raj**
Working at the intersection of AI, generative modeling, scientific ML, and engineering optimization.

GitHub: [@rraj-io](https://github.com/rraj-io)

---

## License

MIT License

---

## Acknowledgments

Inspired by the original Transformer paper and the broader open-source ML education community — particularly the Annotated Transformer and nanoGPT projects.
