# Performer Language Model

A PyTorch implementation of [Performer](https://arxiv.org/abs/2009.14794) for language modeling alongside "[Attention is All You Need](https://arxiv.org/abs/1706.03762)". Performer
computes fast self-attention (linear insteaf of quadratic) by approximating the softmax kernel.

In this code, I've used a hybrid approach by utilizing Fast Attention Via positive Orthogonal Random features approach (FAVOR+) in the encoder, and using basic (softmax-based) masked self attention & cross attention in the decoder.

## Includes

- Basic self attention module from "Attention is All You Need"
- FAVOR+ attention module from "Performer"
- BPE-tokenizer that trains on WikiText-103


## Quick Start

Train a model:
```bash
python examples/example.py
```

Generate text:
```bash
python -m performer.inference --prompt="Once upon a time" --temperature=0.7
```

## Advanced Usage

### Core Parameters

- `--model_path`: Path to model weights (default: `data/trained_model.pt`)
- `--tokenizer_path`: Path to tokenizer file (default: `data/tokenizer.json`)
- `--prompt`: Starting text for generation
- `--temperature`: Controls randomness (lower = more predictable)
- `--max_length`: Maximum tokens to generate
- `--top_k`: Limits vocabulary to top k options per step
- `--top_p`: Uses nucleus sampling with probability p