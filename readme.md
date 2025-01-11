# Performer Language Model

A PyTorch implementation of [Performer](https://arxiv.org/abs/2009.14794) for language modeling. Performer wcomputes fast self-attention (linear insteaf of quadratic) by approximating the softmax kernel.

In this code, I've used a hybrid approach by utilizing Fast Attention Via positive Orthogonal Random features approach (FAVOR+) in the encoder, and using basic (softmax-based) masked self attention & cross attention in the decoder.

## Includes

- Basic self attention module from "Attention is All You Need"
- FAVOR+ attention module from "Performer"
- BPE-tokenizer that trains on WikiText-103