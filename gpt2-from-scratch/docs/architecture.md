# Architecture Notes

## 1. Input and target

For a token sequence of length `T`:

```text
x = [t0, t1, ..., t(T-1)]
y = [t1, t2, ..., tT]
```

The model predicts every target position in parallel during training. Causal masking ensures that position `i` cannot inspect positions after `i`.

## 2. Embeddings

The model adds token and learned absolute position embeddings:

```text
h[b, t] = token_embedding[x[b, t]] + position_embedding[t]
```

Both have channel dimension `n_embd`.

## 3. Multi-head causal self-attention

A fused linear layer creates `Q`, `K`, and `V`:

```text
(B, T, C) -> (B, T, 3C)
```

They are split into heads:

```text
(B, T, C) -> (B, n_head, T, head_size)
```

where:

```text
head_size = C / n_head
```

Scores are scaled to avoid softmax saturation:

```text
scores = QK^T / sqrt(head_size)
```

A lower-triangular mask sets future scores to negative infinity before softmax.

## 4. Transformer block

This repository uses pre-LayerNorm blocks:

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

The MLP expands from `C` to `4C`, applies GELU, and projects back to `C`.

## 5. Language-model head

The final normalized hidden state is projected to vocabulary logits:

```text
(B, T, C) -> (B, T, V)
```

The LM-head weight is tied to the token-embedding weight.

## 6. Training loss

The token-level cross-entropy is computed after flattening batch and time:

```text
logits:  (B*T, V)
targets: (B*T)
```

## 7. Autoregressive generation

At every generation step:

1. Keep at most the latest `block_size` tokens.
2. Run the model.
3. Select logits at the final sequence position.
4. Apply temperature and optional top-k filtering.
5. Sample one token.
6. Append it to the sequence.

The implementation does not add fake left padding to short prompts.
