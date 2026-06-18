# From Bigram to GPT-2-style Transformer

이 문서는 여섯 개 노트북의 변화가 왜 필요한지 연결해서 설명합니다.

## 1. Bigram

```text
current character -> next character
```

각 현재 문자에 대해 다음 문자 logits 한 행을 학습합니다. 문맥은 오직 마지막 문자 하나입니다.

## 2. MLP on names

```text
fixed context of 3 characters -> next character
```

문자를 embedding으로 바꾸고 여러 위치의 embedding을 이어 붙여 MLP에 입력합니다. Bigram보다 긴 문맥을 사용하지만 입력 길이는 고정됩니다.

## 3. MLP on Tiny Shakespeare

모델은 그대로 두고 dataset을 연속 문서 sliding window로 바꿉니다.

```text
16 characters -> one next character
```

긴 문서를 학습할 수 있지만 한 window에서 target 하나만 사용하며, `block_size`보다 오래된 정보는 완전히 사라집니다.

## 4. GPT-style target interface

```text
x = [t1, t2, ..., tT]
y = [t2, t3, ..., t(T+1)]
```

이제 모든 위치에서 다음 token을 예측합니다. 출력은 `(B, T, V)`입니다. 그러나 attention이 없다면 각 위치는 여전히 자기 token만 봅니다.

## 5. Single-head masked self-attention

각 위치가 현재 및 과거 위치의 value를 weighted sum으로 모읍니다.

```text
weights = softmax(mask(QK^T / sqrt(d)))
out = weights @ V
```

Causal mask 때문에 미래 token은 볼 수 없습니다. 처음으로 문맥을 선택적으로 결합합니다.

## 6. Tiny GPT

Single-head attention을 다음 구성으로 확장합니다.

- Multi-head attention
- Feed-forward network
- Residual connection
- Layer normalization
- Block stacking

각 head는 다른 관계를 학습할 수 있고, feed-forward network는 각 위치의 feature를 비선형적으로 가공합니다. 여러 block을 통과하며 문맥 표현이 반복적으로 정제됩니다.

## 이 저장소의 추가 개선

노트북 코드에서 교육용 프로젝트로 옮기며 다음을 보완했습니다.

1. Fused QKV projection
2. GELU activation
3. Input/output weight tying
4. Residual projection scaling
5. Separate train/validation continuous splits
6. AdamW decay/no-decay parameter groups
7. Warmup and cosine decay
8. Gradient clipping
9. Checkpoint and resume
10. Top-k and temperature generation
11. Prompt를 가짜 zero-padding 없이 처리
12. Causal behavior unit test

## 가장 중요한 개념 변화

```text
Bigram:
P(x[t+1] | x[t])
```

```text
Fixed-context MLP:
P(x[t+1] | x[t-k+1], ..., x[t])
```

```text
Causal Transformer:
P(x[t+1] | x[0], ..., x[t])
```

Transformer는 block 안의 각 위치가 이전 위치 전체를 직접 참조할 수 있고, attention weight를 통해 어떤 위치가 중요한지를 학습합니다.
