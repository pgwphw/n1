# GPT-2 from Scratch: Tiny Shakespeare

PyTorch만 사용해 **문자 단위 GPT-2 스타일 언어모델**을 처음부터 구현하고 학습하는 교육용 저장소입니다.

이 저장소는 다음 학습 경로를 하나의 실행 가능한 프로젝트로 정리합니다.

1. Bigram language model
2. Embedding + MLP on `names.txt`
3. Fixed-context MLP on Tiny Shakespeare
4. Shifted sequence target: `x[t] -> y[t+1]`
5. Single-head masked self-attention
6. Multi-head attention, MLP, residual connection, LayerNorm, block stacking
7. GPT-2 스타일 decoder-only Transformer

> 이 프로젝트는 OpenAI GPT-2의 대규모 사전학습 모델이나 공식 가중치를 복제하는 프로젝트가 아닙니다. GPT-2의 핵심 구조를 작은 문자 단위 모델로 재현해 원리를 학습하는 구현입니다.

## 핵심 구현

- Character-level tokenizer
- GPT-style shifted sequence dataset
- Learned token embedding
- Learned absolute positional embedding
- Causal multi-head self-attention
- Pre-LayerNorm Transformer block
- Position-wise feed-forward network
- Residual connections
- GELU activation
- Dropout
- Input/output embedding weight tying
- GPT-2식 residual projection 초기화 스케일링
- AdamW parameter grouping
- Warmup + cosine learning-rate decay
- Gradient clipping
- Train/validation split
- Checkpoint 저장 및 재개
- Temperature / top-k sampling
- Causal-mask unit test

## 모델이 학습하는 문제

길이 `T`의 입력 sequence가 있을 때 target을 한 칸 왼쪽으로 이동시킵니다.

```text
x = [t1, t2, t3, ..., tT]
y = [t2, t3, t4, ..., t(T+1)]
```

모델 출력은 다음 shape을 가집니다.

```text
input:   (B, T)
logits:  (B, T, V)
targets: (B, T)
```

각 위치 `t`는 causal mask 때문에 자기 자신과 이전 위치만 볼 수 있습니다.

```text
position 0 -> [0]
position 1 -> [0, 1]
position 2 -> [0, 1, 2]
...
```

따라서 모델은 다음 조건부 확률을 학습합니다.

```text
P(x[t+1] | x[0], x[1], ..., x[t])
```

## 구조

```text
Token IDs
   │
   ├─ Token Embedding
   └─ Position Embedding
           │
           ▼
      Embedding Sum
           │
           ▼
  ┌───────────────────────┐
  │ Pre-LN Transformer    │ × N
  │                       │
  │ x + Causal MHA(LN(x)) │
  │ x + MLP(LN(x))        │
  └───────────────────────┘
           │
           ▼
       Final LayerNorm
           │
           ▼
        LM Head
           │
           ▼
      Next-token logits
```

## 저장소 구조

```text
gpt2-from-scratch/
├── configs/
│   └── tiny_shakespeare.json
├── data/
├── checkpoints/
├── docs/
│   ├── architecture.md
│   └── learning_path.md
├── notebooks/
│   └── gpt2_from_scratch_walkthrough.ipynb
├── scripts/
│   └── download_data.py
├── src/gpt2_scratch/
│   ├── config.py
│   ├── data.py
│   ├── generate.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── Makefile
├── pyproject.toml
└── README.md
```

## 설치

Python 3.10 이상을 권장합니다.

```bash
git clone <YOUR_REPOSITORY_URL>
cd gpt2-from-scratch
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

패키지와 개발 의존성을 설치합니다.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 데이터 다운로드

학습 명령을 실행하면 `data/input.txt`가 없을 때 Tiny Shakespeare를 자동으로 받습니다.

직접 다운로드하려면:

```bash
python scripts/download_data.py
```

또는:

```bash
make data
```

## 학습

GPU 또는 Apple Silicon 가속 환경의 기본 설정:

```bash
python -m gpt2_scratch.train \
  --config configs/tiny_shakespeare.json \
  --device auto
```

CPU에서 구조를 빠르게 확인하는 작은 설정:

```bash
python -m gpt2_scratch.train \
  --config configs/cpu_demo.json \
  --device cpu
```

설치된 CLI를 사용할 수도 있습니다.

```bash
gpt2-train --config configs/tiny_shakespeare.json
```

학습 중 validation loss가 개선되면 다음 위치에 checkpoint가 저장됩니다.

```text
checkpoints/tiny_shakespeare.pt
```

학습을 이어서 진행하려면:

```bash
python -m gpt2_scratch.train \
  --config configs/tiny_shakespeare.json \
  --resume
```

## 텍스트 생성

```bash
python -m gpt2_scratch.generate \
  --checkpoint checkpoints/tiny_shakespeare.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 500 \
  --temperature 0.8 \
  --top-k 40
```

설치된 CLI:

```bash
gpt2-generate --prompt "ROMEO:"
```

### Sampling 파라미터

- `temperature < 1`: 더 보수적이고 반복적인 출력
- `temperature > 1`: 더 다양하지만 불안정할 수 있는 출력
- `top-k`: 매 단계에서 확률이 높은 상위 `k`개 token만 sampling

생성 코드는 가짜 zero-padding을 앞에 붙이지 않습니다. 실제 prompt를 그대로 입력하고, 길이가 `block_size`를 넘으면 가장 최근 token만 사용합니다.

```python
idx_cond = idx[:, -block_size:]
```

## 설정 변경

`configs/tiny_shakespeare.json`의 주요 모델 설정:

```json
{
  "block_size": 128,
  "n_layer": 4,
  "n_head": 4,
  "n_embd": 128,
  "dropout": 0.1
}
```

조건:

```text
n_embd % n_head == 0
```

각 head의 dimension은 다음과 같습니다.

```text
head_size = n_embd / n_head
```

모델을 크게 만들면 품질이 좋아질 수 있지만, attention 계산량과 메모리 사용량이 빠르게 증가합니다.

```text
attention memory/time ≈ O(T²)
```

## 테스트

```bash
pytest -q
```

테스트 항목:

- Dataset target이 정확히 한 token shift되었는지
- 출력 shape이 `(B, T, V)`인지
- loss가 scalar인지
- 미래 token 변경이 과거 위치 logits에 영향을 주지 않는지
- prompt가 생성 결과 앞부분에 보존되는지
- token embedding과 LM head가 weight tying 되었는지

## 핵심 코드 읽는 순서

1. `src/gpt2_scratch/data.py`
2. `src/gpt2_scratch/model.py::CausalSelfAttention`
3. `src/gpt2_scratch/model.py::Block`
4. `src/gpt2_scratch/model.py::GPT`
5. `src/gpt2_scratch/train.py`
6. `src/gpt2_scratch/generate.py`

## Attention 계산

입력 표현을 query, key, value로 투영합니다.

```text
Q, K, V: (B, n_head, T, head_size)
```

Attention score:

```text
scores = Q @ Kᵀ / sqrt(head_size)
```

Causal mask 적용 후:

```text
weights = softmax(mask(scores))
```

Value 결합:

```text
output = weights @ V
```

각 위치는 미래 token에는 확률 0을 부여하고, 과거와 현재 위치 중 어떤 정보를 얼마나 가져올지 학습합니다.

## Notebook 1~6과 이 저장소의 연결

| 단계 | 문맥 처리 | target | 모델 |
|---|---|---|---|
| 1 | 현재 문자 1개 | 다음 문자 1개 | Bigram |
| 2 | 최근 3문자 | 다음 문자 1개 | Embedding + MLP |
| 3 | 최근 16문자 | 다음 문자 1개 | Fixed-context MLP |
| 4 | 각 위치의 현재 문자 | shifted sequence | Minimal sequence LM |
| 5 | 과거 전체를 선택적으로 참조 | shifted sequence | Single-head attention |
| 6 | 여러 attention 관계 + 깊은 변환 | shifted sequence | Tiny GPT |

자세한 비교는 [`docs/learning_path.md`](docs/learning_path.md)를 참고하세요.

## GPT-2와 비슷한 점과 다른 점

### 비슷한 점

- Decoder-only Transformer
- Causal self-attention
- Learned positional embedding
- Pre-LayerNorm residual block
- GELU MLP
- Weight tying
- Autoregressive next-token generation

### 다른 점

- Character tokenizer를 사용함
- 모델 규모가 매우 작음
- Tiny Shakespeare만 학습함
- 분산 학습, mixed precision, 대규모 corpus pipeline이 없음
- 공식 GPT-2 checkpoint와 호환되지 않음

## GitHub에 올리기

이 폴더는 이미 Git 저장소로 초기화할 수 있는 형태입니다.

```bash
git init
git add .
git commit -m "Implement GPT-2-style language model from scratch"
git branch -M main
git remote add origin https://github.com/<USERNAME>/gpt2-from-scratch.git
git push -u origin main
```

GitHub CLI가 설치되어 있다면:

```bash
gh repo create gpt2-from-scratch --public --source=. --remote=origin --push
```

## License

MIT License
