from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GPTConfig:
    """Architecture configuration for the decoder-only Transformer."""

    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_layer <= 0 or self.n_head <= 0 or self.n_embd <= 0:
            raise ValueError("n_layer, n_head, and n_embd must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
