from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(path: str | Path) -> Path:
    """Download Tiny Shakespeare only when the file is missing."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading Tiny Shakespeare to {destination} ...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, destination)
    return destination


class CharTokenizer:
    """Deterministic character tokenizer built from a training corpus."""

    def __init__(self, chars: Iterable[str]) -> None:
        self.chars = sorted(set(chars))
        if not self.chars:
            raise ValueError("Tokenizer vocabulary cannot be empty")
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(text)

    @classmethod
    def from_dict(cls, payload: dict) -> "CharTokenizer":
        return cls(payload["chars"])

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            display = ", ".join(repr(ch) for ch in unknown)
            raise ValueError(f"Unknown character(s): {display}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(idx)] for idx in ids)

    def to_dict(self) -> dict[str, list[str]]:
        return {"chars": self.chars}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class NextTokenDataset(Dataset):
    """Return x[t:t+T] and the one-token-shifted target sequence."""

    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        if data.ndim != 1:
            raise ValueError("data must be a one-dimensional token tensor")
        if len(data) <= block_size:
            raise ValueError("data must be longer than block_size")
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def build_dataloaders(
    encoded: torch.Tensor,
    block_size: int,
    batch_size: int,
    train_split: float = 0.9,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Split the continuous text first, then create non-leaking windows."""

    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")

    split_index = int(len(encoded) * train_split)
    train_data = encoded[:split_index]
    val_data = encoded[split_index:]

    train_dataset = NextTokenDataset(train_data, block_size)
    val_dataset = NextTokenDataset(val_data, block_size)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader
