import torch

from gpt2_scratch.data import CharTokenizer, NextTokenDataset


def test_shifted_targets() -> None:
    tokenizer = CharTokenizer.from_text("abcdef")
    encoded = torch.tensor(tokenizer.encode("abcdef"), dtype=torch.long)
    dataset = NextTokenDataset(encoded, block_size=3)
    x, y = dataset[1]
    assert tokenizer.decode(x.tolist()) == "bcd"
    assert tokenizer.decode(y.tolist()) == "cde"


def test_unknown_character_raises() -> None:
    tokenizer = CharTokenizer.from_text("abc")
    try:
        tokenizer.encode("abd")
    except ValueError as error:
        assert "Unknown character" in str(error)
    else:
        raise AssertionError("Expected ValueError")
