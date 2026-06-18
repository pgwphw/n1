import torch

from gpt2_scratch.config import GPTConfig
from gpt2_scratch.model import GPT


def tiny_model() -> GPT:
    config = GPTConfig(
        vocab_size=17,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
    )
    return GPT(config)


def test_forward_shape_and_loss() -> None:
    model = tiny_model()
    x = torch.randint(0, 17, (4, 8))
    y = torch.randint(0, 17, (4, 8))
    logits, loss = model(x, y)
    assert logits.shape == (4, 8, 17)
    assert loss is not None
    assert loss.ndim == 0


def test_causal_mask_blocks_future_information() -> None:
    model = tiny_model().eval()
    x1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    x2 = x1.clone()
    x2[:, 4:] = torch.tensor([[9, 10, 11, 12]])

    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)

    # Altering future tokens cannot change logits at positions 0..3.
    assert torch.allclose(logits1[:, :4], logits2[:, :4], atol=1e-6)


def test_generate_preserves_prompt_and_length() -> None:
    model = tiny_model().eval()
    prompt = torch.tensor([[1, 2, 3]])
    output = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=5)
    assert output.shape == (1, 8)
    assert torch.equal(output[:, :3], prompt)


def test_tied_token_and_output_weights() -> None:
    model = tiny_model()
    assert model.transformer["wte"].weight.data_ptr() == model.lm_head.weight.data_ptr()
