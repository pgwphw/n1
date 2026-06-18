from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .config import GPTConfig
from .data import CharTokenizer, build_dataloaders, download_tiny_shakespeare
from .model import GPT
from .utils import (
    build_optimizer,
    cosine_learning_rate,
    load_json,
    resolve_device,
    set_seed,
)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    step: int,
    best_val_loss: float,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPT-2-style character LM")
    parser.add_argument("--config", default="configs/tiny_shakespeare.json")
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_json(args.config)
    train_config = settings["train"]
    set_seed(int(train_config["seed"]))
    device = resolve_device(args.device)

    data_path = download_tiny_shakespeare(args.data)
    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        **settings["model"],
    )
    train_loader, val_loader = build_dataloaders(
        encoded=encoded,
        block_size=model_config.block_size,
        batch_size=int(train_config["batch_size"]),
        train_split=float(train_config["train_split"]),
        num_workers=int(train_config["num_workers"]),
    )

    model = GPT(model_config).to(device)
    optimizer = build_optimizer(
        model,
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        beta1=float(train_config["beta1"]),
        beta2=float(train_config["beta2"]),
    )

    checkpoint_path = Path(train_config["checkpoint_path"])
    start_step = 0
    best_val_loss = float("inf")
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])
        print(f"Resumed from step {start_step}")

    print(f"device: {device}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"parameters: {model.num_parameters():,}")

    model.train()
    train_iterator = iter(train_loader)
    max_steps = int(train_config["max_steps"])
    eval_interval = int(train_config["eval_interval"])

    progress = tqdm(range(start_step, max_steps), desc="training")
    for step in progress:
        lr = cosine_learning_rate(
            step=step,
            max_steps=max_steps,
            warmup_steps=int(train_config["warmup_steps"]),
            learning_rate=float(train_config["learning_rate"]),
            min_learning_rate=float(train_config["min_learning_rate"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["grad_clip"]))
        optimizer.step()

        progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        should_evaluate = step % eval_interval == 0 or step == max_steps - 1
        if should_evaluate:
            val_loss = estimate_loss(
                model,
                val_loader,
                device,
                max_batches=int(train_config["eval_batches"]),
            )
            print(f"\nstep {step:5d} | train {loss.item():.4f} | val {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    tokenizer,
                    step,
                    best_val_loss,
                )
                print(f"saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
