from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import build_dataloader
from model.lora import LoRAConfig, apply_lora, print_trainable_params, save_lora
from model.nanogpt import NanoGPT


def iter_forever(loader: Iterable[tuple[torch.Tensor, torch.Tensor]]):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def estimate_val_loss(
    model: NanoGPT,
    val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    n_batches: int,
) -> float:
    model.eval()
    losses = []
    it = iter_forever(val_loader)
    for _ in range(n_batches):
        x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss, _ = model(x, y)
        if loss is None:
            raise RuntimeError("Validation loss is None")
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    print(f"device={device} amp_enabled={amp_enabled}")

    base_ckpt = Path(args.base_checkpoint)
    if not base_ckpt.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt}")

    train_bin = Path(args.train_bin)
    val_bin = Path(args.val_bin)
    if not train_bin.exists():
        raise FileNotFoundError(f"Train bin not found: {train_bin}")
    if not val_bin.exists():
        raise FileNotFoundError(f"Val bin not found: {val_bin}")

    model = NanoGPT.from_checkpoint(str(base_ckpt), device=str(device)).to(device)
    lora_cfg = LoRAConfig(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=set(args.target_modules),
        bias=args.bias,
    )
    apply_lora(model, lora_cfg)
    stats = print_trainable_params(model, verbose=False)
    print(
        f"LoRA trainable params: {stats['trainable']:,}/{stats['total']:,} "
        f"({100.0*stats['trainable']/stats['total']:.4f}%)"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    context_len = model.config.context_len
    train_loader = build_dataloader(
        bin_path=train_bin,
        context_len=context_len,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = build_dataloader(
        bin_path=val_bin,
        context_len=context_len,
        batch_size=args.batch_size,
        shuffle=False,
    )
    train_iter = iter_forever(train_loader)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    last_loss = float("nan")
    for step in range(1, args.max_steps + 1):
        x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
            _, loss, _ = model(x, y)
        if loss is None:
            raise RuntimeError("Training loss is None")
        last_loss = float(loss.item())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0 or step == 1:
            print(
                f"step={step:4d} train_loss={last_loss:.4f} "
                f"grad_norm={float(grad_norm):.4f}"
            )
        if step % args.eval_interval == 0 or step == args.max_steps:
            val_loss = estimate_val_loss(model, val_loader, device, args.eval_batches)
            print(f"[eval] step={step:4d} val_loss={val_loss:.4f} ppl={math.exp(val_loss):.2f}")

    adapter_path = out_dir / args.adapter_name
    save_lora(model, lora_cfg, str(adapter_path), step=args.max_steps, loss=last_loss)
    print(f"Saved adapter: {adapter_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning trainer")
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--train_bin", type=str, default="data/fine_tune_train.bin")
    parser.add_argument("--val_bin", type=str, default="data/fine_tune_val.bin")
    parser.add_argument("--out_dir", type=str, default="results/lora")
    parser.add_argument("--adapter_name", type=str, default="lora_poetry_rank4.pt")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=20)

    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["W_q", "W_k", "W_v", "W_o"],
        help="Attention projection modules to wrap with LoRA",
    )
    parser.add_argument("--bias", type=str, default="none", choices=["none", "lora", "all"])
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

