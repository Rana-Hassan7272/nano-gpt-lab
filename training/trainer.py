from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import mlflow
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import build_dataloader
from model.nanogpt import NanoGPT, NanoGPTConfig
from training.scheduler import WarmupCosineScheduler


def load_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format: {path}")
    return cfg


def iter_forever(loader: Iterable[tuple[torch.Tensor, torch.Tensor]]):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def estimate_val_loss(
    model: NanoGPT,
    val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    eval_batches: int,
) -> float:
    model.eval()
    losses = []
    it = iter_forever(val_loader)
    for _ in range(eval_batches):
        x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss, _ = model(x, y)
        if loss is None:
            raise RuntimeError("Validation loss is None")
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def build_model_config(cfg: Dict[str, Any], vocab_size: int) -> NanoGPTConfig:
    model_cfg = cfg.get("model", {})
    return NanoGPTConfig(
        vocab_size=vocab_size,
        context_len=int(model_cfg.get("context_length", 256)),
        d_model=int(model_cfg.get("d_model", 128)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        ffn_variant="standard",
        norm_type="rmsnorm",
        pos_encoding="rope",
        use_flash=torch.cuda.is_available(),
    )


def train(config_path: Path) -> None:
    cfg = load_config(config_path)

    train_cfg = cfg.get("training", {})
    context_len = int(cfg.get("model", {}).get("context_length", 256))
    batch_size = int(train_cfg.get("batch_size", 32))
    max_steps = int(train_cfg.get("max_steps", 5000))
    learning_rate = float(train_cfg.get("learning_rate", 3e-4))
    warmup_steps = int(train_cfg.get("warmup_steps", 100))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    weight_decay = float(train_cfg.get("weight_decay", 0.1))
    eval_interval = int(train_cfg.get("eval_interval", 100))
    eval_batches = int(train_cfg.get("eval_batches", 20))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 500))
    mlflow_log_interval = int(train_cfg.get("mlflow_log_interval", 100))

    data_dir = PROJECT_ROOT / "data"
    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    tokenizer_json = data_dir / "tokenizer.json"
    assert train_bin.exists(), f"Missing {train_bin}"
    assert val_bin.exists(), f"Missing {val_bin}"
    assert tokenizer_json.exists(), f"Missing {tokenizer_json}"

    tokenizer_meta = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab_size = int(tokenizer_meta["vocab_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    amp_dtype = torch.float16
    print(f"amp_enabled={amp_enabled} amp_dtype={amp_dtype if amp_enabled else 'n/a'}")

    model = NanoGPT(build_model_config(cfg, vocab_size=vocab_size)).to(device)
    optimizer = model.configure_optimizer(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device_type=("cuda" if device.type == "cuda" else "cpu"),
    )
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr_ratio=min_lr_ratio,
    )

    train_loader = build_dataloader(
        bin_path=train_bin,
        context_len=context_len,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = build_dataloader(
        bin_path=val_bin,
        context_len=context_len,
        batch_size=batch_size,
        shuffle=False,
    )
    train_iter = iter_forever(train_loader)

    ckpt_dir = PROJECT_ROOT / "results" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = str(mlflow_cfg.get("tracking_uri", str(PROJECT_ROOT / "results" / "mlruns")))
    experiment_name = str(mlflow_cfg.get("experiment_name", "nanogpt-lab"))
    run_name = str(mlflow_cfg.get("run_name", f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(
        {
            "context_len": context_len,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "grad_clip": grad_clip,
            "weight_decay": weight_decay,
            "min_lr_ratio": min_lr_ratio,
            "vocab_size": vocab_size,
            "amp_enabled": amp_enabled,
            "device": str(device),
        }
    )

    model.train()
    try:
        for step in range(1, max_steps + 1):
            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                _, loss, _ = model(x, y)
            if loss is None:
                raise RuntimeError("Training loss is None")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 20 == 0 or step == 1:
                print(
                    f"step={step:5d} "
                    f"train_loss={loss.item():.4f} "
                    f"grad_norm={float(grad_norm):.4f} "
                    f"lr={scheduler.current_lr():.6g}"
                )

            val_loss: float | None = None
            if step % eval_interval == 0 or step == max_steps:
                val_loss = estimate_val_loss(model, val_loader, device, eval_batches)
                print(f"[eval] step={step:5d} val_loss={val_loss:.4f} ppl={math.exp(val_loss):.2f}")

            if step % mlflow_log_interval == 0 or step == max_steps:
                if val_loss is None:
                    val_loss = estimate_val_loss(model, val_loader, device, eval_batches)
                mlflow.log_metrics(
                    {
                        "train_loss": float(loss.item()),
                        "val_loss": float(val_loss),
                        "perplexity": float(math.exp(val_loss)),
                        "learning_rate": float(scheduler.current_lr()),
                        "grad_norm": float(grad_norm),
                    },
                    step=step,
                )

            if step % checkpoint_interval == 0 or step == max_steps:
                ckpt_path = ckpt_dir / f"step_{step}.pt"
                model.save_checkpoint(str(ckpt_path), step=step, optimizer=optimizer, loss=loss.item())
    finally:
        mlflow.end_run()

    print("Training finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="NanoGPT trainer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_config.yaml"),
        help="YAML config path",
    )
    args = parser.parse_args()
    train(Path(args.config))


if __name__ == "__main__":
    main()
