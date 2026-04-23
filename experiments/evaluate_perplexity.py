from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TokenChunkDataset
from model.lora import LoRAConfig, apply_lora, load_lora
from model.nanogpt import NanoGPT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_eval_indices(
    dataset_len: int,
    batch_size: int,
    eval_batches: int,
    seed: int,
) -> List[torch.Tensor]:
    total_needed = batch_size * eval_batches
    g = torch.Generator().manual_seed(seed)
    # If dataset is smaller than requested samples, sample with replacement.
    if dataset_len >= total_needed:
        flat = torch.randperm(dataset_len, generator=g)[:total_needed]
    else:
        flat = torch.randint(0, dataset_len, (total_needed,), generator=g)
    return [flat[i * batch_size : (i + 1) * batch_size] for i in range(eval_batches)]


@torch.no_grad()
def estimate_val_loss(
    model: NanoGPT,
    dataset: TokenChunkDataset,
    batch_indices: List[torch.Tensor],
    device: torch.device,
) -> float:
    model.eval()
    losses: List[float] = []
    for idx_batch in batch_indices:
        xs, ys = [], []
        for i in idx_batch.tolist():
            x, y = dataset[int(i)]
            xs.append(x)
            ys.append(y)
        x_batch = torch.stack(xs, dim=0).to(device, non_blocking=True)
        y_batch = torch.stack(ys, dim=0).to(device, non_blocking=True)
        _, loss, _ = model(x_batch, y_batch)
        if loss is None:
            raise RuntimeError("Validation loss is None")
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


def run_multi_seed_eval(
    model: NanoGPT,
    dataset: TokenChunkDataset,
    seeds: List[int],
    batch_size: int,
    eval_batches: int,
    device: torch.device,
) -> Dict[str, Any]:
    per_seed: List[Dict[str, float]] = []
    for seed in seeds:
        set_seed(seed)
        batches = build_eval_indices(
            dataset_len=len(dataset),
            batch_size=batch_size,
            eval_batches=eval_batches,
            seed=seed,
        )
        val_loss = estimate_val_loss(model=model, dataset=dataset, batch_indices=batches, device=device)
        ppl = float(math.exp(val_loss))
        per_seed.append({"seed": float(seed), "val_loss": val_loss, "perplexity": ppl})

    losses = [x["val_loss"] for x in per_seed]
    ppls = [x["perplexity"] for x in per_seed]
    return {
        "per_seed": per_seed,
        "aggregate": {
            "mean_val_loss": float(statistics.mean(losses)),
            "std_val_loss": float(statistics.pstdev(losses)) if len(losses) > 1 else 0.0,
            "mean_perplexity": float(statistics.mean(ppls)),
            "std_perplexity": float(statistics.pstdev(ppls)) if len(ppls) > 1 else 0.0,
        },
    }


def load_model(
    checkpoint_path: Path,
    device_str: str,
    adapter_path: Optional[Path],
    rank: int,
    alpha: float,
    target_modules: List[str],
    bias: str,
) -> NanoGPT:
    model = NanoGPT.from_checkpoint(str(checkpoint_path), device=device_str).to(device_str)
    if adapter_path is not None:
        lora_cfg = LoRAConfig(
            rank=rank,
            alpha=alpha,
            dropout=0.0,
            target_modules=set(target_modules),
            bias=bias,
        )
        apply_lora(model, lora_cfg)
        model, _ = load_lora(model, str(adapter_path), device=device_str)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic held-out perplexity evaluation for baseline and LoRA models."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to base checkpoint (.pt)")
    parser.add_argument("--val_bin", type=str, default="data/val.bin", help="Validation token binary path")
    parser.add_argument("--context_len", type=int, default=None, help="Optional override for context length")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default="results/eval/perplexity_eval.json")
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional LoRA adapter path")
    parser.add_argument("--compare_lora", action="store_true", help="Evaluate both baseline and LoRA")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--target_modules", nargs="+", default=["W_q", "W_k", "W_v", "W_o"])
    parser.add_argument("--bias", type=str, default="none", choices=["none", "lora", "all"])
    return parser.parse_args()


def resolve_val_bin_path(val_bin_arg: str) -> Path:
    candidate = Path(val_bin_arg)
    if candidate.exists():
        return candidate

    fallback = Path("data/fine_tune_val.bin")
    if fallback.exists():
        print(
            f"[warn] Validation binary '{candidate}' not found. "
            f"Falling back to '{fallback}'."
        )
        return fallback

    raise FileNotFoundError(
        f"Validation binary not found: {candidate}. "
        "Also checked fallback: data/fine_tune_val.bin"
    )


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    val_bin = resolve_val_bin_path(args.val_bin)

    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    if args.compare_lora and adapter_path is None:
        raise ValueError("--compare_lora requires --adapter_path")
    if adapter_path is not None and not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    device = torch.device(args.device)

    base_model = NanoGPT.from_checkpoint(str(checkpoint_path), device=str(device))
    context_len = int(args.context_len) if args.context_len is not None else int(base_model.config.context_len)
    del base_model

    dataset = TokenChunkDataset(bin_path=val_bin, context_len=context_len)

    baseline_model = load_model(
        checkpoint_path=checkpoint_path,
        device_str=str(device),
        adapter_path=None,
        rank=args.rank,
        alpha=args.alpha,
        target_modules=args.target_modules,
        bias=args.bias,
    )
    baseline_result = run_multi_seed_eval(
        model=baseline_model,
        dataset=dataset,
        seeds=args.seeds,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        device=device,
    )

    payload: Dict[str, Any] = {
        "setup": {
            "checkpoint": str(checkpoint_path),
            "val_bin": str(val_bin),
            "context_len": context_len,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "seeds": args.seeds,
            "device": str(device),
        },
        "baseline": baseline_result,
    }

    if args.compare_lora and adapter_path is not None:
        lora_model = load_model(
            checkpoint_path=checkpoint_path,
            device_str=str(device),
            adapter_path=adapter_path,
            rank=args.rank,
            alpha=args.alpha,
            target_modules=args.target_modules,
            bias=args.bias,
        )
        lora_result = run_multi_seed_eval(
            model=lora_model,
            dataset=dataset,
            seeds=args.seeds,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            device=device,
        )
        payload["lora"] = lora_result
        payload["delta"] = {
            "val_loss_mean_delta": float(
                lora_result["aggregate"]["mean_val_loss"] - baseline_result["aggregate"]["mean_val_loss"]
            ),
            "perplexity_mean_delta": float(
                lora_result["aggregate"]["mean_perplexity"] - baseline_result["aggregate"]["mean_perplexity"]
            ),
        }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved evaluation report: {out_path}")
    print(
        f"Baseline mean ppl: {payload['baseline']['aggregate']['mean_perplexity']:.4f} "
        f"(std={payload['baseline']['aggregate']['std_perplexity']:.4f})"
    )
    if "lora" in payload:
        print(
            f"LoRA mean ppl: {payload['lora']['aggregate']['mean_perplexity']:.4f} "
            f"(std={payload['lora']['aggregate']['std_perplexity']:.4f})"
        )
        print(
            f"Perplexity delta (LoRA - baseline): {payload['delta']['perplexity_mean_delta']:.4f}"
        )


if __name__ == "__main__":
    main()

