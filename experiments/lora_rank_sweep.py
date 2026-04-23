from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from evaluate_perplexity import (
    TokenChunkDataset,
    load_model,
    run_multi_seed_eval,
)


def parse_rank_adapter(spec: str) -> Tuple[int, Path]:
    # Format: "<rank>:<adapter_path>"
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid adapter spec '{spec}'. Expected '<rank>:<path>'.")
    rank = int(parts[0])
    adapter_path = Path(parts[1])
    if rank <= 0:
        raise ValueError(f"Rank must be > 0, got {rank}")
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    return rank, adapter_path


def count_trainable(model: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def build_decision_sentence(
    rank: int,
    ppl: float,
    best_ppl: float,
    trainable_pct: float,
) -> str:
    delta_pct = ((ppl - best_ppl) / best_ppl) * 100.0
    if abs(delta_pct) <= 1.0:
        return (
            f"Rank {rank} is near-best (within 1% of best perplexity) with "
            f"{trainable_pct:.3f}% trainable parameters, so it is a strong efficiency candidate."
        )
    if delta_pct > 0:
        return (
            f"Rank {rank} underperforms the best rank by {delta_pct:.2f}% perplexity; "
            f"only keep it if parameter budget is stricter than quality target."
        )
    return (
        f"Rank {rank} improves over the previous best by {-delta_pct:.2f}% perplexity, "
        f"at {trainable_pct:.3f}% trainable parameters."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible LoRA rank sweep evaluation on held-out validation data."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Base checkpoint path")
    parser.add_argument("--val_bin", type=str, default="data/val.bin", help="Validation binary path")
    parser.add_argument("--context_len", type=int, default=None, help="Optional context length override")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        help="Adapter spec '<rank>:<path>' (repeat for each rank)",
    )
    parser.add_argument("--alpha", type=float, default=4.0, help="LoRA alpha used for loading")
    parser.add_argument("--target_modules", nargs="+", default=["W_q", "W_k", "W_v", "W_o"])
    parser.add_argument("--bias", type=str, default="none", choices=["none", "lora", "all"])
    parser.add_argument("--output_json", type=str, default="results/eval/lora_rank_sweep.json")
    parser.add_argument("--output_md", type=str, default="results/eval/lora_rank_sweep.md")
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
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    adapter_specs = [parse_rank_adapter(spec) for spec in args.adapter]
    adapter_specs = sorted(adapter_specs, key=lambda x: x[0])

    device = torch.device(args.device)

    probe_model = load_model(
        checkpoint_path=checkpoint,
        device_str=str(device),
        adapter_path=None,
        rank=4,
        alpha=args.alpha,
        target_modules=args.target_modules,
        bias=args.bias,
    )
    context_len = int(args.context_len) if args.context_len is not None else int(probe_model.config.context_len)
    val_bin = resolve_val_bin_path(args.val_bin)
    dataset = TokenChunkDataset(bin_path=val_bin, context_len=context_len)

    baseline_eval = run_multi_seed_eval(
        model=probe_model,
        dataset=dataset,
        seeds=args.seeds,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        device=device,
    )

    rank_rows: List[Dict[str, Any]] = []
    for rank, adapter_path in adapter_specs:
        model = load_model(
            checkpoint_path=checkpoint,
            device_str=str(device),
            adapter_path=adapter_path,
            rank=rank,
            alpha=args.alpha,
            target_modules=args.target_modules,
            bias=args.bias,
        )
        eval_result = run_multi_seed_eval(
            model=model,
            dataset=dataset,
            seeds=args.seeds,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            device=device,
        )
        trainable, total = count_trainable(model)
        rank_rows.append(
            {
                "rank": rank,
                "adapter_path": str(adapter_path),
                "mean_val_loss": eval_result["aggregate"]["mean_val_loss"],
                "std_val_loss": eval_result["aggregate"]["std_val_loss"],
                "mean_perplexity": eval_result["aggregate"]["mean_perplexity"],
                "std_perplexity": eval_result["aggregate"]["std_perplexity"],
                "trainable_params": trainable,
                "total_params": total,
                "trainable_pct": (100.0 * trainable / max(1, total)),
                "per_seed": eval_result["per_seed"],
            }
        )

    if not rank_rows:
        raise RuntimeError("No rank results were produced.")

    best = min(rank_rows, key=lambda x: x["mean_perplexity"])
    for row in rank_rows:
        row["delta_vs_best_ppl"] = row["mean_perplexity"] - best["mean_perplexity"]
        row["decision"] = build_decision_sentence(
            rank=row["rank"],
            ppl=row["mean_perplexity"],
            best_ppl=best["mean_perplexity"],
            trainable_pct=row["trainable_pct"],
        )

    payload: Dict[str, Any] = {
        "setup": {
            "checkpoint": str(checkpoint),
            "val_bin": str(val_bin),
            "context_len": context_len,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "seeds": args.seeds,
            "device": str(device),
            "alpha": args.alpha,
            "target_modules": args.target_modules,
            "bias": args.bias,
        },
        "baseline": baseline_eval["aggregate"],
        "best_rank": {
            "rank": best["rank"],
            "mean_perplexity": best["mean_perplexity"],
            "trainable_pct": best["trainable_pct"],
        },
        "ranks": rank_rows,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_md = Path(args.output_md)
    lines = [
        "# LoRA Rank Sweep Report",
        "",
        f"- Baseline mean perplexity: **{baseline_eval['aggregate']['mean_perplexity']:.4f}**",
        f"- Best rank: **r={best['rank']}** (mean perplexity **{best['mean_perplexity']:.4f}**)",
        "",
        "| Rank | Mean PPL | Std PPL | Trainable Params | Trainable % | Delta vs Best PPL | Decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rank_rows:
        lines.append(
            f"| {row['rank']} | {row['mean_perplexity']:.4f} | {row['std_perplexity']:.4f} | "
            f"{row['trainable_params']:,} | {row['trainable_pct']:.4f}% | "
            f"{row['delta_vs_best_ppl']:.4f} | {row['decision']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    print(f"Best rank: r={best['rank']} mean_ppl={best['mean_perplexity']:.4f}")


if __name__ == "__main__":
    main()

