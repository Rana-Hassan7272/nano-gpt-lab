from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DIMENSIONS = ["style_adherence", "coherence", "relevance", "fluency"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score blinded prompt benchmark ratings.")
    p.add_argument("--outputs_json", type=str, default="results/eval/prompt_benchmark_outputs.json")
    p.add_argument("--ratings_json", type=str, default="results/eval/prompt_benchmark_ratings_template.json")
    p.add_argument("--output_json", type=str, default="results/eval/prompt_benchmark_summary.json")
    p.add_argument("--output_md", type=str, default="results/eval/prompt_benchmark_summary.md")
    return p.parse_args()


def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def main() -> None:
    args = parse_args()
    outputs_path = Path(args.outputs_json)
    ratings_path = Path(args.ratings_json)
    if not outputs_path.exists():
        raise FileNotFoundError(f"Missing outputs file: {outputs_path}")
    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing ratings file: {ratings_path}")

    outputs_payload = json.loads(outputs_path.read_text(encoding="utf-8"))
    ratings_payload = json.loads(ratings_path.read_text(encoding="utf-8"))
    mapping = outputs_payload.get("blind_mapping", {})
    ratings = ratings_payload.get("ratings", [])

    wins = {"baseline": 0, "lora": 0, "tie": 0}
    model_dim_scores: Dict[str, Dict[str, List[float]]] = {
        "baseline": {d: [] for d in DIMENSIONS},
        "lora": {d: [] for d in DIMENSIONS},
    }
    scored_count = 0

    for row in ratings:
        pid = row.get("id")
        if pid not in mapping:
            continue
        map_row = mapping[pid]
        winner = row.get("winner")
        scores = row.get("scores", {})

        # Parse dimension scores
        for label in ["A", "B"]:
            model = map_row.get(label)
            if model not in ("baseline", "lora"):
                continue
            dim_map = scores.get(label, {})
            for d in DIMENSIONS:
                v = dim_map.get(d)
                if isinstance(v, (int, float)):
                    model_dim_scores[model][d].append(float(v))

        if winner in ("A", "B"):
            model = map_row[winner]
            wins[model] += 1
            scored_count += 1
        elif winner == "tie":
            wins["tie"] += 1
            scored_count += 1

    baseline_total = wins["baseline"]
    lora_total = wins["lora"]
    tie_total = wins["tie"]
    decisive = baseline_total + lora_total
    lora_win_rate = (lora_total / decisive) if decisive > 0 else 0.0

    agg = {
        "baseline": {d: mean(model_dim_scores["baseline"][d]) for d in DIMENSIONS},
        "lora": {d: mean(model_dim_scores["lora"][d]) for d in DIMENSIONS},
    }
    agg_delta = {d: agg["lora"][d] - agg["baseline"][d] for d in DIMENSIONS}

    summary: Dict[str, Any] = {
        "counts": {
            "scored_prompts": scored_count,
            "baseline_wins": baseline_total,
            "lora_wins": lora_total,
            "ties": tie_total,
            "lora_win_rate_on_decisive": lora_win_rate,
        },
        "dimension_scores": {
            "baseline": agg["baseline"],
            "lora": agg["lora"],
            "delta_lora_minus_baseline": agg_delta,
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Prompt Benchmark Summary",
        "",
        f"- Scored prompts: **{scored_count}**",
        f"- Baseline wins: **{baseline_total}**",
        f"- LoRA wins: **{lora_total}**",
        f"- Ties: **{tie_total}**",
        f"- LoRA win rate (decisive only): **{100*lora_win_rate:.2f}%**",
        "",
        "| Dimension | Baseline | LoRA | Delta (LoRA-Baseline) |",
        "|---|---:|---:|---:|",
    ]
    for d in DIMENSIONS:
        md_lines.append(
            f"| {d} | {agg['baseline'][d]:.3f} | {agg['lora'][d]:.3f} | {agg_delta[d]:.3f} |"
        )

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Saved summary JSON: {out_json}")
    print(f"Saved summary MD: {out_md}")


if __name__ == "__main__":
    main()

