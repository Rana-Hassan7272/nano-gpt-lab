from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient


def _metric_history(client: MlflowClient, run_id: str, key: str) -> List[Dict[str, Any]]:
    hist = client.get_metric_history(run_id, key)
    rows: List[Dict[str, Any]] = []
    for h in hist:
        rows.append({"step": int(h.step), "value": float(h.value)})
    return rows


def plot_metric_curves(
    runs: List[Dict[str, Any]],
    metric_key: str,
    out_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 6))
    for r in runs:
        series = r["history"].get(metric_key, [])
        if not series:
            continue
        steps = [p["step"] for p in series]
        values = [p["value"] for p in series]
        plt.plot(steps, values, label=r["run_name"] or r["run_id"])

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(metric_key)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export loss curves from MLflow file store")
    parser.add_argument(
        "--summary_json",
        type=str,
        default="results/mlflow_runs_summary.json",
        help="Path to results/mlflow_runs_summary.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/loss_curves",
        help="Output directory for PNG plots",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    summary_path = Path(args.summary_json) if Path(args.summary_json).is_absolute() else repo_root / args.summary_json
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else repo_root / args.out_dir

    tracking_dir = repo_root / "results" / "mlruns"
    tracking_uri = f"file://{tracking_dir}"
    client = MlflowClient(tracking_uri=tracking_uri)

    runs = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(runs, list):
        raise ValueError(f"Expected list in {summary_path}")

    metric_keys = ["train_loss", "val_loss"]
    enriched_runs: List[Dict[str, Any]] = []
    for r in runs:
        run_id = r["run_id"]
        run_name = r.get("run_name") or run_id
        history: Dict[str, Any] = {}
        for k in metric_keys:
            history[k] = _metric_history(client, run_id, k)
        enriched_runs.append({"run_id": run_id, "run_name": run_name, "history": history})

    # Save overlay plots
    plot_metric_curves(
        runs=enriched_runs,
        metric_key="train_loss",
        out_path=out_dir / "train_loss_overlay.png",
        title="Train loss vs step",
    )
    plot_metric_curves(
        runs=enriched_runs,
        metric_key="val_loss",
        out_path=out_dir / "val_loss_overlay.png",
        title="Validation loss vs step",
    )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()

