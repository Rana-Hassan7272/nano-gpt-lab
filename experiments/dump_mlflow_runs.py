from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from mlflow.tracking import MlflowClient


def _final_metric(client: MlflowClient, run_id: str, key: str) -> Optional[float]:
    hist = client.get_metric_history(run_id, key)
    if not hist:
        return None
    # metric history is time-ordered; last entry is final
    return float(hist[-1].value)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    tracking_dir = repo_root / "results" / "mlruns"
    tracking_uri = f"file://{tracking_dir}"

    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = client.list_experiments()
    if not experiments:
        print("No MLflow experiments found.")
        return

    metrics_keys = ["train_loss", "val_loss", "perplexity", "learning_rate", "grad_norm"]
    rows: List[Dict[str, Any]] = []

    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["attributes.start_time DESC"])
        for r in runs:
            row: Dict[str, Any] = {
                "experiment_name": exp.name,
                "run_id": r.info.run_id,
                "run_name": r.data.tags.get("mlflow.runName"),
                "final_step": None,
            }
            for k in metrics_keys:
                row[k] = _final_metric(client, r.info.run_id, k)
            rows.append(row)

    # Print a compact view
    rows_sorted = sorted(rows, key=lambda x: (x.get("val_loss") is None, x.get("val_loss") if x.get("val_loss") is not None else 1e18))
    print("MLflow runs summary (sorted by final val_loss asc):")
    for row in rows_sorted:
        print(
            f"- {row['run_name'] or row['run_id']}: "
            f"train_loss={row.get('train_loss')} val_loss={row.get('val_loss')} "
            f"ppl={row.get('perplexity')} lr={row.get('learning_rate')} grad_norm={row.get('grad_norm')}"
        )

    out_path = repo_root / "results" / "mlflow_runs_summary.json"
    out_path.write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

