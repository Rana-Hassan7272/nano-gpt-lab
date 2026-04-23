from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import _build_normalized_experiment_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized experiments payload from committed artifacts."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing mlflow_runs_summary.json and other result artifacts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/experiments_payload.json",
        help="Output JSON path for normalized experiments payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    payload = _build_normalized_experiment_payload(results_dir)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")
    print(f"Experiments: {len(payload.get('experiments', []))}")
    print(f"Summary rows: {len(payload.get('summary_table', []))}")


if __name__ == "__main__":
    main()

