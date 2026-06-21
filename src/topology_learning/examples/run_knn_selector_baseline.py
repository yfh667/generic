from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.policies import NearestNeighborPolicy, action_value_report, classification_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dependency-free KNN baseline on a topology learning dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    return parser.parse_args()


def _write_predictions(path: Path, *, steps: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, names: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "true_idx", "true_topology", "pred_idx", "pred_topology", "correct"],
        )
        writer.writeheader()
        for row, step in enumerate(steps):
            true_idx = int(y_true[row])
            pred_idx = int(y_pred[row])
            writer.writerow(
                {
                    "step": int(step),
                    "true_idx": true_idx,
                    "true_topology": str(names[true_idx]),
                    "pred_idx": pred_idx,
                    "pred_topology": str(names[pred_idx]),
                    "correct": int(true_idx == pred_idx),
                }
            )


def main() -> int:
    args = parse_args()
    data = np.load(args.dataset, allow_pickle=False)
    steps = np.asarray(data["steps"], dtype=np.int64)
    features = np.asarray(data["sample_features"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.int32)
    action_values = np.asarray(data["action_values"], dtype=np.float64)
    topology_names = np.asarray(data["topology_names"])

    split = int(round(float(args.train_fraction) * int(steps.size)))
    split = max(1, min(split, int(steps.size) - 1))

    model = NearestNeighborPolicy(k=int(args.k)).fit(features[:split], labels[:split])
    pred = model.predict(features[split:])
    report = {
        **classification_report(labels[split:], pred),
        **action_value_report(action_values=action_values[split:], y_true=labels[split:], y_pred=pred),
        "dataset": str(args.dataset),
        "k": int(args.k),
        "train_rows": int(split),
        "test_rows": int(steps.size - split),
    }

    out_dir = Path(args.out_dir or (args.dataset.parent / f"knn_baseline_k{int(args.k)}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_predictions(
        out_dir / "predictions.csv",
        steps=steps[split:],
        y_true=labels[split:],
        y_pred=pred,
        names=topology_names,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

