from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence


DEFAULT_PREDICTIONS = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_knn_reward_policy_lambda050_time_k5"
    r"\row_mask_knn_reward_policy_predictions.csv"
)
DEFAULT_OUT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_knn_reward_policy_lambda050_time_k5"
    r"\row_mask_knn_reward_policy_schedule.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert row_mask_knn_reward_policy_predictions.csv to dynamic-topology schedule format."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--policy", default="knn_reward_time_k5")
    parser.add_argument("--n", type=int, default=36)
    return parser.parse_args()


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    with Path(args.predictions).open("r", encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.DictReader(f))
    if not raw_rows:
        raise ValueError(f"empty predictions: {args.predictions}")

    out_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        out: dict[str, Any] = {
            "step": int(float(row["step"])),
            "policy": str(args.policy),
            "action": row.get("pred_action", ""),
            "action_index": int(float(row["pred_action_index"])),
            "mask": row.get("pred_mask", ""),
            "row_count": int(float(row.get("pred_count", 0))),
            "mean_hops": float(row["pred_hops_lookup"]),
            "mean_delay_ms": float(row["pred_delay_ms_lookup"]),
            "target_action": row.get("true_action", ""),
            "target_action_index": int(float(row.get("true_action_index", -1))),
            "target_hops": float(row.get("target_hops", "nan")),
            "target_delay_ms": float(row.get("target_delay_ms", "nan")),
            "split": row.get("split", ""),
            "neighbors": row.get("neighbors", ""),
        }
        for y in range(int(args.n)):
            out[f"y{y:02d}"] = int(float(row[f"pred_y{y:02d}"]))
        out_rows.append(out)

    write_rows(Path(args.out), out_rows)
    print(f"wrote {args.out} rows={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
