from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.full_link_gap_selector import _summary_row, _write_rows


def _load_selector_npz(selector_dir: Path) -> dict[str, np.ndarray]:
    with np.load(selector_dir / "selector_arrays.npz", allow_pickle=False) as data:
        return {str(key): np.asarray(data[key]) for key in data.files}


def _read_by_step_actions(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty by-step CSV: {path}")
    if "action_idx" not in rows[0]:
        raise ValueError(f"by-step CSV must contain action_idx column: {path}")
    return np.asarray([int(row["action_idx"]) for row in rows], dtype=np.int32)


def _schedule_from_arrays(arrays: Mapping[str, np.ndarray], schedule_name: str) -> np.ndarray:
    key = f"schedule_{schedule_name}"
    if key not in arrays:
        raise KeyError(f"{key!r} not found in selector_arrays.npz")
    return np.asarray(arrays[key], dtype=np.int32)


def _metric_maps(arrays: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    metric_gaps = {
        key[len("metric_") : -len("_gap")]: np.asarray(value, dtype=np.float64)
        for key, value in arrays.items()
        if key.startswith("metric_") and key.endswith("_gap")
    }
    metric_values = {
        key[len("metric_") :]: np.asarray(value, dtype=np.float64)
        for key, value in arrays.items()
        if key.startswith("metric_") and not key.endswith("_gap") and not key.endswith("_full_link")
    }
    metric_refs = {
        key[len("metric_") : -len("_full_link")]: np.asarray(value, dtype=np.float64)
        for key, value in arrays.items()
        if key.startswith("metric_") and key.endswith("_full_link")
    }
    return metric_gaps, metric_values, metric_refs


def score_selected_schedule(
    *,
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
    name: str,
    quality_threshold: float | None = None,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=np.int32)
    stage_cost = np.asarray(arrays["stage_cost"], dtype=np.float32)
    transition_counts = np.asarray(arrays["transition_counts"], dtype=np.float32)
    metric_gaps, metric_values, metric_refs = _metric_maps(arrays)
    row = _summary_row(
        name=str(name),
        selected=selected,
        stage_cost=stage_cost,
        transition_cost=transition_counts,
        metric_gaps=metric_gaps,
        metric_values=metric_values,
        metric_references=metric_refs,
    )
    setup_counts = np.zeros(selected.shape[0], dtype=np.int32)
    if selected.size > 1:
        setup_counts[1:] = transition_counts[selected[:-1], selected[1:]].astype(np.int32, copy=False)
    row["total_setup_commands"] = int(np.sum(setup_counts))
    row["max_setup_commands_per_step"] = int(np.max(setup_counts)) if setup_counts.size else 0
    row["mean_setup_commands_per_step"] = float(np.mean(setup_counts)) if setup_counts.size else 0.0
    row["num_switches"] = int(np.count_nonzero(selected[1:] != selected[:-1])) if selected.size > 1 else 0
    row["num_unique_topologies"] = int(len(np.unique(selected)))
    if quality_threshold is not None:
        gap = float(row.get("mean_all_global_gap", row.get("mean_all_gap", np.nan)))
        row["quality_threshold"] = float(quality_threshold)
        row["within_quality_threshold"] = bool(gap <= float(quality_threshold))
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score selector schedules with report-style global full-link gaps and setup counts."
    )
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--schedule", action="append", default=None, help="Schedule name stored in selector_arrays.npz.")
    parser.add_argument("--by-step-csv", type=Path, action="append", default=None)
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arrays = _load_selector_npz(Path(args.selector_dir))
    rows: list[dict[str, Any]] = []
    for schedule in args.schedule or []:
        selected = _schedule_from_arrays(arrays, str(schedule))
        rows.append(
            score_selected_schedule(
                arrays=arrays,
                selected=selected,
                name=str(schedule),
                quality_threshold=args.quality_threshold,
            )
        )
    for path in args.by_step_csv or []:
        selected = _read_by_step_actions(Path(path))
        rows.append(
            score_selected_schedule(
                arrays=arrays,
                selected=selected,
                name=Path(path).stem.removesuffix("_by_step"),
                quality_threshold=args.quality_threshold,
            )
        )
    if not rows:
        raise ValueError("provide --schedule and/or --by-step-csv")

    if args.out_csv is not None:
        _write_rows(Path(args.out_csv), rows)
    if args.out_json is not None:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
