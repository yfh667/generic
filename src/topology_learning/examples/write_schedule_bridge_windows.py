from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.bridge_windows import (
    bridge_windows_from_schedule,
    write_dataclass_rows,
)


DEFAULT_SELECTOR_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning"
    r"\g60_w4h3_full_link_gap_three_pairs_hop_delay_budget_frontier_curve_topkscore120_2pct_fine1721_1739"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning"
    r"\schedule_bridge_windows_b1728"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write bridge-window diagnostics for a selector schedule.")
    parser.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR_DIR)
    parser.add_argument("--schedule", default="budget_dp_b1728_cap408_meanref")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sort-by", choices=("direct_saving", "bridge_setup", "middle_length"), default="direct_saving")
    return parser.parse_args()


def _load_arrays(selector_dir: Path) -> dict[str, np.ndarray]:
    with np.load(Path(selector_dir) / "selector_arrays.npz", allow_pickle=False) as data:
        return {str(key): np.asarray(data[key]) for key in data.files}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    arrays = _load_arrays(args.selector_dir)
    schedule_key = f"schedule_{args.schedule}"
    if schedule_key not in arrays:
        raise KeyError(f"{schedule_key!r} not found in selector_arrays.npz")
    selected = np.asarray(arrays[schedule_key], dtype=np.int32)
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    topology_names = np.asarray(arrays["topology_names"])
    transition_counts = np.asarray(arrays["transition_counts"], dtype=np.float64)
    stage_cost = np.asarray(arrays.get("stage_cost"), dtype=np.float64) if "stage_cost" in arrays else None

    segments, windows = bridge_windows_from_schedule(
        selected=selected,
        steps=steps,
        topology_names=topology_names,
        transition_counts=transition_counts,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataclass_rows(out_dir / "segments.csv", segments)

    rows = []
    for window in windows:
        row = asdict(window)
        if stage_cost is not None:
            row["middle_mean_stage_cost"] = float(np.mean(stage_cost[window.left_row : window.right_row + 1, window.middle_action_idx]))
            row["prev_mean_stage_cost"] = float(np.mean(stage_cost[window.left_row : window.right_row + 1, window.prev_action_idx]))
            row["next_mean_stage_cost"] = float(np.mean(stage_cost[window.left_row : window.right_row + 1, window.next_action_idx]))
        rows.append(row)

    reverse = str(args.sort_by) in {"direct_saving", "bridge_setup", "middle_length"}
    sort_key = {
        "direct_saving": lambda row: (int(row["direct_saving"]), int(row["bridge_setup"])),
        "bridge_setup": lambda row: (int(row["bridge_setup"]), int(row["direct_saving"])),
        "middle_length": lambda row: (int(row["middle_length_rows"]), int(row["direct_saving"])),
    }[str(args.sort_by)]
    rows_sorted = sorted(rows, key=sort_key, reverse=reverse)
    _write_rows(out_dir / "bridge_windows.csv", rows)
    _write_rows(out_dir / "bridge_windows_ranked.csv", rows_sorted)

    setup_counts = np.zeros(selected.size, dtype=np.float64)
    if selected.size > 1:
        setup_counts[1:] = transition_counts[selected[:-1], selected[1:]]
    meta = {
        "selector_dir": str(args.selector_dir),
        "schedule": str(args.schedule),
        "num_steps": int(selected.size),
        "num_segments": int(len(segments)),
        "num_bridge_windows": int(len(windows)),
        "total_setup_commands": int(np.sum(setup_counts)),
        "max_setup_commands_per_step": int(np.max(setup_counts)) if setup_counts.size else 0,
        "outputs": {
            "segments_csv": str(out_dir / "segments.csv"),
            "bridge_windows_csv": str(out_dir / "bridge_windows.csv"),
            "bridge_windows_ranked_csv": str(out_dir / "bridge_windows_ranked.csv"),
            "meta_json": str(out_dir / "meta.json"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
