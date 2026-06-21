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

from src.topology_learning.module.local_metric_replacement import (
    apply_metric_replacement,
    read_candidate_meta_from_summary,
    read_replacement_metric_series,
    replacement_by_step_rows,
    setup_counts_with_local_replacement,
    summarize_local_metric_replacement,
)


DEFAULT_SELECTOR_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning"
    r"\g60_w4h3_full_link_gap_three_pairs_hop_delay_budget_frontier_curve_topkscore120_2pct_fine1721_1739"
)
DEFAULT_METRIC_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning"
    r"\local_hybrid_bridge_metrics_466_to_621_b489_full_top3_quality"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning"
    r"\local_hybrid_replacement_schedule_466_to_621_b489"
)


PAIR_TO_PREFIX = {
    "china_europe": "ce",
    "china_africa": "ca",
    "china_america": "cam",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace one contiguous schedule window with a local hybrid candidate, "
            "using precomputed by-step hop/delay metrics for that candidate."
        )
    )
    parser.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR_DIR)
    parser.add_argument("--base-schedule", default="budget_dp_b1728_cap408_meanref")
    parser.add_argument("--metric-dir", type=Path, default=DEFAULT_METRIC_DIR)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--by-step-csv", type=Path, default=None)
    parser.add_argument("--replacement-topology", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--entry-setup", type=int, default=None)
    parser.add_argument("--exit-setup", type=int, default=None)
    parser.add_argument("--left-row", type=int, default=None)
    parser.add_argument("--right-row", type=int, default=None)
    return parser.parse_args()


def _load_selector_arrays(selector_dir: Path) -> dict[str, np.ndarray]:
    with np.load(Path(selector_dir) / "selector_arrays.npz", allow_pickle=False) as data:
        return {str(key): np.asarray(data[key]) for key in data.files}


def _write_rows(path: Path, rows: list[dict]) -> None:
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
    metric_dir = Path(args.metric_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv is not None else metric_dir / "summary.csv"
    by_step_csv = Path(args.by_step_csv) if args.by_step_csv is not None else metric_dir / "by_step.csv"
    arrays = _load_selector_arrays(Path(args.selector_dir))
    schedule_key = f"schedule_{args.base_schedule}"
    if schedule_key not in arrays:
        raise KeyError(f"{schedule_key!r} not found in {Path(args.selector_dir) / 'selector_arrays.npz'}")
    selected = np.asarray(arrays[schedule_key], dtype=np.int32)

    candidate_meta = read_candidate_meta_from_summary(summary_csv, topology=str(args.replacement_topology))
    entry_setup = int(args.entry_setup if args.entry_setup is not None else candidate_meta["entry_setup"])
    exit_setup = int(args.exit_setup if args.exit_setup is not None else candidate_meta["exit_setup"])
    replacement = read_replacement_metric_series(
        by_step_csv,
        topology=str(args.replacement_topology),
        pair_to_metric_prefix=PAIR_TO_PREFIX,
    )
    summary, metric_values = summarize_local_metric_replacement(
        name=f"{args.base_schedule}__replace__{args.replacement_topology}",
        arrays=arrays,
        selected=selected,
        replacement=replacement,
        entry_setup=entry_setup,
        exit_setup=exit_setup,
        left_row=args.left_row,
        right_row=args.right_row,
        quality_threshold=args.quality_threshold,
    )
    _metric_values, left, right = apply_metric_replacement(
        arrays=arrays,
        selected=selected,
        replacement=replacement,
        left_row=args.left_row,
        right_row=args.right_row,
    )
    setup_counts = setup_counts_with_local_replacement(
        selected=selected,
        transition_counts=np.asarray(arrays["transition_counts"], dtype=np.float64),
        left_row=left,
        right_row=right,
        entry_setup=entry_setup,
        exit_setup=exit_setup,
    )
    rows = replacement_by_step_rows(
        arrays=arrays,
        selected=selected,
        replacement_name=str(args.replacement_topology),
        left_row=left,
        right_row=right,
        metric_values=metric_values,
        setup_counts=setup_counts,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(out_dir / "score.csv", [summary])
    _write_rows(out_dir / "by_step.csv", rows)
    np.save(out_dir / "base_selected_action.npy", selected.astype(np.int32, copy=False))
    np.save(out_dir / "setup_counts.npy", setup_counts.astype(np.float32, copy=False))

    meta = {
        "selector_dir": str(args.selector_dir),
        "base_schedule": str(args.base_schedule),
        "replacement_topology": str(args.replacement_topology),
        "summary_csv": str(summary_csv),
        "by_step_csv": str(by_step_csv),
        "candidate_meta": candidate_meta,
        "left_row": int(left),
        "right_row": int(right),
        "entry_setup": int(entry_setup),
        "exit_setup": int(exit_setup),
        "score": summary,
        "outputs": {
            "score_csv": str(out_dir / "score.csv"),
            "by_step_csv": str(out_dir / "by_step.csv"),
            "setup_counts_npy": str(out_dir / "setup_counts.npy"),
            "base_selected_action_npy": str(out_dir / "base_selected_action.npy"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
