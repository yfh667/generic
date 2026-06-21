from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.budget_dp_selector import run_budget_dp_selector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run budget-constrained DP over existing selector schedules.")
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, action="append", required=True)
    parser.add_argument("--score-kind", action="append", default=None)
    parser.add_argument("--max-transition-count", type=int, action="append", default=None)
    parser.add_argument("--top-k-stage", type=int, default=0)
    parser.add_argument(
        "--top-k-score",
        type=int,
        default=0,
        help="Also add the top-k actions under each requested score-kind at every row.",
    )
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--copy-config-from", type=Path, default=None)
    parser.add_argument("--solver", choices=("dense", "frontier"), default="dense")
    parser.add_argument(
        "--max-labels-per-action",
        type=int,
        default=None,
        help="Optional approximate frontier cap per action. Omit for exact frontier pruning.",
    )
    parser.add_argument("--progress-every", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_budget_dp_selector(
        source_dirs=args.source_dir,
        out_dir=args.out_dir,
        budgets=args.budget,
        score_kinds=args.score_kind or ["default_stage"],
        max_transition_counts=args.max_transition_count or [None],
        top_k_stage=int(args.top_k_stage),
        top_k_score=int(args.top_k_score),
        quality_threshold=args.quality_threshold,
        copy_config_from=args.copy_config_from,
        solver=str(args.solver),
        max_labels_per_action=args.max_labels_per_action,
        progress_every=int(args.progress_every),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
