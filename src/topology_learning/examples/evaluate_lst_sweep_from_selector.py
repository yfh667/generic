from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.lst_sweep import run_lst_sweep_from_selector


def parse_pair(text: str) -> tuple[int, int, str]:
    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) == 2:
        return int(parts[0]), int(parts[1]), f"group{parts[0]}_group{parts[1]}"
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), str(parts[2])
    raise argparse.ArgumentTypeError("pair must be source:target or source:target:key")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate selector schedules after generic LST active/building simulation.")
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--schedule", action="append", default=None, help="Schedule name. Repeat to evaluate a subset. Default: all schedules.")
    parser.add_argument("--setup-time", type=float, required=True)
    parser.add_argument("--delay-store-dir", type=Path, required=True)
    parser.add_argument("--position-cache-dir", type=Path, default=None)
    parser.add_argument("--group-xml", type=Path, required=True)
    parser.add_argument("--group-cache-dir", type=Path, required=True)
    parser.add_argument("--pair", type=parse_pair, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument(
        "--setup-mode",
        choices=("break_before_make", "make_before_break"),
        default="break_before_make",
        help="LST state machine mode. Default preserves the original evaluator.",
    )
    parser.add_argument(
        "--setup-timing",
        choices=("reactive", "advance"),
        default="reactive",
        help=(
            "How to interpret the selected topology sequence. Default 'reactive' issues setup commands when "
            "an edge first appears; the edge becomes active only after LST seconds. 'advance' treats the "
            "sequence as desired active topology and is only for offline known-future upper-bound runs."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source, target, key = args.pair
    summary_path = run_lst_sweep_from_selector(
        selector_dir=args.selector_dir,
        schedules=args.schedule,
        setup_time_seconds=float(args.setup_time),
        delay_store_dir=args.delay_store_dir,
        position_cache_dir=args.position_cache_dir,
        group_xml=args.group_xml,
        group_cache_dir=args.group_cache_dir,
        source_group_id=int(source),
        target_group_id=int(target),
        pair_key=str(key),
        out_root=args.out_root,
        setup_mode=str(args.setup_mode),
        setup_timing=str(args.setup_timing),
        force=bool(args.force),
        force_group_cache=bool(args.force_group_cache),
    )
    print(f"[topology-learning] lst sweep summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
