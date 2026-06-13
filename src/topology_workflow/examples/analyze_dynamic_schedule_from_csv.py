from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[3]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_workflow.module import write_dynamic_schedule_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build oracle and minimum-dwell dynamic topology schedules from a comparison CSV."
    )
    parser.add_argument("--compare-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--topology-prefix", type=str, default="motif_")
    parser.add_argument("--metric-name", type=str, default="mean shortest delay (ms)")
    parser.add_argument(
        "--min-dwell-minutes",
        type=float,
        nargs="+",
        default=[10.0, 30.0, 60.0, 120.0],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    meta = write_dynamic_schedule_outputs(
        compare_csv=args.compare_csv,
        out_dir=args.out_dir,
        topology_prefix=args.topology_prefix,
        min_dwell_minutes=args.min_dwell_minutes,
        metric_name=args.metric_name,
    )
    print(
        "[topology-workflow] dynamic schedules written: "
        f"steps={meta['num_steps']} candidates={meta['num_candidate_topologies']} out={args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
