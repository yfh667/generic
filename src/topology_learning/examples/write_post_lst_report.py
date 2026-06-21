from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.post_lst_report import write_post_lst_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Pareto/recommendation report from post-LST sweep summaries.")
    parser.add_argument("--summary", type=Path, action="append", required=True, help="Post-LST sweep summary CSV. Repeat for multiple pairs.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quality-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--frontier-tolerance",
        type=float,
        action="append",
        default=None,
        help="Quality tolerance band for the priority frontier. Repeat to override defaults.",
    )
    parser.add_argument("--prefix", type=str, default="post_lst")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = write_post_lst_report(
        summary_paths=args.summary,
        out_dir=args.out_dir,
        quality_tolerance=float(args.quality_tolerance),
        frontier_tolerances=args.frontier_tolerance,
        prefix=str(args.prefix),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
