from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.local_replacement_ranking import (
    read_replacement_score_rows,
    write_replacement_ranking_outputs,
)


DEFAULT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning"
)
DEFAULT_COMPARISON = DEFAULT_ROOT / "local_hybrid_replacement_schedule_466_to_621_b489_comparison.csv"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "local_hybrid_replacement_schedule_466_to_621_b489_ranked"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank local hybrid replacement schedules by quality/setup/peak.")
    parser.add_argument("--score-csv", type=Path, action="append", default=None)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--scan-dir", type=Path, action="append", default=None)
    parser.add_argument("--glob", default="local_hybrid_replacement_schedule_*/score.csv")
    parser.add_argument("--quality-threshold", type=float, default=0.1420148851114186)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="local_hybrid_replacement")
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    for row in rows:
        row.setdefault("_source_path", str(path))
    return rows


def main() -> int:
    args = parse_args()
    rows: list[dict] = []
    if args.comparison_csv is not None and Path(args.comparison_csv).exists():
        rows.extend(_read_csv_rows(Path(args.comparison_csv)))
    if args.score_csv:
        rows.extend(read_replacement_score_rows(args.score_csv))
    for scan_dir in args.scan_dir or []:
        rows.extend(read_replacement_score_rows(sorted(Path(scan_dir).glob(str(args.glob)))))
    if not rows:
        raise ValueError("no score rows were provided")

    result = write_replacement_ranking_outputs(
        rows=rows,
        out_dir=Path(args.out_dir),
        quality_threshold=args.quality_threshold,
        prefix=str(args.prefix),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
