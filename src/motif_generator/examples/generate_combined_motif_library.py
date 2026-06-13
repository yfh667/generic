from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.motif_generator.module.library import write_combined_canonical_motif_library_csv


DEFAULT_OUT_DIR = THIS_DIR / "outputs" / "combined_motif_library"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a combined canonical motif library over all sizes up to max_w,max_h."
    )
    parser.add_argument("--max-w", type=int, required=True)
    parser.add_argument("--max-h", type=int, required=True)
    parser.add_argument("--min-w", type=int, default=2)
    parser.add_argument("--min-h", type=int, default=1)
    parser.add_argument("--exclude-max-size", action="store_true", help="Use this for the small102-style library.")
    parser.add_argument("--phase-count", type=int, default=None)
    parser.add_argument("--row-pitch", type=int, default=36)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--csv-name", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    include_max_size = not bool(args.exclude_max_size)
    default_name = (
        f"combined_canonical_w{int(args.min_w)}_{int(args.max_w)}_"
        f"h{int(args.min_h)}_{int(args.max_h)}"
        f"{'' if include_max_size else '_excluding_max'}.csv"
    )
    meta = write_combined_canonical_motif_library_csv(
        out_dir / (args.csv_name or default_name),
        max_w=int(args.max_w),
        max_h=int(args.max_h),
        min_w=int(args.min_w),
        min_h=int(args.min_h),
        include_max_size=include_max_size,
        row_pitch=int(args.row_pitch),
        phase_count=args.phase_count,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
