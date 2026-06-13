from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.motif_generator.module.library import write_canonical_motif_library_csv


DEFAULT_OUT_DIR = THIS_DIR / "outputs" / "canonical_motif_library"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the experiment motif library for one w,h: primitive exact-box "
            "motifs followed by global translation canonicalization."
        )
    )
    parser.add_argument("--w", type=int, required=True, help="Motif box width, including target boundary.")
    parser.add_argument("--h", type=int, required=True, help="Motif box height.")
    parser.add_argument("--phase-count", type=int, default=None, help="Optional N; checks h divides N.")
    parser.add_argument("--row-pitch", type=int, default=36, help="Representative node-id pitch in edge strings.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--csv-name", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    csv_name = args.csv_name or f"canonical_w{int(args.w)}_h{int(args.h)}.csv"
    meta = write_canonical_motif_library_csv(
        out_dir / csv_name,
        w=int(args.w),
        h=int(args.h),
        row_pitch=int(args.row_pitch),
        phase_count=args.phase_count,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
