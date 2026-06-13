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

from src.motif_generator.module.exact_box import (
    count_maximal_exact_box,
    enumerate_maximal_exact_box,
    enumerate_primitive_exact_box,
    motif_edges_as_user_ids,
    pretty_motif,
)


DEFAULT_OUT_DIR = THIS_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enumerate exact-box motifs for a given w,h.")
    parser.add_argument("--w", type=int, default=3, help="Motif box width.")
    parser.add_argument("--h", type=int, default=3, help="Motif box height.")
    parser.add_argument("--phase-count", type=int, default=None, help="Optional N; checks h divides N.")
    parser.add_argument("--row-pitch", type=int, default=36, help="Node ID pitch used in representative edge strings.")
    parser.add_argument("--mode", choices=("maximal", "primitive", "both"), default="both")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit-print", type=int, default=20)
    return parser.parse_args()


def write_csv(path: Path, motifs, *, row_pitch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["motif_id", "motif", "edges"])
        writer.writeheader()
        for idx, motif in enumerate(motifs, start=1):
            writer.writerow(
                {
                    "motif_id": idx,
                    "motif": pretty_motif(motif),
                    "edges": ",".join(motif_edges_as_user_ids(motif, row_pitch=row_pitch)),
                }
            )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maximal_count = count_maximal_exact_box(args.w, args.h, phase_count=args.phase_count)
    maximal = enumerate_maximal_exact_box(args.w, args.h, phase_count=args.phase_count)
    primitive = enumerate_primitive_exact_box(args.w, args.h, phase_count=args.phase_count)

    payload = {
        "w": int(args.w),
        "h": int(args.h),
        "phase_count": None if args.phase_count is None else int(args.phase_count),
        "row_pitch": int(args.row_pitch),
        "maximal_count": int(maximal_count),
        "primitive_count": len(primitive),
    }
    (out_dir / f"exact_box_w{args.w}_h{args.h}_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.mode in ("maximal", "both"):
        write_csv(out_dir / f"exact_box_w{args.w}_h{args.h}_maximal.csv", maximal, row_pitch=args.row_pitch)
    if args.mode in ("primitive", "both"):
        write_csv(out_dir / f"exact_box_w{args.w}_h{args.h}_primitive.csv", primitive, row_pitch=args.row_pitch)

    print(f"[motif-generator] w={args.w}, h={args.h}")
    print(f"[motif-generator] maximal_count={maximal_count}")
    print(f"[motif-generator] primitive_count={len(primitive)}")
    print(f"[motif-generator] out_dir={out_dir}")

    preview = maximal if args.mode != "primitive" else primitive
    for idx, motif in enumerate(preview[: max(0, int(args.limit_print))], start=1):
        edges = ",".join(motif_edges_as_user_ids(motif, row_pitch=args.row_pitch))
        print(f"#{idx:02d} {pretty_motif(motif)}  edges={edges}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
