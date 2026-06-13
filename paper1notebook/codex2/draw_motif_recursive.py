from __future__ import annotations

import argparse
import csv
from pathlib import Path

import motif_recursive as mr
from draw_box_motif_v2 import draw_all


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR / "outputs" / "motif_recursive_drawings"


def motif_to_assign(motif: mr.Motif) -> dict[tuple[int, int], str | None]:
    return {
        (col_idx, row_idx): symbol
        for col_idx, column in enumerate(motif)
        for row_idx, symbol in enumerate(column)
    }


def write_motif_csv(motifs: list[mr.Motif], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["motif_id", "motif", "box_edges"])
        writer.writeheader()
        for idx, motif in enumerate(motifs, start=1):
            writer.writerow(
                {
                    "motif_id": idx,
                    "motif": mr.pretty_motif(motif),
                    "box_edges": ", ".join(mr.representative_box_edges_user_ids(motif)),
                }
            )


def draw_motifs(
    motifs: list[mr.Motif],
    *,
    w: int,
    h: int,
    out_dir: Path,
    class_label: str,
    row_pitch: int,
    per_page: int,
    cols: int,
    dpi: int,
    write_pdf: bool,
) -> None:
    assign_classes = [motif_to_assign(motif) for motif in motifs]
    draw_all(
        assign_classes,
        w,
        h,
        row_pitch,
        out_dir,
        per_page=per_page,
        cols=cols,
        write_pdf=write_pdf,
        dpi=dpi,
        class_label=class_label,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw motif_recursive exact-box motifs.")
    parser.add_argument("--w", type=int, default=3, help="Exact-box width.")
    parser.add_argument("--h", type=int, default=3, help="Exact-box height.")
    parser.add_argument(
        "--mode",
        choices=("maximal", "primitive", "both"),
        default="both",
        help="Which motif set to draw.",
    )
    parser.add_argument("--row-pitch", type=int, default=36)
    parser.add_argument("--per-page", type=int, default=24)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("maximal", "both"):
        maximal = mr.enumerate_maximal_exact_box(args.w, args.h)
        maximal_dir = out_dir / f"w{args.w}_h{args.h}_maximal"
        write_motif_csv(maximal, maximal_dir / f"motif_recursive_w{args.w}_h{args.h}_maximal.csv")
        draw_motifs(
            maximal,
            w=args.w,
            h=args.h,
            out_dir=maximal_dir,
            class_label="maximal exact-box motifs",
            row_pitch=args.row_pitch,
            per_page=args.per_page,
            cols=args.cols,
            dpi=args.dpi,
            write_pdf=not args.no_pdf,
        )
        print(f"[motif-recursive-draw] maximal exact-box motifs: {len(maximal)} -> {maximal_dir}")

    if args.mode in ("primitive", "both"):
        primitive = mr.enumerate_primitive_exact_box(args.w, args.h)
        primitive_dir = out_dir / f"w{args.w}_h{args.h}_primitive"
        write_motif_csv(primitive, primitive_dir / f"motif_recursive_w{args.w}_h{args.h}_primitive.csv")
        draw_motifs(
            primitive,
            w=args.w,
            h=args.h,
            out_dir=primitive_dir,
            class_label="primitive exact-box motifs",
            row_pitch=args.row_pitch,
            per_page=args.per_page,
            cols=args.cols,
            dpi=args.dpi,
            write_pdf=not args.no_pdf,
        )
        print(f"[motif-recursive-draw] primitive exact-box motifs: {len(primitive)} -> {primitive_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
