from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch


GENERIC_ROOT = Path(__file__).resolve().parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from box_motif_enumerator_v2 import OFFS, canon_graph
from src.motif_generator.module.exact_box import (
    enumerate_primitive_exact_box,
    motif_edges_as_user_ids,
    pretty_motif,
)


DEFAULT_OUT_DIR = Path(r"E:\paper11\data\linshi\motif_w4h3_primitive_translation_dedup")
DIR_COLORS = {
    "A": "#1f77b4",
    "B": "#7b2cbf",
    "C": "#f77f00",
    "D": "#2a9d8f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw W4 H3 primitive motifs after global translation deduplication."
    )
    parser.add_argument("--w", type=int, default=4)
    parser.add_argument("--h", type=int, default=3)
    parser.add_argument("--row-pitch", type=int, default=36)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--per-page", type=int, default=24)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)
    return parser.parse_args()


def motif_to_assign(motif):
    return {
        (int(c), int(r)): motif[c][r]
        for c in range(len(motif))
        for r in range(len(motif[0]))
    }


def canonical_primitive_representatives(w: int, h: int):
    primitive = enumerate_primitive_exact_box(w, h)
    by_key = {}
    for motif in primitive:
        key = canon_graph(motif_to_assign(motif), w, h)
        current = by_key.get(key)
        if current is None or pretty_motif(motif) < pretty_motif(current):
            by_key[key] = motif
    return sorted(by_key.values(), key=pretty_motif), len(primitive)


def motif_support_label(motif) -> str:
    entries = []
    for c, column in enumerate(motif):
        for r, symbol in enumerate(column):
            if symbol is not None:
                entries.append(f"({c},{r},{symbol})")
    return "[" + ", ".join(entries) + "]"


def node_label(col: int, row: int, row_pitch: int) -> int:
    return int(col) * int(row_pitch) + int(row) + 1


def draw_one(ax, motif, motif_id: int, row_pitch: int) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    edge_count = sum(1 for col in motif for symbol in col if symbol is not None)
    ax.set_title(f"#{motif_id:03d}  {pretty_motif(motif)}  edges={edge_count}", fontsize=7.6, pad=2)

    def xy(col: int, row: int) -> tuple[float, float]:
        return float(col) * 1.42, -float(row) * 0.98

    for col, column in enumerate(motif):
        for row, symbol in enumerate(column):
            if symbol is None:
                continue
            dx, dy = OFFS[str(symbol)]
            x0, y0 = xy(col, row)
            x1, y1 = xy(col + dx, row + dy)
            arrow = FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=1.05,
                color=DIR_COLORS[str(symbol)],
                connectionstyle=f"arc3,rad={0.08 if symbol == 'D' else 0.0}",
                shrinkA=7,
                shrinkB=7,
                zorder=2,
            )
            ax.add_patch(arrow)
            ax.text(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0 + 0.08,
                str(symbol),
                color=DIR_COLORS[str(symbol)],
                fontsize=6.8,
                ha="center",
                va="bottom",
            )

    box_w = len(motif) + 1
    box_h = len(motif[0])
    for col in range(box_w):
        for row in range(box_h):
            x, y = xy(col, row)
            planned = col < box_w - 1
            ax.scatter(
                [x],
                [y],
                s=44 if planned else 36,
                facecolors="white" if planned else "#f0f0f0",
                edgecolors="#222222" if planned else "#999999",
                linewidths=0.75,
                zorder=4,
            )
            ax.text(
                x,
                y,
                str(node_label(col, row, row_pitch)),
                ha="center",
                va="center",
                fontsize=5.4,
                color="#111111" if planned else "#666666",
                zorder=5,
            )

    ax.set_xlim(-0.45, (box_w - 1) * 1.42 + 0.5)
    ax.set_ylim(-(box_h - 1) * 0.98 - 0.6, 0.62)


def write_csv(path: Path, motifs, row_pitch: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["motif_id", "motif", "edge_count", "support", "edges"],
        )
        writer.writeheader()
        for idx, motif in enumerate(motifs, start=1):
            writer.writerow(
                {
                    "motif_id": idx,
                    "motif": pretty_motif(motif),
                    "edge_count": sum(1 for col in motif for symbol in col if symbol is not None),
                    "support": motif_support_label(motif),
                    "edges": ", ".join(motif_edges_as_user_ids(motif, row_pitch=row_pitch)),
                }
            )


def draw_pages(
    motifs,
    *,
    out_dir: Path,
    w: int,
    h: int,
    row_pitch: int,
    per_page: int,
    cols: int,
    dpi: int,
    write_pdf: bool,
    start_page: int,
    end_page: int | None,
) -> None:
    rows = math.ceil(per_page / cols)
    total_pages = math.ceil(len(motifs) / per_page)
    first_page = max(1, int(start_page))
    last_page = total_pages if end_page is None else min(total_pages, int(end_page))
    pdf = PdfPages(out_dir / f"w{w}_h{h}_primitive_translation_dedup.pdf") if write_pdf else None
    try:
        for page_idx in range(first_page - 1, last_page):
            start = page_idx * per_page
            page_motifs = motifs[start : start + per_page]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.65, rows * 2.35))
            axes_list = list(axes.flat if hasattr(axes, "flat") else [axes])
            for ax in axes_list:
                ax.axis("off")

            for offset, motif in enumerate(page_motifs):
                draw_one(axes_list[offset], motif, start + offset + 1, row_pitch)

            handles = [
                FancyArrowPatch(
                    (0, 0),
                    (0.5, 0),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    color=color,
                    label=f"{symbol}={OFFS[symbol]}",
                )
                for symbol, color in DIR_COLORS.items()
            ]
            fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
            fig.suptitle(
                f"W{w} H{h} primitive + translation-dedup motifs "
                f"({len(motifs)} total), page {page_idx + 1}/{total_pages}",
                fontsize=13,
            )
            fig.tight_layout(rect=(0, 0.055, 1, 0.94))
            png_path = out_dir / f"w{w}_h{h}_primitive_translation_dedup_page_{page_idx + 1:02d}.png"
            fig.savefig(png_path, dpi=dpi)
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)
            print(f"[draw-w4h3] wrote {png_path}", flush=True)
    finally:
        if pdf is not None:
            pdf.close()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    motifs, primitive_count = canonical_primitive_representatives(args.w, args.h)

    csv_path = out_dir / f"w{args.w}_h{args.h}_primitive_translation_dedup.csv"
    write_csv(csv_path, motifs, int(args.row_pitch))
    draw_pages(
        motifs,
        out_dir=out_dir,
        w=int(args.w),
        h=int(args.h),
        row_pitch=int(args.row_pitch),
        per_page=int(args.per_page),
        cols=int(args.cols),
        dpi=int(args.dpi),
        write_pdf=not bool(args.no_pdf),
        start_page=int(args.start_page),
        end_page=args.end_page,
    )
    print(
        f"[draw-w4h3] primitive_matrix_count={primitive_count} "
        f"translation_dedup_count={len(motifs)} csv={csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
