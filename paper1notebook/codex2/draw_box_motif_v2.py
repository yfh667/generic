from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch

from box_motif_enumerator_v2 import OFFS, describe, edge_labels, enumerate_mn, write_outputs


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR / "outputs" / "box_motif_v2_drawings"

DIR_COLORS = {
    "A": "#1f77b4",
    "B": "#7b2cbf",
    "C": "#f77f00",
    "D": "#2a9d8f",
}


def node_label(col: int, row: int, row_pitch: int) -> int:
    return int(col) * int(row_pitch) + int(row) + 1


def draw_one(ax, assign, m: int, n: int, motif_id: int, row_pitch: int) -> None:
    ax.set_aspect("equal")
    ax.axis("off")

    cols, edge_count, saturation = describe(assign, m, n)
    edge_count = edge_count.replace("边", " edges")
    saturation = "full" if saturation == "饱和" else "idle"
    ax.set_title(f"#{motif_id} [{cols}] {edge_count} {saturation}", fontsize=8, pad=2)

    def xy(col: int, row: int) -> tuple[float, float]:
        return float(col) * 1.38, -float(row) * 0.96

    for (col, row), direction in sorted(assign.items()):
        if direction is None:
            continue
        dp, dn = OFFS[direction]
        dst_col = col + dp
        dst_row = row + dn
        x0, y0 = xy(col, row)
        x1, y1 = xy(dst_col, dst_row)
        rad = 0.08 if direction == "D" else 0.0
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=1.05,
            color=DIR_COLORS[direction],
            alpha=0.9,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=7,
            shrinkB=7,
            zorder=2,
        )
        ax.add_patch(arrow)
        ax.text(
            (x0 + x1) / 2.0,
            (y0 + y1) / 2.0 + 0.08,
            direction,
            color=DIR_COLORS[direction],
            fontsize=7,
            ha="center",
            va="bottom",
        )

    for col in range(m):
        for row in range(n):
            x, y = xy(col, row)
            is_planned = col < m - 1
            ax.scatter(
                [x],
                [y],
                s=46 if is_planned else 38,
                facecolors="white" if is_planned else "#f2f2f2",
                edgecolors="#222222" if is_planned else "#b8b8b8",
                linewidths=0.85,
                zorder=4,
            )
            ax.text(
                x,
                y,
                str(node_label(col, row, row_pitch)),
                ha="center",
                va="center",
                fontsize=6,
                color="#111111" if is_planned else "#777777",
                zorder=5,
            )

    ax.set_xlim(-0.45, (m - 1) * 1.38 + 0.55)
    ax.set_ylim(-(n - 1) * 0.96 - 0.65, 0.65)


def draw_page(
    classes,
    m: int,
    n: int,
    row_pitch: int,
    start_idx: int,
    page_classes,
    page_number: int,
    total_pages: int,
    cols: int,
    rows: int,
    class_label: str,
):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.65))
    axes_list = list(axes.flat if hasattr(axes, "flat") else [axes])
    for ax in axes_list:
        ax.axis("off")

    for offset, assign in enumerate(page_classes):
        draw_one(axes_list[offset], assign, m, n, start_idx + offset, row_pitch)

    legend_handles = [
        FancyArrowPatch(
            (0, 0),
            (0.5, 0),
            arrowstyle="-|>",
            mutation_scale=9,
            color=color,
            label=f"{label}={OFFS[label]}",
        )
        for label, color in DIR_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(
        f"Box motif v2 m={m}, n={n}: {len(classes)} {class_label} "
        f"(page {page_number}/{total_pages})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    return fig


def draw_all(
    classes,
    m: int,
    n: int,
    row_pitch: int,
    out_dir: Path,
    per_page: int,
    cols: int,
    *,
    write_pdf: bool,
    dpi: int,
    class_label: str = "translation-deduped maximal classes",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = (per_page + cols - 1) // cols
    total_pages = (len(classes) + per_page - 1) // per_page
    pdf_path = out_dir / f"box_motif_v2_m{m}_n{n}.pdf"

    pdf = PdfPages(pdf_path) if write_pdf else None
    try:
        for page_idx in range(total_pages):
            start = page_idx * per_page
            end = min(start + per_page, len(classes))
            fig = draw_page(
                classes,
                m,
                n,
                row_pitch,
                start + 1,
                classes[start:end],
                page_idx + 1,
                total_pages,
                cols,
                rows,
                class_label,
            )
            png_path = out_dir / f"box_motif_v2_m{m}_n{n}_page_{page_idx + 1:02d}.png"
            fig.savefig(png_path, dpi=dpi)
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)
            print(f"[draw-box-motif-v2] wrote {png_path}")
    finally:
        if pdf is not None:
            pdf.close()

    if write_pdf:
        print(f"[draw-box-motif-v2] wrote {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw box motif v2 classes for arbitrary m,n.")
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--row-pitch", type=int, default=36)
    parser.add_argument("--per-page", type=int, default=24)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--class-label", default="translation-deduped maximal classes")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    legal_count, maximal_count, classes = enumerate_mn(args.m, args.n)
    out_dir = Path(args.out_dir)
    write_outputs(out_dir, args.m, args.n, legal_count, maximal_count, classes, args.row_pitch)
    draw_all(
        classes,
        args.m,
        args.n,
        args.row_pitch,
        out_dir,
        args.per_page,
        args.cols,
        write_pdf=not args.no_pdf,
        dpi=args.dpi,
        class_label=args.class_label,
    )
    print(
        f"[draw-box-motif-v2] m={args.m}, n={args.n}, legal={legal_count}, "
        f"maximal={maximal_count}, translation_deduped={len(classes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
