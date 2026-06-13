from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from motif_enumerator import (
    DIRECTIONS,
    g60_node_label,
    motif_edges_for_cell,
    motif_rows,
    primitive_motifs,
    summarize_motifs,
    valid_motifs,
)


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR / "outputs" / "motif_p2_y3"

DIR_COLORS = {
    "A": "#1f77b4",
    "B": "#7b2cbf",
    "C": "#f77f00",
    "D": "#2a9d8f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enumerate and visualize primitive p-by-y motifs.")
    parser.add_argument("--p", type=int, default=2, help="Columns in orbital planes, i.e. x direction.")
    parser.add_argument("--y", type=int, default=3, help="Rows in y direction.")
    parser.add_argument("--kn", type=int, default=None, help="Legacy alias for --y.")
    parser.add_argument("--kp", type=int, default=None, help="Legacy alias for --p.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if args.kn is not None:
        args.y = args.kn
    if args.kp is not None:
        args.p = args.kp
    return args


def output_stem(p: int, y: int) -> str:
    return f"primitive_motifs_p{int(p)}_y{int(y)}"


def write_outputs(motifs: tuple[tuple[str, ...], ...], kn: int, kp: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize_motifs(motifs, kn, kp)
    stem = output_stem(kp, kn)

    with (out_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["motif_id", "y", "p", "kn", "kp", "rows", "directions", "edges"])
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "y": int(kn),
                "p": int(kp),
            }
            for row in rows
        )

    payload = {
        "p": int(kp),
        "y": int(kn),
        "kn": int(kn),
        "kp": int(kp),
        "directions": {key: {"dp": value[0], "dy": value[1]} for key, value in DIRECTIONS.items()},
        "valid_count_before_primitive_filter": len(valid_motifs(kn, kp)),
        "primitive_count": len(motifs),
        "motifs": rows,
    }
    (out_dir / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def draw_one(ax, motif: tuple[str, ...], kn: int, kp: int, motif_id: int) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"#{motif_id}  {' / '.join(motif_rows(motif, kn, kp))}", fontsize=9, pad=2)

    def xy(row: int, col: int) -> tuple[float, float]:
        return float(col) * 1.35, -float(row) * 1.05

    edges = motif_edges_for_cell(motif, kn, kp)
    ghost_nodes = {(edge["dst_row"], edge["dst_col"]) for edge in edges if not (0 <= edge["dst_row"] < kn and 0 <= edge["dst_col"] < kp)}

    for row, col in sorted(ghost_nodes):
        x, y = xy(row, col)
        ax.scatter([x], [y], s=42, facecolors="#f2f2f2", edgecolors="#b0b0b0", linewidths=0.8, zorder=1)
        ax.text(x, y - 0.23, str(g60_node_label(row, col)), ha="center", va="top", fontsize=6, color="#777777")

    for edge in edges:
        x0, y0 = xy(edge["src_row"], edge["src_col"])
        x1, y1 = xy(edge["dst_row"], edge["dst_col"])
        label = edge["direction"]
        rad = 0.0
        if label == "D":
            rad = 0.08
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=1.1,
            color=DIR_COLORS[label],
            alpha=0.88,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=7,
            shrinkB=7,
            zorder=2,
        )
        ax.add_patch(arrow)
        mx = (x0 + x1) / 2.0
        my = (y0 + y1) / 2.0
        ax.text(mx, my + 0.08, label, color=DIR_COLORS[label], fontsize=7, ha="center", va="bottom")

    for row in range(kn):
        for col in range(kp):
            x, y = xy(row, col)
            ax.scatter([x], [y], s=58, facecolors="white", edgecolors="#222222", linewidths=0.9, zorder=4)
            ax.text(x, y, str(g60_node_label(row, col)), ha="center", va="center", fontsize=7, color="#111111", zorder=5)

    ax.set_xlim(-0.45, (kp + 1.95) * 1.35)
    ax.set_ylim(-(kn + 0.85) * 1.05, 0.80)


def draw_motif_sheet(motifs: tuple[tuple[str, ...], ...], kn: int, kp: int, out_dir: Path) -> None:
    cols = 4
    rows = (len(motifs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.55))
    axes_list = list(axes.flat if hasattr(axes, "flat") else [axes])

    for ax in axes_list:
        ax.axis("off")

    for idx, motif in enumerate(motifs, start=1):
        draw_one(axes_list[idx - 1], motif, kn, kp, idx)

    legend_handles = [
        FancyArrowPatch((0, 0), (0.5, 0), arrowstyle="-|>", mutation_scale=9, color=color, label=f"{label}={DIRECTIONS[label]}")
        for label, color in DIR_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle(
        f"Primitive p={kp}, y={kn} motifs: {len(motifs)} combinations "
        "(translation deduplicated; smaller rectangular tilings removed)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))

    stem = output_stem(kp, kn)
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"[motif] wrote {png_path}")
    print(f"[motif] wrote {svg_path}")


def main() -> int:
    args = parse_args()
    kn = int(args.y)
    kp = int(args.p)
    motifs = primitive_motifs(kn, kp)
    out_dir = Path(args.out_dir)
    write_outputs(motifs, kn, kp, out_dir)
    draw_motif_sheet(motifs, kn, kp, out_dir)
    print(f"[motif] p={kp}, y={kn}")
    print(f"[motif] valid_count_before_primitive_filter={len(valid_motifs(kn, kp))}")
    print(f"[motif] primitive_count={len(motifs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
