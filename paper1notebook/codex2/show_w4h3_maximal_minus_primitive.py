from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


GENERIC_ROOT = Path(__file__).resolve().parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from box_motif_enumerator_v2 import OFFS, canon_graph
from src.motif_generator.module.exact_box import (
    enumerate_maximal_exact_box,
    enumerate_primitive_exact_box,
    motif_edges_as_user_ids,
    pretty_motif,
    smaller_repeat_factors,
)


OUT_DIR = Path(r"E:\paper11\data\linshi\motif_w4h3_maximal_minus_primitive")
ROW_PITCH = 36
DIR_COLORS = {
    "A": "#1f77b4",
    "B": "#7b2cbf",
    "C": "#f77f00",
    "D": "#2a9d8f",
}


def motif_to_assign(motif):
    return {
        (int(c), int(r)): motif[c][r]
        for c in range(len(motif))
        for r in range(len(motif[0]))
    }


def motif_support_label(motif) -> str:
    entries = []
    for c, column in enumerate(motif):
        for r, symbol in enumerate(column):
            if symbol is not None:
                entries.append(f"({c},{r},{symbol})")
    return "[" + ", ".join(entries) + "]"


def node_label(col: int, row: int) -> int:
    return int(col) * ROW_PITCH + int(row) + 1


def draw_one(ax, motif, title: str) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=5)

    def xy(col: int, row: int) -> tuple[float, float]:
        return float(col) * 1.45, -float(row) * 1.05

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
                mutation_scale=10,
                linewidth=1.4,
                color=DIR_COLORS[str(symbol)],
                connectionstyle=f"arc3,rad={0.10 if symbol == 'D' else 0.0}",
                shrinkA=8,
                shrinkB=8,
                zorder=2,
            )
            ax.add_patch(arrow)
            ax.text(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0 + 0.11,
                str(symbol),
                color=DIR_COLORS[str(symbol)],
                fontsize=9,
                ha="center",
                va="bottom",
            )

    w = len(motif) + 1
    h = len(motif[0])
    for col in range(w):
        for row in range(h):
            x, y = xy(col, row)
            planned = col < w - 1
            ax.scatter(
                [x],
                [y],
                s=74 if planned else 60,
                facecolors="white" if planned else "#f0f0f0",
                edgecolors="#222222" if planned else "#999999",
                linewidths=1.0,
                zorder=4,
            )
            ax.text(
                x,
                y,
                str(node_label(col, row)),
                ha="center",
                va="center",
                fontsize=7,
                color="#111111" if planned else "#666666",
                zorder=5,
            )

    ax.set_xlim(-0.5, (w - 1) * 1.45 + 0.5)
    ax.set_ylim(-(h - 1) * 1.05 - 0.75, 0.75)


def main() -> int:
    w, h = 4, 3
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    maximal = enumerate_maximal_exact_box(w, h)
    primitive = enumerate_primitive_exact_box(w, h)
    primitive_keys = {canon_graph(motif_to_assign(motif), w, h) for motif in primitive}

    seen = set()
    gap = []
    for motif in maximal:
        key = canon_graph(motif_to_assign(motif), w, h)
        if key in primitive_keys or key in seen:
            continue
        seen.add(key)
        gap.append(motif)

    gap = sorted(gap, key=pretty_motif)

    csv_path = OUT_DIR / "w4h3_maximal_minus_primitive_6.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gap_id",
                "motif",
                "repeat_factors_as_smaller_wh",
                "support",
                "edges",
            ],
        )
        writer.writeheader()
        for idx, motif in enumerate(gap, start=1):
            writer.writerow(
                {
                    "gap_id": idx,
                    "motif": pretty_motif(motif),
                    "repeat_factors_as_smaller_wh": repr(smaller_repeat_factors(motif)),
                    "support": motif_support_label(motif),
                    "edges": ", ".join(motif_edges_as_user_ids(motif, row_pitch=ROW_PITCH)),
                }
            )

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    for idx, (ax, motif) in enumerate(zip(axes.flat, gap), start=1):
        title = f"#{idx}  {pretty_motif(motif)}\nrepeat={smaller_repeat_factors(motif)}"
        draw_one(ax, motif, title)

    handles = [
        FancyArrowPatch(
            (0, 0),
            (0.5, 0),
            arrowstyle="-|>",
            mutation_scale=11,
            color=color,
            label=f"{symbol}={OFFS[symbol]}",
        )
        for symbol, color in DIR_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("W4 H3: maximal translation classes removed by matrix-primitive filtering", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    png_path = OUT_DIR / "w4h3_maximal_minus_primitive_6.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"maximal={len(maximal)} primitive_matrix={len(primitive)} gap={len(gap)}")
    print(f"csv={csv_path}")
    print(f"png={png_path}")
    for idx, motif in enumerate(gap, start=1):
        print(
            f"#{idx}: {pretty_motif(motif)} | repeat={smaller_repeat_factors(motif)} | "
            f"support={motif_support_label(motif)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
