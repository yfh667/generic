from __future__ import annotations

from pathlib import Path
from typing import Any

from .exact_box import Motif
from .support import MotifSupport, motif_support_from_dict, motif_support_to_edge_records
from .tiling import node_id


SYMBOL_COLORS = {
    "A": "#1f77b4",
    "B": "#7b2cbf",
    "C": "#f77f00",
    "D": "#2a9d8f",
}


def motif_matrix_to_support(
    motif: Motif,
    *,
    name: str = "",
) -> MotifSupport:
    if not motif:
        raise ValueError("motif matrix must contain at least one planning column")
    h = len(motif[0])
    if any(len(col) != h for col in motif):
        raise ValueError("all motif columns must have the same height")

    support = []
    for col, column in enumerate(motif):
        for row, symbol in enumerate(column):
            if symbol is not None:
                support.append([int(col), int(row), str(symbol)])
    return motif_support_from_dict(
        {
            "motif": {
                "name": str(name),
                "w": len(motif) + 1,
                "h": h,
                "support": support,
            }
        }
    )


def normalize_drawable_motif(motif: MotifSupport | Motif | dict[str, Any]) -> MotifSupport:
    """Normalize supported motif inputs for local drawing.

    Accepted inputs:
    - ``MotifSupport``
    - support-style dict: ``{"w": 3, "h": 3, "support": [[0, 0, "D"], ...]}``
    - wrapped dict: ``{"motif": {"w": 3, "h": 3, "support": ...}}``
    - motif matrix returned by ``enumerate_*`` or ``motif_from_columns``
    """

    if isinstance(motif, MotifSupport):
        return motif
    if isinstance(motif, dict):
        return motif_support_from_dict(motif)
    return motif_matrix_to_support(motif)  # type: ignore[arg-type]


def draw_motif_support(
    motif: MotifSupport | Motif | dict[str, Any],
    *,
    out_path: str | Path | None = None,
    ax=None,
    row_pitch: int = 36,
    title: str | None = None,
    show_node_ids: bool = True,
    show_symbols: bool = True,
    show_grid: bool = False,
    figsize: tuple[float, float] | None = None,
    dpi: int = 180,
):
    """Draw one local motif box as nodes and arrows.

    The function deliberately draws only the local motif, not the full tiled
    constellation. Use ``tile_motif_on_grid`` and the 2D viewer for full-network
    topology inspection. Row ``0`` is drawn at the bottom, so ``y`` increases
    upward in the picture.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    motif_support = normalize_drawable_motif(motif)
    local_edges = motif_support_to_edge_records(motif_support)

    if figsize is None:
        figsize = (max(4.0, motif_support.w * 1.55), max(3.2, motif_support.h * 1.25))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_aspect("equal")
    ax.axis("off")

    def xy(col: int, row: int) -> tuple[float, float]:
        return float(col) * 1.45, float(row) * 1.05

    def edge_rad(edge) -> float:
        if edge.symbol != "D":
            return 0.0
        center_row = (motif_support.h - 1) / 2.0
        if edge.src_row < center_row:
            return 0.24
        if edge.src_row > center_row:
            return -0.24
        return 0.20

    if show_grid:
        xmin = -0.35
        xmax = (motif_support.w - 1) * 1.45 + 0.35
        ymin = -0.35
        ymax = (motif_support.h - 1) * 1.05 + 0.35
        ax.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            color="#d1d5db",
            linewidth=0.8,
            linestyle=(0, (3, 3)),
            zorder=0,
        )

    for edge in local_edges:
        x0, y0 = xy(edge.src_col, edge.src_row)
        x1, y1 = xy(edge.dst_col, edge.dst_row)
        color = SYMBOL_COLORS.get(edge.symbol, "#111827")
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=2.0,
            color=color,
            connectionstyle=f"arc3,rad={edge_rad(edge):.3f}",
            shrinkA=12,
            shrinkB=12,
            zorder=2,
        )
        ax.add_patch(arrow)
        if show_symbols:
            ax.text(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0 + 0.13,
                edge.symbol,
                color=color,
                fontsize=11,
                ha="center",
                va="bottom",
                zorder=3,
            )

    for col in range(motif_support.w):
        for row in range(motif_support.h):
            x, y = xy(col, row)
            planned = col < motif_support.w - 1
            ax.scatter(
                [x],
                [y],
                s=110 if planned else 92,
                facecolors="#ffffff" if planned else "#f3f4f6",
                edgecolors="#111827" if planned else "#9ca3af",
                linewidths=1.2,
                zorder=4,
            )
            if show_node_ids:
                ax.text(
                    x,
                    y,
                    str(node_id(col, row, int(row_pitch))),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#111827" if planned else "#6b7280",
                    zorder=5,
                )

    resolved_title = title
    if resolved_title is None:
        resolved_title = motif_support.name or f"motif w={motif_support.w}, h={motif_support.h}"
    ax.set_title(resolved_title, fontsize=12, pad=8)
    ax.set_xlim(-0.55, (motif_support.w - 1) * 1.45 + 0.55)
    ax.set_ylim(-0.65, (motif_support.h - 1) * 1.05 + 0.7)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    return fig, ax
