from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .exact_box import EdgeRecord, Motif, motif_edge_records, pretty_motif
from .support import MotifSupport, motif_support_from_dict, motif_support_label, motif_support_to_edge_records


@dataclass(frozen=True)
class PlacedEdge:
    src_col: int
    src_row: int
    dst_col: int
    dst_row: int
    origin_col: int
    origin_row: int
    symbol: str

    def user_id_pair(self, row_pitch: int) -> tuple[int, int]:
        src = node_id(self.src_col, self.src_row, row_pitch)
        dst = node_id(self.dst_col, self.dst_row, row_pitch)
        return src, dst


@dataclass(frozen=True)
class PlacementAttempt:
    origin_col: int
    origin_row: int
    accepted: bool
    reason: str
    added_edges: int


@dataclass(frozen=True)
class TiledMotifResult:
    p: int
    n: int
    motif: Motif
    motif_width: int
    motif_height: int
    placed_edges: list[PlacedEdge]
    placements: list[PlacementAttempt]
    motif_label: str = ""

    @property
    def edge_count(self) -> int:
        return len(self.placed_edges)

    @property
    def accepted_count(self) -> int:
        return sum(1 for item in self.placements if item.accepted)

    @property
    def rejected_count(self) -> int:
        return sum(1 for item in self.placements if not item.accepted)


def node_id(col: int, row: int, row_pitch: int) -> int:
    """Column-major 1-based node id: ``id = col * row_pitch + row + 1``."""

    return int(col) * int(row_pitch) + int(row) + 1


def node_coord(node: int, row_pitch: int) -> tuple[int, int]:
    zero = int(node) - 1
    if zero < 0:
        raise ValueError(f"node id must be positive: {node}")
    return zero // int(row_pitch), zero % int(row_pitch)


def try_place_edges(
    local_edges: list[EdgeRecord],
    *,
    origin_col: int,
    origin_row: int,
    p: int,
    n: int,
    outgoing: dict[tuple[int, int], tuple[int, int]],
    incoming: dict[tuple[int, int], tuple[int, int]],
    allow_clipped_right: bool = True,
) -> tuple[bool, str, list[PlacedEdge]]:
    """Try placing one motif patch at one origin without mutating state."""

    placed: list[PlacedEdge] = []
    seen_sources: set[tuple[int, int]] = set()
    seen_targets: set[tuple[int, int]] = set()

    for edge in local_edges:
        src = (int(origin_col) + int(edge.src_col), int(origin_row) + int(edge.src_row))
        dst = (int(origin_col) + int(edge.dst_col), int(origin_row) + int(edge.dst_row))

        if not (0 <= src[0] < int(p) and 0 <= src[1] < int(n)):
            if allow_clipped_right and src[0] >= int(p) and 0 <= src[1] < int(n):
                continue
            return False, f"source outside grid: {src}", []

        if not (0 <= dst[0] < int(p) and 0 <= dst[1] < int(n)):
            if allow_clipped_right and dst[0] >= int(p) and 0 <= dst[1] < int(n):
                continue
            return False, f"target outside grid: {dst}", []

        if src in seen_sources:
            return False, f"motif source used twice: {src}", []
        if dst in seen_targets:
            return False, f"motif target used twice: {dst}", []
        seen_sources.add(src)
        seen_targets.add(dst)

        old_dst = outgoing.get(src)
        if old_dst is not None and old_dst != dst:
            return False, f"out conflict at {src}: existing {old_dst}, new {dst}", []

        old_src = incoming.get(dst)
        if old_src is not None and old_src != src:
            return False, f"in conflict at {dst}: existing {old_src}, new {src}", []

        placed.append(
            PlacedEdge(
                src_col=src[0],
                src_row=src[1],
                dst_col=dst[0],
                dst_row=dst[1],
                origin_col=int(origin_col),
                origin_row=int(origin_row),
                symbol=str(edge.symbol),
            )
        )

    return True, "ok", placed


def tile_edge_records_on_grid(
    *,
    p: int,
    n: int,
    motif_width: int,
    motif_height: int,
    local_edges: list[EdgeRecord],
    motif: Motif | None = None,
    motif_label: str | None = None,
    horizontal_step: int | None = None,
    allow_vertical_overlap: bool = True,
    allow_clipped_right: bool = True,
) -> TiledMotifResult:
    """Greedily tile local motif edges onto a ``p x n`` grid.

    Candidate origins are tried from bottom to top and left to right. Vertical
    origins are tried every row when overlap is enabled; otherwise they jump by
    ``motif_height``. Horizontal origins jump by ``motif_width - 1`` by default.
    """

    p = int(p)
    n = int(n)
    motif_width = int(motif_width)
    motif_height = int(motif_height)
    if p < 1 or n < 1:
        raise ValueError("p and n must both be positive")
    if motif_width < 2 or motif_height < 1:
        raise ValueError("motif_width must be >= 2 and motif_height must be >= 1")

    if horizontal_step is None:
        horizontal_step = max(1, motif_width - 1)
    vertical_step = 1 if allow_vertical_overlap else max(1, motif_height)

    outgoing: dict[tuple[int, int], tuple[int, int]] = {}
    incoming: dict[tuple[int, int], tuple[int, int]] = {}
    all_edges: dict[tuple[tuple[int, int], tuple[int, int]], PlacedEdge] = {}
    placements: list[PlacementAttempt] = []

    row_origins = range(0, max(0, n - motif_height) + 1, vertical_step)
    col_origins = range(0, p, int(horizontal_step))

    for row0 in row_origins:
        for col0 in col_origins:
            ok, reason, placed = try_place_edges(
                local_edges,
                origin_col=col0,
                origin_row=row0,
                p=p,
                n=n,
                outgoing=outgoing,
                incoming=incoming,
                allow_clipped_right=allow_clipped_right,
            )
            if not ok:
                placements.append(PlacementAttempt(col0, row0, False, reason, 0))
                continue

            added = 0
            for edge in placed:
                src = (edge.src_col, edge.src_row)
                dst = (edge.dst_col, edge.dst_row)
                outgoing[src] = dst
                incoming[dst] = src
                key = (src, dst)
                if key not in all_edges:
                    all_edges[key] = edge
                    added += 1
            placements.append(PlacementAttempt(col0, row0, True, "ok", added))

    empty_motif: Motif = tuple()
    return TiledMotifResult(
        p=p,
        n=n,
        motif=motif if motif is not None else empty_motif,
        motif_width=motif_width,
        motif_height=motif_height,
        placed_edges=sorted(
            all_edges.values(),
            key=lambda item: (item.src_col, item.src_row, item.dst_col, item.dst_row),
        ),
        placements=placements,
        motif_label=str(motif_label or (pretty_motif(motif) if motif else "edge-record motif")),
    )


def normalize_motif_for_tiling(
    motif: Motif | MotifSupport | dict[str, Any],
) -> tuple[int, int, list[EdgeRecord], Motif, str]:
    """Normalize public motif inputs to local edge records for tiling.

    User-facing code should normally pass the support-style dict:

    ``{"w": 3, "h": 3, "support": [(0, 0, "D"), ...]}``

    The symbol-matrix form is kept for compatibility with the exact-box
    enumerator output.
    """

    if isinstance(motif, MotifSupport):
        return (
            motif.w,
            motif.h,
            motif_support_to_edge_records(motif),
            tuple(),
            motif.name or motif_support_label(motif),
        )
    if isinstance(motif, dict):
        motif_support = motif_support_from_dict(motif)
        return (
            motif_support.w,
            motif_support.h,
            motif_support_to_edge_records(motif_support),
            tuple(),
            motif_support.name or motif_support_label(motif_support),
        )

    motif_matrix = motif
    if not motif_matrix:
        raise ValueError("motif must contain at least one planning column")
    motif_height = len(motif_matrix[0])
    if any(len(col) != motif_height for col in motif_matrix):
        raise ValueError("all motif columns must have the same height")
    motif_width = len(motif_matrix) + 1
    return (
        motif_width,
        motif_height,
        motif_edge_records(motif_matrix),
        motif_matrix,
        pretty_motif(motif_matrix),
    )


def tile_motif_on_grid(
    *,
    p: int,
    n: int,
    motif: Motif | MotifSupport | dict[str, Any],
    horizontal_step: int | None = None,
    allow_vertical_overlap: bool = True,
    allow_clipped_right: bool = True,
) -> TiledMotifResult:
    """Tile one motif onto a full ``p x n`` 2D grid.

    Preferred user-facing input is a support-style dict with ``w``, ``h``, and
    ``support``. Matrix motifs from the exact-box enumerator are also accepted.
    """

    motif_width, motif_height, local_edges, motif_matrix, motif_label = normalize_motif_for_tiling(motif)
    return tile_edge_records_on_grid(
        p=p,
        n=n,
        motif_width=motif_width,
        motif_height=motif_height,
        local_edges=local_edges,
        motif=motif_matrix,
        motif_label=motif_label,
        horizontal_step=horizontal_step,
        allow_vertical_overlap=allow_vertical_overlap,
        allow_clipped_right=allow_clipped_right,
    )


def write_tiled_motif_outputs(result: TiledMotifResult, out_dir: str | Path) -> None:
    """Write placed edges, placement attempts, and metadata to CSV/JSON."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "placed_edges.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "src_id",
                "dst_id",
                "src_col",
                "src_row",
                "dst_col",
                "dst_row",
                "origin_col",
                "origin_row",
                "symbol",
            ],
        )
        writer.writeheader()
        for edge in result.placed_edges:
            src_id, dst_id = edge.user_id_pair(result.n)
            writer.writerow(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "src_col": edge.src_col,
                    "src_row": edge.src_row,
                    "dst_col": edge.dst_col,
                    "dst_row": edge.dst_row,
                    "origin_col": edge.origin_col,
                    "origin_row": edge.origin_row,
                    "symbol": edge.symbol,
                }
            )

    with (out_dir / "placements.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["origin_col", "origin_row", "accepted", "reason", "added_edges"],
        )
        writer.writeheader()
        for item in result.placements:
            writer.writerow(asdict(item))

    payload = {
        "p": result.p,
        "n": result.n,
        "motif": result.motif_label or (pretty_motif(result.motif) if result.motif else None),
        "motif_matrix": pretty_motif(result.motif) if result.motif else None,
        "motif_width": result.motif_width,
        "motif_height": result.motif_height,
        "accepted_placements": result.accepted_count,
        "rejected_placements": result.rejected_count,
        "placed_edge_count": result.edge_count,
    }
    (out_dir / "meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_tiled_motif(
    result: TiledMotifResult,
    out_path: str | Path,
    *,
    show_patch_boxes: bool = True,
    show_node_labels: bool | None = None,
    dpi: int = 180,
) -> None:
    """Draw a tiled motif result to a PNG/SVG/PDF path with matplotlib."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if show_node_labels is None:
        show_node_labels = result.p * result.n <= 180

    fig_width = max(7.0, min(24.0, result.p * 0.72))
    fig_height = max(5.0, min(28.0, result.n * 0.34))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect("equal")
    ax.axis("off")

    def xy(col: int, row: int) -> tuple[float, float]:
        return float(col) * 1.0, float(row) * 1.0

    if show_patch_boxes:
        for placement in result.placements:
            if not placement.accepted:
                continue
            x0, y0 = xy(placement.origin_col, placement.origin_row)
            rect = plt.Rectangle(
                (x0 - 0.25, y0 - 0.25),
                (result.motif_width - 1) + 0.5,
                (result.motif_height - 1) + 0.5,
                fill=False,
                edgecolor="#d0d0d0",
                linewidth=0.55,
                linestyle="--",
                zorder=0,
            )
            ax.add_patch(rect)

    colors = {
        "A": "#1f77b4",
        "B": "#7b2cbf",
        "C": "#f77f00",
        "D": "#2a9d8f",
    }
    for edge in result.placed_edges:
        x0, y0 = xy(edge.src_col, edge.src_row)
        x1, y1 = xy(edge.dst_col, edge.dst_row)
        rad = 0.08 if edge.symbol == "D" else 0.0
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=7 if result.p * result.n > 200 else 11,
                linewidth=0.8 if result.p * result.n > 200 else 1.4,
                color=colors.get(edge.symbol, "#222222"),
                alpha=0.88,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=4,
                shrinkB=4,
                zorder=2,
            )
        )

    node_size = 10 if result.p * result.n > 200 else 70
    for col in range(result.p):
        for row in range(result.n):
            x, y = xy(col, row)
            ax.scatter([x], [y], s=node_size, facecolors="white", edgecolors="#333333", linewidths=0.5, zorder=4)
            if show_node_labels:
                ax.text(
                    x,
                    y,
                    str(node_id(col, row, result.n)),
                    ha="center",
                    va="center",
                    fontsize=6,
                    zorder=5,
                )

    ax.set_xlim(-0.8, result.p - 0.2)
    ax.set_ylim(-0.8, result.n - 0.2)
    title_motif = result.motif_label or (pretty_motif(result.motif) if result.motif else "edge-record motif")
    ax.set_title(
        f"tiled motif {title_motif} on p={result.p}, n={result.n} | "
        f"accepted={result.accepted_count}, edges={result.edge_count}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
