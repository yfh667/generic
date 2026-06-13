from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = THIS_DIR / "outputs" / "tile_motif_to_grid_demo"


@dataclass(frozen=True)
class DirectedEdge:
    src_col: int
    src_row: int
    dst_col: int
    dst_row: int


@dataclass(frozen=True)
class PlacedEdge:
    src_col: int
    src_row: int
    dst_col: int
    dst_row: int
    origin_col: int
    origin_row: int


@dataclass(frozen=True)
class Placement:
    origin_col: int
    origin_row: int
    accepted: bool
    reason: str
    added_edges: int


def node_id(col: int, row: int, rows: int) -> int:
    return int(col) * int(rows) + int(row) + 1


def node_coord(node: int, rows: int) -> tuple[int, int]:
    zero = int(node) - 1
    if zero < 0:
        raise ValueError(f"node id must be positive: {node}")
    return zero // int(rows), zero % int(rows)


def parse_edge_ids(text: str) -> list[tuple[int, int]]:
    edges = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"edge must use 'u-v' format: {chunk!r}")
        a, b = chunk.split("-", 1)
        edges.append((int(a.strip()), int(b.strip())))
    return edges


def local_edges_from_global_ids(
    edge_ids: list[tuple[int, int]],
    *,
    full_rows: int,
    motif_width: int,
    motif_height: int,
    origin_col: int | None = None,
    origin_row: int | None = None,
    auto_orient: bool = True,
) -> tuple[list[DirectedEdge], tuple[int, int]]:
    coords = []
    for u, v in edge_ids:
        uc, ur = node_coord(u, full_rows)
        vc, vr = node_coord(v, full_rows)
        if auto_orient and vc < uc:
            uc, ur, vc, vr = vc, vr, uc, ur
        coords.append((uc, ur, vc, vr))

    if origin_col is None:
        origin_col = min(min(uc, vc) for uc, _, vc, _ in coords)
    if origin_row is None:
        origin_row = min(min(ur, vr) for _, ur, _, vr in coords)

    local = []
    for uc, ur, vc, vr in coords:
        edge = DirectedEdge(
            src_col=uc - origin_col,
            src_row=ur - origin_row,
            dst_col=vc - origin_col,
            dst_row=vr - origin_row,
        )
        if edge.dst_col <= edge.src_col:
            raise ValueError(f"motif edge must point to the right: {edge}")
        for col, row in ((edge.src_col, edge.src_row), (edge.dst_col, edge.dst_row)):
            if not (0 <= col < motif_width and 0 <= row < motif_height):
                raise ValueError(
                    f"edge {edge} is outside motif box width={motif_width}, height={motif_height}"
                )
        local.append(edge)
    return local, (origin_col, origin_row)


def try_place(
    local_edges: list[DirectedEdge],
    *,
    origin_col: int,
    origin_row: int,
    full_cols: int,
    full_rows: int,
    outgoing: dict[tuple[int, int], tuple[int, int]],
    incoming: dict[tuple[int, int], tuple[int, int]],
    allow_clipped_right: bool = True,
) -> tuple[bool, str, list[PlacedEdge]]:
    placed = []
    seen_local_sources = set()
    seen_local_targets = set()

    for edge in local_edges:
        src = (origin_col + edge.src_col, origin_row + edge.src_row)
        dst = (origin_col + edge.dst_col, origin_row + edge.dst_row)

        if not (0 <= src[0] < full_cols and 0 <= src[1] < full_rows):
            if allow_clipped_right and src[0] >= full_cols and 0 <= src[1] < full_rows:
                continue
            return False, f"source outside grid: {src}", []

        if not (0 <= dst[0] < full_cols and 0 <= dst[1] < full_rows):
            if allow_clipped_right and dst[0] >= full_cols and 0 <= dst[1] < full_rows:
                continue
            return False, f"target outside grid: {dst}", []

        if src in seen_local_sources and outgoing.get(src, dst) != dst:
            return False, f"motif source used twice: {src}", []
        if dst in seen_local_targets and incoming.get(dst, src) != src:
            return False, f"motif target used twice: {dst}", []
        seen_local_sources.add(src)
        seen_local_targets.add(dst)

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
                origin_col=origin_col,
                origin_row=origin_row,
            )
        )

    return True, "ok", placed


def tile_motif_greedy(
    *,
    full_cols: int,
    full_rows: int,
    motif_width: int,
    motif_height: int,
    local_edges: list[DirectedEdge],
    horizontal_step: int | None = None,
    allow_vertical_overlap: bool = True,
    allow_clipped_right: bool = True,
) -> tuple[list[PlacedEdge], list[Placement]]:
    """Greedily place motif patches over the full grid.

    Candidate origins are tried from bottom to top and left to right.
    Vertical origins use every row when overlap is enabled; otherwise they jump
    by motif_height. Horizontal origins jump by motif_width-1 by default,
    matching the self-contained-box tiling rule.
    """

    if horizontal_step is None:
        horizontal_step = max(1, motif_width - 1)
    vertical_step = 1 if allow_vertical_overlap else max(1, motif_height)

    outgoing: dict[tuple[int, int], tuple[int, int]] = {}
    incoming: dict[tuple[int, int], tuple[int, int]] = {}
    all_edges: dict[tuple[tuple[int, int], tuple[int, int]], PlacedEdge] = {}
    placements: list[Placement] = []

    row_origins = range(0, max(0, full_rows - motif_height) + 1, vertical_step)
    col_origins = range(0, full_cols, horizontal_step)

    for row0 in row_origins:
        for col0 in col_origins:
            ok, reason, placed = try_place(
                local_edges,
                origin_col=col0,
                origin_row=row0,
                full_cols=full_cols,
                full_rows=full_rows,
                outgoing=outgoing,
                incoming=incoming,
                allow_clipped_right=allow_clipped_right,
            )
            if not ok:
                placements.append(Placement(col0, row0, False, reason, 0))
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
            placements.append(Placement(col0, row0, True, "ok", added))

    return list(all_edges.values()), placements


def write_outputs(
    *,
    out_dir: Path,
    full_cols: int,
    full_rows: int,
    motif_width: int,
    motif_height: int,
    local_edges: list[DirectedEdge],
    placed_edges: list[PlacedEdge],
    placements: list[Placement],
) -> None:
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
            ],
        )
        writer.writeheader()
        for edge in placed_edges:
            writer.writerow(
                {
                    "src_id": node_id(edge.src_col, edge.src_row, full_rows),
                    "dst_id": node_id(edge.dst_col, edge.dst_row, full_rows),
                    "src_col": edge.src_col,
                    "src_row": edge.src_row,
                    "dst_col": edge.dst_col,
                    "dst_row": edge.dst_row,
                    "origin_col": edge.origin_col,
                    "origin_row": edge.origin_row,
                }
            )

    with (out_dir / "placements.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["origin_col", "origin_row", "accepted", "reason", "added_edges"],
        )
        writer.writeheader()
        for item in placements:
            writer.writerow(item.__dict__)

    payload = {
        "full_cols": full_cols,
        "full_rows": full_rows,
        "motif_width": motif_width,
        "motif_height": motif_height,
        "local_edges": [edge.__dict__ for edge in local_edges],
        "accepted_placements": sum(1 for item in placements if item.accepted),
        "rejected_placements": sum(1 for item in placements if not item.accepted),
        "placed_edge_count": len(placed_edges),
    }
    (out_dir / "meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_full_grid(
    *,
    out_path: Path,
    full_cols: int,
    full_rows: int,
    motif_width: int,
    motif_height: int,
    placed_edges: list[PlacedEdge],
    placements: list[Placement],
) -> None:
    fig, ax = plt.subplots(figsize=(max(7, full_cols * 1.4), max(5, full_rows * 0.9)))
    ax.set_aspect("equal")
    ax.axis("off")

    def xy(col: int, row: int) -> tuple[float, float]:
        return float(col) * 1.45, float(row) * 1.05

    for placement in placements:
        if not placement.accepted:
            continue
        x0, y0 = xy(placement.origin_col, placement.origin_row)
        width = (motif_width - 1) * 1.45
        height = (motif_height - 1) * 1.05
        rect = plt.Rectangle(
            (x0 - 0.33, y0 - 0.33),
            width + 0.66,
            height + 0.66,
            fill=False,
            edgecolor="#d0d0d0",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )
        ax.add_patch(rect)

    for edge in placed_edges:
        x0, y0 = xy(edge.src_col, edge.src_row)
        x1, y1 = xy(edge.dst_col, edge.dst_row)
        dy = edge.dst_row - edge.src_row
        color = "#1f77b4" if dy == 0 else ("#7b2cbf" if dy < 0 else "#f77f00")
        rad = 0.08 if edge.dst_col - edge.src_col > 1 else 0.0
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.5,
                color=color,
                alpha=0.9,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=9,
                shrinkB=9,
                zorder=2,
            )
        )

    for col in range(full_cols):
        for row in range(full_rows):
            x, y = xy(col, row)
            ax.scatter([x], [y], s=82, facecolors="white", edgecolors="#222222", linewidths=1.0, zorder=4)
            ax.text(x, y, str(node_id(col, row, full_rows)), ha="center", va="center", fontsize=9, zorder=5)

    ax.set_xlim(-0.9, (full_cols - 1) * 1.45 + 0.9)
    ax.set_ylim(-0.8, (full_rows - 1) * 1.05 + 0.8)
    ax.set_title(
        f"Tiled motif on {full_cols}x{full_rows} grid | "
        f"accepted patches={sum(1 for p in placements if p.accepted)} | edges={len(placed_edges)}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile a motif edge list onto a 2D satellite grid.")
    parser.add_argument("--full-cols", type=int, default=3)
    parser.add_argument("--full-rows", type=int, default=4)
    parser.add_argument("--motif-width", type=int, default=3)
    parser.add_argument("--motif-height", type=int, default=2)
    parser.add_argument("--motif-edges", default="3-7,4-8,8-11,7-12")
    parser.add_argument("--motif-origin-col", type=int, default=None)
    parser.add_argument("--motif-origin-row", type=int, default=None)
    parser.add_argument("--horizontal-step", type=int, default=None)
    parser.add_argument("--no-vertical-overlap", action="store_true")
    parser.add_argument("--no-clipped-right", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    edge_ids = parse_edge_ids(args.motif_edges)
    local_edges, inferred_origin = local_edges_from_global_ids(
        edge_ids,
        full_rows=args.full_rows,
        motif_width=args.motif_width,
        motif_height=args.motif_height,
        origin_col=args.motif_origin_col,
        origin_row=args.motif_origin_row,
    )
    placed_edges, placements = tile_motif_greedy(
        full_cols=args.full_cols,
        full_rows=args.full_rows,
        motif_width=args.motif_width,
        motif_height=args.motif_height,
        local_edges=local_edges,
        horizontal_step=args.horizontal_step,
        allow_vertical_overlap=not args.no_vertical_overlap,
        allow_clipped_right=not args.no_clipped_right,
    )
    out_dir = Path(args.out_dir)
    write_outputs(
        out_dir=out_dir,
        full_cols=args.full_cols,
        full_rows=args.full_rows,
        motif_width=args.motif_width,
        motif_height=args.motif_height,
        local_edges=local_edges,
        placed_edges=placed_edges,
        placements=placements,
    )
    png_path = out_dir / "tiled_motif.png"
    draw_full_grid(
        out_path=png_path,
        full_cols=args.full_cols,
        full_rows=args.full_rows,
        motif_width=args.motif_width,
        motif_height=args.motif_height,
        placed_edges=placed_edges,
        placements=placements,
    )
    print(f"[tile-motif] motif_edges={args.motif_edges}")
    print(f"[tile-motif] inferred_origin={inferred_origin}")
    print(f"[tile-motif] accepted={sum(1 for item in placements if item.accepted)}")
    print(f"[tile-motif] rejected={sum(1 for item in placements if not item.accepted)}")
    print(f"[tile-motif] placed_edges={len(placed_edges)}")
    print(f"[tile-motif] wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
