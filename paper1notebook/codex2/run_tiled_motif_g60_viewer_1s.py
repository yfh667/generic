from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer

from tile_motif_to_grid import (
    PlacedEdge,
    local_edges_from_global_ids,
    parse_edge_ids,
    tile_motif_greedy,
    write_outputs as write_tiling_outputs,
)


DEFAULT_OUT_DIR = THIS_DIR / "outputs" / "g60_tiled_motif_viewer_1s"


def option_from_delta(dp: int, dy: int) -> int:
    if (int(dp), int(dy)) == (1, 0):
        return 0
    if (int(dp), int(dy)) == (1, -1):
        return 1
    if (int(dp), int(dy)) == (2, 0):
        return 2
    if (int(dp), int(dy)) == (1, 1):
        return 4
    return 99


def edge_table_from_placed_edges(placed_edges: list[PlacedEdge], *, rows: int, cols: int) -> EdgeTable:
    src: list[int] = []
    dst: list[int] = []
    option: list[int] = []
    src_plane: list[int] = []
    src_y: list[int] = []
    dst_plane: list[int] = []
    dst_y: list[int] = []

    for edge in sorted(
        placed_edges,
        key=lambda item: (item.src_col, item.src_row, item.dst_col, item.dst_row),
    ):
        u = int(edge.src_col) * int(rows) + int(edge.src_row)
        v = int(edge.dst_col) * int(rows) + int(edge.dst_row)
        dp = int(edge.dst_col) - int(edge.src_col)
        dy = int(edge.dst_row) - int(edge.src_row)
        src.append(u)
        dst.append(v)
        option.append(option_from_delta(dp, dy))
        src_plane.append(int(edge.src_col))
        src_y.append(int(edge.src_row))
        dst_plane.append(int(edge.dst_col))
        dst_y.append(int(edge.dst_row))

    total = int(cols) * int(rows)
    return EdgeTable(
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        option=np.asarray(option, dtype=np.int16),
        src_plane=np.asarray(src_plane, dtype=np.int16),
        src_y=np.asarray(src_y, dtype=np.int16),
        dst_plane=np.asarray(dst_plane, dtype=np.int16),
        dst_y=np.asarray(dst_y, dtype=np.int16),
        sat_ids=[str(i + 1) for i in range(total)],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile one motif onto G60 P=18,N=36 and view it in the 2D topology GUI.")
    parser.add_argument("--full-cols", type=int, default=18, help="P / plane count.")
    parser.add_argument("--full-rows", type=int, default=36, help="N / y phase count.")
    parser.add_argument("--motif-width", type=int, default=3)
    parser.add_argument("--motif-height", type=int, default=2)
    parser.add_argument(
        "--motif-edges",
        default="1-37,2-38,38-73,37-74",
        help=(
            "Motif edge list in full-grid node IDs. Default is the G60-N=36 "
            "version of the 3x4 sketch edge pattern."
        ),
    )
    parser.add_argument("--motif-origin-col", type=int, default=None)
    parser.add_argument("--motif-origin-row", type=int, default=None)
    parser.add_argument("--horizontal-step", type=int, default=None)
    parser.add_argument("--no-vertical-overlap", action="store_true")
    parser.add_argument("--no-clipped-right", action="store_true")
    parser.add_argument("--step", type=int, default=1, help="Single viewer time step, default 1s.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=860)
    parser.add_argument("--edge-width", type=float, default=0.045)
    parser.add_argument("--edge-alpha", type=int, default=190)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.full_cols) != int(G60_CONFIG.P) or int(args.full_rows) != int(G60_CONFIG.N):
        print(
            f"[tiled-motif-viewer] warning: viewer config is G60 P={G60_CONFIG.P},N={G60_CONFIG.N}; "
            f"requested full_cols={args.full_cols}, full_rows={args.full_rows}",
            flush=True,
        )

    edge_ids = parse_edge_ids(args.motif_edges)
    local_edges, inferred_origin = local_edges_from_global_ids(
        edge_ids,
        full_rows=int(args.full_rows),
        motif_width=int(args.motif_width),
        motif_height=int(args.motif_height),
        origin_col=args.motif_origin_col,
        origin_row=args.motif_origin_row,
    )
    placed_edges, placements = tile_motif_greedy(
        full_cols=int(args.full_cols),
        full_rows=int(args.full_rows),
        motif_width=int(args.motif_width),
        motif_height=int(args.motif_height),
        local_edges=local_edges,
        horizontal_step=args.horizontal_step,
        allow_vertical_overlap=not bool(args.no_vertical_overlap),
        allow_clipped_right=not bool(args.no_clipped_right),
    )

    out_dir = Path(args.out_dir)
    write_tiling_outputs(
        out_dir=out_dir,
        full_cols=int(args.full_cols),
        full_rows=int(args.full_rows),
        motif_width=int(args.motif_width),
        motif_height=int(args.motif_height),
        local_edges=local_edges,
        placed_edges=placed_edges,
        placements=placements,
    )
    edge_table = edge_table_from_placed_edges(
        placed_edges,
        rows=int(args.full_rows),
        cols=int(args.full_cols),
    )
    write_edges_csv(edge_table, out_dir / "viewer_edges.csv")

    print(f"[tiled-motif-viewer] motif_edges={args.motif_edges}", flush=True)
    print(f"[tiled-motif-viewer] inferred_origin={inferred_origin}", flush=True)
    print(
        f"[tiled-motif-viewer] accepted={sum(1 for item in placements if item.accepted)} "
        f"rejected={sum(1 for item in placements if not item.accepted)} "
        f"viewer_edges={edge_table.num_edges} out_dir={out_dir}",
        flush=True,
    )

    if args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=[int(args.step)],
        edge_table=edge_table,
        window_title=f"G60 tiled motif topology at {int(args.step)}s",
        group_data={},
        show_groups=False,
    )
    viewer.edge_width = float(args.edge_width)
    viewer.edge_alpha = int(args.edge_alpha)
    viewer.width_slider.setValue(int(max(8, min(50, round(float(args.edge_width) * 1000)))))
    viewer.alpha_slider.setValue(int(max(25, min(190, int(args.edge_alpha)))))
    viewer.update_step(0)
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
