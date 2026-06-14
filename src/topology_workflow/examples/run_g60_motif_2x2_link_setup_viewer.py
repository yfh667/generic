from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.module import (
    build_dynamic_region_internal_option_constraint_series,
    build_single_motif_edge_table,
    write_dynamic_region_constraint_series,
)
from src.topology_workflow.module.config import group_name


MOTIF_2X2_CB = {
    "name": "motif_2x2_CB",
    "w": 2,
    "h": 2,
    "offsets": {
        "A": [1, 0],
        "B": [1, -1],
        "C": [1, 1],
        "D": [2, 0],
    },
    "support": [
        [0, 0, "C"],
        [0, 1, "B"],
    ],
}

DEFAULT_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE = PROJECT_ROOT / "data" / "satnet_experiments" / "caches" / "G60" / "group_data_cache"
DEFAULT_OUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "runs"
    / "paper1"
    / "G60"
    / "link_setup_time"
    / "motif_2x2_CB_china_europe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a 2x2 motif with region-internal +grid and link-setup-time building edges."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--link-setup-time", type=float, default=30.0, help="LST in seconds.")
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--constrained-groups", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--grid-option", type=int, default=0)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true", help="Load existing output masks instead of recomputing.")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument(
        "--skip-edge-state-csv",
        action="store_true",
        help="Do not write per-step per-edge active/building CSV rows. Keep this on for long full-day runs.",
    )
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def build_base_edge_table():
    return build_single_motif_edge_table(
        topology_raw={
            "kind": "single_motif",
            "motif": MOTIF_2X2_CB,
            "tiling": {
                "allow_vertical_overlap": True,
                "allow_clipped_right": True,
            },
            "add_intra_ring": True,
        },
        config=G60_CONFIG,
    )


def series_files_exist(out_dir: Path) -> bool:
    required = [
        out_dir / "steps.npy",
        out_dir / "edge_active_mask.npy",
        out_dir / "edge_building_mask.npy",
        out_dir / "union_edges.csv",
    ]
    return all(path.exists() for path in required)


def load_edge_table_from_csv(path: Path) -> EdgeTable:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f))
    sat_ids = [str(i + 1) for i in range(int(G60_CONFIG.total_sats))]
    return EdgeTable(
        src=np.asarray([int(row["src_node"]) for row in rows], dtype=np.int32),
        dst=np.asarray([int(row["dst_node"]) for row in rows], dtype=np.int32),
        option=np.asarray([int(row["option"]) for row in rows], dtype=np.int16),
        src_plane=np.asarray([int(row["src_plane"]) for row in rows], dtype=np.int16),
        src_y=np.asarray([int(row["src_y"]) for row in rows], dtype=np.int16),
        dst_plane=np.asarray([int(row["dst_plane"]) for row in rows], dtype=np.int16),
        dst_y=np.asarray([int(row["dst_y"]) for row in rows], dtype=np.int16),
        sat_ids=sat_ids,
    )


def load_existing_series_for_viewer(out_dir: Path):
    steps = [int(x) for x in np.load(out_dir / "steps.npy")]
    edge_table = load_edge_table_from_csv(out_dir / "union_edges.csv")
    active_mask = np.load(out_dir / "edge_active_mask.npy")
    building_mask = np.load(out_dir / "edge_building_mask.npy")
    return steps, edge_table, active_mask, building_mask


def main() -> int:
    args = parse_args()
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty time range")

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT / f"t{steps[0]}_{steps[-1]}_stride{int(args.stride)}_lst{float(args.link_setup_time):g}s"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    group_data = load_or_build_group_data(
        xml_file=args.xml_file,
        group_cache_dir=args.group_cache_dir,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )
    constrained_groups = tuple(int(x) for x in args.constrained_groups)
    if args.reuse_existing and series_files_exist(out_dir):
        steps, edge_table, active_mask, building_mask = load_existing_series_for_viewer(out_dir)
        print(
            f"[link-setup-demo] reused existing series steps={len(steps)} "
            f"union_edges={edge_table.num_edges} active_state_rows={int(active_mask.sum())} "
            f"building_state_rows={int(building_mask.sum())} out={out_dir}",
            flush=True,
        )
    else:
        base_edge_table = build_base_edge_table()
        series = build_dynamic_region_internal_option_constraint_series(
            config=G60_CONFIG,
            base_edge_table=base_edge_table,
            group_data=group_data,
            steps=steps,
            constrained_groups=constrained_groups,
            setup_time_seconds=float(args.link_setup_time),
            forced_option=int(args.grid_option),
            group_names={gid: group_name(G60_CONFIG, gid) for gid in constrained_groups},
        )
        write_dynamic_region_constraint_series(
            series,
            out_dir,
            meta={
                "constellation": G60_CONFIG.name,
                "P": int(G60_CONFIG.P),
                "N": int(G60_CONFIG.N),
                "motif": MOTIF_2X2_CB,
                "constrained_groups": list(constrained_groups),
                "forced_option": int(args.grid_option),
                "link_setup_time_seconds": float(args.link_setup_time),
                "rule": (
                    "Target topology uses actual group membership at each step. "
                    "Region-internal +grid links are anticipatory and occupy ports during their LST window. "
                    "Motif links that re-enter the target topology are reactive and remain building until LST elapses. "
                    "Building links are exported separately and drawn as dashed links, but are not active routing links."
                ),
            },
            write_edge_state_csv=not bool(args.skip_edge_state_csv),
        )
        steps = list(series.steps)
        edge_table = series.edge_table
        active_mask = series.edge_active_mask
        building_mask = series.edge_building_mask
        print(
            f"[link-setup-demo] steps={len(steps)} union_edges={edge_table.num_edges} "
            f"active_state_rows={int(active_mask.sum())} building_state_rows={int(building_mask.sum())} out={out_dir}",
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
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=active_mask,
        edge_building_mask=building_mask,
        window_title=(
            f"G60 {MOTIF_2X2_CB['name']} China/Europe +grid LST={float(args.link_setup_time):g}s "
            f"{steps[0]}..{steps[-1]}"
        ),
        group_data=group_data,
        show_groups=True,
        topology_edge_color="#000000",
        topology_edge_alpha=170,
        topology_edge_width=0.022,
        building_edge_color="#2F6FED",
        building_edge_alpha=175,
        building_edge_width=0.028,
        show_grid_lines=False,
    )
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
