from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


GENERIC_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import write_edges_csv
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.satellite_topology_viewer.module.topology_edges import INTRA_OPTION, build_full_option_plus_intra_edges


DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "starlink_72_22_1_550_delta_gridplus_viewer"
DEFAULT_XML_FILE = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "Starlink_72_22_1_550"
    / "satellitesposition"
    / "station_visible_satellites_72_22_1_delta.xml"
)
DEFAULT_GROUP_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "Starlink_72_22_1_550"
    / "satellitesposition"
    / "full_option_edge_delay"
    / "group_data_cache"
)


STATION_GROUPS = {
    0: {"name": "America", "stations": list(range(0, 5))},
    1: {"name": "Africa", "stations": list(range(5, 11))},
    2: {"name": "China", "stations": list(range(11, 23))},
    3: {"name": "Europe", "stations": list(range(23, 31))},
}
GROUP_COLORS = ["#FF0000", "#00FF00", "#0000FF", "#FFA500"]


TOPOLOGY_OPTIONS = {
    "plus_grid": (0,),
    "xgrid": (4,),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Starlink 72x22 Walker-delta topologies with plane wrap enabled."
    )
    parser.add_argument(
        "--topology",
        choices=sorted(TOPOLOGY_OPTIONS),
        default="xgrid",
        help="plus_grid uses option 0 + intra links; xgrid uses option 4 + intra links.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML_FILE)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ViewerConfig(
        name="Starlink_72_22",
        P=72,
        N=22,
        station_groups=STATION_GROUPS,
        group_colors=GROUP_COLORS,
    )
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty step range")

    inter_options = TOPOLOGY_OPTIONS[str(args.topology)]
    edge_table = build_full_option_plus_intra_edges(
        config,
        inter_options=inter_options,
        include_intra=True,
        wrap_planes=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(edge_table, out_dir / f"starlink_delta_{args.topology}_edges.csv")

    option = np.asarray(edge_table.option)
    src_plane = np.asarray(edge_table.src_plane)
    dst_plane = np.asarray(edge_table.dst_plane)
    inter_mask = option != INTRA_OPTION
    seam_mask = inter_mask & (np.abs(src_plane - dst_plane) == int(config.P) - 1)
    print(
        f"[starlink-delta-{args.topology}] P={config.P} N={config.N} "
        f"steps={len(steps)} range={steps[0]}..{steps[-1]} stride={args.stride} "
        f"edges={edge_table.num_edges} inter={int(np.count_nonzero(inter_mask))} "
        f"intra={int(np.count_nonzero(~inter_mask))} seam_inter={int(np.count_nonzero(seam_mask))} "
        f"out_dir={out_dir}",
        flush=True,
    )
    if int(np.count_nonzero(seam_mask)) != int(config.N):
        raise RuntimeError(f"Expected {config.N} seam inter edges, got {int(np.count_nonzero(seam_mask))}")

    if args.check_only:
        return 0

    group_data = {}
    if not bool(args.no_groups):
        group_data = load_or_build_group_data(
            xml_file=args.xml_file,
            group_cache_dir=args.group_cache_dir,
            steps=steps,
            station_groups=config.station_groups,
            total_sats=config.total_sats,
            constellation_name=config.name,
            stride=int(args.stride),
            enabled=True,
            force=False,
        )
        print(f"[starlink-delta-{args.topology}] group_steps={len(group_data)}", flush=True)

    edge_values = np.zeros((1, int(edge_table.num_edges)), dtype=np.float32)
    edge_values[0, seam_mask] = 1.0
    edge_active_mask = np.ones((1, int(edge_table.num_edges)), dtype=bool)

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = SatelliteTopology2DViewer(
        config,
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=edge_active_mask,
        edge_values=edge_values,
        value_min=0.0,
        value_max=1.0,
        edge_value_label="plane_wrap_seam",
        scale_edge_width_by_value=True,
        value_width_min=0.010,
        value_width_max=0.095,
        value_color_mode="red_alpha",
        value_solid_color="#C1121F",
        value_alpha_min=30,
        value_alpha_max=255,
        zero_value_edges_visible=False,
        show_topology_under_edge_values=True,
        window_title=(
            f"Starlink 72x22 Walker-delta {args.topology} "
            f"{steps[0]}..{steps[-1]}s; red = plane 71 -> 0 seam edges"
        ),
        group_data=group_data,
        show_groups=not bool(args.no_groups),
        show_grid_lines=False,
        hide_y_wrap_edges=True,
        topology_edge_color="#000000",
        topology_edge_alpha=130,
        topology_edge_width=0.018,
    )
    screenshot = args.screenshot
    if screenshot is None and args.offscreen:
        screenshot = out_dir / f"starlink_delta_{args.topology}_t{steps[0]}_{steps[-1]}_stride{args.stride}_viewer.png"
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
