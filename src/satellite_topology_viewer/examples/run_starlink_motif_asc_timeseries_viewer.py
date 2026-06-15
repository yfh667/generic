from __future__ import annotations

import argparse
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

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import write_edges_csv
from src.motif_generator.module.config_io import load_yaml_dict
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.module.edge_tables import build_single_motif_edge_table


DEFAULT_CONFIG = (
    GENERIC_ROOT
    / "src"
    / "motif_generator"
    / "examples"
    / "configs"
    / "starlink_72_22_w2_h2_cross_cb.yaml"
)
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
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "views"
    / "Starlink_72_22_1_550"
    / "motif_w2_h2_cross_cb_access_timeseries"
)


STATION_GROUPS = {
    0: {"name": "America", "stations": [0, 1, 2, 3, 4]},
    1: {"name": "Africa", "stations": [5, 6, 7, 8, 9, 10]},
    2: {"name": "China", "stations": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
    3: {"name": "Europe", "stations": [23, 24, 25, 26, 27, 28, 29, 30]},
}
GROUP_COLORS = ["#FF0000", "#00FF00", "#0000FF", "#FFA500"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a Starlink tiled motif with the src 2D topology viewer and access-region groups."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML_FILE)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--include-intra", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=880)
    parser.add_argument("--edge-width", type=float, default=0.045)
    parser.add_argument("--edge-alpha", type=int, default=185)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = load_yaml_dict(args.config)
    grid_raw = raw.get("grid", {}) if isinstance(raw.get("grid", {}), dict) else {}
    tiling_raw = raw.get("tiling", {}) if isinstance(raw.get("tiling", {}), dict) else {}
    motif_raw = raw.get("motif")
    if not isinstance(motif_raw, dict):
        raise ValueError("YAML must contain motif mapping")

    constellation_name = str(
        raw.get("constellation_name")
        or grid_raw.get("constellation_name")
        or grid_raw.get("name")
        or "Starlink_72_22"
    )
    config = ViewerConfig(
        name=constellation_name,
        P=int(grid_raw.get("p", 72)),
        N=int(grid_raw.get("n", 22)),
        station_groups=STATION_GROUPS,
        group_colors=GROUP_COLORS,
    )
    topology_raw = {
        "kind": "single_motif",
        "motif": motif_raw,
        "tiling": {
            "horizontal_step": tiling_raw.get("horizontal_step"),
            "allow_vertical_overlap": bool(tiling_raw.get("allow_vertical_overlap", True)),
            "allow_clipped_right": bool(tiling_raw.get("allow_clipped_right", True)),
        },
        "wrap_planes": bool(tiling_raw.get("wrap_cols", True)),
        "add_intra_ring": bool(args.include_intra),
    }
    edge_table = build_single_motif_edge_table(topology_raw=topology_raw, config=config)

    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty time range")
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
            force=bool(args.force_group_cache),
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(edge_table, out_dir / "viewer_edges.csv")

    src_plane = np.asarray(edge_table.src_plane, dtype=np.int16)
    dst_plane = np.asarray(edge_table.dst_plane, dtype=np.int16)
    seam_count = int(np.count_nonzero(np.abs(src_plane - dst_plane) == int(config.P) - 1))
    print(
        f"[starlink-motif-asc-viewer] steps={len(steps)} range={steps[0]}..{steps[-1]} stride={args.stride} "
        f"edges={edge_table.num_edges} include_intra={bool(args.include_intra)} seam_edges={seam_count} "
        f"group_steps={len(group_data)} out_dir={out_dir}",
        flush=True,
    )

    if args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    motif_name = str(motif_raw.get("name", "motif"))
    viewer = SatelliteTopology2DViewer(
        config,
        steps=steps,
        edge_table=edge_table,
        window_title=f"Starlink 72x22 {motif_name} access-region time-series {steps[0]}..{steps[-1]}s",
        group_data=group_data,
        show_groups=not bool(args.no_groups),
        show_grid_lines=False,
        hide_y_wrap_edges=True,
        topology_edge_color="#000000",
        topology_edge_alpha=int(args.edge_alpha),
        topology_edge_width=float(args.edge_width),
    )
    viewer.edge_width = float(args.edge_width)
    viewer.edge_alpha = int(args.edge_alpha)
    viewer.width_slider.setValue(int(max(8, min(85, round(float(args.edge_width) * 1000)))))
    viewer.alpha_slider.setValue(int(max(25, min(220, int(args.edge_alpha)))))
    viewer.update_step(0)

    screenshot = args.screenshot
    if screenshot is None and args.offscreen:
        screenshot = out_dir / f"starlink_{motif_name}_access_timeseries_t{steps[0]}_{steps[-1]}_stride{args.stride}.png"

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
