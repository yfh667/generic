from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_g60_motif_gridplus_shortest_path_timeseries import build_support_motif_edge_table
from src.config.viewer_config import G60_CONFIG
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data


DEFAULT_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE = THIS_DIR / "cache" / "group_data_cache"
DEFAULT_MOTIF_CONFIG = GENERIC_ROOT / "src" / "motif_generator" / "examples" / "configs" / "dad_cxx_support.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open dynamic G60 2D viewer for the support-motif topology.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86164, help="Inclusive end step.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--initial-step", type=int, default=86100)
    parser.add_argument("--motif-config", type=Path, default=DEFAULT_MOTIF_CONFIG)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--edge-width", type=float, default=0.045)
    parser.add_argument("--edge-alpha", type=int, default=170)
    parser.add_argument("--hide-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty step range")

    edge_table = build_support_motif_edge_table(
        motif_config=Path(args.motif_config),
        p=int(G60_CONFIG.P),
        n=int(G60_CONFIG.N),
    )
    group_data = load_or_build_group_data(
        xml_file=Path(args.xml_file),
        group_cache_dir=Path(args.group_cache_dir),
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=not bool(args.hide_groups),
        force=bool(args.force_group_cache),
    )

    print(
        f"[support-motif-viewer] steps={len(steps)} range={steps[0]}..{steps[-1]} "
        f"edges={edge_table.num_edges} group_steps={len(group_data)}",
        flush=True,
    )
    if args.check_only:
        return 0

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        window_title=f"G60 support motif DAD_Cxx dynamic topology {steps[0]}..{steps[-1]}s",
        group_data=group_data,
        show_groups=not bool(args.hide_groups),
    )
    viewer.edge_width = float(args.edge_width)
    viewer.edge_alpha = int(args.edge_alpha)
    viewer.width_slider.setValue(int(max(8, min(50, round(float(args.edge_width) * 1000)))))
    viewer.alpha_slider.setValue(int(max(25, min(190, int(args.edge_alpha)))))

    if int(args.initial_step) in steps:
        row = steps.index(int(args.initial_step))
        viewer.slider.setValue(row)
        viewer.update_step(row)
    else:
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
