from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


GENERIC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from g60_paper2_config import (
    DEFAULT_GROUP_CACHE_DIR,
    DEFAULT_MAPPING_DIR,
    DEFAULT_SCREENSHOT_DIR,
    DEFAULT_XML,
    build_paper2_g60_config,
    write_station_group_mapping,
)
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.satellite_topology_viewer.module.topology_edges import build_full_option_plus_intra_edges


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show Paper2 G60 station-visibility region groups in the reusable 2D topology viewer."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--mapping-dir", type=Path, default=DEFAULT_MAPPING_DIR)
    parser.add_argument("--width", type=int, default=1300)
    parser.add_argument("--height", type=int, default=820)
    parser.add_argument("--show-reference-links", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    if int(args.end) < int(args.start):
        raise ValueError("--end must be >= --start")

    config = build_paper2_g60_config()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    write_station_group_mapping(out_dir=Path(args.mapping_dir), config=config)

    group_data = load_or_build_group_data(
        xml_file=Path(args.xml_file),
        group_cache_dir=Path(args.group_cache_dir),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )

    edge_table = build_full_option_plus_intra_edges(config)
    if args.show_reference_links:
        edge_active_mask = np.ones((1, edge_table.num_edges), dtype=bool)
    else:
        edge_active_mask = np.zeros((1, edge_table.num_edges), dtype=bool)

    print(
        f"[paper2-g60-viewer] steps={len(steps)} range={steps[0]}..{steps[-1]} stride={args.stride} "
        f"groups={len(config.station_groups)} group_steps={len(group_data)} "
        f"station_mapping={Path(args.mapping_dir)}",
        flush=True,
    )
    for gid, info in config.station_groups.items():
        stations = [int(x) + 1 for x in info["stations"]]
        print(
            f"[paper2-g60-viewer] group_id={gid} name={info['name']} "
            f"stations_1based={stations[0]}..{stations[-1]} count={len(stations)}",
            flush=True,
        )

    if args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    screenshot = args.screenshot
    if screenshot is None and args.offscreen:
        DEFAULT_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        screenshot = DEFAULT_SCREENSHOT_DIR / f"paper2_g60_groups_t{steps[0]}_{steps[-1]}_stride{args.stride}.png"

    viewer = SatelliteTopology2DViewer(
        config,
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=edge_active_mask,
        window_title=f"Paper2 G60 station VOC groups {steps[0]}..{steps[-1]}s",
        group_data=group_data,
        show_groups=True,
        topology_edge_color="#111111",
        topology_edge_alpha=110,
        topology_edge_width=0.012,
    )
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
