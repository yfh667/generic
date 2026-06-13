from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from topology_edges import build_full_option_plus_intra_edges
from src.config.viewer_config import G60_CONFIG
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data


DEFAULT_STORE_DIR = THIS_DIR / "outputs" / "g60_group_betweenness_t0_86164"
DEFAULT_XML = PROJECT_ROOT / "data" / "basic_file" / "G60" / "satellitesposition" / "station_visible_satellites_20250106.xml"
DEFAULT_GROUP_CACHE = THIS_DIR / "cache" / "group_data_cache"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the full 0..86164s G60 China-Europe edge-betweenness GUI."
    )
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--no-groups", action="store_true")
    return parser.parse_args(argv)


def load_meta(store_dir: Path) -> dict:
    meta_path = store_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store_dir = Path(args.store_dir)
    delay_like_path = store_dir / "edge_betweenness.npy"
    steps_path = store_dir / "time_indices.npy"
    if not delay_like_path.exists():
        raise FileNotFoundError(f"Missing edge_betweenness.npy: {delay_like_path}")
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing time_indices.npy: {steps_path}")

    meta = load_meta(store_dir)
    steps = [int(x) for x in np.load(steps_path)]
    values = np.load(delay_like_path, mmap_mode="r")
    edge_table = build_full_option_plus_intra_edges(G60_CONFIG)
    if tuple(values.shape) != (len(steps), int(edge_table.num_edges)):
        raise ValueError(
            f"edge_betweenness shape {values.shape} != expected {(len(steps), int(edge_table.num_edges))}"
        )

    groups_enabled = not bool(args.no_groups)
    group_data = {}
    if groups_enabled:
        group_data = load_or_build_group_data(
            xml_file=args.xml_file,
            group_cache_dir=args.group_cache_dir,
            steps=steps,
            station_groups=G60_CONFIG.station_groups,
            total_sats=G60_CONFIG.total_sats,
            constellation_name=G60_CONFIG.name,
            stride=1,
            enabled=True,
            force=False,
        )

    value_min = float(meta.get("value_min", 0.0))
    value_max = float(meta.get("value_max", np.nanmax(values)))
    print(
        f"[full-betweenness-viewer] steps={len(steps)} edges={edge_table.num_edges} "
        f"values=({value_min:.4f}, {value_max:.4f}) group_steps={len(group_data)} store={store_dir}",
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
        edge_values=values,
        value_min=value_min,
        value_max=value_max,
        edge_value_label="edge_betweenness",
        scale_edge_width_by_value=True,
        value_width_min=0.006,
        value_width_max=0.085,
        value_color_mode="red_alpha",
        value_solid_color="#C1121F",
        value_alpha_min=28,
        value_alpha_max=230,
        zero_value_edges_visible=False,
        zero_value_threshold=0.0,
        show_topology_under_edge_values=True,
        topology_edge_color="#000000",
        topology_edge_alpha=155,
        topology_edge_width=0.014,
        window_title="G60 China-Europe edge betweenness 0..86164s",
        group_data=group_data,
        show_groups=groups_enabled,
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
