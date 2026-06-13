from __future__ import annotations

import argparse
import csv
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

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data


DEFAULT_STORE_DIR = (
    PROJECT_ROOT
    / "data"
    / "linshi"
    / "g60_multi_region_weighted_betweenness_t0_86164_stride60"
)
DEFAULT_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE = THIS_DIR / "cache" / "group_data_cache"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open G60 multi-region weighted shortest-delay edge-betweenness GUI."
    )
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument(
        "--view",
        default="combined_sum",
        choices=["combined_sum", "combined_max", "china_europe", "china_america", "china_africa"],
    )
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


def read_edge_table(path: Path) -> EdgeTable:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda row: int(row["edge_idx"]))
    total_sats = int(G60_CONFIG.total_sats)
    sat_ids = [str(i + 1) for i in range(total_sats)]
    src_values = []
    dst_values = []
    option_values = []
    src_plane_values = []
    src_y_values = []
    dst_plane_values = []
    dst_y_values = []
    for row in rows:
        src = int(row["src_node"])
        dst = int(row["dst_node"])
        src_values.append(src)
        dst_values.append(dst)
        option_values.append(int(row["option"]))
        src_plane_values.append(int(row["src_plane"]))
        src_y_values.append(int(row["src_y"]))
        dst_plane_values.append(int(row["dst_plane"]))
        dst_y_values.append(int(row["dst_y"]))
        if 0 <= src < total_sats:
            sat_ids[src] = str(row.get("src_sat_id", src + 1))
        if 0 <= dst < total_sats:
            sat_ids[dst] = str(row.get("dst_sat_id", dst + 1))
    return EdgeTable(
        src=np.asarray(src_values, dtype=np.int32),
        dst=np.asarray(dst_values, dtype=np.int32),
        option=np.asarray(option_values, dtype=np.int16),
        src_plane=np.asarray(src_plane_values, dtype=np.int16),
        src_y=np.asarray(src_y_values, dtype=np.int16),
        dst_plane=np.asarray(dst_plane_values, dtype=np.int16),
        dst_y=np.asarray(dst_y_values, dtype=np.int16),
        sat_ids=sat_ids,
    )


def value_path_for_view(store_dir: Path, view: str) -> Path:
    if view == "combined_sum":
        return store_dir / "edge_betweenness_sum.npy"
    if view == "combined_max":
        return store_dir / "edge_betweenness_max.npy"
    return store_dir / view / "edge_betweenness.npy"


def inferred_stride(steps: list[int]) -> int:
    if len(steps) < 2:
        return 1
    diffs = np.diff(np.asarray(steps, dtype=np.int64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1
    return int(np.median(diffs))


def view_label(view: str) -> str:
    return {
        "combined_sum": "China-Europe + China-America + China-Africa",
        "combined_max": "max(China-Europe, China-America, China-Africa)",
        "china_europe": "China-Europe",
        "china_america": "China-America",
        "china_africa": "China-Africa",
    }.get(view, view)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store_dir = Path(args.store_dir)
    value_path = value_path_for_view(store_dir, str(args.view))
    steps_path = store_dir / "time_indices.npy"
    edges_path = store_dir / "edges.csv"
    if not value_path.exists():
        raise FileNotFoundError(f"Missing value matrix for view={args.view}: {value_path}")
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing time_indices.npy: {steps_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges.csv: {edges_path}")

    meta = load_meta(store_dir)
    steps = [int(x) for x in np.load(steps_path)]
    values = np.load(value_path, mmap_mode="r")
    edge_table = read_edge_table(edges_path)
    if tuple(values.shape) != (len(steps), int(edge_table.num_edges)):
        raise ValueError(f"value matrix shape {values.shape} != expected {(len(steps), int(edge_table.num_edges))}")

    if args.view == "combined_sum":
        value_min = float(meta.get("value_min", 0.0))
        value_max = float(meta.get("value_max", np.nanmax(values)))
    elif args.view == "combined_max":
        value_min = float(meta.get("combined_max_value_min", 0.0))
        value_max = float(meta.get("combined_max_value_max", np.nanmax(values)))
    else:
        value_min = 0.0
        value_max = float(np.nanmax(values)) if values.size else 0.0

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
            stride=inferred_stride(steps),
            enabled=True,
            force=False,
        )

    print(
        f"[multi-region-viewer] view={args.view} label={view_label(args.view)!r} "
        f"steps={len(steps)} edges={edge_table.num_edges} values=({value_min:.4f}, {value_max:.4f}) "
        f"group_steps={len(group_data)} store={store_dir}",
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
        edge_value_label="weighted_edge_betweenness",
        scale_edge_width_by_value=True,
        value_width_min=0.006,
        value_width_max=0.095,
        value_color_mode="red_alpha",
        value_solid_color="#C1121F",
        value_alpha_min=28,
        value_alpha_max=235,
        zero_value_edges_visible=False,
        zero_value_threshold=0.0,
        show_topology_under_edge_values=True,
        topology_edge_color="#000000",
        topology_edge_alpha=155,
        topology_edge_width=0.014,
        window_title=f"G60 weighted shortest-delay edge betweenness: {view_label(args.view)}",
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
