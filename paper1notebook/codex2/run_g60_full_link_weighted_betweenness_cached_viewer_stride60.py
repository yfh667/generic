from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.edge_usage_viewer import EdgeUsageTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.multi_viewer import (  # noqa: E402
    Topology2DPanel,
    UnifiedControlTopology2DViewer,
)
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402


DEFAULT_CACHE_DIR = Path(r"E:\paper11\data\linshi\g60_multi_region_weighted_betweenness_t0_86164_stride60")
DEFAULT_GROUP_XML = PROJECT_ROOT / "data" / "basic_file" / "G60" / "satellitesposition" / "station_visible_satellites_20250106.xml"
DEFAULT_GROUP_CACHE_DIR = PROJECT_ROOT / "data" / "satnet_experiments" / "caches" / "G60" / "group_data_cache"

PAIRS = {
    "china_europe": "China-Europe",
    "china_america": "China-America",
    "china_africa": "China-Africa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cached interactive 2D viewer for precomputed G60 full-link weighted edge betweenness."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--group-xml", type=Path, default=DEFAULT_GROUP_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--width", type=int, default=1900)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--no-shared-value-scale", action="store_true")
    return parser.parse_args()


def read_edge_table_csv(path: Path, *, total_sats: int) -> EdgeTable:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty edge CSV: {path}")
    return EdgeTable(
        src=np.asarray([int(row["src_node"]) for row in rows], dtype=np.int32),
        dst=np.asarray([int(row["dst_node"]) for row in rows], dtype=np.int32),
        option=np.asarray([int(row["option"]) for row in rows], dtype=np.int16),
        src_plane=np.asarray([int(row["src_plane"]) for row in rows], dtype=np.int16),
        src_y=np.asarray([int(row["src_y"]) for row in rows], dtype=np.int16),
        dst_plane=np.asarray([int(row["dst_plane"]) for row in rows], dtype=np.int16),
        dst_y=np.asarray([int(row["dst_y"]) for row in rows], dtype=np.int16),
        sat_ids=[str(i + 1) for i in range(int(total_sats))],
    )


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    steps = [int(x) for x in np.asarray(np.load(cache_dir / "time_indices.npy"), dtype=np.int64)]
    edge_table = read_edge_table_csv(cache_dir / "edges.csv", total_sats=int(G60_CONFIG.total_sats))
    values_by_pair = {
        pair_key: np.asarray(np.load(cache_dir / pair_key / "edge_betweenness.npy", mmap_mode="r"), dtype=np.float32)
        for pair_key in PAIRS
    }
    for pair_key, values in values_by_pair.items():
        if values.shape != (len(steps), int(edge_table.num_edges)):
            raise ValueError(
                f"{pair_key} values shape {values.shape} != "
                f"({len(steps)}, {edge_table.num_edges})"
            )

    value_max = max(1.0, *(float(np.nanmax(values)) for values in values_by_pair.values()))
    group_data = load_or_build_group_data(
        xml_file=Path(args.group_xml),
        group_cache_dir=Path(args.group_cache_dir),
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=60 if len(steps) > 1 and steps[1] - steps[0] == 60 else 1,
        enabled=True,
        force=False,
    )

    print(
        f"[cached-full-link-betweenness] steps={len(steps)} range={steps[0]}..{steps[-1]} "
        f"edges={edge_table.num_edges} value_max={value_max:.1f} cache={cache_dir}",
        flush=True,
    )
    for pair_key, label in PAIRS.items():
        values = values_by_pair[pair_key]
        print(
            f"[cached-full-link-betweenness] {label}: "
            f"shape={values.shape} max={float(np.nanmax(values)):.1f} "
            f"mean_nonzero_edges={float(np.mean(np.count_nonzero(values > 0.0, axis=1))):.1f}",
            flush=True,
        )
    if bool(args.check_only):
        return 0

    if bool(args.offscreen):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    active_mask = np.ones((1, int(edge_table.num_edges)), dtype=bool)
    panels: list[Topology2DPanel] = []
    for pair_key, label in PAIRS.items():
        viewer = EdgeUsageTopology2DViewer(
            G60_CONFIG,
            steps=steps,
            edge_table=edge_table,
            edge_usage_values=values_by_pair[pair_key],
            value_max=value_max,
            edge_active_mask=active_mask,
            window_title=f"G60 full-link weighted betweenness cached {label} {steps[0]}..{steps[-1]}s",
            group_data=group_data,
            show_groups=True,
            topology_edge_alpha=95,
            topology_edge_width=0.010,
            value_width_min=0.010,
            value_width_max=0.120,
        )
        panels.append(Topology2DPanel(title=label, viewer=viewer, stretch=1))

    combined = UnifiedControlTopology2DViewer(
        title=f"G60 full-link weighted shortest-delay edge betweenness cached {steps[0]}..{steps[-1]}s",
        panels=panels,
        shared_value_scale=not bool(args.no_shared_value_scale),
        orientation=QtCore.Qt.Horizontal,
    )
    return run_viewer_widget(
        combined,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
