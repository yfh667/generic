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
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\full_link_weighted_betweenness_three_pairs_t0_2d"
)
DEFAULT_GROUP_XML = PROJECT_ROOT / "data" / "basic_file" / "G60" / "satellitesposition" / "station_visible_satellites_20250106.xml"
DEFAULT_GROUP_CACHE_DIR = PROJECT_ROOT / "data" / "satnet_experiments" / "caches" / "G60" / "group_data_cache"

PAIRS = {
    "china_europe": "China-Europe",
    "china_america": "China-America",
    "china_africa": "China-Africa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw G60 full-link weighted shortest-delay edge betweenness at t=0 with the 2D topology viewer."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--group-xml", type=Path, default=DEFAULT_GROUP_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--width", type=int, default=1900)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def read_edge_table_csv(path: Path, *, total_sats: int) -> EdgeTable:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty edge CSV: {path}")

    src = np.asarray([int(row["src_node"]) for row in rows], dtype=np.int32)
    dst = np.asarray([int(row["dst_node"]) for row in rows], dtype=np.int32)
    option = np.asarray([int(row["option"]) for row in rows], dtype=np.int16)
    src_plane = np.asarray([int(row["src_plane"]) for row in rows], dtype=np.int16)
    src_y = np.asarray([int(row["src_y"]) for row in rows], dtype=np.int16)
    dst_plane = np.asarray([int(row["dst_plane"]) for row in rows], dtype=np.int16)
    dst_y = np.asarray([int(row["dst_y"]) for row in rows], dtype=np.int16)
    sat_ids = [str(i + 1) for i in range(int(total_sats))]
    return EdgeTable(
        src=src,
        dst=dst,
        option=option,
        src_plane=src_plane,
        src_y=src_y,
        dst_plane=dst_plane,
        dst_y=dst_y,
        sat_ids=sat_ids,
    )


def load_step_values(cache_dir: Path, *, pair_key: str, step: int) -> np.ndarray:
    steps = np.load(cache_dir / "time_indices.npy", mmap_mode="r")
    matches = np.flatnonzero(np.asarray(steps) == int(step))
    if matches.size == 0:
        raise ValueError(f"step {step} not found in {cache_dir / 'time_indices.npy'}")
    row = int(matches[0])
    values = np.load(cache_dir / pair_key / "edge_betweenness.npy", mmap_mode="r")
    return np.asarray(values[row : row + 1], dtype=np.float32)


def make_viewer(
    *,
    pair_key: str,
    pair_label: str,
    step: int,
    edge_table: EdgeTable,
    values: np.ndarray,
    group_data: dict,
    shared_value_max: float,
) -> EdgeUsageTopology2DViewer:
    return EdgeUsageTopology2DViewer(
        G60_CONFIG,
        steps=[int(step)],
        edge_table=edge_table,
        edge_usage_values=values,
        value_max=float(shared_value_max),
        window_title=f"G60 full-link weighted betweenness t={step}s {pair_label}",
        group_data=group_data,
        show_groups=True,
        show_topology_under_edge_values=True,
        zero_value_edges_visible=False,
        topology_edge_alpha=95,
        topology_edge_width=0.010,
        value_width_min=0.010,
        value_width_max=0.120,
        value_alpha_min=35,
        value_alpha_max=245,
    )


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshot = args.screenshot or out_dir / f"g60_full_link_weighted_betweenness_t{int(args.step)}_three_pairs_2d.png"

    edge_table = read_edge_table_csv(cache_dir / "edges.csv", total_sats=int(G60_CONFIG.total_sats))
    values_by_pair = {
        pair_key: load_step_values(cache_dir, pair_key=pair_key, step=int(args.step))
        for pair_key in PAIRS
    }
    shared_value_max = max(1.0, *(float(np.nanmax(values)) for values in values_by_pair.values()))
    group_data = load_or_build_group_data(
        xml_file=Path(args.group_xml),
        group_cache_dir=Path(args.group_cache_dir),
        steps=[int(args.step)],
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=1,
        enabled=True,
        force=False,
    )

    print(
        f"[full-link-betweenness-2d] step={int(args.step)} edges={edge_table.num_edges} "
        f"shared_value_max={shared_value_max:.3f} screenshot={screenshot}",
        flush=True,
    )
    for pair_key, pair_label in PAIRS.items():
        values = values_by_pair[pair_key]
        print(
            f"[full-link-betweenness-2d] {pair_label}: "
            f"max={float(np.nanmax(values)):.3f} sum={float(np.nansum(values)):.3f} "
            f"nonzero={int(np.count_nonzero(values > 0.0))}",
            flush=True,
        )

    if args.check_only:
        return 0

    if not bool(args.show):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    panels: list[Topology2DPanel] = []
    for pair_key, pair_label in PAIRS.items():
        viewer = make_viewer(
            pair_key=pair_key,
            pair_label=pair_label,
            step=int(args.step),
            edge_table=edge_table,
            values=values_by_pair[pair_key],
            group_data=group_data,
            shared_value_max=shared_value_max,
        )
        max_value = float(np.nanmax(values_by_pair[pair_key]))
        nonzero = int(np.count_nonzero(values_by_pair[pair_key] > 0.0))
        panels.append(
            Topology2DPanel(
                title=f"{pair_label} | max={max_value:.0f}, used_edges={nonzero}",
                viewer=viewer,
                stretch=1,
            )
        )

    combined = UnifiedControlTopology2DViewer(
        title=f"G60 full-link weighted shortest-delay edge betweenness t={int(args.step)}s",
        panels=panels,
        shared_value_scale=True,
        orientation=QtCore.Qt.Horizontal,
    )

    return run_viewer_widget(
        combined,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=not bool(args.show),
        screenshot=screenshot if not bool(args.show) else args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
