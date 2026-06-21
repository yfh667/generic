from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

if "--offscreen" in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[3]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.full_link_node_usage_viewer import (  # noqa: E402
    FullLinkNodeUsageData,
    FullLinkNodeUsageTopologyViewer,
    read_edge_table_csv,
)
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.stores import MetricStoreLayout, expand_unique_state_values  # noqa: E402


DEFAULT_USAGE_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\full_link_edge_usage_share_three_pairs_t0_86160_stride60"
)
GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show G60 full-link topology time series with four-option node usage inspector. "
            "Click a node to inspect option 0/1/2/4 right-neighbor usage separately."
        )
    )
    parser.add_argument("--pair", choices=("china_europe", "china_africa", "china_america"), default="china_europe")
    parser.add_argument("--usage-root", type=Path, default=DEFAULT_USAGE_ROOT)
    parser.add_argument("--initial-step", type=int, default=43020)
    parser.add_argument("--initial-node", type=int, default=None)
    parser.add_argument("--value-mode", choices=("hops", "hop", "delay", "any", "primary", "secondary"), default="delay")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def load_full_link_usage(*, usage_root: Path, pair: str) -> FullLinkNodeUsageData:
    usage_root = Path(usage_root)
    hop_dir = usage_root / "hop_shortest" / str(pair)
    delay_dir = usage_root / "delay_shortest"
    if not hop_dir.exists():
        raise FileNotFoundError(f"hop usage cache not found: {hop_dir}")
    if not delay_dir.exists():
        raise FileNotFoundError(f"delay usage cache not found: {delay_dir}")

    layout = MetricStoreLayout(hop_dir)
    steps = np.asarray(np.load(layout.time_indices_npy), dtype=np.int64)
    delay_steps = np.asarray(np.load(delay_dir / "time_indices.npy"), dtype=np.int64)
    if not np.array_equal(steps, delay_steps):
        raise ValueError("hop and delay full-link caches have different time axes")

    edge_table = read_edge_table_csv(layout.edges_csv, total_nodes=int(G60_CONFIG.total_sats))
    hop_usage = expand_unique_state_values(
        unique_state_values=np.load(layout.unique_state_values_npy, mmap_mode="r"),
        state_ids=np.load(layout.state_ids_npy, mmap_mode="r"),
    ).astype(np.float32)
    delay_usage = np.asarray(np.load(delay_dir / str(pair) / "edge_betweenness.npy", mmap_mode="r"), dtype=np.float32)
    return FullLinkNodeUsageData(
        steps=steps,
        edge_table=edge_table,
        hop_usage=hop_usage,
        delay_usage=delay_usage,
    )


def main() -> int:
    args = parse_args()
    data = load_full_link_usage(usage_root=Path(args.usage_root), pair=str(args.pair))
    steps = [int(x) for x in data.steps]
    initial_row = int(np.argmin(np.abs(np.asarray(data.steps, dtype=np.int64) - int(args.initial_step))))
    actual_initial_step = int(data.steps[initial_row])

    group_data = {}
    if not bool(args.no_groups):
        raw_groups = load_or_build_group_data(
            xml_file=GROUP_XML,
            group_cache_dir=GROUP_CACHE_DIR,
            steps=steps,
            station_groups=G60_CONFIG.station_groups,
            total_sats=G60_CONFIG.total_sats,
            constellation_name=G60_CONFIG.name,
            stride=60,
            enabled=True,
            force=bool(args.force_group_cache),
        )
        wanted = set(steps)
        group_data = {int(step): payload for step, payload in raw_groups.items() if int(step) in wanted}

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = FullLinkNodeUsageTopologyViewer(
        G60_CONFIG,
        data=data,
        value_mode=str(args.value_mode),
        window_title=f"G60 full-link node usage | {args.pair}",
        group_data=group_data,
        show_groups=not bool(args.no_groups),
    )
    viewer.edge_width = 0.033
    viewer.edge_alpha = 185
    viewer.node_radius = 0.115
    viewer.update_step(initial_row)
    if args.initial_node is not None:
        viewer.pick_node(int(args.initial_node))

    print(
        "[full-link-node-usage-viewer] "
        f"pair={args.pair} steps={len(steps)} range={steps[0]}..{steps[-1]} "
        f"initial_step={actual_initial_step} edges={data.edge_table.num_edges} group_steps={len(group_data)}",
        flush=True,
    )
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
