from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from node_working_inspector_viewer import (  # noqa: E402
    NodeWorkingInspectorEdgeUsageViewer,
    build_viewer_edge_payload,
    load_right_link_state_store,
    nearest_row_for_step,
    row_indices_for_window,
)
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402


DEFAULT_STORE_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\ideal_splice_056_061_056_china_europe\t0_86160_stride60"
    r"\switch36000_54000\right_link_state"
)
GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show the ideal no-LST motif000056 -> motif000061 -> motif000056 splice "
            "and inspect each node's right-link usage by clicking the node."
        )
    )
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86164)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--initial-step", type=int, default=43020)
    parser.add_argument("--initial-node", type=int, default=None)
    parser.add_argument("--usage-mode", choices=("hop", "delay", "any"), default="delay")
    parser.add_argument("--working-mode", choices=("hop", "delay", "any"), default="any")
    parser.add_argument("--inspector-mode", choices=("window", "right-panel"), default="window")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1700)
    parser.add_argument("--height", type=int, default=980)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    raw_store = load_right_link_state_store(args.store_dir)
    rows = row_indices_for_window(raw_store, start=int(args.start), end=int(args.end), stride=int(args.stride))
    store = raw_store.slice_rows(rows)
    steps = [int(x) for x in np.asarray(store.steps, dtype=np.int64)]
    initial_row = nearest_row_for_step(store.steps, int(args.initial_step))
    actual_initial_step = int(store.steps[initial_row])

    edge_table, active_mask, edge_values, right_to_viewer_edge_idx = build_viewer_edge_payload(
        store,
        config=G60_CONFIG,
        usage_mode=str(args.usage_mode),
    )

    group_data = {}
    if not bool(args.no_groups):
        raw_groups = load_or_build_group_data(
            xml_file=GROUP_XML,
            group_cache_dir=GROUP_CACHE_DIR,
            steps=steps,
            station_groups=G60_CONFIG.station_groups,
            total_sats=G60_CONFIG.total_sats,
            constellation_name=G60_CONFIG.name,
            stride=int(args.stride),
            enabled=True,
            force=bool(args.force_group_cache),
        )
        wanted = set(steps)
        group_data = {int(step): data for step, data in raw_groups.items() if int(step) in wanted}

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    value_max = max(1.0, float(np.nanmax(edge_values)) if edge_values.size else 1.0)
    viewer = NodeWorkingInspectorEdgeUsageViewer(
        G60_CONFIG,
        store=store,
        right_to_viewer_edge_idx=right_to_viewer_edge_idx,
        working_mode=str(args.working_mode),
        inspector_mode=str(args.inspector_mode),
        steps=steps,
        edge_table=edge_table,
        edge_usage_values=edge_values,
        value_max=value_max,
        edge_active_mask=active_mask,
        window_title=(
            "G60 ideal no-LST splice node inspector: "
            f"motif000056 -> motif000061 -> motif000056, usage={args.usage_mode}"
        ),
        group_data=group_data,
        show_groups=not bool(args.no_groups),
        topology_edge_color="#000000",
        topology_edge_alpha=168,
        topology_edge_width=0.018,
        value_width_min=0.010,
        value_width_max=0.105,
        value_alpha_min=34,
        value_alpha_max=245,
        hide_y_wrap_edges=True,
        show_grid_lines=False,
    )
    viewer.edge_width = 0.040
    viewer.edge_alpha = 210
    viewer.node_radius = 0.125
    if str(args.inspector_mode) == "right-panel" and hasattr(viewer, "main_splitter"):
        viewer.main_splitter.setSizes([620, 360])
    viewer.update_step(initial_row)
    if args.initial_node is not None:
        viewer.pick_node(int(args.initial_node))
    elif str(args.inspector_mode) == "window":
        viewer.show_inspector_window()

    screenshot = args.screenshot
    if screenshot is None and bool(args.offscreen):
        screenshot = Path(args.store_dir) / (
            f"ideal_splice_node_inspector_t{actual_initial_step}_"
            f"node{int(args.initial_node) if args.initial_node is not None else 'none'}_"
            f"{args.usage_mode}.png"
        )

    print(
        "[ideal-splice-node-inspector] "
        f"store={Path(args.store_dir)} "
        f"requested={int(args.start)}..{int(args.end)} stride={int(args.stride)} "
        f"actual={steps[0]}..{steps[-1]} steps={len(steps)} "
        f"initial_step={actual_initial_step} usage_mode={args.usage_mode} "
        f"inspector_mode={args.inspector_mode} edges={edge_table.num_edges} group_steps={len(group_data)}",
        flush=True,
    )

    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        offscreen=bool(args.offscreen),
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
