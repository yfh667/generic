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
from src.satellite_topology_viewer.module.link_switch_viewer import (  # noqa: E402
    LinkSwitchTopologyViewer,
    build_viewer_edge_payload,
    load_right_link_state_store,
    make_link_switch_viewer_data,
    nearest_row_for_step,
    row_indices_for_window,
)
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
            "Show a dynamic link-switch topology with two edge-betweenness layers "
            "and a pop-up node working-state inspector."
        )
    )
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--initial-step", type=int, default=43020)
    parser.add_argument("--initial-node", type=int, default=None)
    parser.add_argument("--value-mode", choices=("hop", "delay", "any", "primary", "secondary"), default="delay")
    parser.add_argument("--working-mode", choices=("hop", "delay", "any", "primary", "secondary"), default="any")
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

    edge_table, active_mask, hop_values, _right_to_viewer = build_viewer_edge_payload(
        store,
        config=G60_CONFIG,
        usage_mode="hop",
    )
    _edge_table2, _active2, delay_values, _right_to_viewer2 = build_viewer_edge_payload(
        store,
        config=G60_CONFIG,
        usage_mode="delay",
    )
    data = make_link_switch_viewer_data(
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=active_mask,
        edge_betweenness_primary=hop_values,
        edge_betweenness_secondary=delay_values,
        primary_label="hop",
        secondary_label="delay",
        meta={
            "source_store": str(Path(args.store_dir)),
            "description": "edge masks plus two edge-betweenness layers loaded from right_link_state store",
        },
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

    viewer = LinkSwitchTopologyViewer(
        G60_CONFIG,
        data=data,
        value_mode=str(args.value_mode),
        working_mode=str(args.working_mode),
        inspector_mode=str(args.inspector_mode),
        topology_motif_id=np.asarray(store.topology_motif_id, dtype=np.int16),
        window_title=(
            "G60 link-switch topology: "
            f"value={args.value_mode}, inspector={args.inspector_mode}"
        ),
        group_data=group_data,
        show_groups=not bool(args.no_groups),
    )
    viewer.edge_width = 0.040
    viewer.edge_alpha = 210
    viewer.node_radius = 0.125
    viewer.update_step(initial_row)
    if args.initial_node is not None:
        viewer.pick_node(int(args.initial_node))

    screenshot = args.screenshot
    if screenshot is None and bool(args.offscreen):
        screenshot = Path(args.store_dir) / (
            f"link_switch_node_inspector_t{actual_initial_step}_{args.value_mode}.png"
        )

    print(
        "[link-switch-node-inspector] "
        f"store={Path(args.store_dir)} requested={int(args.start)}..{int(args.end)} "
        f"stride={int(args.stride)} actual={steps[0]}..{steps[-1]} steps={len(steps)} "
        f"initial_step={actual_initial_step} value_mode={args.value_mode} "
        f"edges={edge_table.num_edges} group_steps={len(group_data)}",
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
