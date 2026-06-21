from __future__ import annotations

import argparse
import csv
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

import plan_motif0056_0061_roundtrip_link_setup as roundtrip  # noqa: E402
import plan_motif0056_to_0061_link_setup as one_way  # noqa: E402
from run_motif0056_to_0061_setup_plan_viewer import edge_key_index, union_edge_table  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.edge_usage_viewer import EdgeUsageTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402


GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")
DEFAULT_PLAN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\m0056_to_m0061_to_m0056_china_europe\t0_86160_stride60\switch36000_54000_dur600_c20"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show round-trip motif000056 -> motif000061 -> motif000056 setup plan.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--return-switch-step", type=int, default=54000)
    parser.add_argument("--initial-step", type=int, default=53100)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def read_events(path: Path) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    first: dict[int, tuple[int, int]] = {}
    second: dict[int, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            target = first if row["transition"] == "switch_056_to_061" else second
            target[int(row["owner"])] = (int(row["plan_start"]), int(row["plan_end"]))
    return first, second


def link_usage_for_display(
    *,
    row: int,
    step: int,
    key: tuple[int, int],
    topo56: one_way.TopologyData,
    topo61: one_way.TopologyData,
    return_switch_step: int,
) -> float:
    if int(step) >= int(return_switch_step) and key in topo56.edge_key_to_idx:
        return float(topo56.values[row, topo56.edge_key_to_idx[key]])
    if key in topo61.edge_key_to_idx:
        return float(topo61.values[row, topo61.edge_key_to_idx[key]])
    if key in topo56.edge_key_to_idx:
        return float(topo56.values[row, topo56.edge_key_to_idx[key]])
    return 0.0


def build_masks_and_values(
    *,
    steps: list[int],
    edge_table,
    topo56: one_way.TopologyData,
    topo61: one_way.TopologyData,
    first_events: dict[int, tuple[int, int]],
    second_events: dict[int, tuple[int, int]],
    return_switch_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_to_idx = edge_key_index(edge_table)
    intra_keys = {
        one_way.edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]))
        for idx in range(edge_table.num_edges)
        if int(edge_table.option[idx]) == -1
    }
    active_mask = np.zeros((len(steps), edge_table.num_edges), dtype=bool)
    building_mask = np.zeros((len(steps), edge_table.num_edges), dtype=bool)
    values = np.zeros((len(steps), edge_table.num_edges), dtype=np.float32)
    owners = sorted(set(topo56.right_by_owner) | set(topo61.right_by_owner))

    for row, step in enumerate(steps):
        active_keys = set(intra_keys)
        building_keys: set[tuple[int, int]] = set()
        for owner in owners:
            e1 = first_events.get(owner)
            e2 = second_events.get(owner)
            link56 = topo56.right_by_owner.get(owner)
            link61 = topo61.right_by_owner.get(owner)
            if e1 is not None and int(step) < e1[1]:
                if int(step) < e1[0]:
                    if link56 is not None:
                        active_keys.add(link56.edge_key)
                elif link61 is not None:
                    building_keys.add(link61.edge_key)
                continue
            if e2 is not None:
                if int(step) < e2[0]:
                    if link61 is not None:
                        active_keys.add(link61.edge_key)
                elif int(step) < e2[1]:
                    if link56 is not None:
                        building_keys.add(link56.edge_key)
                else:
                    if link56 is not None:
                        active_keys.add(link56.edge_key)
                continue
            link = link61 if e1 is not None and int(step) >= e1[1] else (link56 or link61)
            if link is not None:
                active_keys.add(link.edge_key)

        for key in active_keys:
            idx = key_to_idx.get(key)
            if idx is not None:
                active_mask[row, idx] = True
                values[row, idx] = link_usage_for_display(
                    row=row,
                    step=int(step),
                    key=key,
                    topo56=topo56,
                    topo61=topo61,
                    return_switch_step=int(return_switch_step),
                )
        for key in building_keys:
            idx = key_to_idx.get(key)
            if idx is not None:
                building_mask[row, idx] = True
    return active_mask, building_mask, values


def main() -> int:
    args = parse_args()
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if int(args.initial_step) not in set(steps):
        raise ValueError("--initial-step must be on the requested time axis")

    topo56, topo61 = roundtrip.build_topologies()
    first_events, second_events = read_events(Path(args.plan_dir) / "link_setup_events.csv")
    edge_table = union_edge_table(topo56, topo61)
    active_mask, building_mask, values = build_masks_and_values(
        steps=steps,
        edge_table=edge_table,
        topo56=topo56,
        topo61=topo61,
        first_events=first_events,
        second_events=second_events,
        return_switch_step=int(args.return_switch_step),
    )
    group_data = load_or_build_group_data(
        xml_file=GROUP_XML,
        group_cache_dir=GROUP_CACHE_DIR,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=not bool(args.no_groups),
        force=bool(args.force_group_cache),
    )
    group_data = {int(step): data for step, data in group_data.items() if int(step) in set(steps)}

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    viewer = EdgeUsageTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        edge_usage_values=values,
        value_max=max(1.0, float(np.nanmax(values))),
        edge_active_mask=active_mask,
        edge_building_mask=building_mask,
        window_title="G60 planned setup: motif000056 -> motif000061 -> motif000056",
        group_data=group_data,
        show_groups=not bool(args.no_groups),
        topology_edge_color="#000000",
        topology_edge_alpha=160,
        topology_edge_width=0.018,
        building_edge_color="#1D4ED8",
        building_edge_alpha=230,
        building_edge_width=0.060,
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
    viewer.update_step(steps.index(int(args.initial_step)))

    screenshot = args.screenshot
    if screenshot is None and bool(args.offscreen):
        screenshot = Path(args.plan_dir) / f"roundtrip_setup_plan_viewer_t{int(args.initial_step)}.png"
    print(
        "[roundtrip-setup-viewer] "
        f"steps={len(steps)} initial_step={args.initial_step} edges={edge_table.num_edges} "
        f"building_max={int(np.max(np.sum(building_mask, axis=1)))} plan_dir={args.plan_dir}",
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
