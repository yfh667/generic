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

import plan_motif0056_to_0061_link_setup as plan  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.edge_usage_viewer import EdgeUsageTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module.edge_tables import INTRA_OPTION, make_edge_table_from_records  # noqa: E402


GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")
DEFAULT_PLAN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\m0056_to_m0061_china_europe\t0_86160_stride60\switch36000_dur600_c20"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the planned motif000056 -> motif000061 link setup process in the original 2D viewer."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--initial-step", type=int, default=35400)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def read_plan_events(path: Path) -> dict[int, tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    events: dict[int, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            owner = int(row["owner"])
            events[owner] = (int(row["plan_start"]), int(row["plan_end"]))
    return events


def build_source_target() -> tuple[plan.TopologyData, plan.TopologyData]:
    rows = plan.read_motif_rows(plan.MOTIF_LIBRARY_CSV)
    source_spec = plan.build_topology_spec(56, rows[56])
    target_spec = plan.build_topology_spec(61, rows[61])
    source = plan.load_topology_data(source_spec, start=0, end=86160, stride=60)
    target = plan.load_topology_data(target_spec, start=0, end=86160, stride=60)
    return source, target


def union_edge_table(source, target):
    records: list[tuple[int, int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for edge_table in (source.spec.edge_table, target.spec.edge_table):
        for idx in range(edge_table.num_edges):
            src = int(edge_table.src[idx])
            dst = int(edge_table.dst[idx])
            key = plan.edge_key(src, dst)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                (
                    int(edge_table.src_plane[idx]),
                    int(edge_table.src_y[idx]),
                    int(edge_table.dst_plane[idx]),
                    int(edge_table.dst_y[idx]),
                    int(edge_table.option[idx]),
                )
            )
    return make_edge_table_from_records(p=int(G60_CONFIG.P), n=int(G60_CONFIG.N), records=records)


def edge_key_index(edge_table) -> dict[tuple[int, int], int]:
    return {
        plan.edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])): int(idx)
        for idx in range(edge_table.num_edges)
    }


def all_intra_keys(edge_table) -> set[tuple[int, int]]:
    return {
        plan.edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]))
        for idx in range(edge_table.num_edges)
        if int(edge_table.option[idx]) == INTRA_OPTION
    }


def link_usage_for_display(
    *,
    row: int,
    step: int,
    switch_step: int,
    key: tuple[int, int],
    source: plan.TopologyData,
    target: plan.TopologyData,
) -> float:
    if int(step) < int(switch_step) and key in source.edge_key_to_idx:
        return float(source.values[row, source.edge_key_to_idx[key]])
    if key in target.edge_key_to_idx:
        return float(target.values[row, target.edge_key_to_idx[key]])
    if key in source.edge_key_to_idx:
        return float(source.values[row, source.edge_key_to_idx[key]])
    return 0.0


def build_viewer_arrays(
    *,
    steps: list[int],
    switch_step: int,
    union_table,
    source: plan.TopologyData,
    target: plan.TopologyData,
    events_by_owner: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    union_idx = edge_key_index(union_table)
    intra_keys = all_intra_keys(union_table)
    active_mask = np.zeros((len(steps), union_table.num_edges), dtype=bool)
    building_mask = np.zeros((len(steps), union_table.num_edges), dtype=bool)
    edge_values = np.zeros((len(steps), union_table.num_edges), dtype=np.float32)

    all_owners = sorted(set(source.right_by_owner) | set(target.right_by_owner))
    for row, step in enumerate(steps):
        active_keys = set(intra_keys)
        building_keys: set[tuple[int, int]] = set()

        for owner in all_owners:
            old_link = source.right_by_owner.get(owner)
            new_link = target.right_by_owner.get(owner)
            event = events_by_owner.get(owner)
            if event is None:
                link = old_link or new_link
                if link is not None:
                    active_keys.add(link.edge_key)
                continue

            plan_start, plan_end = event
            if int(step) < int(plan_start):
                if old_link is not None:
                    active_keys.add(old_link.edge_key)
            elif int(plan_start) <= int(step) < int(plan_end):
                if new_link is not None:
                    building_keys.add(new_link.edge_key)
            else:
                if new_link is not None:
                    active_keys.add(new_link.edge_key)

        for key in active_keys:
            idx = union_idx.get(key)
            if idx is None:
                continue
            active_mask[row, idx] = True
            edge_values[row, idx] = link_usage_for_display(
                row=row,
                step=int(step),
                switch_step=int(switch_step),
                key=key,
                source=source,
                target=target,
            )
        for key in building_keys:
            idx = union_idx.get(key)
            if idx is not None:
                building_mask[row, idx] = True
    return active_mask, building_mask, edge_values


def main() -> int:
    args = parse_args()
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if int(args.initial_step) not in set(steps):
        raise ValueError("--initial-step must be on the requested time axis")

    source, target = build_source_target()
    events_by_owner = read_plan_events(Path(args.plan_dir) / "link_setup_events.csv")
    edge_table = union_edge_table(source, target)
    active_mask, building_mask, edge_values = build_viewer_arrays(
        steps=steps,
        switch_step=int(args.switch_step),
        union_table=edge_table,
        source=source,
        target=target,
        events_by_owner=events_by_owner,
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
    wanted = set(steps)
    group_data = {int(step): data for step, data in group_data.items() if int(step) in wanted}

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = EdgeUsageTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        edge_usage_values=edge_values,
        value_max=max(1.0, float(np.nanmax(edge_values))),
        edge_active_mask=active_mask,
        edge_building_mask=building_mask,
        window_title="G60 planned link setup: motif000056 -> motif000061",
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
        screenshot = Path(args.plan_dir) / f"setup_plan_viewer_t{int(args.initial_step)}.png"

    print(
        "[setup-plan-viewer] "
        f"steps={len(steps)} initial_step={args.initial_step} "
        f"edges={edge_table.num_edges} active_shape={active_mask.shape} "
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
