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
from run_motif0056_to_0061_setup_plan_viewer import all_intra_keys, edge_key_index, union_edge_table  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402


GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")
DEFAULT_LST_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_lst_sweep_056_061_056_china_europe\t0_86160_stride60"
    r"\lst_1s_schedule_resolution_guard60"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show usage-driven LST=60s setup schedule in the G60 2D topology viewer."
    )
    parser.add_argument("--mode", choices=("delay", "any"), default="delay")
    parser.add_argument("--lst", type=int, default=60)
    parser.add_argument("--lst-root", type=Path, default=DEFAULT_LST_ROOT)
    parser.add_argument("--start", type=int, default=35000)
    parser.add_argument("--end", type=int, default=36200)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--return-switch-step", type=int, default=54000)
    parser.add_argument("--initial-step", type=int, default=35940)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--focus-owner", type=int, default=None)
    parser.add_argument("--focus-zoom", type=float, default=6.0)
    return parser.parse_args()


def parse_edge_text(text: str) -> tuple[int, int] | None:
    text = str(text).strip()
    if not text or text.lower() == "nan":
        return None
    left, right = text.split("-", 1)
    return one_way.edge_key(int(left), int(right))


def load_schedule(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = parse_edge_text(row["new_edge"])
            if key is None:
                continue
            rows.append(
                {
                    "transition": str(row["transition"]),
                    "owner": int(row["owner"]),
                    "edge_key": key,
                    "plan_start": int(row["plan_start"]),
                    "plan_end": int(row["plan_end"]),
                }
            )
    return rows


def current_building_owner(schedule_rows: list[dict[str, object]], step: int) -> int | None:
    for row in schedule_rows:
        if int(row["plan_start"]) <= int(step) < int(row["plan_end"]):
            return int(row["owner"])
    return None


def events_by_owner(schedule_rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    out: dict[int, list[dict[str, object]]] = {}
    for row in schedule_rows:
        out.setdefault(int(row["owner"]), []).append(row)
    for rows in out.values():
        rows.sort(key=lambda item: (int(item["plan_start"]), int(item["plan_end"])))
    return out


def source_target_link_for_event(*, event: dict[str, object], owner: int, topo56, topo61):
    if str(event["transition"]) == "switch_056_to_061":
        return topo56.right_by_owner.get(int(owner)), topo61.right_by_owner.get(int(owner))
    if str(event["transition"]) == "switch_061_to_056":
        return topo61.right_by_owner.get(int(owner)), topo56.right_by_owner.get(int(owner))
    raise ValueError(f"unknown transition: {event['transition']}")


def active_and_building_for_owner(*, owner: int, step: int, owner_events: list[dict[str, object]], topo56, topo61):
    active_link = None
    for event in owner_events:
        source_link, target_link = source_target_link_for_event(
            event=event,
            owner=int(owner),
            topo56=topo56,
            topo61=topo61,
        )
        if int(step) < int(event["plan_start"]):
            return source_link, None
        if int(event["plan_start"]) <= int(step) < int(event["plan_end"]):
            return None, target_link
        active_link = target_link
    return active_link, None


def ideal_active_right_link_for_step(*, step: int, owner: int, topo56, topo61, switch_step: int, return_switch_step: int):
    topo = topo61 if int(switch_step) <= int(step) < int(return_switch_step) else topo56
    return topo.right_by_owner.get(int(owner))


def build_masks(
    *,
    steps: list[int],
    edge_table,
    topo56,
    topo61,
    schedule_rows: list[dict[str, object]],
    switch_step: int,
    return_switch_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    key_to_idx = edge_key_index(edge_table)
    intra_keys = all_intra_keys(edge_table)
    scheduled_by_owner = events_by_owner(schedule_rows)
    all_owners = sorted(set(topo56.right_by_owner) | set(topo61.right_by_owner) | set(scheduled_by_owner))
    active_mask = np.zeros((len(steps), edge_table.num_edges), dtype=bool)
    building_mask = np.zeros((len(steps), edge_table.num_edges), dtype=bool)

    for row_idx, step in enumerate(steps):
        active_keys = set(intra_keys)
        building_keys: set[tuple[int, int]] = set()
        for owner in all_owners:
            owner_events = scheduled_by_owner.get(int(owner))
            if owner_events:
                active_link, building_link = active_and_building_for_owner(
                    owner=int(owner),
                    step=int(step),
                    owner_events=owner_events,
                    topo56=topo56,
                    topo61=topo61,
                )
            else:
                active_link = ideal_active_right_link_for_step(
                    step=int(step),
                    owner=int(owner),
                    topo56=topo56,
                    topo61=topo61,
                    switch_step=int(switch_step),
                    return_switch_step=int(return_switch_step),
                )
                building_link = None
            if active_link is not None:
                active_keys.add(active_link.edge_key)
            if building_link is not None:
                building_keys.add(building_link.edge_key)
        for key in active_keys:
            idx = key_to_idx.get(key)
            if idx is not None:
                active_mask[row_idx, idx] = True
        for key in building_keys:
            idx = key_to_idx.get(key)
            if idx is not None:
                building_mask[row_idx, idx] = True
    return active_mask, building_mask


def main() -> int:
    args = parse_args()
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty time axis")
    if int(args.initial_step) not in set(steps):
        raise ValueError("--initial-step must be on the requested time axis")

    topo56, topo61 = roundtrip.build_topologies()
    edge_table = union_edge_table(topo56, topo61)
    schedule_path = Path(args.lst_root) / str(args.mode) / f"lst{int(args.lst):03d}" / "schedule_1s.csv"
    schedule_rows = load_schedule(schedule_path)
    active_mask, building_mask = build_masks(
        steps=steps,
        edge_table=edge_table,
        topo56=topo56,
        topo61=topo61,
        schedule_rows=schedule_rows,
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
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
    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=active_mask,
        edge_building_mask=building_mask,
        window_title=f"G60 usage-driven {args.mode} LST={int(args.lst)}s setup, 1s schedule grid",
        group_data=group_data,
        show_groups=not bool(args.no_groups),
        topology_edge_color="#000000",
        topology_edge_alpha=170,
        topology_edge_width=0.028,
        building_edge_color="#2563EB",
        building_edge_alpha=235,
        building_edge_width=0.095,
        hide_y_wrap_edges=True,
        show_grid_lines=False,
    )
    viewer.node_radius = 0.125
    viewer.update_step(steps.index(int(args.initial_step)))

    screenshot = args.screenshot
    if screenshot is None and bool(args.offscreen):
        screenshot = Path(args.lst_root) / str(args.mode) / f"lst{int(args.lst):03d}" / (
            f"usage_driven_lst{int(args.lst):03d}_viewer_t{int(args.initial_step)}.png"
        )
    print(
        "[usage-driven-lst-viewer] "
        f"mode={args.mode} lst={args.lst} steps={len(steps)} initial={args.initial_step} "
        f"edges={edge_table.num_edges} building_max={int(np.max(np.sum(building_mask, axis=1)))} "
        f"schedule={schedule_path}",
        flush=True,
    )
    focus_owner = args.focus_owner
    if focus_owner is None:
        focus_owner = current_building_owner(schedule_rows, int(args.initial_step))
    if args.screenshot is not None and focus_owner is not None:
        screenshot_path = Path(args.screenshot)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        viewer.resize(int(args.width), int(args.height))
        viewer.show()
        app.processEvents()
        viewer.fit_scene()
        viewer.fit_on_next_resize = False
        viewer.zoom_view(float(args.focus_zoom))
        x, y = viewer.node_grid_pos(int(focus_owner))
        viewer.view.centerOn(float(x) + 0.8, float(y))
        app.processEvents()
        ok = viewer.grab().save(str(screenshot_path))
        print(
            f"[usage-driven-lst-viewer] focused_owner={focus_owner} "
            f"screenshot={screenshot_path} ok={ok}",
            flush=True,
        )
        return 0 if ok else 1
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        offscreen=bool(args.offscreen),
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
