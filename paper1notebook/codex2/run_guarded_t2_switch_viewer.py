from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
for path in (GENERIC_ROOT, THIS_FILE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.edge_usage_viewer import EdgeUsageTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.full_link_node_usage_viewer import read_edge_table_csv  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402


GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")
DEFAULT_PLAN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\guarded_t2_switch_056_061_china_europe"
    r"\t0_86160_stride1\switch36000_segend54000_lst060_oldzero_delay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the guarded T2 motif000056 -> motif000061 switch in the 2D viewer.")
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--start", type=int, default=35900)
    parser.add_argument("--end", type=int, default=36200)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--initial-step", type=int, default=35940)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def row_slice(steps: np.ndarray, *, start: int, end: int, stride: int) -> np.ndarray:
    rows = np.flatnonzero(
        (steps >= int(start))
        & (steps <= int(end))
        & (((steps - int(start)) % int(stride)) == 0)
    )
    if rows.size == 0:
        raise ValueError(f"no rows for start={start}, end={end}, stride={stride}")
    return rows


def main() -> int:
    args = parse_args()
    if bool(args.offscreen):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    plan_dir = Path(args.plan_dir)
    steps_all = np.load(plan_dir / "steps.npy")
    rows = row_slice(steps_all, start=int(args.start), end=int(args.end), stride=int(args.stride))
    steps = np.asarray(steps_all[rows], dtype=np.int64)
    if int(args.initial_step) not in set(int(x) for x in steps):
        raise ValueError("--initial-step must be on the selected time axis")

    edge_table = read_edge_table_csv(plan_dir / "union_edges.csv", total_nodes=int(G60_CONFIG.total_sats))
    active = np.asarray(np.load(plan_dir / "edge_active_mask.npy", mmap_mode="r")[rows, :], dtype=bool)
    building = np.asarray(np.load(plan_dir / "edge_building_mask.npy", mmap_mode="r")[rows, :], dtype=bool)
    values = np.asarray(np.load(plan_dir / "edge_usage_values.npy", mmap_mode="r")[rows, :], dtype=np.float32)

    group_data = {}
    if not bool(args.no_groups):
        raw_groups = load_or_build_group_data(
            xml_file=GROUP_XML,
            group_cache_dir=GROUP_CACHE_DIR,
            steps=[int(x) for x in steps],
            station_groups=G60_CONFIG.station_groups,
            total_sats=G60_CONFIG.total_sats,
            constellation_name=G60_CONFIG.name,
            stride=int(args.stride),
            enabled=True,
            force=bool(args.force_group_cache),
        )
        wanted = set(int(x) for x in steps)
        group_data = {int(step): data for step, data in raw_groups.items() if int(step) in wanted}

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = EdgeUsageTopology2DViewer(
        G60_CONFIG,
        steps=[int(x) for x in steps],
        edge_table=edge_table,
        edge_usage_values=values,
        value_max=max(1.0, float(np.nanmax(values)) if values.size else 1.0),
        edge_active_mask=active,
        edge_building_mask=building,
        window_title=f"G60 guarded T2 switch 000056 -> 000061 | {int(steps[0])}..{int(steps[-1])}s",
        group_data=group_data,
        show_groups=not bool(args.no_groups),
        topology_edge_color="#000000",
        topology_edge_alpha=165,
        topology_edge_width=0.020,
        building_edge_color="#2563EB",
        building_edge_alpha=240,
        building_edge_width=0.080,
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
    viewer.update_step(int(np.where(steps == int(args.initial_step))[0][0]))

    screenshot = args.screenshot
    if screenshot is None and bool(args.offscreen):
        screenshot = plan_dir / f"guarded_t2_switch_viewer_t{int(args.initial_step)}.png"
    print(
        "[guarded-t2-viewer] "
        f"plan_dir={plan_dir} steps={len(steps)} initial={args.initial_step} "
        f"edges={edge_table.num_edges} building_max={int(np.max(np.sum(building, axis=1)))}",
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
