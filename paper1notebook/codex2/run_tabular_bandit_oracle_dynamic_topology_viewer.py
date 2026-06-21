from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PIPELINE_DIR = GENERIC_ROOT / "paper1notebook" / "pipeline"
for path in (GENERIC_ROOT, THIS_DIR, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_paper1_motif_shortest_hops import load_yaml, path_from  # noqa: E402
from search_motif0040_0056_hybrid_region_grid import DEFAULT_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module.config import viewer_config_from_workflow  # noqa: E402


DEFAULT_DYNAMIC_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_tabular_bandit_oracle_dynamic_topology_lambda050"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the row-mask tabular-bandit oracle dynamic topology in the 2D viewer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dynamic-dir", type=Path, default=DEFAULT_DYNAMIC_DIR)
    parser.add_argument("--start", type=int, default=None, help="Optional inclusive step filter.")
    parser.add_argument("--end", type=int, default=None, help="Optional inclusive step filter.")
    parser.add_argument("--stride", type=int, default=None, help="Group-cache stride metadata only.")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--edge-width", type=float, default=0.035)
    parser.add_argument("--edge-alpha", type=int, default=175)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def read_edge_table_csv(path: Path, *, total_sats: int) -> EdgeTable:
    src: list[int] = []
    dst: list[int] = []
    option: list[int] = []
    src_plane: list[int] = []
    src_y: list[int] = []
    dst_plane: list[int] = []
    dst_y: list[int] = []
    sat_ids = [str(i + 1) for i in range(int(total_sats))]
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            src.append(int(row["src_node"]))
            dst.append(int(row["dst_node"]))
            option.append(int(row["option"]))
            src_plane.append(int(row["src_plane"]))
            src_y.append(int(row["src_y"]))
            dst_plane.append(int(row["dst_plane"]))
            dst_y.append(int(row["dst_y"]))
    return EdgeTable(
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        option=np.asarray(option, dtype=np.int16),
        src_plane=np.asarray(src_plane, dtype=np.int16),
        src_y=np.asarray(src_y, dtype=np.int16),
        dst_plane=np.asarray(dst_plane, dtype=np.int16),
        dst_y=np.asarray(dst_y, dtype=np.int16),
        sat_ids=sat_ids,
    )


def filter_rows(steps: np.ndarray, *, start: int | None, end: int | None) -> np.ndarray:
    mask = np.ones(steps.shape[0], dtype=bool)
    if start is not None:
        mask &= steps >= int(start)
    if end is not None:
        mask &= steps <= int(end)
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        raise ValueError(f"empty step filter start={start} end={end}")
    return rows.astype(np.int64)


def main() -> int:
    args = parse_args()
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    raw = load_yaml(Path(args.config))
    config = viewer_config_from_workflow(raw)
    paths = raw.get("paths", {})
    dynamic_dir = Path(args.dynamic_dir)
    edge_table = read_edge_table_csv(dynamic_dir / "union_edges.csv", total_sats=int(config.total_sats))
    steps_all = np.load(dynamic_dir / "steps.npy")
    active_all = np.load(dynamic_dir / "edge_active_mask.npy", mmap_mode="r")
    selected_rows = filter_rows(steps_all, start=args.start, end=args.end)
    steps = [int(steps_all[idx]) for idx in selected_rows.tolist()]
    active_mask = np.asarray(active_all[selected_rows, :], dtype=bool)

    stride = int(args.stride if args.stride is not None else raw.get("time", {}).get("stride", 60))
    group_data = None
    if not args.no_groups:
        group_data = load_or_build_group_data(
            xml_file=path_from(paths, "group_xml"),
            group_cache_dir=path_from(paths, "group_cache_dir"),
            steps=steps,
            station_groups=config.station_groups,
            total_sats=config.total_sats,
            constellation_name=config.name,
            stride=stride,
            enabled=True,
            force=bool(args.force_group_cache),
        )

    if args.check_only:
        print(
            "check ok | "
            f"steps={len(steps)} step_range={steps[0]}..{steps[-1]} "
            f"union_edges={edge_table.num_edges} active_shape={active_mask.shape} "
            f"active_min={int(active_mask.sum(axis=1).min())} "
            f"active_max={int(active_mask.sum(axis=1).max())} groups={group_data is not None}"
        )
        return 0

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = SatelliteTopology2DViewer(
        config,
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=active_mask,
        window_title=f"G60 tabular oracle dynamic topology lambda0.50 {steps[0]}..{steps[-1]}s",
        group_data=group_data,
        show_groups=not args.no_groups,
        topology_edge_color="#000000",
        topology_edge_alpha=int(args.edge_alpha),
        topology_edge_width=float(args.edge_width),
        hide_y_wrap_edges=True,
        show_grid_lines=False,
    )
    viewer.resize(int(args.width), int(args.height))

    if args.screenshot is not None:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        viewer.show()
        app.processEvents()
        ok = viewer.grab().save(str(args.screenshot))
        print(f"screenshot={args.screenshot} ok={ok}")
        if args.offscreen:
            return 0

    return run_viewer_widget(viewer, app=app)


if __name__ == "__main__":
    raise SystemExit(main())
