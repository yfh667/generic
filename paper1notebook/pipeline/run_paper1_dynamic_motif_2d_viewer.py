from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.link_delay.module.edge_options import EdgeTable
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.module.batch_shortest_hops import topology_specs_from_motif_csv
from src.topology_workflow.module.config import load_workflow_yaml, time_axis_from_config, viewer_config_from_workflow


DEFAULT_CONFIG = THIS_DIR / "configs" / "g60_w_le4_h_le3_shortest_hops.yaml"


class DynamicMotifTopologyViewer(SatelliteTopology2DViewer):
    def __init__(
        self,
        *args,
        topology_names: list[str],
        topology_values: list[float],
        topology_value_label: str,
        **kwargs,
    ):
        self.dynamic_topology_names = [str(x) for x in topology_names]
        self.dynamic_topology_values = [float(x) for x in topology_values]
        self.dynamic_topology_value_label = str(topology_value_label)
        super().__init__(*args, **kwargs)

    def update_step(self, row: int, *, sync_slider: bool = True):
        super().update_step(row, sync_slider=sync_slider)
        row = int(max(0, min(int(row), len(self.dynamic_topology_names) - 1)))
        name = self.dynamic_topology_names[row]
        value = self.dynamic_topology_values[row]
        self.step_label.setText(
            self.step_label.text()
            + f" | dynamic motif {name} | {self.dynamic_topology_value_label} {value:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the time-varying best motif topology with SatelliteTopology2DViewer."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pair", default="china_europe", choices=("china_europe", "china_america", "china_africa"))
    parser.add_argument("--dynamic-csv", type=Path, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=860)
    parser.add_argument("--edge-width", type=float, default=0.045)
    parser.add_argument("--edge-alpha", type=int, default=190)
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


def path_from(raw: dict[str, Any], key: str) -> Path:
    value = raw.get(key)
    if value in (None, ""):
        raise ValueError(f"missing paths.{key} in workflow config")
    return Path(str(value))


def load_dynamic_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"dynamic CSV is empty: {path}")
    required = {"step", "dynamic_best_topology", "dynamic_best_hops"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"dynamic CSV missing columns {sorted(missing)}: {path}")
    return rows


def filter_dynamic_rows(
    rows: list[dict[str, str]],
    *,
    start: int,
    end: int,
    stride: int,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        step = int(float(row["step"]))
        if step < int(start) or step > int(end):
            continue
        if (step - int(start)) % int(stride) != 0:
            continue
        out.append(row)
    if not out:
        raise ValueError(f"no dynamic rows left after filtering start={start}, end={end}, stride={stride}")
    return out


def edge_key(edge_table: EdgeTable, idx: int) -> tuple[int, int]:
    src = int(edge_table.src[idx])
    dst = int(edge_table.dst[idx])
    return (src, dst) if src < dst else (dst, src)


def union_edge_table_for_specs(specs_by_name: dict[str, Any], names: list[str]) -> tuple[EdgeTable, dict[str, np.ndarray]]:
    records: dict[tuple[int, int], tuple[int, int]] = {}
    for name in names:
        table = specs_by_name[name].edge_table
        for idx in range(table.num_edges):
            key = edge_key(table, idx)
            records.setdefault(key, (int(table.option[idx]), int(idx)))

    src_values: list[int] = []
    dst_values: list[int] = []
    option_values: list[int] = []
    src_plane_values: list[int] = []
    src_y_values: list[int] = []
    dst_plane_values: list[int] = []
    dst_y_values: list[int] = []
    for src, dst in sorted(records):
        option, _ = records[(src, dst)]
        src_values.append(int(src))
        dst_values.append(int(dst))
        option_values.append(int(option))
        src_plane_values.append(int(src) // len(specs_by_name[names[0]].edge_table.sat_ids))
        src_y_values.append(0)
        dst_plane_values.append(int(dst) // len(specs_by_name[names[0]].edge_table.sat_ids))
        dst_y_values.append(0)

    # Recompute plane/y using the first table's constellation dimensions.
    first_table = specs_by_name[names[0]].edge_table
    total_sats = len(first_table.sat_ids)
    max_y = int(max(np.max(first_table.src_y), np.max(first_table.dst_y))) + 1
    if total_sats % max_y != 0:
        raise ValueError("cannot infer N from motif edge table")
    n = max_y
    src_plane_values = [int(src) // n for src in src_values]
    src_y_values = [int(src) % n for src in src_values]
    dst_plane_values = [int(dst) // n for dst in dst_values]
    dst_y_values = [int(dst) % n for dst in dst_values]

    union = EdgeTable(
        src=np.asarray(src_values, dtype=np.int32),
        dst=np.asarray(dst_values, dtype=np.int32),
        option=np.asarray(option_values, dtype=np.int16),
        src_plane=np.asarray(src_plane_values, dtype=np.int16),
        src_y=np.asarray(src_y_values, dtype=np.int16),
        dst_plane=np.asarray(dst_plane_values, dtype=np.int16),
        dst_y=np.asarray(dst_y_values, dtype=np.int16),
        sat_ids=list(first_table.sat_ids),
    )
    union_index = {edge_key(union, idx): idx for idx in range(union.num_edges)}

    active_indices: dict[str, np.ndarray] = {}
    for name in names:
        table = specs_by_name[name].edge_table
        idxs = [union_index[edge_key(table, idx)] for idx in range(table.num_edges)]
        active_indices[name] = np.asarray(sorted(set(idxs)), dtype=np.int32)
    return union, active_indices


def build_active_mask(
    *,
    topology_names_by_row: list[str],
    edge_count: int,
    active_indices_by_name: dict[str, np.ndarray],
) -> np.ndarray:
    mask = np.zeros((len(topology_names_by_row), int(edge_count)), dtype=bool)
    for row, name in enumerate(topology_names_by_row):
        mask[row, active_indices_by_name[name]] = True
    return mask


def main() -> int:
    args = parse_args()
    workflow = load_workflow_yaml(args.config)
    config = viewer_config_from_workflow(workflow)
    cfg_start, cfg_end, cfg_stride = time_axis_from_config(workflow)
    start = cfg_start if args.start is None else int(args.start)
    end = cfg_end if args.end is None else int(args.end)
    stride = cfg_stride if args.stride is None else int(args.stride)

    paths_raw = workflow.get("paths", {})
    motif_raw = workflow.get("motif_library", {})
    out_dir = path_from(paths_raw, "out_dir")
    motif_csv = path_from(paths_raw, "motif_library_dir") / str(motif_raw.get("csv_name"))
    dynamic_csv = (
        Path(args.dynamic_csv)
        if args.dynamic_csv is not None
        else out_dir / str(args.pair) / "paper_style_dynamic_best_mean_shortest_hops.csv"
    )

    dynamic_rows = filter_dynamic_rows(
        load_dynamic_rows(dynamic_csv),
        start=start,
        end=end,
        stride=stride,
    )
    steps = [int(float(row["step"])) for row in dynamic_rows]
    topology_names = [str(row["dynamic_best_topology"]) for row in dynamic_rows]
    topology_values = [float(row["dynamic_best_hops"]) for row in dynamic_rows]
    unique_topology_names = sorted(set(topology_names))

    specs = topology_specs_from_motif_csv(
        motif_csv,
        config=config,
        library=str(motif_raw.get("name_prefix", "motif")),
        name_prefix=str(motif_raw.get("name_prefix", "motif")),
        add_intra_ring=True,
        wrap_planes=False,
    )
    specs_by_name = {spec.name: spec for spec in specs}
    missing = [name for name in unique_topology_names if name not in specs_by_name]
    if missing:
        raise ValueError(f"dynamic CSV references topologies not in motif library: {missing[:8]}")

    edge_table, active_indices_by_name = union_edge_table_for_specs(specs_by_name, unique_topology_names)
    active_mask = build_active_mask(
        topology_names_by_row=topology_names,
        edge_count=edge_table.num_edges,
        active_indices_by_name=active_indices_by_name,
    )

    group_data = {}
    if not args.no_groups:
        group_data = load_or_build_group_data(
            xml_file=path_from(paths_raw, "group_xml"),
            group_cache_dir=path_from(paths_raw, "group_cache_dir"),
            steps=steps,
            station_groups=config.station_groups,
            total_sats=config.total_sats,
            constellation_name=config.name,
            stride=stride,
            enabled=True,
            force=bool(args.force_group_cache),
        )

    print(
        f"[dynamic-motif-2d] pair={args.pair}, steps={len(steps)}, "
        f"unique_motifs={len(unique_topology_names)}, union_edges={edge_table.num_edges}, "
        f"dynamic_csv={dynamic_csv}",
        flush=True,
    )
    print(f"[dynamic-motif-2d] first motifs={', '.join(unique_topology_names[:8])}", flush=True)

    if args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = DynamicMotifTopologyViewer(
        config,
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=active_mask,
        window_title=f"{config.name} dynamic motif 2D | {args.pair} | shortest hops",
        group_data=group_data,
        show_groups=not bool(args.no_groups),
        show_grid_lines=False,
        topology_names=topology_names,
        topology_values=topology_values,
        topology_value_label="hops",
    )
    viewer.edge_width = float(args.edge_width)
    viewer.edge_alpha = int(args.edge_alpha)
    viewer.width_slider.setValue(int(max(8, min(85, round(float(args.edge_width) * 1000)))))
    viewer.alpha_slider.setValue(int(max(25, min(220, int(args.edge_alpha)))))
    viewer.update_step(0)

    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
