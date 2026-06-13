from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_g60_selected_motifs_shortest_delay_parallel import edge_table_for_motif
from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data


DEFAULT_SCHEDULE_DIR = (
    PROJECT_ROOT
    / "data"
    / "linshi"
    / "g60_w4h3_dynamic_schedule_selected100_t0_86164_stride60"
)
DEFAULT_SELECTED_MOTIFS = (
    PROJECT_ROOT
    / "data"
    / "linshi"
    / "g60_w4h3_selected100_shortest_delay_t0_86164_stride60"
    / "selected_motifs_original_rows.csv"
)
DEFAULT_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE = PROJECT_ROOT / "data" / "linshi" / "cache" / "group_data_cache"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_dynamic_topology_schedule_120min_viewer"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the G60 dynamic topology sequence selected by the 120-minute dwell schedule."
    )
    parser.add_argument("--schedule-dir", type=Path, default=DEFAULT_SCHEDULE_DIR)
    parser.add_argument("--by-step", type=Path, default=None)
    parser.add_argument("--segments", type=Path, default=None)
    parser.add_argument("--selected-motifs", type=Path, default=DEFAULT_SELECTED_MOTIFS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--hide-groups", action="store_true")
    parser.add_argument(
        "--show-background-grid",
        action="store_true",
        help="Show pale background grid lines. The default keeps only topology links and axis labels.",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args(argv)


def edge_key(src: int, dst: int) -> tuple[int, int]:
    src = int(src)
    dst = int(dst)
    return (src, dst) if src < dst else (dst, src)


def edge_keys_for_table(edge_table: EdgeTable) -> set[tuple[int, int]]:
    return {edge_key(int(edge_table.src[i]), int(edge_table.dst[i])) for i in range(edge_table.num_edges)}


def make_union_edge_table(edge_tables: dict[str, EdgeTable]) -> tuple[EdgeTable, dict[str, set[tuple[int, int]]]]:
    key_to_option: dict[tuple[int, int], int] = {}
    topology_keys: dict[str, set[tuple[int, int]]] = {}
    for topology, table in edge_tables.items():
        keys = set()
        for idx in range(table.num_edges):
            key = edge_key(int(table.src[idx]), int(table.dst[idx]))
            keys.add(key)
            key_to_option.setdefault(key, int(table.option[idx]))
        topology_keys[str(topology)] = keys

    src_values: list[int] = []
    dst_values: list[int] = []
    option_values: list[int] = []
    for (src, dst), option in sorted(key_to_option.items()):
        src_values.append(int(src))
        dst_values.append(int(dst))
        option_values.append(int(option))

    src_arr = np.asarray(src_values, dtype=np.int32)
    dst_arr = np.asarray(dst_values, dtype=np.int32)
    n = int(G60_CONFIG.N)
    union = EdgeTable(
        src=src_arr,
        dst=dst_arr,
        option=np.asarray(option_values, dtype=np.int16),
        src_plane=(src_arr // n).astype(np.int16),
        src_y=(src_arr % n).astype(np.int16),
        dst_plane=(dst_arr // n).astype(np.int16),
        dst_y=(dst_arr % n).astype(np.int16),
        sat_ids=[str(i + 1) for i in range(int(G60_CONFIG.total_sats))],
    )
    return union, topology_keys


def build_active_mask(
    *,
    union_edge_table: EdgeTable,
    topology_keys: dict[str, set[tuple[int, int]]],
    topology_by_row: list[str],
) -> np.ndarray:
    union_keys = [edge_key(int(union_edge_table.src[i]), int(union_edge_table.dst[i])) for i in range(union_edge_table.num_edges)]
    mask = np.zeros((len(topology_by_row), union_edge_table.num_edges), dtype=bool)
    for row, topology in enumerate(topology_by_row):
        keys = topology_keys[str(topology)]
        for edge_idx, key in enumerate(union_keys):
            if key in keys:
                mask[row, edge_idx] = True
    return mask


def load_motif_info(selected_motifs_csv: Path, segments_csv: Path) -> dict[str, dict[str, str]]:
    info: dict[str, dict[str, str]] = {}
    for path in [selected_motifs_csv, segments_csv]:
        if not Path(path).exists():
            continue
        with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if "topology" in row and row.get("topology"):
                    name = str(row["topology"])
                elif row.get("motif_id"):
                    name = f"motif_{int(row['motif_id']):06d}"
                else:
                    continue
                info[name] = {
                    "motif_id": str(row.get("motif_id", "")),
                    "motif": str(row.get("motif", "")),
                    "edges": str(row.get("edges", "")),
                }
    return info


def set_background_grid_visible(viewer: SatelliteTopology2DViewer, visible: bool) -> None:
    for item in viewer.scene.items():
        if isinstance(item, QtWidgets.QGraphicsLineItem):
            item.setVisible(bool(visible))


class DynamicScheduleTopologyViewer(SatelliteTopology2DViewer):
    def __init__(self, *args, topology_by_row: list[str], motif_info: dict[str, dict[str, str]], **kwargs):
        self.topology_by_row = list(topology_by_row)
        self.motif_info = dict(motif_info)
        super().__init__(*args, **kwargs)

    def update_step(self, row: int, *, sync_slider: bool = True):
        super().update_step(row, sync_slider=sync_slider)
        topology = self.topology_by_row[self.current_row]
        info = self.motif_info.get(topology, {})
        motif_text = info.get("motif", "")
        text = self.step_label.text()
        self.step_label.setText(f"{text} | topology {topology} | motif {motif_text}")


def write_topology_by_step(path: Path, by_step: pd.DataFrame) -> None:
    cols = [col for col in ["step", "topology", "delay_ms", "static_best_delay_ms", "oracle_delay_ms"] if col in by_step.columns]
    by_step[cols].to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schedule_dir = Path(args.schedule_dir)
    by_step_path = Path(args.by_step) if args.by_step else schedule_dir / "min_dwell_120min_by_step.csv"
    segments_path = Path(args.segments) if args.segments else schedule_dir / "min_dwell_120min_segments.csv"
    if not by_step_path.exists():
        raise FileNotFoundError(f"Missing schedule by-step CSV: {by_step_path}")
    if not segments_path.exists():
        raise FileNotFoundError(f"Missing schedule segment CSV: {segments_path}")

    by_step = pd.read_csv(by_step_path)
    if "step" not in by_step.columns or "topology" not in by_step.columns:
        raise ValueError("by-step CSV must contain 'step' and 'topology' columns.")

    steps = [int(x) for x in by_step["step"].tolist()]
    step_stride = int(steps[1] - steps[0]) if len(steps) > 1 else 1
    topology_by_row = [str(x) for x in by_step["topology"].tolist()]
    unique_topologies = list(dict.fromkeys(topology_by_row))
    motif_info = load_motif_info(Path(args.selected_motifs), segments_path)
    missing = [name for name in unique_topologies if name not in motif_info or not motif_info[name].get("motif")]
    if missing:
        raise ValueError(f"Missing motif definitions for: {missing}")

    edge_tables = {name: edge_table_for_motif(motif_info[name]["motif"]) for name in unique_topologies}
    union_edge_table, topology_keys = make_union_edge_table(edge_tables)
    active_mask = build_active_mask(
        union_edge_table=union_edge_table,
        topology_keys=topology_keys,
        topology_by_row=topology_by_row,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(union_edge_table, out_dir / "union_edges.csv")
    np.save(out_dir / "edge_active_mask.npy", active_mask)
    write_topology_by_step(out_dir / "topology_by_step.csv", by_step)
    pd.read_csv(segments_path).to_csv(out_dir / "topology_segments.csv", index=False)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "by_step": str(by_step_path),
                "segments": str(segments_path),
                "selected_motifs": str(args.selected_motifs),
                "steps": len(steps),
                "start_step": int(steps[0]),
                "end_step": int(steps[-1]),
                "step_stride": int(step_stride),
                "unique_topologies": unique_topologies,
                "union_edges": int(union_edge_table.num_edges),
                "active_edges_min": int(active_mask.sum(axis=1).min()),
                "active_edges_max": int(active_mask.sum(axis=1).max()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[dynamic-topology] steps={len(steps)} topologies={unique_topologies} "
        f"union_edges={union_edge_table.num_edges} active_edges={int(active_mask.sum(axis=1).min())}.."
        f"{int(active_mask.sum(axis=1).max())} out_dir={out_dir}",
        flush=True,
    )

    if args.check_only:
        return 0

    group_data = load_or_build_group_data(
        xml_file=Path(args.xml_file),
        group_cache_dir=Path(args.group_cache_dir),
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=int(G60_CONFIG.total_sats),
        constellation_name=G60_CONFIG.name,
        stride=int(step_stride),
        enabled=not bool(args.hide_groups),
        force=False,
    )

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = DynamicScheduleTopologyViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=union_edge_table,
        edge_active_mask=active_mask,
        topology_by_row=topology_by_row,
        motif_info=motif_info,
        window_title="G60 dynamic topology schedule: min dwell 120 min",
        group_data=group_data,
        show_groups=not bool(args.hide_groups),
    )
    set_background_grid_visible(viewer, bool(args.show_background_grid))
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
