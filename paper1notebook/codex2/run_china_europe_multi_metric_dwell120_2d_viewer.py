from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


GENERIC_ROOT = Path(__file__).resolve().parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module.edge_tables import build_motif_text_edge_table, make_edge_table_from_records  # noqa: E402


RUN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_w_le4_h_le3\multi_metric_dwell_china_europe_static056_t0_86160_stride60"
)
SCHEDULE_CSV = RUN_DIR / "china_europe_multi_metric_dwell120_by_step.csv"
SEGMENTS_CSV = RUN_DIR / "china_europe_multi_metric_dwell120_segments.csv"
MOTIF_LIBRARY_CSV = Path(
    r"E:\paper11\data\satnet_experiments\libraries\motif\exact_box"
    r"\w_le_4_h_le_3\combined_w_le4_h_le3_808.csv"
)
GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")
DEFAULT_SCREENSHOT = RUN_DIR / "china_europe_multi_metric_dwell120_2d_t0_86160_stride60.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the China-Europe multi-metric dwell120 dynamic schedule in the G60 2D viewer."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--schedule-csv", type=Path, default=SCHEDULE_CSV)
    parser.add_argument("--motif-library-csv", type=Path, default=MOTIF_LIBRARY_CSV)
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--width", type=int, default=1300)
    parser.add_argument("--height", type=int, default=860)
    return parser.parse_args()


def normalize_motif_name(name: str) -> str:
    text = str(name).strip()
    if text.startswith("combined_motif_"):
        return text
    if text.startswith("motif_"):
        return "combined_" + text
    return text


def read_schedule(path: Path, *, start: int, end: int, stride: int) -> tuple[list[int], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"schedule csv not found: {path}")
    wanted_steps = list(range(int(start), int(end) + 1, int(stride)))
    wanted_set = set(wanted_steps)
    by_step: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            if step in wanted_set:
                by_step[step] = normalize_motif_name(row["topology"])
    missing = [step for step in wanted_steps if step not in by_step]
    if missing:
        preview = ", ".join(str(x) for x in missing[:8])
        raise ValueError(f"schedule does not contain {len(missing)} requested steps; first missing: {preview}")
    return wanted_steps, [by_step[step] for step in wanted_steps]


def read_motif_texts(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"motif library csv not found: {path}")
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            motif_id = int(row["motif_id"])
            out[f"combined_motif_{motif_id:06d}"] = str(row["motif"]).strip()
    return out


def edge_key(edge_table, idx: int) -> tuple[int, int]:
    src = int(edge_table.src[idx])
    dst = int(edge_table.dst[idx])
    return (src, dst) if src < dst else (dst, src)


def edge_table_to_records(edge_table) -> list[tuple[int, int, int, int, int]]:
    records = []
    for idx in range(int(edge_table.num_edges)):
        records.append(
            (
                int(edge_table.src_plane[idx]),
                int(edge_table.src_y[idx]),
                int(edge_table.dst_plane[idx]),
                int(edge_table.dst_y[idx]),
                int(edge_table.option[idx]),
            )
        )
    return records


def build_dynamic_edge_table_and_mask(
    *,
    topology_names: list[str],
    motif_text_by_name: dict[str, str],
) -> tuple[object, np.ndarray, dict[str, str]]:
    unique_names = list(dict.fromkeys(topology_names))
    missing = [name for name in unique_names if name not in motif_text_by_name]
    if missing:
        raise KeyError(f"motif names missing in library: {missing[:8]}")

    edge_sets: dict[str, set[tuple[int, int]]] = {}
    union_records_by_key: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
    used_motif_text: dict[str, str] = {}
    for name in unique_names:
        motif_text = motif_text_by_name[name]
        used_motif_text[name] = motif_text
        table = build_motif_text_edge_table(
            motif_text=motif_text,
            config=G60_CONFIG,
            allow_vertical_overlap=True,
            allow_clipped_right=True,
            wrap_planes=False,
            add_intra_ring=True,
        )
        current_edges: set[tuple[int, int]] = set()
        for idx, record in enumerate(edge_table_to_records(table)):
            key = edge_key(table, idx)
            current_edges.add(key)
            union_records_by_key.setdefault(key, record)
        edge_sets[name] = current_edges

    union_table = make_edge_table_from_records(
        p=int(G60_CONFIG.P),
        n=int(G60_CONFIG.N),
        records=list(union_records_by_key.values()),
    )
    union_index = {edge_key(union_table, idx): idx for idx in range(int(union_table.num_edges))}
    active_mask = np.zeros((len(topology_names), int(union_table.num_edges)), dtype=bool)
    for row_idx, name in enumerate(topology_names):
        for key in edge_sets[name]:
            active_mask[row_idx, union_index[key]] = True
    return union_table, active_mask, used_motif_text


def load_group_data_for_steps(*, steps: list[int], stride: int, force: bool) -> dict:
    group_data = load_or_build_group_data(
        xml_file=GROUP_XML,
        group_cache_dir=GROUP_CACHE_DIR,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(stride),
        enabled=True,
        force=bool(force),
    )
    wanted = set(int(step) for step in steps)
    return {int(step): data for step, data in group_data.items() if int(step) in wanted}


def main() -> int:
    args = parse_args()
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    if int(args.end) < int(args.start):
        raise ValueError("--end must be >= --start")
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    steps, topology_names = read_schedule(
        args.schedule_csv,
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    motif_texts = read_motif_texts(args.motif_library_csv)
    edge_table, active_mask, used_motif_text = build_dynamic_edge_table_and_mask(
        topology_names=topology_names,
        motif_text_by_name=motif_texts,
    )

    group_data = {}
    if not bool(args.no_groups):
        group_data = load_group_data_for_steps(
            steps=steps,
            stride=int(args.stride),
            force=bool(args.force_group_cache),
        )

    unique_sequence = list(dict.fromkeys(topology_names))
    print(
        "[multi-metric-dwell-2d] "
        f"steps={len(steps)} start={steps[0]} end={steps[-1]} stride={args.stride} | "
        f"unique_motifs={len(unique_sequence)} | union_edges={edge_table.num_edges} | "
        f"active_edges_first={int(active_mask[0].sum())} | group_steps={len(group_data)}",
        flush=True,
    )
    print("[multi-metric-dwell-2d] sequence:", flush=True)
    for name in unique_sequence:
        print(f"  {name}: {used_motif_text[name]}", flush=True)
    print(f"[multi-metric-dwell-2d] segments={SEGMENTS_CSV}", flush=True)

    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=active_mask,
        window_title="G60 China-Europe multi-metric dwell120 dynamic topology",
        group_data=group_data,
        show_groups=bool(group_data),
        show_grid_lines=False,
        topology_edge_color="#000000",
        topology_edge_alpha=185,
        topology_edge_width=0.028,
        hide_y_wrap_edges=True,
    )
    viewer.edge_width = 0.036
    viewer.edge_alpha = 215
    viewer.node_radius = 0.125

    screenshot = args.screenshot
    if screenshot is None and bool(args.offscreen):
        screenshot = DEFAULT_SCREENSHOT
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        offscreen=bool(args.offscreen),
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
