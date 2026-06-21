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


OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_w_le4_h_le3\two_motif_056_330_china_europe_t0_86160_stride60"
)
MOTIF_LIBRARY_CSV = Path(
    r"E:\paper11\data\satnet_experiments\libraries\motif\exact_box"
    r"\w_le_4_h_le_3\combined_w_le4_h_le3_808.csv"
)
GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")

MOTIF_056 = "combined_motif_000056"
MOTIF_330 = "combined_motif_000330"
SWITCH_START = 41100
SWITCH_END = 48240
DEFAULT_INITIAL_STEP = 43020


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a two-motif G60 dynamic topology: motif 000056, switching to 000330 around t=43020s."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--switch-start", type=int, default=SWITCH_START)
    parser.add_argument("--switch-end", type=int, default=SWITCH_END)
    parser.add_argument("--initial-step", type=int, default=DEFAULT_INITIAL_STEP)
    parser.add_argument("--motif-library-csv", type=Path, default=MOTIF_LIBRARY_CSV)
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--width", type=int, default=1300)
    parser.add_argument("--height", type=int, default=860)
    return parser.parse_args()


def read_motif_texts(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"motif library csv not found: {path}")
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            motif_id = int(row["motif_id"])
            out[f"combined_motif_{motif_id:06d}"] = str(row["motif"]).strip()
    return out


def make_topology_names(steps: list[int], *, switch_start: int, switch_end: int) -> list[str]:
    return [
        MOTIF_330 if int(switch_start) <= int(step) <= int(switch_end) else MOTIF_056
        for step in steps
    ]


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

    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    topology_names = make_topology_names(
        steps,
        switch_start=int(args.switch_start),
        switch_end=int(args.switch_end),
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

    initial_step = int(args.initial_step)
    if initial_step not in set(steps):
        raise ValueError(f"--initial-step {initial_step} is not on the requested time axis")
    initial_row = steps.index(initial_step)

    print(
        "[two-motif-056-330-2d] "
        f"steps={len(steps)} start={steps[0]} end={steps[-1]} stride={args.stride} | "
        f"switch={args.switch_start}..{args.switch_end} | initial_step={initial_step} | "
        f"union_edges={edge_table.num_edges} | active_edges_initial={int(active_mask[initial_row].sum())} | "
        f"group_steps={len(group_data)}",
        flush=True,
    )
    for name in [MOTIF_056, MOTIF_330]:
        print(f"[two-motif-056-330-2d] {name}: {used_motif_text[name]}", flush=True)

    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=active_mask,
        window_title="G60 China-Europe two-motif topology: 000056 + 000330",
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
    viewer.update_step(initial_row)

    screenshot = args.screenshot
    if screenshot is None and bool(args.offscreen):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        screenshot = OUT_DIR / f"two_motif_056_330_2d_t{initial_step}.png"
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        offscreen=bool(args.offscreen),
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
