from __future__ import annotations

import argparse
import csv
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
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.link_switch_viewer import (  # noqa: E402
    LinkSwitchTopologyViewer,
    make_link_switch_viewer_data,
)
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.stores import MetricStoreLayout, expand_unique_state_values  # noqa: E402


DEFAULT_HOP_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3"
    r"\dual_edge_usage_056_061_china_europe\edge_betweenness_cache"
)
DEFAULT_DELAY_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\delay_edge_usage_056_061_china_europe"
)
DEFAULT_DELAY_TRACKED_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_switch_056_061_056_china_europe_1s_delay"
    r"\t0_86160_stride1\switch36000_54000\setup600_delay\tracked_usage"
)
GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show a static G60 motif topology time series with China-Europe "
            "hop/delay edge usage and pop-up node usage inspector."
        )
    )
    parser.add_argument("--motif-id", type=int, choices=(56, 61), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--initial-step", type=int, default=43020)
    parser.add_argument("--initial-node", type=int, default=None)
    parser.add_argument("--value-mode", choices=("hops", "delay", "any", "primary", "secondary"), default="delay")
    parser.add_argument("--working-mode", choices=("hops", "delay", "any", "primary", "secondary"), default="any")
    parser.add_argument("--inspector-mode", choices=("window", "right-panel"), default="window")
    parser.add_argument("--hop-root", type=Path, default=DEFAULT_HOP_ROOT)
    parser.add_argument("--delay-root", type=Path, default=DEFAULT_DELAY_ROOT)
    parser.add_argument(
        "--delay-tracked-root",
        type=Path,
        default=DEFAULT_DELAY_TRACKED_ROOT,
        help=(
            "Fallback 1-second delay edge-usage store. Used when --delay-root does not contain "
            "the requested t{start}_{end}_stride{stride} cache."
        ),
    )
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def parse_edge_key_text(text: str) -> tuple[int, int]:
    left, right = str(text).strip().split("-", 1)
    a = int(left)
    b = int(right)
    return (a, b) if a <= b else (b, a)


def read_edge_table(edges_csv: Path) -> EdgeTable:
    src: list[int] = []
    dst: list[int] = []
    option: list[int] = []
    src_plane: list[int] = []
    src_y: list[int] = []
    dst_plane: list[int] = []
    dst_y: list[int] = []
    with Path(edges_csv).open("r", encoding="utf-8-sig", newline="") as f:
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
        sat_ids=[str(idx + 1) for idx in range(int(G60_CONFIG.total_sats))],
    )


def load_tracked_delay_values(
    *,
    tracked_root: Path,
    topology_name: str,
    steps: np.ndarray,
    edge_table: EdgeTable,
) -> np.ndarray:
    tracked_dir = Path(tracked_root) / topology_name
    steps_path = tracked_dir / "time_indices.npy"
    counts_path = tracked_dir / "tracked_edge_usage_counts.npy"
    edges_path = tracked_dir / "tracked_edges.csv"
    missing = [path for path in (steps_path, counts_path, edges_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "delay usage cache was not found in --delay-root, and fallback tracked store is missing files: "
            + ", ".join(str(path) for path in missing)
        )

    tracked_steps = np.asarray(np.load(steps_path, mmap_mode="r"), dtype=np.int64)
    wanted_steps = np.asarray(steps, dtype=np.int64)
    rows = np.searchsorted(tracked_steps, wanted_steps)
    bad = (rows >= tracked_steps.size) | (np.asarray(tracked_steps[rows], dtype=np.int64) != wanted_steps)
    if bool(np.any(bad)):
        examples = wanted_steps[np.flatnonzero(bad)[:10]]
        raise KeyError(f"tracked delay store does not contain requested steps, examples={examples.tolist()}")

    key_to_edge_idx = {
        parse_edge_key_text(f"{int(edge_table.src[idx])}-{int(edge_table.dst[idx])}"): int(idx)
        for idx in range(int(edge_table.num_edges))
    }
    tracked_cols: list[int] = []
    target_edge_indices: list[int] = []
    with edges_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            edge_idx = key_to_edge_idx.get(parse_edge_key_text(row["edge_key"]))
            if edge_idx is None:
                continue
            tracked_cols.append(int(row["tracked_col"]))
            target_edge_indices.append(int(edge_idx))

    counts = np.load(counts_path, mmap_mode="r")
    delay_values = np.zeros((int(wanted_steps.size), int(edge_table.num_edges)), dtype=np.float32)
    if tracked_cols:
        delay_values[:, np.asarray(target_edge_indices, dtype=np.int64)] = np.asarray(
            counts[np.ix_(rows.astype(np.int64), np.asarray(tracked_cols, dtype=np.int64))],
            dtype=np.float32,
        )
    print(
        "[static-motif-node-usage-viewer] "
        f"using fallback tracked delay usage: {tracked_dir} "
        f"tracked_edges={len(tracked_cols)} full_edges={edge_table.num_edges}",
        flush=True,
    )
    return delay_values


def load_static_motif_payload(
    *,
    motif_id: int,
    start: int,
    end: int,
    stride: int,
    hop_root: Path,
    delay_root: Path,
    delay_tracked_root: Path,
):
    topology_name = f"combined_motif_{int(motif_id):06d}"
    window_name = f"t{int(start)}_{int(end)}_stride{int(stride)}"
    hop_dir = Path(hop_root) / window_name / "china_europe" / topology_name
    delay_dir = Path(delay_root) / window_name / topology_name
    if not hop_dir.exists():
        raise FileNotFoundError(f"hop usage cache not found: {hop_dir}")

    layout = MetricStoreLayout(hop_dir)
    steps = np.asarray(np.load(layout.time_indices_npy), dtype=np.int64)
    edge_table = read_edge_table(layout.edges_csv)
    hop_values = expand_unique_state_values(
        unique_state_values=np.load(layout.unique_state_values_npy, mmap_mode="r"),
        state_ids=np.load(layout.state_ids_npy, mmap_mode="r"),
    ).astype(np.float32)

    if delay_dir.exists():
        delay_steps = np.asarray(np.load(delay_dir / "time_indices.npy"), dtype=np.int64)
        delay_values = np.asarray(
            np.load(delay_dir / "china_europe" / "edge_betweenness.npy", mmap_mode="r"),
            dtype=np.float32,
        )
        if not np.array_equal(steps, delay_steps):
            raise ValueError("hop and delay usage caches have different time axes")
    else:
        delay_values = load_tracked_delay_values(
            tracked_root=Path(delay_tracked_root),
            topology_name=topology_name,
            steps=steps,
            edge_table=edge_table,
        )
    if hop_values.shape != delay_values.shape:
        raise ValueError(f"hop shape {hop_values.shape} != delay shape {delay_values.shape}")

    active_mask = np.ones((int(steps.size), int(edge_table.num_edges)), dtype=bool)
    return topology_name, steps, edge_table, active_mask, hop_values, delay_values


def main() -> int:
    args = parse_args()
    topology_name, steps_array, edge_table, active_mask, hop_values, delay_values = load_static_motif_payload(
        motif_id=int(args.motif_id),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
        hop_root=Path(args.hop_root),
        delay_root=Path(args.delay_root),
        delay_tracked_root=Path(args.delay_tracked_root),
    )
    steps = [int(x) for x in steps_array]
    initial_row = int(np.argmin(np.abs(steps_array - int(args.initial_step))))
    actual_initial_step = int(steps_array[initial_row])

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

    data = make_link_switch_viewer_data(
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=active_mask,
        edge_betweenness_primary=hop_values,
        edge_betweenness_secondary=delay_values,
        primary_label="hops",
        secondary_label="delay",
        meta={"topology": topology_name, "pair": "china_europe"},
    )
    viewer = LinkSwitchTopologyViewer(
        G60_CONFIG,
        data=data,
        value_mode=str(args.value_mode),
        working_mode=str(args.working_mode),
        inspector_mode=str(args.inspector_mode),
        topology_motif_id=np.full(len(steps), int(args.motif_id), dtype=np.int16),
        window_title=f"G60 motif{int(args.motif_id):06d} China-Europe node usage",
        group_data=group_data,
        show_groups=not bool(args.no_groups),
    )
    viewer.edge_width = 0.040
    viewer.edge_alpha = 210
    viewer.node_radius = 0.125
    viewer.update_step(initial_row)
    if args.initial_node is not None:
        viewer.pick_node(int(args.initial_node))

    print(
        "[static-motif-node-usage-viewer] "
        f"motif={int(args.motif_id):06d} steps={len(steps)} actual_range={steps[0]}..{steps[-1]} "
        f"initial_step={actual_initial_step} edges={edge_table.num_edges} group_steps={len(group_data)}",
        flush=True,
    )
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
