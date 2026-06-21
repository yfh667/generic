from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.edge_usage_viewer import EdgeUsageTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.multi_viewer import (  # noqa: E402
    Topology2DPanel,
    UnifiedControlTopology2DViewer,
)
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module.edge_tables import build_motif_text_edge_table  # noqa: E402


DEFAULT_EXPERIMENT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_0056_0061_common_spike_analysis\full_link_guided_spike_patch"
    r"\t0_86160_stride60"
)
DEFAULT_FULL_LINK_CACHE = Path(r"E:\paper11\data\linshi\g60_multi_region_weighted_betweenness_t0_86164_stride60")
DEFAULT_GROUP_XML = PROJECT_ROOT / "data" / "basic_file" / "G60" / "satellitesposition" / "station_visible_satellites_20250106.xml"
DEFAULT_GROUP_CACHE_DIR = PROJECT_ROOT / "data" / "satnet_experiments" / "caches" / "G60" / "group_data_cache"
PAIR_KEY = "china_europe"
MOTIF000056 = "DBD | --B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive 2D viewer for the full-link-guided dynamic one-right topology. "
            "Panels share the same full-link edge table: full_link, dynamic selected one-right, and motif000056."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--full-link-cache", type=Path, default=DEFAULT_FULL_LINK_CACHE)
    parser.add_argument("--pair", choices=[PAIR_KEY], default=PAIR_KEY)
    parser.add_argument("--group-xml", type=Path, default=DEFAULT_GROUP_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--step", type=int, default=None, help="Initial displayed time step, e.g. 5220.")
    parser.add_argument("--width", type=int, default=1900)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--no-shared-value-scale", action="store_true")
    return parser.parse_args()


def edge_key(src: int, dst: int) -> tuple[int, int]:
    src = int(src)
    dst = int(dst)
    return (src, dst) if src <= dst else (dst, src)


def read_edge_table_csv(path: Path, *, total_sats: int) -> EdgeTable:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty edge CSV: {path}")
    return EdgeTable(
        src=np.asarray([int(row["src_node"]) for row in rows], dtype=np.int32),
        dst=np.asarray([int(row["dst_node"]) for row in rows], dtype=np.int32),
        option=np.asarray([int(row["option"]) for row in rows], dtype=np.int16),
        src_plane=np.asarray([int(row["src_plane"]) for row in rows], dtype=np.int16),
        src_y=np.asarray([int(row["src_y"]) for row in rows], dtype=np.int16),
        dst_plane=np.asarray([int(row["dst_plane"]) for row in rows], dtype=np.int16),
        dst_y=np.asarray([int(row["dst_y"]) for row in rows], dtype=np.int16),
        sat_ids=[str(i + 1) for i in range(int(total_sats))],
    )


def motif_active_mask_on_full_table(*, full_edge_table: EdgeTable, motif_text: str) -> np.ndarray:
    motif_table = build_motif_text_edge_table(motif_text=motif_text, config=G60_CONFIG, add_intra_ring=True)
    motif_keys = {edge_key(int(motif_table.src[idx]), int(motif_table.dst[idx])) for idx in range(motif_table.num_edges)}
    mask = np.zeros((1, int(full_edge_table.num_edges)), dtype=bool)
    for idx in range(int(full_edge_table.num_edges)):
        if edge_key(int(full_edge_table.src[idx]), int(full_edge_table.dst[idx])) in motif_keys:
            mask[0, idx] = True
    return mask


def make_panel(
    *,
    title: str,
    steps: list[int],
    edge_table: EdgeTable,
    values: np.ndarray,
    active_mask: np.ndarray,
    group_data: dict,
    value_max: float,
) -> Topology2DPanel:
    viewer = EdgeUsageTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        edge_usage_values=values,
        value_max=float(value_max),
        edge_active_mask=active_mask,
        window_title=title,
        group_data=group_data,
        show_groups=True,
        show_topology_under_edge_values=True,
        zero_value_edges_visible=False,
        topology_edge_alpha=145,
        topology_edge_width=0.014,
        value_width_min=0.010,
        value_width_max=0.120,
        value_alpha_min=35,
        value_alpha_max=245,
        show_grid_lines=False,
    )
    return Topology2DPanel(title=title, viewer=viewer, stretch=1)


def main() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    cache_dir = Path(args.full_link_cache)

    steps = [int(x) for x in np.asarray(np.load(cache_dir / PAIR_KEY / "time_indices.npy"), dtype=np.int64)]
    edge_table = read_edge_table_csv(cache_dir / "edges.csv", total_sats=int(G60_CONFIG.total_sats))
    values = np.load(cache_dir / PAIR_KEY / "edge_betweenness.npy", mmap_mode="r")
    if values.shape != (len(steps), int(edge_table.num_edges)):
        raise ValueError(f"values shape {values.shape} != ({len(steps)}, {edge_table.num_edges})")

    dynamic_mask = np.load(experiment_dir / "dynamic_one_right_active_full_edge_mask.npy", mmap_mode="r")
    if dynamic_mask.shape != (len(steps), int(edge_table.num_edges)):
        raise ValueError(f"dynamic active mask shape {dynamic_mask.shape} != ({len(steps)}, {edge_table.num_edges})")
    full_mask = np.ones((1, int(edge_table.num_edges)), dtype=bool)
    motif56_mask = motif_active_mask_on_full_table(full_edge_table=edge_table, motif_text=MOTIF000056)

    stride = int(steps[1] - steps[0]) if len(steps) > 1 else 1
    group_data = load_or_build_group_data(
        xml_file=Path(args.group_xml),
        group_cache_dir=Path(args.group_cache_dir),
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=stride,
        enabled=True,
        force=False,
    )

    value_max = max(1.0, float(np.nanmax(values)))
    initial_row = 0
    if args.step is not None:
        matches = [idx for idx, step in enumerate(steps) if int(step) == int(args.step)]
        if not matches:
            raise ValueError(f"step {args.step} not found in cached steps {steps[0]}..{steps[-1]} stride {stride}")
        initial_row = int(matches[0])

    print(
        f"[dynamic-one-right-viewer] steps={len(steps)} range={steps[0]}..{steps[-1]} stride={stride} "
        f"edges={edge_table.num_edges} dynamic_mask={dynamic_mask.shape} value_max={value_max:.1f}",
        flush=True,
    )
    print(
        f"[dynamic-one-right-viewer] motif000056 active edges={int(np.count_nonzero(motif56_mask))} "
        f"dynamic selected mean={float(np.mean(np.count_nonzero(dynamic_mask, axis=1))):.1f}",
        flush=True,
    )
    if bool(args.check_only):
        return 0

    if bool(args.offscreen):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    panels = [
        make_panel(
            title="full_link weighted usage",
            steps=steps,
            edge_table=edge_table,
            values=values,
            active_mask=full_mask,
            group_data=group_data,
            value_max=value_max,
        ),
        make_panel(
            title="dynamic one-right selected from full_link usage",
            steps=steps,
            edge_table=edge_table,
            values=values,
            active_mask=dynamic_mask,
            group_data=group_data,
            value_max=value_max,
        ),
        make_panel(
            title="motif000056 DBD | --B on full-link edge table",
            steps=steps,
            edge_table=edge_table,
            values=values,
            active_mask=motif56_mask,
            group_data=group_data,
            value_max=value_max,
        ),
    ]
    combined = UnifiedControlTopology2DViewer(
        title="G60 China-Europe full-link-guided dynamic one-right spike study",
        panels=panels,
        shared_value_scale=not bool(args.no_shared_value_scale),
        orientation=QtCore.Qt.Horizontal,
    )
    if initial_row:
        combined.master.update_step(initial_row)
    return run_viewer_widget(
        combined,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
