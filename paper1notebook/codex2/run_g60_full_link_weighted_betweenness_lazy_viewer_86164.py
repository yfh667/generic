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
from src.satellite_topology_viewer.module.edge_usage_viewer import LazyEdgeUsageTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.multi_viewer import (  # noqa: E402
    Topology2DPanel,
    UnifiedControlTopology2DViewer,
)
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_metrics.module.weighted_edge_betweenness import (  # noqa: E402
    build_weighted_adjacency,
    weighted_edge_betweenness_between_node_sets,
)


DEFAULT_DELAY_STORE = Path(r"E:\paper11\data\linshi\G60_full_options_plus_intra_t0_86164_stride1")
DEFAULT_GROUP_XML = PROJECT_ROOT / "data" / "basic_file" / "G60" / "satellitesposition" / "station_visible_satellites_20250106.xml"
DEFAULT_GROUP_CACHE_DIR = PROJECT_ROOT / "data" / "satnet_experiments" / "caches" / "G60" / "group_data_cache"

PAIRS = {
    "china_europe": {
        "label": "China-Europe",
        "source_group_id": 2,
        "target_group_id": 3,
    },
    "china_america": {
        "label": "China-America",
        "source_group_id": 2,
        "target_group_id": 0,
    },
    "china_africa": {
        "label": "China-Africa",
        "source_group_id": 2,
        "target_group_id": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive lazy 2D viewer for G60 full-link weighted shortest-delay "
            "edge betweenness over 0..86164s."
        )
    )
    parser.add_argument("--delay-store", type=Path, default=DEFAULT_DELAY_STORE)
    parser.add_argument("--group-xml", type=Path, default=DEFAULT_GROUP_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86164)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=1900)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument(
        "--initial-value-max",
        type=float,
        default=300.0,
        help="Initial red scale before the first row is computed; shared scale updates per row.",
    )
    parser.add_argument(
        "--no-shared-value-scale",
        action="store_true",
        help="Disable per-row shared color/width scale across the three panels.",
    )
    return parser.parse_args()


def read_edge_table_csv(path: Path, *, total_sats: int) -> EdgeTable:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty edge CSV: {path}")
    sat_ids = [str(i + 1) for i in range(int(total_sats))]
    return EdgeTable(
        src=np.asarray([int(row["src_node"]) for row in rows], dtype=np.int32),
        dst=np.asarray([int(row["dst_node"]) for row in rows], dtype=np.int32),
        option=np.asarray([int(row["option"]) for row in rows], dtype=np.int16),
        src_plane=np.asarray([int(row["src_plane"]) for row in rows], dtype=np.int16),
        src_y=np.asarray([int(row["src_y"]) for row in rows], dtype=np.int16),
        dst_plane=np.asarray([int(row["dst_plane"]) for row in rows], dtype=np.int16),
        dst_y=np.asarray([int(row["dst_y"]) for row in rows], dtype=np.int16),
        sat_ids=sat_ids,
    )


def validate_steps(requested_steps: list[int], store_steps: np.ndarray) -> dict[int, int]:
    index = {int(step): idx for idx, step in enumerate(np.asarray(store_steps, dtype=np.int64))}
    missing = [step for step in requested_steps if int(step) not in index]
    if missing:
        preview = ", ".join(str(x) for x in missing[:8])
        raise ValueError(f"delay store does not contain requested steps: {preview}")
    return {int(step): int(index[int(step)]) for step in requested_steps}


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty step range")

    delay_store = Path(args.delay_store)
    edge_table = read_edge_table_csv(delay_store / "edges.csv", total_sats=int(G60_CONFIG.total_sats))
    delay_ms = np.load(delay_store / "edge_delay_ms.npy", mmap_mode="r")
    store_steps = np.load(delay_store / "time_indices.npy", mmap_mode="r")
    step_to_row = validate_steps(steps, store_steps)
    if int(delay_ms.shape[1]) != int(edge_table.num_edges):
        raise ValueError(f"delay cols {delay_ms.shape[1]} != edge count {edge_table.num_edges}")

    group_data = load_or_build_group_data(
        xml_file=Path(args.group_xml),
        group_cache_dir=Path(args.group_cache_dir),
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=False,
    )
    adjacency = build_weighted_adjacency(edge_table, int(G60_CONFIG.total_sats))
    active_mask = np.ones((1, int(edge_table.num_edges)), dtype=bool)

    print(
        f"[lazy-full-link-betweenness] steps={len(steps)} range={steps[0]}..{steps[-1]} "
        f"stride={int(args.stride)} edges={edge_table.num_edges} delay_store={delay_store}",
        flush=True,
    )

    if bool(args.check_only):
        for pair_key, pair in PAIRS.items():
            weights = np.asarray(delay_ms[step_to_row[int(steps[0])]], dtype=np.float32)
            sources = group_nodes_for_step(group_data, int(steps[0]), int(pair["source_group_id"]))
            targets = group_nodes_for_step(group_data, int(steps[0]), int(pair["target_group_id"]))
            values, summary, _samples = weighted_edge_betweenness_between_node_sets(
                edge_table,
                total_nodes=int(G60_CONFIG.total_sats),
                source_nodes=sources,
                target_nodes=targets,
                weights=weights,
                adjacency=adjacency,
                sample_path_limit=0,
            )
            print(
                f"[lazy-full-link-betweenness] check {pair['label']}: "
                f"src={summary.source_nodes} dst={summary.target_nodes} "
                f"mean_delay={summary.mean_shortest_weight:.4f} "
                f"max_usage={summary.max_edge_betweenness:.1f} "
                f"nonzero={summary.nonzero_edges} sum={float(np.sum(values)):.1f}",
                flush=True,
            )
        return 0

    if bool(args.offscreen):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    def make_provider(pair_key: str):
        pair = PAIRS[pair_key]

        def provider(row: int, step: int):
            row = int(row)
            step = int(step)
            store_row = step_to_row[step]
            weights = np.asarray(delay_ms[store_row], dtype=np.float32)
            sources = group_nodes_for_step(group_data, step, int(pair["source_group_id"]))
            targets = group_nodes_for_step(group_data, step, int(pair["target_group_id"]))
            values, summary, _samples = weighted_edge_betweenness_between_node_sets(
                edge_table,
                total_nodes=int(G60_CONFIG.total_sats),
                source_nodes=sources,
                target_nodes=targets,
                weights=weights,
                adjacency=adjacency,
                sample_path_limit=0,
            )
            status = (
                f"{pair['label']} weighted SP | "
                f"src={summary.source_nodes}, dst={summary.target_nodes}, "
                f"mean_delay={summary.mean_shortest_weight:.3f} ms, "
                f"max_usage={summary.max_edge_betweenness:.0f}, "
                f"used_edges={summary.nonzero_edges}"
            )
            return values, status

        return provider

    panels: list[Topology2DPanel] = []
    for pair_key, pair in PAIRS.items():
        viewer = LazyEdgeUsageTopology2DViewer(
            G60_CONFIG,
            steps=steps,
            edge_table=edge_table,
            edge_usage_provider=make_provider(pair_key),
            initial_value_max=float(args.initial_value_max),
            edge_active_mask=active_mask,
            window_title=f"G60 full-link weighted betweenness {pair['label']} {steps[0]}..{steps[-1]}s",
            group_data=group_data,
            show_groups=True,
        )
        viewer.topology_edge_alpha = 95
        viewer.topology_edge_width = 0.010
        viewer.value_width_min = 0.010
        viewer.value_width_max = 0.120
        panels.append(Topology2DPanel(title=str(pair["label"]), viewer=viewer, stretch=1))

    combined = UnifiedControlTopology2DViewer(
        title=f"G60 full-link weighted shortest-delay edge betweenness {steps[0]}..{steps[-1]}s",
        panels=panels,
        shared_value_scale=not bool(args.no_shared_value_scale),
        orientation=QtCore.Qt.Horizontal,
    )
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
