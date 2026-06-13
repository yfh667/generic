from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


GENERIC_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.group_edge_betweenness import (
    compute_group_pair_edge_betweenness_by_step,
    write_group_betweenness_outputs,
)
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.satellite_topology_viewer.module.topology_edges import build_full_option_plus_intra_edges


DEFAULT_XML = PROJECT_ROOT / "data" / "basic_file" / "G60" / "satellitesposition" / "station_visible_satellites_20250106.xml"
DEFAULT_GROUP_CACHE = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "full_option_edge_delay"
    / "group_data_cache"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "postprocess"
    / "satellite_topology_viewer"
    / "g60_group_betweenness_t0_10"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize G60 China-Europe group-pair edge betweenness with intra y-ring links."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--source-group", type=int, default=2, help="Default G60 group 2 is China.")
    parser.add_argument("--target-group", type=int, default=3, help="Default G60 group 3 is Europe.")
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args(argv)


def group_name(group_id: int) -> str:
    return str(G60_CONFIG.station_groups.get(int(group_id), {}).get("name", f"Group {group_id}"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    edge_table = build_full_option_plus_intra_edges(G60_CONFIG)
    group_data = load_or_build_group_data(
        xml_file=args.xml_file,
        group_cache_dir=args.group_cache_dir,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )

    values, summaries, path_samples = compute_group_pair_edge_betweenness_by_step(
        edge_table=edge_table,
        total_nodes=G60_CONFIG.total_sats,
        steps=steps,
        group_data=group_data,
        source_group_id=int(args.source_group),
        target_group_id=int(args.target_group),
        sample_path_limit_per_step=50,
    )
    write_group_betweenness_outputs(
        out_dir=args.out_dir,
        edge_table=edge_table,
        steps=steps,
        values=values,
        summaries=summaries,
        path_samples=path_samples,
        source_group_name=group_name(args.source_group),
        target_group_name=group_name(args.target_group),
    )

    print(
        f"[group-betweenness] steps={len(steps)} edges={edge_table.num_edges} "
        f"values=({float(np.nanmin(values)):.4f}, {float(np.nanmax(values)):.4f}) "
        f"out_dir={args.out_dir}",
        flush=True,
    )
    for summary in summaries[: min(3, len(summaries))]:
        print(
            f"[group-betweenness] step={summary.step} "
            f"{group_name(args.source_group)}_nodes={summary.source_nodes} "
            f"{group_name(args.target_group)}_nodes={summary.target_nodes} "
            f"pairs={summary.reachable_pairs} mean_hops={summary.mean_shortest_distance_hops:.3f} "
            f"max_edge={summary.max_edge_betweenness:.3f}",
            flush=True,
        )

    if args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        edge_values=values,
        value_min=0.0,
        value_max=float(np.nanmax(values)) if values.size else 0.0,
        edge_value_label="edge_betweenness",
        scale_edge_width_by_value=True,
        value_width_min=0.006,
        value_width_max=0.085,
        value_color_mode="red_alpha",
        value_solid_color="#C1121F",
        value_alpha_min=28,
        value_alpha_max=230,
        zero_value_edges_visible=False,
        zero_value_threshold=0.0,
        show_topology_under_edge_values=True,
        topology_edge_color="#000000",
        topology_edge_alpha=155,
        topology_edge_width=0.014,
        window_title=(
            f"G60 {group_name(args.source_group)}-{group_name(args.target_group)} "
            f"edge betweenness {steps[0]}..{steps[-1]}s"
        ),
        group_data=group_data,
        show_groups=True,
    )
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
