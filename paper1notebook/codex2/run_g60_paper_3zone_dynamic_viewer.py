from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.config.viewer_config import G60_CONFIG
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.examples.run_g60_motif_2x2_link_setup_viewer import DEFAULT_GROUP_CACHE, DEFAULT_XML

from paper_3zone_dynamic_viewer import (
    Paper3ZoneDynamicViewer,
    load_or_build_paper_3zone_topology_cache,
    read_level_wise_best_motif,
)


REPRO_ROOT = Path("E:/") / "\u590d\u73b0" / "satnetwork.github.io"
DEFAULT_AUTHOR_MOTIF_FILE = (
    REPRO_ROOT
    / "output_data"
    / "multi_motif_40_40_53deg_5014"
    / "level_wise_best_motif.txt"
)
DEFAULT_POSITION_CACHE = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "_position_cache"
    / "cache_0_86164_1s"
)
DEFAULT_SCREENSHOT = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "runs"
    / "paper1"
    / "G60"
    / "paper_3zone_dynamic_demo"
    / "g60_paper_3zone_dynamic_0_10s.png"
)
DEFAULT_TOPOLOGY_CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "runs"
    / "paper1"
    / "G60"
    / "paper_3zone_dynamic_demo"
    / "cache"
)


def default_topology_cache_dir(
    *,
    start: int,
    end: int,
    stride: int,
    author_motif_file: Path,
    max_isl_km: float | None,
    wrap_planes: bool,
) -> Path:
    range_tag = "range_none" if max_isl_km is None else f"range{float(max_isl_km):g}km"
    wrap_tag = "wrap1" if wrap_planes else "wrap0"
    motif_tag = Path(author_motif_file).parent.name
    return DEFAULT_TOPOLOGY_CACHE_ROOT / f"{motif_tag}_t{int(start)}_{int(end)}_stride{int(stride)}_{range_tag}_{wrap_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce satnetwork 3-zone multi-motif links on the G60 2D topology viewer."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--author-motif-file", type=Path, default=DEFAULT_AUTHOR_MOTIF_FILE)
    parser.add_argument("--position-cache-dir", type=Path, default=DEFAULT_POSITION_CACHE)
    parser.add_argument("--max-isl-km", type=float, default=5014.0)
    parser.add_argument("--no-range-filter", action="store_true")
    parser.add_argument("--no-plane-wrap", action="store_true")
    parser.add_argument("--topology-cache-dir", type=Path, default=None)
    parser.add_argument("--force-topology-cache", action="store_true")
    parser.add_argument("--no-topology-cache", action="store_true")
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty time range; check --start/--end/--stride")
    motifs = read_level_wise_best_motif(Path(args.author_motif_file))
    max_isl_km = None if bool(args.no_range_filter) else float(args.max_isl_km)
    topology_cache_dir = None if bool(args.no_topology_cache) else args.topology_cache_dir
    if topology_cache_dir is None and not bool(args.no_topology_cache):
        topology_cache_dir = default_topology_cache_dir(
            start=int(args.start),
            end=int(args.end),
            stride=int(args.stride),
            author_motif_file=Path(args.author_motif_file),
            max_isl_km=max_isl_km,
            wrap_planes=not bool(args.no_plane_wrap),
        )
    series = load_or_build_paper_3zone_topology_cache(
        config=G60_CONFIG,
        steps=steps,
        position_cache_dir=Path(args.position_cache_dir),
        motifs=motifs,
        cache_dir=topology_cache_dir,
        force=bool(args.force_topology_cache),
        max_isl_km=max_isl_km,
        wrap_planes=not bool(args.no_plane_wrap),
        author_motif_file=Path(args.author_motif_file),
    )

    active_counts = series.edge_active_mask.sum(axis=1)
    print(
        f"[paper-3zone] steps={steps[0]}..{steps[-1]} rows={len(steps)} "
        f"union_edges={series.edge_table.num_edges} "
        f"active_edges_min={int(active_counts.min())} active_edges_max={int(active_counts.max())} "
        f"range_filter={max_isl_km} author_motif={args.author_motif_file}",
        flush=True,
    )
    for idx, motif in enumerate(motifs):
        print(f"[paper-3zone] zone{idx}: {motif.label}", flush=True)

    if args.check_only:
        return 0

    group_data = {}
    if not args.no_groups:
        group_data = load_or_build_group_data(
            xml_file=Path(args.xml_file),
            group_cache_dir=Path(args.group_cache_dir),
            steps=steps,
            station_groups=G60_CONFIG.station_groups,
            total_sats=G60_CONFIG.total_sats,
            constellation_name=G60_CONFIG.name,
            stride=int(args.stride),
            enabled=True,
            force=bool(args.force_group_cache),
        )

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = Paper3ZoneDynamicViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=series.edge_table,
        edge_active_mask=series.edge_active_mask,
        edge_zone_ids=series.edge_zone_ids,
        latitude_deg=series.latitude_deg,
        latitude_zone_ids=series.latitude_zone_ids,
        latitude_zones=series.latitude_zones,
        zone_motifs=series.motifs,
        group_data=group_data,
        show_groups=not bool(args.no_groups),
        show_grid_lines=False,
        window_title=(
            f"G60 paper 3-zone dynamic links | {Path(args.author_motif_file).parent.name} | "
            f"{steps[0]}..{steps[-1]}s"
        ),
        topology_edge_alpha=150,
    )

    screenshot = args.screenshot
    if screenshot is None and args.offscreen:
        screenshot = DEFAULT_SCREENSHOT
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
