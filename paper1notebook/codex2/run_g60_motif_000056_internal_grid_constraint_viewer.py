from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import write_edges_csv
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.module.edge_tables import build_single_motif_edge_table
from src.topology_workflow.module.region_constraints import (
    apply_region_internal_option_constraint,
    build_region_internal_option_edges,
)


DEFAULT_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE = PROJECT_ROOT / "data" / "satnet_experiments" / "caches" / "G60" / "group_data_cache"
DEFAULT_OUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "runs"
    / "paper1"
    / "G60"
    / "internal_grid_constraint"
    / "motif_000056_DBD_xxB"
)

MOTIF_000056 = {
    "name": "motif_000056_DBD_xxB",
    "w": 3,
    "h": 3,
    "offsets": {
        "A": [1, 0],
        "B": [1, -1],
        "C": [1, 1],
        "D": [2, 0],
    },
    "support": [
        [0, 0, "D"],
        [0, 1, "B"],
        [0, 2, "D"],
        [1, 2, "B"],
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw motif_000056 after enforcing China/Europe internal +grid links. "
            "The base topology is the motif tiled over the full G60 grid, plus intra y-ring links."
        )
    )
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--constrained-groups", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--grid-option", type=int, default=0)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--hide-groups", action="store_true")
    return parser.parse_args()


def group_name(group_id: int) -> str:
    return str(G60_CONFIG.station_groups.get(int(group_id), {}).get("name", f"Group {group_id}"))


def default_out_dir(step: int, constrained_groups: list[int]) -> Path:
    group_text = "_".join(str(int(x)) for x in constrained_groups)
    return DEFAULT_OUT_ROOT / f"step{int(step)}_groups{group_text}"


def build_base_motif_edge_table():
    return build_single_motif_edge_table(
        topology_raw={
            "kind": "single_motif",
            "motif": MOTIF_000056,
            "tiling": {
                "allow_vertical_overlap": True,
                "allow_clipped_right": True,
            },
            "add_intra_ring": True,
        },
        config=G60_CONFIG,
    )


def main() -> int:
    args = parse_args()
    step = int(args.step)
    constrained_groups = [int(x) for x in args.constrained_groups]
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_out_dir(step, constrained_groups)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_data = load_or_build_group_data(
        xml_file=args.xml_file,
        group_cache_dir=args.group_cache_dir,
        steps=[step],
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=1,
        enabled=True,
        force=bool(args.force_group_cache),
    )
    if step not in group_data:
        raise ValueError(f"missing group data for step={step}")

    groups_obj = group_data[step].get("groups", {})
    group_nodes = {gid: set(int(x) for x in groups_obj.get(gid, set()) or set()) for gid in constrained_groups}
    base_edge_table = build_base_motif_edge_table()
    internal_grid_edge_table = build_region_internal_option_edges(
        config=G60_CONFIG,
        group_nodes=group_nodes,
        constrained_groups=constrained_groups,
        option=int(args.grid_option),
    )
    constrained_edge_table, stats_obj = apply_region_internal_option_constraint(
        base_edge_table=base_edge_table,
        internal_option_edge_table=internal_grid_edge_table,
        group_nodes=group_nodes,
        constrained_groups=constrained_groups,
        total_nodes=int(G60_CONFIG.total_sats),
        p=int(G60_CONFIG.P),
        n=int(G60_CONFIG.N),
        forced_option=int(args.grid_option),
        group_names={gid: group_name(gid) for gid in constrained_groups},
    )
    stats = stats_obj.to_dict()

    edges_csv = out_dir / "edges.csv"
    meta_json = out_dir / "meta.json"
    write_edges_csv(constrained_edge_table, edges_csv)
    meta = {
        "constellation": G60_CONFIG.name,
        "P": int(G60_CONFIG.P),
        "N": int(G60_CONFIG.N),
        "step": step,
        "motif": MOTIF_000056,
        "rule": (
            "Start from motif_000056 tiled over the full grid plus intra y-ring links. "
            "For selected China/Europe group-internal inter-plane pairs, add +grid option 0 links. "
            "If a selected-region node side has such an internal +grid link, all other inter-plane links "
            "on that same side are removed. Same selected-region non-grid inter-plane links are removed."
        ),
        **stats,
        "outputs": {
            "edges_csv": str(edges_csv),
            "meta_json": str(meta_json),
        },
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[internal-grid-motif] step={step} motif={MOTIF_000056['name']} "
        f"edges={stats['kept_edges']} base={stats['base_edges']} forced_grid={stats['forced_internal_option_edges']} "
        f"dropped={stats['dropped_edges']} out={out_dir}",
        flush=True,
    )
    print(
        f"[internal-grid-motif] validation={stats['validation']} option_counts={stats['option_counts_kept']}",
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
        steps=[step],
        edge_table=constrained_edge_table,
        edge_values=None,
        window_title=f"G60 motif_000056 with China/Europe internal +grid constraint step={step}",
        group_data={} if args.hide_groups else group_data,
        show_groups=not bool(args.hide_groups),
        topology_edge_color="#000000",
        topology_edge_alpha=190,
        topology_edge_width=0.024,
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
