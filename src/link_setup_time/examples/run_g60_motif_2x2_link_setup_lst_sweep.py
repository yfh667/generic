from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_setup_time.module import (
    build_region_internal_target_series,
    group_state_sequence_from_group_data,
    plot_sweep_summary,
    plot_sweep_timeseries,
    sweep_link_setup_times,
    write_sweep_outputs,
)
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.module import build_single_motif_edge_table


MOTIF_2X2_CB = {
    "name": "motif_2x2_CB",
    "w": 2,
    "h": 2,
    "offsets": {
        "A": [1, 0],
        "B": [1, -1],
        "C": [1, 1],
        "D": [2, 0],
    },
    "support": [
        [0, 0, "C"],
        [0, 1, "B"],
    ],
}

DEFAULT_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE = PROJECT_ROOT / "data" / "satnet_experiments" / "caches" / "G60" / "group_data_cache"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "runs"
    / "paper1"
    / "G60"
    / "link_setup_time"
    / "motif_2x2_CB_china_europe"
    / "lst_sweep_10_140"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G60 motif_2x2_CB LST sweep example using src.link_setup_time.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86400)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--lst-values", type=int, nargs="*", default=list(range(10, 141, 10)))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--constrained-groups", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--grid-option", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=6000)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def build_base_edge_table():
    return build_single_motif_edge_table(
        topology_raw={
            "kind": "single_motif",
            "motif": MOTIF_2X2_CB,
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
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    constrained_groups = tuple(int(x) for x in args.constrained_groups)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    state_sequence = group_state_sequence_from_group_data(
        group_data=group_data,
        steps=steps,
        constrained_groups=constrained_groups,
    )
    target_series = build_region_internal_target_series(
        config=G60_CONFIG,
        base_edge_table=build_base_edge_table(),
        state_sequence=state_sequence,
        forced_option=int(args.grid_option),
        wrap_planes=False,
    )
    result = sweep_link_setup_times(
        target_series=target_series,
        steps=steps,
        lst_values=[int(x) for x in args.lst_values],
        chunk_size=int(args.chunk_size),
    )
    result = write_sweep_outputs(out_dir=out_dir, result=result)
    plot_sweep_timeseries(
        path=out_dir / "lst_sweep_building_link_timeseries.png",
        result=result,
        title=f"G60 {MOTIF_2X2_CB['name']} China/Europe building-link count under LST sweep",
    )
    plot_sweep_summary(
        path=out_dir / "lst_sweep_summary.png",
        rows=result.rows,
        title="Building-link count summary vs LST",
    )

    for row in result.rows:
        print(
            f"[link-setup-time] LST={row.lst_s}s "
            f"mean={row.mean_building_edges:.3f} max={row.max_building_edges}",
            flush=True,
        )
    print(
        {
            "out_dir": str(out_dir),
            "summary_csv": str(out_dir / "lst_sweep_summary.csv"),
            "timeseries_plot": str(out_dir / "lst_sweep_building_link_timeseries.png"),
            "summary_plot": str(out_dir / "lst_sweep_summary.png"),
            "lst_values": [int(x) for x in args.lst_values],
            "steps": len(steps),
            "union_edges": int(target_series.edge_table.num_edges),
            "motif": MOTIF_2X2_CB["name"],
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
