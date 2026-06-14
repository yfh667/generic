from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_paper1_motif_shortest_delay import load_yaml, path_from, region_pair_specs, wrap_planes_from_config
from run_paper1_motif_shortest_hops import ensure_motif_library
from src.link_delay.module.position_cache import open_position_cache_for_interval
from src.link_delay.module.query import open_delay_store_for_interval
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.module import topology_specs_from_motif_csv
from src.topology_workflow.module.batch_shortest_delay import _topology_pair_delay_complete
from src.topology_workflow.module.config import time_axis_from_config, viewer_config_from_workflow
from src.topology_workflow.module.shortest_delay import compute_shortest_delay_timeseries


DEFAULT_CONFIG = THIS_DIR / "configs" / "g60_w4_h3_shortest_delay.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serial resume runner for Paper1 shortest-delay motif results.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pair", required=True, help="One region pair key, e.g. china_africa.")
    parser.add_argument("--motif-offset", type=int, default=0)
    parser.add_argument("--limit-motifs", type=int, default=0)
    parser.add_argument("--engine", choices=("auto", "scipy", "heapq"), default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = load_yaml(args.config)
    config = viewer_config_from_workflow(raw)
    start, end, stride = time_axis_from_config(raw)
    steps = list(range(int(start), int(end) + 1, int(stride)))

    paths = raw.get("paths", {}) if isinstance(raw.get("paths", {}), dict) else {}
    run_raw = raw.get("run", {}) if isinstance(raw.get("run", {}), dict) else {}
    engine = str(args.engine if args.engine is not None else run_raw.get("engine", "auto"))

    csv_path = ensure_motif_library(raw, force=False)
    library_raw = raw.get("motif_library", {}) if isinstance(raw.get("motif_library", {}), dict) else {}
    motif_specs = topology_specs_from_motif_csv(
        csv_path,
        config=config,
        library="combined_motif",
        name_prefix=str(library_raw.get("name_prefix", "combined")),
        limit=int(args.limit_motifs),
        offset=int(args.motif_offset),
        add_intra_ring=True,
        wrap_planes=wrap_planes_from_config(raw),
    )
    if not motif_specs:
        raise ValueError("No motif specs selected")

    group_data = load_or_build_group_data(
        xml_file=path_from(paths, "group_xml"),
        group_cache_dir=path_from(paths, "group_cache_dir"),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(stride),
        enabled=True,
        force=False,
    )
    pair_specs = region_pair_specs(raw, subset=[args.pair])
    if len(pair_specs) != 1:
        raise ValueError(f"Expected exactly one pair for {args.pair!r}, got {len(pair_specs)}")
    pair = pair_specs[0]

    out_dir = path_from(paths, "out_dir")
    pair_dir = out_dir / str(pair.key)
    pair_dir.mkdir(parents=True, exist_ok=True)

    delay_store = open_delay_store_for_interval(
        int(start),
        int(end),
        stride=int(stride),
        store_dir=path_from(paths, "delay_store_dir"),
        constellation_name=config.name,
    )
    delay_rows = delay_store.rows_for_interval(int(start), int(end), int(stride))

    position_store = open_position_cache_for_interval(
        int(start),
        int(end),
        stride=int(stride),
        cache_dir=Path(paths["position_cache_dir"]) if paths.get("position_cache_dir") else None,
    )
    position_rows = position_store.rows_for_interval(int(start), int(end), int(stride))

    completed = 0
    for idx, spec in enumerate(motif_specs, start=1):
        topology_dir = pair_dir / str(spec.name)
        if not bool(args.force) and _topology_pair_delay_complete(topology_dir, steps):
            completed += 1
            print(f"[serial-resume] skip {idx}/{len(motif_specs)} {spec.name}", flush=True)
            continue
        print(f"[serial-resume] compute {idx}/{len(motif_specs)} {spec.name}", flush=True)
        compute_shortest_delay_timeseries(
            topology_name=str(spec.name),
            edge_table=spec.edge_table,
            config=config,
            group_data=group_data,
            steps=steps,
            delay_rows=np.asarray(delay_rows, dtype=np.int64),
            position_rows=np.asarray(position_rows, dtype=np.int64),
            delay_store=delay_store,
            position_store=position_store,
            source_group_id=int(pair.source_group_id),
            target_group_id=int(pair.target_group_id),
            out_dir=topology_dir,
            engine=engine,
            sample_steps=0,
            sample_pairs_per_step=0,
            progress_every=0,
        )
        completed += 1
    print(f"[serial-resume] done pair={pair.key} selected={len(motif_specs)} completed_or_skipped={completed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
