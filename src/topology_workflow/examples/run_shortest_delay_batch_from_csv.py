from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[3]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.module import (
    RegionPairSpec,
    compute_shortest_delay_batch,
    full_link_topology_spec,
    topology_specs_from_motif_csv,
)


def parse_pair(text: str) -> RegionPairSpec:
    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) == 2:
        source, target = parts
        key = f"group{source}_group{target}"
    elif len(parts) == 3:
        source, target, key = parts
    else:
        raise argparse.ArgumentTypeError("pair must be source:target or source:target:key")
    return RegionPairSpec(key=str(key), label=str(key), source_group_id=int(source), target_group_id=int(target))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shortest-delay batch simulation for motif CSV topologies."
    )
    parser.add_argument("--motif-csv", type=Path, required=True)
    parser.add_argument("--delay-store-dir", type=Path, required=True)
    parser.add_argument("--group-xml", type=Path, required=True)
    parser.add_argument("--group-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--position-cache-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=600)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--pair", type=parse_pair, action="append", default=None)
    parser.add_argument("--library", type=str, default="motif_library")
    parser.add_argument("--name-prefix", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-full-link", action="store_true")
    parser.add_argument("--engine", choices=("auto", "scipy", "heapq"), default="auto")
    parser.add_argument("--sample-steps", type=int, default=1)
    parser.add_argument("--sample-pairs-per-step", type=int, default=20)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    pair_specs = args.pair or [
        RegionPairSpec(key="china_europe", label="China-Europe", source_group_id=2, target_group_id=3)
    ]
    topology_specs = topology_specs_from_motif_csv(
        args.motif_csv,
        config=G60_CONFIG,
        library=args.library,
        name_prefix=args.name_prefix,
        limit=int(args.limit),
        add_intra_ring=True,
    )
    if args.include_full_link:
        topology_specs.append(full_link_topology_spec(config=G60_CONFIG, name="full_link_plus_intra"))

    group_data = load_or_build_group_data(
        xml_file=args.group_xml,
        group_cache_dir=args.group_cache_dir,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )
    meta = compute_shortest_delay_batch(
        topology_specs=topology_specs,
        pair_specs=pair_specs,
        config=G60_CONFIG,
        group_data=group_data,
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
        out_dir=args.out_dir,
        delay_store_dir=args.delay_store_dir,
        position_cache_dir=args.position_cache_dir,
        engine=args.engine,
        sample_steps=int(args.sample_steps),
        sample_pairs_per_step=int(args.sample_pairs_per_step),
    )
    print(f"[topology-workflow] shortest-delay batch written: {meta['num_topologies']} topologies -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
