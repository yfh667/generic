from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_paper1_motif_shortest_hops import (
    build_baselines,
    ensure_motif_library,
    load_yaml,
    path_from,
    region_pair_specs,
)
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.module import compute_shortest_delay_batch, topology_specs_from_motif_csv
from src.topology_workflow.module.config import time_axis_from_config, viewer_config_from_workflow


DEFAULT_CONFIG = THIS_DIR / "configs" / "g60_w_le4_h_le3_shortest_delay.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper1 motif-library shortest-delay pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit-motifs", type=int, default=None)
    parser.add_argument("--engine", choices=("auto", "scipy", "heapq"), default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--sample-pairs-per-step", type=int, default=None)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--regenerate-library", action="store_true")
    parser.add_argument("--skip-gridplus", action="store_true")
    parser.add_argument("--pairs", nargs="*", default=None, help="Optional subset of region pair keys.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = load_yaml(args.config)
    config = viewer_config_from_workflow(raw)

    start, end, stride = time_axis_from_config(raw)
    if args.start is not None:
        start = int(args.start)
    if args.end is not None:
        end = int(args.end)
    if args.stride is not None:
        stride = int(args.stride)
    if end < start:
        raise ValueError("end must be >= start")
    if stride <= 0:
        raise ValueError("stride must be positive")
    steps = list(range(int(start), int(end) + 1, int(stride)))

    paths = raw.get("paths", {}) if isinstance(raw.get("paths", {}), dict) else {}
    run_raw = raw.get("run", {}) if isinstance(raw.get("run", {}), dict) else {}
    out_dir = Path(args.out_dir) if args.out_dir is not None else path_from(paths, "out_dir")
    limit_motifs = int(args.limit_motifs if args.limit_motifs is not None else run_raw.get("limit_motifs", 0))
    engine = str(args.engine if args.engine is not None else run_raw.get("engine", "auto"))
    sample_steps = int(args.sample_steps if args.sample_steps is not None else run_raw.get("sample_steps", 0))
    sample_pairs = int(
        args.sample_pairs_per_step
        if args.sample_pairs_per_step is not None
        else run_raw.get("sample_pairs_per_step", 0)
    )
    force_group_cache = bool(args.force_group_cache or run_raw.get("force_group_cache", False))

    csv_path = ensure_motif_library(raw, force=bool(args.regenerate_library))
    library_raw = raw.get("motif_library", {}) if isinstance(raw.get("motif_library", {}), dict) else {}
    motif_specs = topology_specs_from_motif_csv(
        csv_path,
        config=config,
        library="combined_motif",
        name_prefix=str(library_raw.get("name_prefix", "combined")),
        limit=limit_motifs,
        add_intra_ring=True,
    )
    baseline_specs = build_baselines(raw, config=config, skip_gridplus=bool(args.skip_gridplus))
    topology_specs = motif_specs + baseline_specs

    group_data = load_or_build_group_data(
        xml_file=path_from(paths, "group_xml"),
        group_cache_dir=path_from(paths, "group_cache_dir"),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(stride),
        enabled=True,
        force=force_group_cache,
    )
    pair_specs = region_pair_specs(raw, subset=args.pairs)

    out_dir.mkdir(parents=True, exist_ok=True)
    effective = {
        "config_file": str(args.config),
        "start": int(start),
        "end": int(end),
        "stride": int(stride),
        "motif_csv": str(csv_path),
        "out_dir": str(out_dir),
        "num_topologies": len(topology_specs),
        "pairs": [pair.key for pair in pair_specs],
        "engine": engine,
        "sample_steps": sample_steps,
        "sample_pairs_per_step": sample_pairs,
    }
    (out_dir / "effective_config.json").write_text(json.dumps(effective, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = compute_shortest_delay_batch(
        topology_specs=topology_specs,
        pair_specs=pair_specs,
        config=config,
        group_data=group_data,
        start=int(start),
        end=int(end),
        stride=int(stride),
        out_dir=out_dir,
        delay_store_dir=path_from(paths, "delay_store_dir"),
        position_cache_dir=Path(paths["position_cache_dir"]) if paths.get("position_cache_dir") else None,
        engine=engine,
        sample_steps=sample_steps,
        sample_pairs_per_step=sample_pairs,
        progress_every=int(run_raw.get("progress_every", 200)),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
