from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


GENERIC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
THIS_DIR = Path(__file__).resolve().parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from g60_paper2_config import (
    DATA_ROOT,
    DEFAULT_EPHEM_DIR,
    DEFAULT_GROUP_CACHE_DIR,
    DEFAULT_GW_EPHEM_DIR,
    DEFAULT_GW_GROUP_CACHE_DIR,
    DEFAULT_GW_MAPPING_DIR,
    DEFAULT_GW_POSITION_CACHE_ROOT,
    DEFAULT_GW_XML,
    DEFAULT_MAPPING_DIR,
    DEFAULT_POSITION_CACHE_ROOT,
    DEFAULT_XML,
    build_paper2_g60_config,
    build_paper2_gw_config,
    write_station_group_mapping,
)
from plot_g60_china_europe_shortest_metrics import (
    build_or_load_delay_weights,
    build_topology_edge_table,
    ensure_paper2_position_cache,
    group_key,
    group_label,
    normalize_topology_name,
    topology_plot_label,
)
from src.link_delay.module.edge_options import write_edges_csv
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_metrics.module import (
    build_group_pair_node_arrays,
    compute_group_pair_shortest_timeseries,
    plot_group_pair_shortest_timeseries,
    result_summary,
    write_group_pair_shortest_timeseries,
)


DEFAULT_OUT_ROOT = DATA_ROOT / "outputs" / "paper2_shortest_metrics"
DEFAULT_DELAY_CACHE_ROOT = DATA_ROOT / "cache" / "paper2_shortest_metrics" / "edge_delay"


@dataclass(frozen=True)
class ConstellationBundle:
    name: str
    xml_file: Path
    ephem_dir: Path
    position_cache_root: Path
    group_cache_dir: Path
    mapping_dir: Path
    config: Any


@dataclass(frozen=True)
class PairSpec:
    key: str
    label: str
    source_group_id: int
    target_group_id: int
    source_group_key: str
    target_group_key: str


def get_constellation_bundle(name: str) -> ConstellationBundle:
    normalized = str(name).strip().lower()
    if normalized in {"g60", "g60_paper2"}:
        return ConstellationBundle(
            name="G60",
            xml_file=DEFAULT_XML,
            ephem_dir=DEFAULT_EPHEM_DIR,
            position_cache_root=DEFAULT_POSITION_CACHE_ROOT,
            group_cache_dir=DEFAULT_GROUP_CACHE_DIR,
            mapping_dir=DEFAULT_MAPPING_DIR,
            config=build_paper2_g60_config(),
        )
    if normalized in {"gw", "gw_paper2"}:
        return ConstellationBundle(
            name="GW",
            xml_file=DEFAULT_GW_XML,
            ephem_dir=DEFAULT_GW_EPHEM_DIR,
            position_cache_root=DEFAULT_GW_POSITION_CACHE_ROOT,
            group_cache_dir=DEFAULT_GW_GROUP_CACHE_DIR,
            mapping_dir=DEFAULT_GW_MAPPING_DIR,
            config=build_paper2_gw_config(),
        )
    raise ValueError(f"Unsupported constellation={name!r}. Configured: G60, GW")


def normalize_group_token(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def group_lookup(config) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for gid, info in config.station_groups.items():
        gid = int(gid)
        tokens = {
            str(gid),
            normalize_group_token(str(info.get("key", ""))),
            normalize_group_token(str(info.get("name", ""))),
            normalize_group_token(f"group_{gid}"),
        }
        for token in tokens:
            if token:
                lookup[token] = gid
    return lookup


def split_pair_token(token: str) -> tuple[str, str]:
    for sep in (":", ",", "->"):
        if sep in token:
            left, right = token.split(sep, 1)
            return left.strip(), right.strip()
    raise ValueError(f"Pair {token!r} must look like china:europe or 0:1")


def parse_pair(token: str, config) -> PairSpec:
    lookup = group_lookup(config)
    left_raw, right_raw = split_pair_token(token)
    left = normalize_group_token(left_raw)
    right = normalize_group_token(right_raw)
    if left not in lookup:
        raise ValueError(f"Unknown source group {left_raw!r}. Available: {sorted(lookup)}")
    if right not in lookup:
        raise ValueError(f"Unknown target group {right_raw!r}. Available: {sorted(lookup)}")
    source_group_id = lookup[left]
    target_group_id = lookup[right]
    src_key = group_key(config, source_group_id)
    dst_key = group_key(config, target_group_id)
    return PairSpec(
        key=f"{src_key}_{dst_key}",
        label=f"{group_label(config, source_group_id)}-{group_label(config, target_group_id)}",
        source_group_id=source_group_id,
        target_group_id=target_group_id,
        source_group_key=src_key,
        target_group_key=dst_key,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def append_pair_rows(rows: list[dict[str, Any]], *, pair: PairSpec, result) -> None:
    for idx, step in enumerate(result.steps):
        rows.append(
            {
                "pair_key": pair.key,
                "pair_label": pair.label,
                "source_group_id": pair.source_group_id,
                "target_group_id": pair.target_group_id,
                "source_group_key": pair.source_group_key,
                "target_group_key": pair.target_group_key,
                "step": int(step),
                "hour": float(int(step) / 3600.0),
                "source_nodes": int(result.source_counts[idx]),
                "target_nodes": int(result.target_counts[idx]),
                "active_edges": int(result.active_edges[idx]),
                "hop_reachable_pairs": int(result.hop_reachable_pairs[idx]),
                "mean_shortest_hops": finite_or_none(float(result.mean_shortest_hops[idx])),
                "min_shortest_hops": finite_or_none(float(result.min_shortest_hops[idx])),
                "p90_shortest_hops": finite_or_none(float(result.p90_shortest_hops[idx])),
                "max_shortest_hops": finite_or_none(float(result.max_shortest_hops[idx])),
                "delay_reachable_pairs": int(result.delay_reachable_pairs[idx]),
                "mean_shortest_delay_ms": finite_or_none(float(result.mean_shortest_delay_ms[idx])),
                "min_shortest_delay_ms": finite_or_none(float(result.min_shortest_delay_ms[idx])),
                "p90_shortest_delay_ms": finite_or_none(float(result.p90_shortest_delay_ms[idx])),
                "max_shortest_delay_ms": finite_or_none(float(result.max_shortest_delay_ms[idx])),
            }
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Paper2 group-pair mean shortest hops and propagation delay."
    )
    parser.add_argument("--constellation", type=str, default="G60")
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Group pairs, e.g. china:europe china:africa europe:north_america. IDs like 0:1 also work.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86400)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--topology", type=str, default="plus_grid")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--delay-cache-root", type=Path, default=DEFAULT_DELAY_CACHE_ROOT)
    parser.add_argument("--xml-file", type=Path, default=None)
    parser.add_argument("--ephem-dir", type=Path, default=None)
    parser.add_argument("--position-cache-root", type=Path, default=None)
    parser.add_argument("--group-cache-dir", type=Path, default=None)
    parser.add_argument("--mapping-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=1)
    parser.add_argument("--chunk-steps", type=int, default=256)
    parser.add_argument("--metric-chunk-size", type=int, default=100)
    parser.add_argument("--force-position-cache", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--force-delay-weights", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    if int(args.end) < int(args.start):
        raise ValueError("--end must be >= --start")

    bundle = get_constellation_bundle(args.constellation)
    config = bundle.config
    xml_file = Path(args.xml_file) if args.xml_file is not None else bundle.xml_file
    ephem_dir = Path(args.ephem_dir) if args.ephem_dir is not None else bundle.ephem_dir
    position_cache_root = (
        Path(args.position_cache_root) if args.position_cache_root is not None else bundle.position_cache_root
    )
    group_cache_dir = Path(args.group_cache_dir) if args.group_cache_dir is not None else bundle.group_cache_dir
    mapping_dir = Path(args.mapping_dir) if args.mapping_dir is not None else bundle.mapping_dir

    write_station_group_mapping(out_dir=mapping_dir, config=config)
    steps = np.arange(int(args.start), int(args.end) + 1, int(args.stride), dtype=np.int64)
    topology = normalize_topology_name(args.topology)
    topology_label = topology_plot_label(topology)
    pairs = [parse_pair(token, config) for token in args.pairs]

    run_key = f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    run_dir = Path(args.out_root) / bundle.name / topology / run_key
    inputs_dir = run_dir / "_inputs"
    delay_cache_dir = Path(args.delay_cache_root) / bundle.name / topology / run_key

    group_data = load_or_build_group_data(
        xml_file=xml_file,
        group_cache_dir=group_cache_dir,
        steps=[int(x) for x in steps],
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )

    edge_table = build_topology_edge_table(config, topology)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    edges_csv = inputs_dir / "edges.csv"
    write_edges_csv(edge_table, edges_csv)
    np.save(inputs_dir / "time_indices.npy", steps.astype(np.int64))
    active_mask = np.ones(int(edge_table.num_edges), dtype=bool)
    np.save(inputs_dir / "active_mask.npy", active_mask)

    position_cache_dir = ensure_paper2_position_cache(
        ephem_dir=ephem_dir,
        cache_root=position_cache_root,
        start=int(args.start),
        end=int(args.end),
        total_sats=config.total_sats,
        workers=int(args.workers),
        force=bool(args.force_position_cache),
    )
    delay_path = build_or_load_delay_weights(
        out_dir=delay_cache_dir,
        position_cache_dir=position_cache_dir,
        edge_table=edge_table,
        steps=steps,
        stride=int(args.stride),
        chunk_steps=int(args.chunk_steps),
        force=bool(args.force_delay_weights),
    )
    edge_delay_ms = np.load(delay_path, mmap_mode="r")

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for pair in pairs:
        print(
            f"[paper2-shortest] pair={pair.key} label={pair.label} topology={topology_label}",
            flush=True,
        )
        group_nodes = build_group_pair_node_arrays(
            group_data=group_data,
            steps=steps,
            source_group_id=pair.source_group_id,
            target_group_id=pair.target_group_id,
        )
        result = compute_group_pair_shortest_timeseries(
            edge_table=edge_table,
            total_nodes=config.total_sats,
            steps=steps,
            edge_active_mask=active_mask,
            edge_weights_ms=edge_delay_ms,
            group_nodes=group_nodes,
            workers=int(args.metric_workers),
            chunk_size=int(args.metric_chunk_size),
            progress_every_chunks=10,
        )

        pair_dir = run_dir / "pairs" / pair.key
        files = write_group_pair_shortest_timeseries(
            result,
            pair_dir,
            meta={
                "constellation": bundle.name,
                "topology": topology,
                "pair_key": pair.key,
                "pair_label": pair.label,
                "source_group": pair.source_group_key,
                "target_group": pair.target_group_key,
                "source_group_id": pair.source_group_id,
                "target_group_id": pair.target_group_id,
                "xml_file": str(xml_file),
                "position_cache_dir": str(position_cache_dir),
                "edge_delay_ms": str(delay_path),
                "edges_csv": str(edges_csv),
                "active_mask": str(inputs_dir / "active_mask.npy"),
            },
        )
        figures_dir = run_dir / "figures" / pair.key
        hop_plot = plot_group_pair_shortest_timeseries(
            {topology_label: result},
            figures_dir,
            metric="mean_shortest_hops",
            ylabel="mean shortest hops",
            title=f"Paper2 {bundle.name} {pair.label} mean shortest hops ({topology_label})",
            filename=f"{pair.key}_mean_shortest_hops.png",
        )
        p90_hop_plot = plot_group_pair_shortest_timeseries(
            {topology_label: result},
            figures_dir,
            metric="p90_shortest_hops",
            ylabel="p90 shortest hops",
            title=f"Paper2 {bundle.name} {pair.label} p90 shortest hops ({topology_label})",
            filename=f"{pair.key}_p90_shortest_hops.png",
        )
        delay_plot = plot_group_pair_shortest_timeseries(
            {topology_label: result},
            figures_dir,
            metric="mean_shortest_delay_ms",
            ylabel="mean shortest propagation delay (ms)",
            title=f"Paper2 {bundle.name} {pair.label} mean shortest propagation delay ({topology_label})",
            filename=f"{pair.key}_mean_shortest_delay_ms.png",
        )

        summary = result_summary(result)
        summary_rows.append(
            {
                "pair_key": pair.key,
                "pair_label": pair.label,
                "source_group_id": pair.source_group_id,
                "target_group_id": pair.target_group_id,
                "source_group_key": pair.source_group_key,
                "target_group_key": pair.target_group_key,
                "steps": int(summary["steps"]),
                "start": int(summary["start"]),
                "end": int(summary["end"]),
                "mean_hops": finite_or_none(float(summary["mean_hops"])),
                "min_hops": finite_or_none(float(summary["min_hops"])),
                "max_hops": finite_or_none(float(summary["max_hops"])),
                "mean_p90_hops": finite_or_none(float(summary["mean_p90_hops"])),
                "min_p90_hops": finite_or_none(float(summary["min_p90_hops"])),
                "max_p90_hops": finite_or_none(float(summary["max_p90_hops"])),
                "mean_shortest_delay_ms": finite_or_none(float(summary["mean_shortest_delay_ms"])),
                "min_delay_ms": finite_or_none(float(summary["min_delay_ms"])),
                "max_delay_ms": finite_or_none(float(summary["max_delay_ms"])),
                "mean_p90_delay_ms": finite_or_none(float(summary["mean_p90_delay_ms"])),
                "min_p90_delay_ms": finite_or_none(float(summary["min_p90_delay_ms"])),
                "max_p90_delay_ms": finite_or_none(float(summary["max_p90_delay_ms"])),
                "timeseries_csv": files["timeseries"],
                "hop_plot": hop_plot,
                "p90_hop_plot": p90_hop_plot,
                "delay_plot": delay_plot,
            }
        )
        append_pair_rows(all_rows, pair=pair, result=result)

    write_csv(run_dir / "all_pairs_timeseries.csv", all_rows)
    write_csv(run_dir / "pair_summary.csv", summary_rows)
    write_json(
        run_dir / "run_meta.json",
        {
            "constellation": bundle.name,
            "topology": topology,
            "topology_label": topology_label,
            "start": int(args.start),
            "end": int(args.end),
            "stride": int(args.stride),
            "pairs": [pair.__dict__ for pair in pairs],
            "xml_file": str(xml_file),
            "ephem_dir": str(ephem_dir),
            "position_cache_dir": str(position_cache_dir),
            "group_cache_dir": str(group_cache_dir),
            "edge_delay_ms": str(delay_path),
            "edges_csv": str(edges_csv),
            "active_mask": str(inputs_dir / "active_mask.npy"),
            "all_pairs_timeseries_csv": str(run_dir / "all_pairs_timeseries.csv"),
            "pair_summary_csv": str(run_dir / "pair_summary.csv"),
        },
    )

    print(f"[paper2-shortest] run_dir={run_dir}", flush=True)
    print(f"[paper2-shortest] all_pairs_csv={run_dir / 'all_pairs_timeseries.csv'}", flush=True)
    print(f"[paper2-shortest] summary_csv={run_dir / 'pair_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
