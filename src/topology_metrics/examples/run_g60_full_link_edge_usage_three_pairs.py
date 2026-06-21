from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[3]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.position_cache import open_position_cache_for_interval
from src.link_delay.module.query import open_delay_store_for_interval
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_metrics.module import (
    WeightedPairSpec,
    compute_edge_betweenness_store,
    compute_weighted_edge_betweenness_timeseries,
    expand_unique_state_values,
    write_delay_hop_combined_usage_store,
)
from src.topology_workflow.module.edge_tables import build_full_option_plus_intra_edge_table
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step


DEFAULT_OUT_DIR = (
    Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60")
    / "full_link_edge_usage_share_three_pairs_t0_86160_stride60"
)
DEFAULT_DELAY_STORE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition\full_option_edge_delay\G60_full_options_t0_86164_stride1"
)
DEFAULT_POSITION_CACHE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition\_position_cache\cache_0_86164_1s"
)
DEFAULT_GROUP_XML = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\cache\G60\group_data")


def parse_pair(text: str) -> WeightedPairSpec:
    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) == 3:
        source, target, key = parts
        label = key
    elif len(parts) == 4:
        source, target, key, label = parts
    else:
        raise argparse.ArgumentTypeError("pair must be source:target:key or source:target:key:label")
    return WeightedPairSpec(key=str(key), source_group_id=int(source), target_group_id=int(target), label=str(label))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute full-link shortest-delay and shortest-hop edge usage share "
            "for the three paper1 G60 region pairs."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE_DIR)
    parser.add_argument("--position-cache-dir", type=Path, default=DEFAULT_POSITION_CACHE_DIR)
    parser.add_argument("--group-xml", type=Path, default=DEFAULT_GROUP_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--sample-steps", type=int, default=0)
    parser.add_argument("--sample-pairs-per-step", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument(
        "--pair",
        type=parse_pair,
        action="append",
        default=None,
        help="Repeatable source:target:key[:label]. Defaults to China-Europe/Africa/America.",
    )
    return parser.parse_args()


def _weights_by_step(
    *,
    steps: list[int],
    stride: int,
    edge_table,
    delay_store_dir: Path,
    position_cache_dir: Path,
) -> np.ndarray:
    delay_store = open_delay_store_for_interval(
        int(steps[0]),
        int(steps[-1]),
        stride=int(stride),
        store_dir=Path(delay_store_dir),
        constellation_name=G60_CONFIG.name,
    )
    delay_rows = delay_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(stride))
    position_store = open_position_cache_for_interval(
        int(steps[0]),
        int(steps[-1]),
        stride=int(stride),
        cache_dir=Path(position_cache_dir),
    )
    position_rows = position_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(stride))
    lookup = build_weight_lookup(
        edge_table,
        delay_store,
        config=G60_CONFIG,
        allow_intra_fallback=True,
    )
    weights = np.empty((len(steps), int(edge_table.num_edges)), dtype=np.float32)
    for row in range(len(steps)):
        weights[row, :] = edge_weights_for_step(
            edge_table=edge_table,
            lookup=lookup,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=int(delay_rows[row]),
            position_row=int(position_rows[row]),
        )
    return weights


def _write_combined_hop_share(
    *,
    out_dir: Path,
    pair_specs: list[WeightedPairSpec],
    num_steps: int,
    num_edges: int,
) -> None:
    combined = np.zeros((int(num_steps), int(num_edges)), dtype=np.float32)
    for pair in pair_specs:
        pair_dir = out_dir / "hop_shortest" / str(pair.key)
        unique_share = np.load(pair_dir / "unique_state_usage_share.npy", mmap_mode="r")
        state_ids = np.load(pair_dir / "state_ids.npy", allow_pickle=False)
        combined += expand_unique_state_values(unique_state_values=np.asarray(unique_share), state_ids=state_ids)
    hop_dir = out_dir / "hop_shortest"
    np.save(hop_dir / "combined_sum_edge_usage_share.npy", combined.astype(np.float32, copy=False))
    np.save(hop_dir / "combined_usage_share_max_over_time.npy", np.max(combined, axis=0).astype(np.float32, copy=False))


def main() -> int:
    args = parse_args()
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty step range")
    pair_specs = args.pair or [
        WeightedPairSpec(key="china_europe", source_group_id=2, target_group_id=3, label="China-Europe"),
        WeightedPairSpec(key="china_africa", source_group_id=2, target_group_id=1, label="China-Africa"),
        WeightedPairSpec(key="china_america", source_group_id=2, target_group_id=0, label="China-America"),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_table = build_full_option_plus_intra_edge_table(config=G60_CONFIG, add_intra_ring=True)
    group_data = load_or_build_group_data(
        xml_file=Path(args.group_xml),
        group_cache_dir=Path(args.group_cache_dir),
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )

    weights = _weights_by_step(
        steps=steps,
        stride=int(args.stride),
        edge_table=edge_table,
        delay_store_dir=Path(args.delay_store_dir),
        position_cache_dir=Path(args.position_cache_dir),
    )
    delay_meta = compute_weighted_edge_betweenness_timeseries(
        topology_name="full_link",
        edge_table=edge_table,
        total_nodes=int(G60_CONFIG.total_sats),
        group_data=group_data,
        steps=steps,
        weights_by_step=weights,
        pair_specs=pair_specs,
        out_dir=out_dir / "delay_shortest",
        sample_steps=int(args.sample_steps),
        sample_pairs_per_step=int(args.sample_pairs_per_step),
        progress_every=int(args.progress_every),
        force=bool(args.force),
    )

    hop_metas = {}
    for pair in pair_specs:
        hop_metas[str(pair.key)] = compute_edge_betweenness_store(
            topology_name="full_link",
            edge_table=edge_table,
            total_nodes=int(G60_CONFIG.total_sats),
            group_data=group_data,
            steps=steps,
            source_group_id=int(pair.source_group_id),
            target_group_id=int(pair.target_group_id),
            out_dir=out_dir / "hop_shortest" / str(pair.key),
            workers=int(args.workers),
            progress_every=int(args.progress_every),
            force=bool(args.force),
            expand_full_matrix=False,
            sample_path_limit=0,
            extra_meta={"pair_key": str(pair.key), "pair_label": pair.label or pair.key},
        )
    _write_combined_hop_share(
        out_dir=out_dir,
        pair_specs=list(pair_specs),
        num_steps=len(steps),
        num_edges=int(edge_table.num_edges),
    )
    combined_meta = write_delay_hop_combined_usage_store(
        out_dir=out_dir / "delay_hop_combined",
        edges_csv=out_dir / "delay_shortest" / "edges.csv",
        delay_usage_share=np.load(out_dir / "delay_shortest" / "combined_usage_share_max_over_time.npy"),
        hop_usage_share=np.load(out_dir / "hop_shortest" / "combined_usage_share_max_over_time.npy"),
        delay_weight=0.5,
        hop_weight=0.5,
        normalize="max",
        extra_meta={"topology": "full_link", "constellation": G60_CONFIG.name},
    )

    meta = {
        "topology": "full_link",
        "constellation": G60_CONFIG.name,
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "num_steps": len(steps),
        "num_edges": int(edge_table.num_edges),
        "pairs": [
            {
                "key": str(pair.key),
                "source_group_id": int(pair.source_group_id),
                "target_group_id": int(pair.target_group_id),
                "label": pair.label or pair.key,
            }
            for pair in pair_specs
        ],
        "outputs": {
            "delay_shortest": str(out_dir / "delay_shortest"),
            "hop_shortest": str(out_dir / "hop_shortest"),
            "delay_hop_combined": str(out_dir / "delay_hop_combined"),
            "delay_combined_usage_share_max": str(out_dir / "delay_shortest" / "combined_usage_share_max_over_time.npy"),
            "hop_combined_usage_share_max": str(out_dir / "hop_shortest" / "combined_usage_share_max_over_time.npy"),
            "delay_hop_combined_usage_share": str(out_dir / "delay_hop_combined" / "combined_usage_share_max_over_time.npy"),
        },
        "delay_meta": delay_meta,
        "hop_meta": hop_metas,
        "delay_hop_combined_meta": combined_meta,
    }
    (out_dir / "full_link_edge_usage_share_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), **meta["outputs"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
