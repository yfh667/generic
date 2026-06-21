from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_motif0056_to_0061_link_setup as one_way  # noqa: E402
from compare_setup_plan_vs_static_metrics import GROUP_CACHE_DIR, GROUP_XML  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.position_cache import open_position_cache_for_interval  # noqa: E402
from src.link_delay.module.query import open_delay_store_for_interval  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module import WeightedPairSpec, compute_weighted_edge_betweenness_timeseries  # noqa: E402
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step  # noqa: E402


DEFAULT_DELAY_STORE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\full_option_edge_delay\G60_full_options_t0_86164_stride1"
)
DEFAULT_POSITION_CACHE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\_position_cache\cache_0_86164_1s"
)
DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\delay_edge_usage_056_061_china_europe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute shortest-delay edge usage for G60 motif000056 and motif000061, China-Europe."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE_DIR)
    parser.add_argument("--position-cache-dir", type=Path, default=DEFAULT_POSITION_CACHE_DIR)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_topology(motif_id: int) -> one_way.TopologyData:
    rows = one_way.read_motif_rows(one_way.MOTIF_LIBRARY_CSV)
    spec = one_way.build_topology_spec(int(motif_id), rows[int(motif_id)])
    right_by_owner, left_by_right = one_way.build_right_links(spec.edge_table)
    return one_way.TopologyData(
        spec=spec,
        values=np.empty((0, int(spec.edge_table.num_edges)), dtype=np.float32),
        right_by_owner=right_by_owner,
        left_by_right=left_by_right,
        edge_key_to_idx=one_way.build_edge_key_index(spec.edge_table),
    )


def weights_by_step(
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


def cache_complete(out_dir: Path, *, num_steps: int, num_edges: int) -> bool:
    path = out_dir / "china_europe" / "edge_betweenness.npy"
    if not path.exists():
        return False
    arr = np.load(path, mmap_mode="r")
    return arr.shape == (int(num_steps), int(num_edges))


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty time range")
    out_root = Path(args.out_root) / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    out_root.mkdir(parents=True, exist_ok=True)
    pair_specs = [
        WeightedPairSpec(key="china_europe", source_group_id=2, target_group_id=3, label="China-Europe")
    ]
    group_data = load_or_build_group_data(
        xml_file=GROUP_XML,
        group_cache_dir=GROUP_CACHE_DIR,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=False,
    )

    for motif_id in (56, 61):
        topo = build_topology(int(motif_id))
        out_dir = out_root / topo.spec.name
        if cache_complete(out_dir, num_steps=len(steps), num_edges=topo.spec.edge_table.num_edges) and not bool(args.force):
            print(f"[delay-usage] reuse {out_dir}", flush=True)
            continue
        print(f"[delay-usage] computing {topo.spec.name} {topo.spec.motif}", flush=True)
        weights = weights_by_step(
            steps=steps,
            stride=int(args.stride),
            edge_table=topo.spec.edge_table,
            delay_store_dir=Path(args.delay_store_dir),
            position_cache_dir=Path(args.position_cache_dir),
        )
        compute_weighted_edge_betweenness_timeseries(
            topology_name=topo.spec.name,
            edge_table=topo.spec.edge_table,
            total_nodes=int(G60_CONFIG.total_sats),
            group_data=group_data,
            steps=steps,
            weights_by_step=weights,
            pair_specs=pair_specs,
            out_dir=out_dir,
            progress_every=int(args.progress_every),
            force=bool(args.force),
        )
        print(f"[delay-usage] wrote {out_dir}", flush=True)

    print(f"out_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
