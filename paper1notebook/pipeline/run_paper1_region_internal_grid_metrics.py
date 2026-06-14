from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_paper1_motif_shortest_hops import (
    ensure_motif_library,
    load_yaml,
    path_from,
    region_pair_specs,
    wrap_planes_from_config,
)
from src.link_delay.module.delay_store import LIGHT_SPEED_KM_S
from src.link_delay.module.edge_options import EdgeTable, OPTION_DELTAS, build_full_option_edges
from src.link_delay.module.position_cache import PositionCacheStore
from src.link_delay.module.query import FullLinkDelayStore
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_metrics.module.group_states import group_nodes_for_step
from src.topology_workflow.module import (
    RegionPairSpec,
    TopologySpec,
    apply_region_internal_option_constraint,
    build_region_internal_option_edges,
    topology_specs_from_motif_csv,
)
from src.topology_workflow.module.config import group_name, time_axis_from_config, viewer_config_from_workflow
from src.topology_workflow.module.shortest_delay import (
    build_heapq_adjacency,
    connected_component_ids,
    dijkstra_heapq,
)


DEFAULT_CONFIG = THIS_DIR / "configs" / "g60_w_le4_h_le3_region_internal_grid_metrics.yaml"
_WORKER_CONTEXT: dict[str, Any] = {}
_SCIPY_TOOLS: tuple[Any, Any] | None | bool = None

LEFT_SIDE = 0
RIGHT_SIDE = 1
INTRA_OPTION = -1


@dataclass(frozen=True)
class StepMetricRow:
    step: int
    source_nodes: int
    target_nodes: int
    reachable_pairs: int
    expected_pairs: int
    mean_shortest_hops: float
    min_shortest_hops: float
    max_shortest_hops: float
    mean_shortest_delay_ms: float
    min_shortest_delay_ms: float
    max_shortest_delay_ms: float
    constrained_edges: int
    forced_internal_option_edges: int
    dropped_edges: int


@dataclass(frozen=True)
class PairStepConstraintContext:
    pair_key: str
    step: int
    sources: tuple[int, ...]
    targets: tuple[int, ...]
    group_bits: np.ndarray
    allowed_neighbor: np.ndarray
    internal_src: np.ndarray
    internal_dst: np.ndarray
    internal_option: np.ndarray
    internal_key: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper1 G60 motif metrics with region-internal +grid/option-0 constraints."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--delay-store-dir", type=Path, default=None)
    parser.add_argument("--position-cache-dir", type=Path, default=None)
    parser.add_argument("--no-position-cache", action="store_true")
    parser.add_argument("--limit-motifs", type=int, default=None)
    parser.add_argument("--motif-offset", type=int, default=None)
    parser.add_argument("--pairs", nargs="*", default=None, help="Optional subset of region pair keys.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--regenerate-library", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute topology-pair outputs even if complete files exist.")
    parser.add_argument("--skip-hops", action="store_true", help="Only compute shortest-delay metrics; hop outputs are written as NaN.")
    parser.add_argument(
        "--delay-engine",
        choices=("auto", "scipy", "heapq"),
        default=None,
        help="Shortest-delay engine. Use heapq to avoid scipy csgraph for fragile topologies.",
    )
    parser.add_argument(
        "--step-heartbeat-seconds",
        type=float,
        default=None,
        help="Print progress while a topology/pair is computing; useful for long fragile repairs.",
    )
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--sample-pairs-per-step", type=int, default=None)
    return parser.parse_args()


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _group_nodes_mapping_for_step(group_data: Mapping, step: int) -> dict[int, set[int]]:
    current = group_data.get(int(step), None)
    if current is None:
        current = group_data.get(str(int(step)), {})
    groups = current.get("groups", {}) if isinstance(current, Mapping) else {}
    return {int(group_id): set(int(node) for node in nodes or []) for group_id, nodes in groups.items()}


def _edge_keys(src: np.ndarray, dst: np.ndarray, total_nodes: int) -> np.ndarray:
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    return a * int(total_nodes) + b


def _build_pair_step_contexts(
    *,
    config,
    group_data: Mapping,
    steps: Sequence[int],
    pair_specs: Sequence[RegionPairSpec],
    forced_option: int,
    wrap_planes: bool = False,
) -> dict[str, list[PairStepConstraintContext]]:
    option_edges = build_full_option_edges(config, options=(int(forced_option),), wrap_planes=bool(wrap_planes))
    option_src = np.asarray(option_edges.src, dtype=np.int32)
    option_dst = np.asarray(option_edges.dst, dtype=np.int32)
    total_nodes = int(config.total_sats)
    contexts: dict[str, list[PairStepConstraintContext]] = {str(pair.key): [] for pair in pair_specs}

    for pair in pair_specs:
        pair_key = str(pair.key)
        for step in steps:
            groups = _group_nodes_mapping_for_step(group_data, int(step))
            group_bits = np.zeros(total_nodes, dtype=np.uint64)
            for bit_idx, group_id in enumerate((int(pair.source_group_id), int(pair.target_group_id))):
                bit = np.uint64(1 << bit_idx)
                for node in groups.get(int(group_id), set()):
                    if 0 <= int(node) < total_nodes:
                        group_bits[int(node)] |= bit

            internal_mask = (group_bits[option_src] & group_bits[option_dst]) != 0
            internal_src = np.asarray(option_src[internal_mask], dtype=np.int32)
            internal_dst = np.asarray(option_dst[internal_mask], dtype=np.int32)
            internal_option = np.full(internal_src.shape[0], int(forced_option), dtype=np.int16)
            internal_key = _edge_keys(internal_src, internal_dst, total_nodes)
            allowed_neighbor = np.full((total_nodes, 2), -1, dtype=np.int32)
            if internal_src.size:
                allowed_neighbor[internal_src, RIGHT_SIDE] = internal_dst
                allowed_neighbor[internal_dst, LEFT_SIDE] = internal_src

            contexts[pair_key].append(
                PairStepConstraintContext(
                    pair_key=pair_key,
                    step=int(step),
                    sources=group_nodes_for_step(group_data, int(step), int(pair.source_group_id)),
                    targets=group_nodes_for_step(group_data, int(step), int(pair.target_group_id)),
                    group_bits=group_bits,
                    allowed_neighbor=allowed_neighbor,
                    internal_src=internal_src,
                    internal_dst=internal_dst,
                    internal_option=internal_option,
                    internal_key=internal_key,
                )
            )
    return contexts


def _edge_table_from_arrays(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    option: np.ndarray,
    n: int,
    sat_ids: list[str],
) -> EdgeTable:
    src = np.asarray(src, dtype=np.int32)
    dst = np.asarray(dst, dtype=np.int32)
    option = np.asarray(option, dtype=np.int16)
    return EdgeTable(
        src=src,
        dst=dst,
        option=option,
        src_plane=(src // int(n)).astype(np.int16),
        src_y=(src % int(n)).astype(np.int16),
        dst_plane=(dst // int(n)).astype(np.int16),
        dst_y=(dst % int(n)).astype(np.int16),
        sat_ids=sat_ids,
    )


def _apply_constraint_context_fast(
    *,
    base_edge_table: EdgeTable,
    context: PairStepConstraintContext,
    total_nodes: int,
    p: int,
    n: int,
    forced_option: int,
    wrap_planes: bool = False,
) -> tuple[EdgeTable, int, int]:
    src = np.asarray(base_edge_table.src, dtype=np.int32)
    dst = np.asarray(base_edge_table.dst, dtype=np.int32)
    option = np.asarray(base_edge_table.option, dtype=np.int16)
    src_plane = np.asarray(base_edge_table.src_plane, dtype=np.int16)
    dst_plane = np.asarray(base_edge_table.dst_plane, dtype=np.int16)
    bits = context.group_bits
    keep = np.ones(src.shape[0], dtype=bool)
    inter_mask = option != INTRA_OPTION

    same_selected_region = (bits[src] & bits[dst]) != 0
    keep &= ~(inter_mask & same_selected_region & (option != int(forced_option)))

    src_side = np.full(src.shape[0], LEFT_SIDE, dtype=np.int8)
    dst_side = np.full(dst.shape[0], RIGHT_SIDE, dtype=np.int8)
    for opt_value in sorted(set(int(x) for x in option.tolist() if int(x) != INTRA_OPTION)):
        opt_mask = option == int(opt_value)
        dp = int(OPTION_DELTAS.get(int(opt_value), (0, 0))[0])
        if bool(wrap_planes):
            src_to_dst = ((src_plane.astype(np.int32) + dp) % int(p)) == dst_plane.astype(np.int32)
            dst_to_src = ((dst_plane.astype(np.int32) + dp) % int(p)) == src_plane.astype(np.int32)
        else:
            src_to_dst = (src_plane.astype(np.int32) + dp) == dst_plane.astype(np.int32)
            dst_to_src = (dst_plane.astype(np.int32) + dp) == src_plane.astype(np.int32)
        src_right = opt_mask & src_to_dst & ~dst_to_src
        dst_right = opt_mask & dst_to_src & ~src_to_dst
        src_side[src_right] = RIGHT_SIDE
        dst_side[src_right] = LEFT_SIDE
        src_side[dst_right] = LEFT_SIDE
        dst_side[dst_right] = RIGHT_SIDE
    allowed_src = context.allowed_neighbor[src, src_side]
    allowed_dst = context.allowed_neighbor[dst, dst_side]
    conflict_src = (bits[src] != 0) & (allowed_src >= 0) & (dst != allowed_src)
    conflict_dst = (bits[dst] != 0) & (allowed_dst >= 0) & (src != allowed_dst)
    keep &= ~(inter_mask & (conflict_src | conflict_dst))

    base_key = _edge_keys(src, dst, int(total_nodes))
    internal_new_count = 0
    if context.internal_key.size:
        internal_is_new = ~np.isin(context.internal_key, base_key, assume_unique=False)
        internal_new_count = int(np.count_nonzero(internal_is_new))
        out_src = np.concatenate([src[keep], context.internal_src[internal_is_new]]).astype(np.int32, copy=False)
        out_dst = np.concatenate([dst[keep], context.internal_dst[internal_is_new]]).astype(np.int32, copy=False)
        out_option = np.concatenate([option[keep], context.internal_option[internal_is_new]]).astype(np.int16, copy=False)
    else:
        out_src = src[keep]
        out_dst = dst[keep]
        out_option = option[keep]

    constrained = _edge_table_from_arrays(
        src=out_src,
        dst=out_dst,
        option=out_option,
        n=int(n),
        sat_ids=list(base_edge_table.sat_ids),
    )
    forced_count = int(context.internal_src.size)
    dropped_count = int(base_edge_table.num_edges + internal_new_count - constrained.num_edges)
    return constrained, forced_count, max(0, dropped_count)


def _build_adjacency(edge_src: np.ndarray, edge_dst: np.ndarray, total_nodes: int) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(int(total_nodes))]
    for src, dst in zip(edge_src, edge_dst):
        a = int(src)
        b = int(dst)
        adjacency[a].append(b)
        adjacency[b].append(a)
    for neighbors in adjacency:
        neighbors.sort()
    return adjacency


def _bfs_distances(adjacency: list[list[int]], source: int) -> np.ndarray:
    dist = np.full(len(adjacency), -1, dtype=np.int16)
    dist[int(source)] = 0
    queue: deque[int] = deque([int(source)])
    while queue:
        node = queue.popleft()
        next_dist = int(dist[node]) + 1
        for neighbor in adjacency[node]:
            if int(dist[neighbor]) >= 0:
                continue
            dist[neighbor] = next_dist
            queue.append(neighbor)
    return dist


def _summarize_hops(
    *,
    adjacency: list[list[int]],
    sources: Sequence[int],
    targets: Sequence[int],
) -> tuple[int, float, float, float]:
    origins = tuple(int(x) for x in sources)
    destinations = tuple(int(x) for x in targets)
    if len(destinations) < len(origins):
        origins, destinations = destinations, origins
    target_arr = np.asarray(destinations, dtype=np.int32)
    values: list[float] = []
    if len(origins) == 0 or target_arr.size == 0:
        return 0, float("nan"), float("nan"), float("nan")
    for source in origins:
        dist = _bfs_distances(adjacency, int(source))
        selected = dist[target_arr]
        reachable = selected[selected >= 0]
        if reachable.size:
            values.extend(float(x) for x in reachable)
    if not values:
        return 0, float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return int(arr.size), float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))


def _edge_weights_fast(
    *,
    edge_table,
    delay_store: FullLinkDelayStore,
    position_store: PositionCacheStore | None,
    delay_row: int,
    position_row: int | None,
) -> np.ndarray:
    src = np.asarray(edge_table.src, dtype=np.int64)
    dst = np.asarray(edge_table.dst, dtype=np.int64)
    store_indices = np.asarray(delay_store.edge_index_matrix[src, dst], dtype=np.int64)
    weights = np.empty(src.shape[0], dtype=np.float32)
    has_store_weight = store_indices >= 0
    if np.any(has_store_weight):
        weights[has_store_weight] = np.asarray(
            delay_store.delay_ms_array[int(delay_row), store_indices[has_store_weight]],
            dtype=np.float32,
        )
    if np.any(~has_store_weight):
        if position_store is None or position_row is None:
            missing = np.where(~has_store_weight)[0][:10].tolist()
            raise KeyError(f"delay store is missing {int(np.count_nonzero(~has_store_weight))} edges; preview={missing}")
        positions = np.asarray(position_store.positions_km[int(position_row)], dtype=np.float32)
        dist_km = np.linalg.norm(positions[src[~has_store_weight]] - positions[dst[~has_store_weight]], axis=1)
        weights[~has_store_weight] = (dist_km / float(LIGHT_SPEED_KM_S) * 1000.0).astype(np.float32)
    return weights


def _scipy_tools() -> tuple[Any, Any] | None:
    global _SCIPY_TOOLS
    if _SCIPY_TOOLS is False:
        return None
    if _SCIPY_TOOLS is None:
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

            _SCIPY_TOOLS = (csr_matrix, scipy_dijkstra)
        except Exception:
            _SCIPY_TOOLS = False
            return None
    return _SCIPY_TOOLS


def _summarize_delay(
    *,
    edge_table,
    config,
    delay_store: FullLinkDelayStore,
    position_store: PositionCacheStore | None,
    delay_row: int,
    position_row: int | None,
    sources: Sequence[int],
    targets: Sequence[int],
    engine: str = "auto",
) -> tuple[int, float, float, float]:
    if len(sources) == 0 or len(targets) == 0:
        return 0, float("nan"), float("nan"), float("nan")
    weights = _edge_weights_fast(
        edge_table=edge_table,
        delay_store=delay_store,
        position_store=position_store,
        delay_row=int(delay_row),
        position_row=int(position_row) if position_row is not None else None,
    )
    origins = tuple(int(x) for x in sources)
    destinations = tuple(int(x) for x in targets)
    if len(destinations) < len(origins):
        origins, destinations = destinations, origins

    selected_engine = str(engine or "auto").lower()
    tools = None if selected_engine == "heapq" else _scipy_tools()
    if selected_engine == "scipy" and tools is None:
        raise RuntimeError("delay engine 'scipy' was requested, but scipy is not available")
    if tools is not None:
        csr_matrix, scipy_dijkstra = tools
        src = np.asarray(edge_table.src, dtype=np.int32)
        dst = np.asarray(edge_table.dst, dtype=np.int32)
        graph = csr_matrix(
            (
                np.concatenate([weights, weights]).astype(np.float32, copy=False),
                (np.concatenate([src, dst]), np.concatenate([dst, src])),
            ),
            shape=(int(config.total_sats), int(config.total_sats)),
        )
        dist = scipy_dijkstra(
            graph,
            directed=True,
            indices=np.asarray(origins, dtype=np.int32),
            return_predecessors=False,
        )
        selected = np.asarray(dist[:, np.asarray(destinations, dtype=np.int32)], dtype=np.float64)
        reachable = selected[np.isfinite(selected)]
        if not reachable.size:
            return 0, float("nan"), float("nan"), float("nan")
        return (
            int(reachable.size),
            float(np.mean(reachable)),
            float(np.min(reachable)),
            float(np.max(reachable)),
        )

    adjacency = build_heapq_adjacency(edge_table, int(config.total_sats))
    components = connected_component_ids(adjacency)
    targets_by_component: dict[int, list[int]] = {}
    for target in destinations:
        comp_id = int(components[int(target)])
        targets_by_component.setdefault(comp_id, []).append(int(target))
    values: list[float] = []
    for source in origins:
        source_comp = int(components[int(source)])
        local_targets = targets_by_component.get(source_comp, [])
        if not local_targets:
            continue
        dist, _prev = dijkstra_heapq(
            adjacency=adjacency,
            weights=weights,
            source=int(source),
            targets=set(local_targets),
        )
        for target in local_targets:
            value = float(dist[int(target)])
            if math.isfinite(value):
                values.append(value)
    if not values:
        return 0, float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return int(arr.size), float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))


def _write_step_rows(path: Path, rows: Sequence[StepMetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(StepMetricRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in fields})


def _topology_pair_complete(topology_dir: Path, steps: Sequence[int]) -> bool:
    time_path = topology_dir / "time_indices.npy"
    hops_path = topology_dir / "mean_shortest_hops.npy"
    delay_path = topology_dir / "mean_shortest_delay_ms.npy"
    metrics_path = topology_dir / "step_metrics.csv"
    if not (time_path.exists() and hops_path.exists() and delay_path.exists() and metrics_path.exists()):
        return False
    try:
        saved_steps = np.load(time_path, mmap_mode="r")
        if not np.array_equal(np.asarray(saved_steps, dtype=np.int64), np.asarray(steps, dtype=np.int64)):
            return False
        hops = np.load(hops_path, mmap_mode="r")
        delay = np.load(delay_path, mmap_mode="r")
        return hops.shape == (len(steps),) and delay.shape == (len(steps),)
    except Exception:
        return False


def _load_topology_pair_values(topology_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(np.load(topology_dir / "mean_shortest_hops.npy"), dtype=np.float32),
        np.asarray(np.load(topology_dir / "mean_shortest_delay_ms.npy"), dtype=np.float32),
    )


def _write_compare_csv(
    path: Path,
    *,
    steps: Sequence[int],
    values_by_topology: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(values_by_topology.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", *names])
        writer.writeheader()
        for idx, step in enumerate(steps):
            row: dict[str, Any] = {"step": int(step)}
            for name in names:
                row[name] = _finite_or_none(float(values_by_topology[name][idx]))
            writer.writerow(row)


def _write_summary_csv(path: Path, values_by_topology: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["topology", "mean", "min", "max", "finite_steps"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, raw_values in values_by_topology.items():
            values = np.asarray(raw_values, dtype=np.float64)
            finite = values[np.isfinite(values)]
            writer.writerow(
                {
                    "topology": str(name),
                    "mean": float(np.mean(finite)) if finite.size else None,
                    "min": float(np.min(finite)) if finite.size else None,
                    "max": float(np.max(finite)) if finite.size else None,
                    "finite_steps": int(finite.size),
                }
            )


def _plot_compare(
    path: Path,
    *,
    steps: Sequence[int],
    values_by_topology: Mapping[str, np.ndarray],
    title: str,
    ylabel: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[paper1-region-grid] plot skipped for {path}: {exc}", flush=True)
        return

    x_hours = np.asarray(steps, dtype=np.float64) / 3600.0
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 7), dpi=180)
    for name, values in values_by_topology.items():
        ax.plot(x_hours, values, linewidth=0.8, alpha=0.8, label=str(name))
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(str(ylabel))
    ax.set_title(str(title))
    ax.grid(True, alpha=0.25)
    if len(values_by_topology) <= 20:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _compute_topology_pair_task(task: dict[str, Any]) -> dict[str, Any]:
    config = task["config"]
    spec: TopologySpec = task["topology_spec"]
    pair: RegionPairSpec = task["pair_spec"]
    group_data = task["group_data"]
    steps = [int(x) for x in task["steps"]]
    delay_store = FullLinkDelayStore(task["delay_store_dir"])
    delay_rows = delay_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(task["stride"]))
    position_store = PositionCacheStore(task["position_cache_dir"]) if task.get("position_cache_dir") else None
    position_rows = (
        position_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(task["stride"]))
        if position_store is not None
        else None
    )
    out_dir = Path(task["out_dir"])
    forced_option = int(task["forced_option"])
    compute_hops = bool(task.get("compute_hops", True))
    delay_engine = str(task.get("delay_engine", "auto"))
    step_heartbeat_seconds = float(task.get("step_heartbeat_seconds", 0.0) or 0.0)

    hop_values = np.full(len(steps), np.nan, dtype=np.float32)
    delay_values = np.full(len(steps), np.nan, dtype=np.float32)
    step_rows: list[StepMetricRow] = []

    step_heartbeat_last = time.perf_counter()
    for local_idx, step in enumerate(steps):
        if step_heartbeat_seconds > 0:
            now = time.perf_counter()
            if now - step_heartbeat_last >= step_heartbeat_seconds:
                step_heartbeat_last = now
                print(
                    f"[paper1-region-grid] heartbeat topology={spec.name} pair={pair.key} "
                    f"step_index={local_idx}/{len(steps)} step={int(step)}",
                    flush=True,
                )
        group_nodes = _group_nodes_mapping_for_step(group_data, int(step))
        internal_edges = build_region_internal_option_edges(
            config=config,
            group_nodes=group_nodes,
            constrained_groups=(int(pair.source_group_id), int(pair.target_group_id)),
            option=forced_option,
            wrap_planes=bool(task.get("wrap_planes", False)),
        )
        constrained_edge_table, stats = apply_region_internal_option_constraint(
            base_edge_table=spec.edge_table,
            internal_option_edge_table=internal_edges,
            group_nodes=group_nodes,
            constrained_groups=(int(pair.source_group_id), int(pair.target_group_id)),
            total_nodes=int(config.total_sats),
            p=int(config.P),
            n=int(config.N),
            forced_option=forced_option,
            group_names={gid: group_name(config, gid) for gid in (pair.source_group_id, pair.target_group_id)},
            wrap_planes=bool(task.get("wrap_planes", False)),
        )
        sources = group_nodes_for_step(group_data, int(step), int(pair.source_group_id))
        targets = group_nodes_for_step(group_data, int(step), int(pair.target_group_id))
        expected_pairs = int(len(sources) * len(targets))

        if compute_hops:
            hop_adjacency = _build_adjacency(constrained_edge_table.src, constrained_edge_table.dst, int(config.total_sats))
            reachable_hops, mean_hops, min_hops, max_hops = _summarize_hops(
                adjacency=hop_adjacency,
                sources=sources,
                targets=targets,
            )
        else:
            reachable_hops, mean_hops, min_hops, max_hops = 0, float("nan"), float("nan"), float("nan")
        reachable_delay, mean_delay, min_delay, max_delay = _summarize_delay(
            edge_table=constrained_edge_table,
            config=config,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=int(delay_rows[local_idx]),
            position_row=int(position_rows[local_idx]) if position_rows is not None else None,
            sources=sources,
            targets=targets,
            engine=delay_engine,
        )
        hop_values[local_idx] = mean_hops
        delay_values[local_idx] = mean_delay
        step_rows.append(
            StepMetricRow(
                step=int(step),
                source_nodes=int(len(sources)),
                target_nodes=int(len(targets)),
                reachable_pairs=int(min(reachable_hops, reachable_delay)),
                expected_pairs=expected_pairs,
                mean_shortest_hops=float(mean_hops),
                min_shortest_hops=float(min_hops),
                max_shortest_hops=float(max_hops),
                mean_shortest_delay_ms=float(mean_delay),
                min_shortest_delay_ms=float(min_delay),
                max_shortest_delay_ms=float(max_delay),
                constrained_edges=int(constrained_edge_table.num_edges),
                forced_internal_option_edges=int(stats.forced_internal_option_edges),
                dropped_edges=int(stats.dropped_edges),
            )
        )

    topology_dir = out_dir / str(pair.key) / "topologies" / str(spec.name)
    _write_step_rows(topology_dir / "step_metrics.csv", step_rows)
    np.save(topology_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))
    np.save(topology_dir / "mean_shortest_hops.npy", hop_values)
    np.save(topology_dir / "mean_shortest_delay_ms.npy", delay_values)
    (topology_dir / "meta.json").write_text(
        json.dumps(
            {
                "topology": str(spec.name),
                "motif_id": spec.motif_id,
                "motif": spec.motif,
                "pair": str(pair.key),
                "source_group_id": int(pair.source_group_id),
                "target_group_id": int(pair.target_group_id),
                "forced_internal_option": forced_option,
                "num_steps": len(steps),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "pair_key": str(pair.key),
        "topology": str(spec.name),
        "hops": hop_values,
        "delay": delay_values,
    }


def _init_worker_context(payload: dict[str, Any]) -> None:
    delay_store = FullLinkDelayStore(payload["delay_store_dir"])
    steps = [int(x) for x in payload["steps"]]
    position_store = PositionCacheStore(payload["position_cache_dir"]) if payload.get("position_cache_dir") else None
    pair_specs = payload["pair_specs"]
    config = payload["config"]
    print(
        f"[paper1-region-grid] worker init contexts pairs={len(pair_specs)} steps={len(steps)}",
        flush=True,
    )
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(
        {
            **payload,
            "steps": steps,
            "delay_store": delay_store,
            "delay_rows": delay_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(payload["stride"])),
            "position_store": position_store,
            "position_rows": (
                position_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(payload["stride"]))
                if position_store is not None
                else None
            ),
        "pair_step_contexts": _build_pair_step_contexts(
            config=config,
            group_data=payload["group_data"],
            steps=steps,
            pair_specs=pair_specs,
            forced_option=int(payload["forced_option"]),
            wrap_planes=bool(payload.get("wrap_planes", False)),
        ),
        }
    )


def _compute_topology_all_pairs_task(task: tuple[TopologySpec, Sequence[RegionPairSpec]]) -> dict[str, Any]:
    if not _WORKER_CONTEXT:
        raise RuntimeError("worker context is not initialized")
    spec, pair_specs = task
    config = _WORKER_CONTEXT["config"]
    group_data = _WORKER_CONTEXT["group_data"]
    steps = _WORKER_CONTEXT["steps"]
    delay_store = _WORKER_CONTEXT["delay_store"]
    delay_rows = _WORKER_CONTEXT["delay_rows"]
    position_store = _WORKER_CONTEXT["position_store"]
    position_rows = _WORKER_CONTEXT["position_rows"]
    out_dir = Path(_WORKER_CONTEXT["out_dir"])
    forced_option = int(_WORKER_CONTEXT["forced_option"])
    force = bool(_WORKER_CONTEXT.get("force", False))
    compute_hops = bool(_WORKER_CONTEXT.get("compute_hops", True))
    delay_engine = str(_WORKER_CONTEXT.get("delay_engine", "auto"))
    step_heartbeat_seconds = float(_WORKER_CONTEXT.get("step_heartbeat_seconds", 0.0) or 0.0)
    pair_step_contexts = _WORKER_CONTEXT["pair_step_contexts"]

    results: dict[str, dict[str, np.ndarray]] = {}
    for pair in pair_specs:
        contexts = pair_step_contexts[str(pair.key)]
        topology_dir = out_dir / str(pair.key) / "topologies" / str(spec.name)
        if not force and _topology_pair_complete(topology_dir, steps):
            hops, delay = _load_topology_pair_values(topology_dir)
            results[str(pair.key)] = {"hops": hops, "delay": delay}
            continue

        hop_values = np.full(len(steps), np.nan, dtype=np.float32)
        delay_values = np.full(len(steps), np.nan, dtype=np.float32)
        step_rows: list[StepMetricRow] = []

        step_heartbeat_last = time.perf_counter()
        for local_idx, step in enumerate(steps):
            if step_heartbeat_seconds > 0:
                now = time.perf_counter()
                if now - step_heartbeat_last >= step_heartbeat_seconds:
                    step_heartbeat_last = now
                    print(
                        f"[paper1-region-grid] heartbeat topology={spec.name} pair={pair.key} "
                        f"step_index={local_idx}/{len(steps)} step={int(step)}",
                        flush=True,
                    )
            context = contexts[local_idx]
            constrained_edge_table, forced_internal_edges, dropped_edges = _apply_constraint_context_fast(
                base_edge_table=spec.edge_table,
                context=context,
                total_nodes=int(config.total_sats),
                p=int(config.P),
                n=int(config.N),
                forced_option=forced_option,
                wrap_planes=bool(_WORKER_CONTEXT.get("wrap_planes", False)),
            )
            sources = context.sources
            targets = context.targets
            expected_pairs = int(len(sources) * len(targets))

            if compute_hops:
                hop_adjacency = _build_adjacency(
                    constrained_edge_table.src,
                    constrained_edge_table.dst,
                    int(config.total_sats),
                )
                reachable_hops, mean_hops, min_hops, max_hops = _summarize_hops(
                    adjacency=hop_adjacency,
                    sources=sources,
                    targets=targets,
                )
            else:
                reachable_hops, mean_hops, min_hops, max_hops = 0, float("nan"), float("nan"), float("nan")
            reachable_delay, mean_delay, min_delay, max_delay = _summarize_delay(
                edge_table=constrained_edge_table,
                config=config,
                delay_store=delay_store,
                position_store=position_store,
                delay_row=int(delay_rows[local_idx]),
                position_row=int(position_rows[local_idx]) if position_rows is not None else None,
                sources=sources,
                targets=targets,
                engine=delay_engine,
            )
            hop_values[local_idx] = mean_hops
            delay_values[local_idx] = mean_delay
            step_rows.append(
                StepMetricRow(
                    step=int(step),
                    source_nodes=int(len(sources)),
                    target_nodes=int(len(targets)),
                    reachable_pairs=int(min(reachable_hops, reachable_delay)),
                    expected_pairs=expected_pairs,
                    mean_shortest_hops=float(mean_hops),
                    min_shortest_hops=float(min_hops),
                    max_shortest_hops=float(max_hops),
                    mean_shortest_delay_ms=float(mean_delay),
                    min_shortest_delay_ms=float(min_delay),
                    max_shortest_delay_ms=float(max_delay),
                    constrained_edges=int(constrained_edge_table.num_edges),
                    forced_internal_option_edges=int(forced_internal_edges),
                    dropped_edges=int(dropped_edges),
                )
            )

        _write_step_rows(topology_dir / "step_metrics.csv", step_rows)
        np.save(topology_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))
        np.save(topology_dir / "mean_shortest_hops.npy", hop_values)
        np.save(topology_dir / "mean_shortest_delay_ms.npy", delay_values)
        (topology_dir / "meta.json").write_text(
            json.dumps(
                {
                    "topology": str(spec.name),
                    "motif_id": spec.motif_id,
                    "motif": spec.motif,
                    "pair": str(pair.key),
                    "source_group_id": int(pair.source_group_id),
                    "target_group_id": int(pair.target_group_id),
                    "forced_internal_option": forced_option,
                    "num_steps": len(steps),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        results[str(pair.key)] = {"hops": hop_values, "delay": delay_values}
    return {"topology": str(spec.name), "pairs": results}


def _worker_count(requested: int) -> int:
    if int(requested) > 0:
        return int(requested)
    return 1


def _write_topology_library(path: Path, specs: Sequence[TopologySpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "library", "motif_id", "source_w", "source_h", "edge_count", "motif", "support"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "name": spec.name,
                    "library": spec.library,
                    "motif_id": spec.motif_id,
                    "source_w": spec.source_w,
                    "source_h": spec.source_h,
                    "edge_count": spec.edge_table.num_edges,
                    "motif": spec.motif,
                    "support": spec.support,
                }
            )


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    raw = load_yaml(args.config)
    config = viewer_config_from_workflow(raw)
    paths = raw.get("paths", {}) if isinstance(raw.get("paths", {}), dict) else {}
    run_raw = raw.get("run", {}) if isinstance(raw.get("run", {}), dict) else {}
    constraint_raw = raw.get("region_internal_constraint", {}) if isinstance(raw.get("region_internal_constraint", {}), dict) else {}

    start, end, stride = time_axis_from_config(raw)
    if args.start is not None:
        start = int(args.start)
    if args.end is not None:
        end = int(args.end)
    if args.stride is not None:
        stride = int(args.stride)
    steps = list(range(int(start), int(end) + 1, int(stride)))
    if not steps:
        raise ValueError("time range produced no steps")

    out_dir = Path(args.out_dir) if args.out_dir is not None else path_from(paths, "out_dir")
    limit_motifs = int(args.limit_motifs if args.limit_motifs is not None else run_raw.get("limit_motifs", 0))
    motif_offset = int(args.motif_offset if args.motif_offset is not None else run_raw.get("motif_offset", 0))
    max_workers = int(args.max_workers if args.max_workers is not None else run_raw.get("max_workers", 1))
    progress_every = int(args.progress_every if args.progress_every is not None else run_raw.get("progress_every", 25))
    forced_option = int(constraint_raw.get("forced_option", 0))
    compute_hops = bool(run_raw.get("compute_hops", True)) and not bool(args.skip_hops)
    delay_engine = str(args.delay_engine or run_raw.get("delay_engine", "auto"))
    step_heartbeat_seconds = float(
        args.step_heartbeat_seconds
        if args.step_heartbeat_seconds is not None
        else run_raw.get("step_heartbeat_seconds", 0.0)
    )
    wrap_planes = wrap_planes_from_config(raw)
    delay_store_dir = Path(args.delay_store_dir) if args.delay_store_dir is not None else path_from(paths, "delay_store_dir")
    if args.no_position_cache:
        position_cache_dir = None
    elif args.position_cache_dir is not None:
        position_cache_dir = Path(args.position_cache_dir)
    elif paths.get("position_cache_dir"):
        position_cache_dir = Path(paths["position_cache_dir"])
    else:
        position_cache_dir = None

    csv_path = ensure_motif_library(raw, force=bool(args.regenerate_library))
    library_raw = raw.get("motif_library", {}) if isinstance(raw.get("motif_library", {}), dict) else {}
    topology_specs = topology_specs_from_motif_csv(
        csv_path,
        config=config,
        library="combined_motif",
        name_prefix=str(library_raw.get("name_prefix", "combined")),
        limit=limit_motifs,
        offset=motif_offset,
        add_intra_ring=True,
        wrap_planes=wrap_planes,
    )
    pair_specs = region_pair_specs(raw, subset=args.pairs)

    group_data = load_or_build_group_data(
        xml_file=path_from(paths, "group_xml"),
        group_cache_dir=path_from(paths, "group_cache_dir"),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(stride),
        enabled=True,
        force=bool(args.force_group_cache or run_raw.get("force_group_cache", False)),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_topology_library(out_dir / "topology_library.csv", topology_specs)
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))

    print(
        f"[paper1-region-grid] topologies={len(topology_specs)} pairs={len(pair_specs)} "
        f"steps={len(steps)} forced_option={forced_option} workers={_worker_count(max_workers)} "
        f"delay_engine={delay_engine}",
        flush=True,
    )
    worker_payload = {
        "config": config,
        "group_data": group_data,
        "pair_specs": pair_specs,
        "steps": steps,
        "stride": int(stride),
        "delay_store_dir": str(delay_store_dir),
        "position_cache_dir": str(position_cache_dir) if position_cache_dir is not None else None,
        "out_dir": str(out_dir),
        "forced_option": forced_option,
        "force": bool(args.force),
        "wrap_planes": bool(wrap_planes),
        "compute_hops": bool(compute_hops),
        "delay_engine": str(delay_engine),
        "step_heartbeat_seconds": float(step_heartbeat_seconds),
    }
    tasks = [(spec, pair_specs) for spec in topology_specs]
    pair_hops: dict[str, dict[str, np.ndarray]] = {str(pair.key): {} for pair in pair_specs}
    pair_delay: dict[str, dict[str, np.ndarray]] = {str(pair.key): {} for pair in pair_specs}

    completed = 0
    workers = _worker_count(max_workers)
    if workers <= 1:
        _init_worker_context(worker_payload)
        for task in tasks:
            result = _compute_topology_all_pairs_task(task)
            topology = str(result["topology"])
            for pair_key, payload in result["pairs"].items():
                pair_hops[str(pair_key)][topology] = payload["hops"]
                pair_delay[str(pair_key)][topology] = payload["delay"]
            completed += 1
            if progress_every > 0 and (completed == len(tasks) or completed % progress_every == 0):
                print(f"[paper1-region-grid] completed {completed}/{len(tasks)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker_context, initargs=(worker_payload,)) as executor:
            futures = [executor.submit(_compute_topology_all_pairs_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                topology = str(result["topology"])
                for pair_key, payload in result["pairs"].items():
                    pair_hops[str(pair_key)][topology] = payload["hops"]
                    pair_delay[str(pair_key)][topology] = payload["delay"]
                completed += 1
                if progress_every > 0 and (completed == len(tasks) or completed % progress_every == 0):
                    print(f"[paper1-region-grid] completed {completed}/{len(tasks)} elapsed={time.perf_counter() - started:.1f}s", flush=True)

    pair_meta: dict[str, Any] = {}
    for pair in pair_specs:
        pair_dir = out_dir / str(pair.key)
        pair_dir.mkdir(parents=True, exist_ok=True)
        label = f"{group_name(config, pair.source_group_id)}-{group_name(config, pair.target_group_id)}"
        _write_compare_csv(
            pair_dir / "compare_mean_shortest_hops.csv",
            steps=steps,
            values_by_topology=pair_hops[str(pair.key)],
        )
        _write_compare_csv(
            pair_dir / "compare_mean_shortest_delay_ms.csv",
            steps=steps,
            values_by_topology=pair_delay[str(pair.key)],
        )
        _write_summary_csv(pair_dir / "summary_mean_shortest_hops.csv", pair_hops[str(pair.key)])
        _write_summary_csv(pair_dir / "summary_mean_shortest_delay_ms.csv", pair_delay[str(pair.key)])
        _plot_compare(
            pair_dir / "compare_mean_shortest_hops.png",
            steps=steps,
            values_by_topology=pair_hops[str(pair.key)],
            title=f"{config.name} {label} region-internal +grid shortest hops",
            ylabel=f"{label} mean shortest hops",
        )
        _plot_compare(
            pair_dir / "compare_mean_shortest_delay_ms.png",
            steps=steps,
            values_by_topology=pair_delay[str(pair.key)],
            title=f"{config.name} {label} region-internal +grid shortest delay",
            ylabel=f"{label} mean shortest delay (ms)",
        )
        pair_meta[str(pair.key)] = {
            "label": label,
            "compare_mean_shortest_hops_csv": str(pair_dir / "compare_mean_shortest_hops.csv"),
            "compare_mean_shortest_delay_ms_csv": str(pair_dir / "compare_mean_shortest_delay_ms.csv"),
            "summary_mean_shortest_hops_csv": str(pair_dir / "summary_mean_shortest_hops.csv"),
            "summary_mean_shortest_delay_ms_csv": str(pair_dir / "summary_mean_shortest_delay_ms.csv"),
        }

    meta = {
        "constellation": config.name,
        "start": int(start),
        "end": int(end),
        "stride": int(stride),
        "num_steps": len(steps),
        "num_topologies": len(topology_specs),
        "motif_offset": int(motif_offset),
        "limit_motifs": int(limit_motifs),
        "num_pairs": len(pair_specs),
        "forced_internal_option": forced_option,
        "wrap_planes": bool(wrap_planes),
        "compute_hops": bool(compute_hops),
        "delay_engine": str(delay_engine),
        "step_heartbeat_seconds": float(step_heartbeat_seconds),
        "motif_csv": str(csv_path),
        "delay_store_dir": str(delay_store_dir),
        "position_cache_dir": str(position_cache_dir) if position_cache_dir is not None else None,
        "out_dir": str(out_dir),
        "pairs": pair_meta,
        "elapsed_s": float(time.perf_counter() - started),
    }
    (out_dir / "region_internal_grid_metrics_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
