from __future__ import annotations

import csv
import heapq
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.delay_store import LIGHT_SPEED_KM_S
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.link_delay.module.position_cache import PositionCacheStore
from src.link_delay.module.query import FullLinkDelayStore
from src.topology_metrics.module.group_states import group_nodes_for_step

from .config import group_name
from .edge_tables import INTRA_OPTION


@dataclass(frozen=True)
class ShortestDelaySummary:
    step: int
    source_nodes: int
    target_nodes: int
    reachable_pairs: int
    total_shortest_delay_ms: float
    mean_shortest_delay_ms: float
    min_shortest_delay_ms: float
    max_shortest_delay_ms: float


@dataclass(frozen=True)
class WeightLookup:
    topology_edge_indices: np.ndarray
    delay_store_edge_indices: np.ndarray
    fallback_intra_edge_indices: np.ndarray
    fallback_intra_src_nodes: np.ndarray
    fallback_intra_dst_nodes: np.ndarray


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def node_xy(node: int, config: ViewerConfig) -> tuple[int, int]:
    node = int(node)
    return node // int(config.N), node % int(config.N)


def path_xy_text(path: list[int], config: ViewerConfig) -> str:
    return "->".join(f"({p},{y})" for p, y in (node_xy(node, config) for node in path))


def build_weight_lookup(
    edge_table: EdgeTable,
    delay_store: FullLinkDelayStore,
    *,
    config: ViewerConfig,
    allow_intra_fallback: bool,
) -> WeightLookup:
    topology_edge_indices: list[int] = []
    store_edge_indices: list[int] = []
    fallback_intra_indices: list[int] = []
    fallback_intra_src: list[int] = []
    fallback_intra_dst: list[int] = []
    missing: list[str] = []

    for idx in range(edge_table.num_edges):
        src = int(edge_table.src[idx])
        dst = int(edge_table.dst[idx])
        option = int(edge_table.option[idx])
        try:
            delay_idx = delay_store.edge_idx(src, dst)
        except KeyError:
            if option == INTRA_OPTION and allow_intra_fallback:
                fallback_intra_indices.append(idx)
                fallback_intra_src.append(src)
                fallback_intra_dst.append(dst)
                continue
            sp, sy = node_xy(src, config)
            dp, dy = node_xy(dst, config)
            missing.append(f"edge_idx={idx} {src}({sp},{sy})-{dst}({dp},{dy}) option={option}")
            continue
        topology_edge_indices.append(idx)
        store_edge_indices.append(int(delay_idx))

    if missing:
        preview = "\n".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise KeyError(
            "Some topology edges are not present in the delay store:\n"
            f"{preview}{more}\n"
            "Use a plus-intra full-option delay store or enable position-cache fallback for intra links."
        )

    return WeightLookup(
        topology_edge_indices=np.asarray(topology_edge_indices, dtype=np.int64),
        delay_store_edge_indices=np.asarray(store_edge_indices, dtype=np.int64),
        fallback_intra_edge_indices=np.asarray(fallback_intra_indices, dtype=np.int64),
        fallback_intra_src_nodes=np.asarray(fallback_intra_src, dtype=np.int64),
        fallback_intra_dst_nodes=np.asarray(fallback_intra_dst, dtype=np.int64),
    )


def edge_weights_for_step(
    *,
    edge_table: EdgeTable,
    lookup: WeightLookup,
    delay_store: FullLinkDelayStore,
    position_store: PositionCacheStore | None,
    delay_row: int,
    position_row: int | None,
) -> np.ndarray:
    weights = np.empty(edge_table.num_edges, dtype=np.float32)
    if lookup.topology_edge_indices.size:
        weights[lookup.topology_edge_indices] = np.asarray(
            delay_store.delay_ms_array[int(delay_row), lookup.delay_store_edge_indices],
            dtype=np.float32,
        )

    if lookup.fallback_intra_edge_indices.size:
        if position_store is None or position_row is None:
            raise RuntimeError("position cache fallback was requested, but no position row is available")
        positions = np.asarray(position_store.positions_km[int(position_row)], dtype=np.float32)
        src_pos = positions[lookup.fallback_intra_src_nodes]
        dst_pos = positions[lookup.fallback_intra_dst_nodes]
        dist_km = np.linalg.norm(src_pos - dst_pos, axis=1)
        weights[lookup.fallback_intra_edge_indices] = (dist_km / float(LIGHT_SPEED_KM_S) * 1000.0).astype(np.float32)
    return weights


def build_bidirectional_sparse_parts(edge_table: EdgeTable) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(edge_table.src, dtype=np.int32)
    dst = np.asarray(edge_table.dst, dtype=np.int32)
    return np.concatenate([src, dst]), np.concatenate([dst, src])


def build_heapq_adjacency(edge_table: EdgeTable, total_nodes: int) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(int(total_nodes))]
    for edge_idx in range(edge_table.num_edges):
        src = int(edge_table.src[edge_idx])
        dst = int(edge_table.dst[edge_idx])
        adjacency[src].append((dst, edge_idx))
        adjacency[dst].append((src, edge_idx))
    return adjacency


def connected_component_ids(adjacency: list[list[tuple[int, int]]]) -> np.ndarray:
    """Return a component id for each node in an undirected static topology."""

    comp = np.full(len(adjacency), -1, dtype=np.int32)
    current = 0
    for start in range(len(adjacency)):
        if comp[start] >= 0:
            continue
        comp[start] = current
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor, _edge_idx in adjacency[node]:
                neighbor = int(neighbor)
                if comp[neighbor] >= 0:
                    continue
                comp[neighbor] = current
                stack.append(neighbor)
        current += 1
    return comp


def choose_engine(requested: str):
    if requested in ("auto", "scipy"):
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

            return "scipy", csr_matrix, scipy_dijkstra
        except Exception as exc:
            if requested == "scipy":
                raise RuntimeError(f"engine=scipy was requested, but scipy is unavailable: {exc}") from exc
    return "heapq", None, None


def dijkstra_heapq(
    *,
    adjacency: list[list[tuple[int, int]]],
    weights: np.ndarray,
    source: int,
    targets: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(adjacency)
    dist = np.full(n, np.inf, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int32)
    dist[int(source)] = 0.0
    heap: list[tuple[float, int]] = [(0.0, int(source))]
    remaining = set(int(x) for x in targets)

    while heap and remaining:
        current_dist, node = heapq.heappop(heap)
        if current_dist != float(dist[node]):
            continue
        remaining.discard(node)
        for neighbor, edge_idx in adjacency[node]:
            next_dist = current_dist + float(weights[int(edge_idx)])
            if next_dist < float(dist[neighbor]):
                dist[neighbor] = next_dist
                prev[neighbor] = node
                heapq.heappush(heap, (next_dist, neighbor))
    return dist, prev


def reconstruct_from_prev(prev: np.ndarray, source: int, target: int) -> list[int]:
    source = int(source)
    target = int(target)
    if source == target:
        return [source]
    path = [target]
    current = target
    guard = 0
    while current != source:
        current = int(prev[current])
        if current < 0:
            return []
        path.append(current)
        guard += 1
        if guard > prev.size:
            return []
    path.reverse()
    return path


def summarize_distances(
    *,
    step: int,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
    values: np.ndarray,
) -> ShortestDelaySummary:
    reachable = np.isfinite(values)
    if sources and targets and bool(np.any(reachable)):
        reachable_values = np.asarray(values[reachable], dtype=np.float64)
        total = float(np.sum(reachable_values))
        count = int(reachable_values.size)
        return ShortestDelaySummary(
            step=int(step),
            source_nodes=len(sources),
            target_nodes=len(targets),
            reachable_pairs=count,
            total_shortest_delay_ms=total,
            mean_shortest_delay_ms=float(total / count),
            min_shortest_delay_ms=float(np.min(reachable_values)),
            max_shortest_delay_ms=float(np.max(reachable_values)),
        )
    return ShortestDelaySummary(
        step=int(step),
        source_nodes=len(sources),
        target_nodes=len(targets),
        reachable_pairs=0,
        total_shortest_delay_ms=0.0,
        mean_shortest_delay_ms=float("nan"),
        min_shortest_delay_ms=float("nan"),
        max_shortest_delay_ms=float("nan"),
    )


def summary_to_row(item: ShortestDelaySummary) -> dict:
    return {
        "step": int(item.step),
        "source_nodes": int(item.source_nodes),
        "target_nodes": int(item.target_nodes),
        "reachable_pairs": int(item.reachable_pairs),
        "total_shortest_delay_ms": finite_or_none(item.total_shortest_delay_ms),
        "mean_shortest_delay_ms": finite_or_none(item.mean_shortest_delay_ms),
        "min_shortest_delay_ms": finite_or_none(item.min_shortest_delay_ms),
        "max_shortest_delay_ms": finite_or_none(item.max_shortest_delay_ms),
    }


def write_path_sample_row(
    writer: csv.DictWriter,
    *,
    step: int,
    source: int,
    target: int,
    delay_ms: float,
    path: list[int],
    config: ViewerConfig,
) -> None:
    writer.writerow(
        {
            "step": int(step),
            "source": int(source),
            "source_xy": str(node_xy(source, config)),
            "target": int(target),
            "target_xy": str(node_xy(target, config)),
            "delay_ms": float(delay_ms),
            "path": "->".join(str(x) for x in path),
            "path_xy": path_xy_text(path, config),
            "path_indexed": ",".join(f"{idx + 1}:{node}" for idx, node in enumerate(path)),
        }
    )


def compute_shortest_delay_timeseries(
    *,
    topology_name: str,
    edge_table: EdgeTable,
    config: ViewerConfig,
    group_data: dict,
    steps: list[int],
    delay_rows: np.ndarray,
    position_rows: np.ndarray | None,
    delay_store: FullLinkDelayStore,
    position_store: PositionCacheStore | None,
    source_group_id: int,
    target_group_id: int,
    out_dir: str | Path,
    engine: str = "auto",
    sample_steps: int = 3,
    sample_pairs_per_step: int = 60,
    progress_every: int = 200,
) -> np.ndarray:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(edge_table, out_dir / "edges.csv")
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))

    engine_name, csr_matrix, scipy_dijkstra = choose_engine(str(engine))
    lookup = build_weight_lookup(
        edge_table,
        delay_store,
        config=config,
        allow_intra_fallback=position_store is not None,
    )
    row_index, col_index = build_bidirectional_sparse_parts(edge_table)
    heapq_adjacency = (
        build_heapq_adjacency(edge_table, int(config.total_sats)) if engine_name == "heapq" else None
    )
    heapq_components = connected_component_ids(heapq_adjacency) if heapq_adjacency is not None else None
    mean_values = np.full(len(steps), np.nan, dtype=np.float32)

    summary_fields = [
        "step",
        "source_nodes",
        "target_nodes",
        "reachable_pairs",
        "total_shortest_delay_ms",
        "mean_shortest_delay_ms",
        "min_shortest_delay_ms",
        "max_shortest_delay_ms",
    ]
    path_fields = ["step", "source", "source_xy", "target", "target_xy", "delay_ms", "path", "path_xy", "path_indexed"]

    print(
        f"[topology-workflow] shortest_delay topology={topology_name} edges={edge_table.num_edges} "
        f"delay_store_edges={lookup.topology_edge_indices.size} "
        f"intra_position_fallback={lookup.fallback_intra_edge_indices.size} engine={engine_name}",
        flush=True,
    )
    started_at = time.time()

    with (out_dir / "step_summary.csv").open("w", encoding="utf-8", newline="") as summary_f, (
        out_dir / "path_samples.csv"
    ).open("w", encoding="utf-8", newline="") as sample_f:
        summary_writer = csv.DictWriter(summary_f, fieldnames=summary_fields)
        sample_writer = csv.DictWriter(sample_f, fieldnames=path_fields)
        summary_writer.writeheader()
        sample_writer.writeheader()

        for local_idx, step in enumerate(steps):
            sources = group_nodes_for_step(group_data, int(step), int(source_group_id))
            targets = group_nodes_for_step(group_data, int(step), int(target_group_id))
            sample_this_step = local_idx < max(0, int(sample_steps)) and int(sample_pairs_per_step) > 0
            sample_written = 0

            if not sources or not targets:
                item = summarize_distances(
                    step=int(step),
                    sources=sources,
                    targets=targets,
                    values=np.asarray([], dtype=np.float64),
                )
                summary_writer.writerow(summary_to_row(item))
                continue

            weights = edge_weights_for_step(
                edge_table=edge_table,
                lookup=lookup,
                delay_store=delay_store,
                position_store=position_store,
                delay_row=int(delay_rows[local_idx]),
                position_row=int(position_rows[local_idx]) if position_rows is not None else None,
            )
            target_array = np.asarray(targets, dtype=np.int32)

            if engine_name == "scipy":
                graph_values = np.concatenate([weights, weights]).astype(np.float32, copy=False)
                graph = csr_matrix(
                    (graph_values, (row_index, col_index)),
                    shape=(int(config.total_sats), int(config.total_sats)),
                )
                source_array = np.asarray(sources, dtype=np.int32)
                if sample_this_step:
                    dist_matrix, predecessors = scipy_dijkstra(
                        graph,
                        directed=True,
                        indices=source_array,
                        return_predecessors=True,
                    )
                    dist_matrix = np.atleast_2d(np.asarray(dist_matrix, dtype=np.float64))
                    predecessors = np.atleast_2d(np.asarray(predecessors, dtype=np.int32))
                else:
                    dist_matrix = scipy_dijkstra(
                        graph,
                        directed=True,
                        indices=source_array,
                        return_predecessors=False,
                    )
                    dist_matrix = np.atleast_2d(np.asarray(dist_matrix, dtype=np.float64))
                    predecessors = None

                values = dist_matrix[:, target_array]
                if sample_this_step and predecessors is not None:
                    for source_i, source in enumerate(sources):
                        for target in targets:
                            delay_ms = float(dist_matrix[source_i, int(target)])
                            if not np.isfinite(delay_ms):
                                continue
                            path = reconstruct_from_prev(predecessors[source_i], int(source), int(target))
                            write_path_sample_row(
                                sample_writer,
                                step=int(step),
                                source=int(source),
                                target=int(target),
                                delay_ms=delay_ms,
                                path=path,
                                config=config,
                            )
                            sample_written += 1
                            if sample_written >= int(sample_pairs_per_step):
                                break
                        if sample_written >= int(sample_pairs_per_step):
                            break
            else:
                values_rows: list[np.ndarray] = []
                target_components = heapq_components[target_array] if heapq_components is not None else None
                for source in sources:
                    row = np.full(len(targets), np.inf, dtype=np.float64)
                    if target_components is not None:
                        same_component = target_components == int(heapq_components[int(source)])
                        if not bool(np.any(same_component)):
                            values_rows.append(row)
                            continue
                        source_target_array = target_array[same_component]
                        target_set = set(int(x) for x in source_target_array)
                    else:
                        same_component = None
                        source_target_array = target_array
                        target_set = set(int(x) for x in targets)

                    dist, prev = dijkstra_heapq(
                        adjacency=heapq_adjacency,
                        weights=weights,
                        source=int(source),
                        targets=target_set,
                    )
                    if same_component is None:
                        row = np.asarray(dist[target_array], dtype=np.float64)
                    else:
                        row[same_component] = dist[source_target_array]
                    values_rows.append(row)
                    if sample_this_step:
                        for target in targets:
                            delay_ms = float(dist[int(target)])
                            if not np.isfinite(delay_ms):
                                continue
                            path = reconstruct_from_prev(prev, int(source), int(target))
                            write_path_sample_row(
                                sample_writer,
                                step=int(step),
                                source=int(source),
                                target=int(target),
                                delay_ms=delay_ms,
                                path=path,
                                config=config,
                            )
                            sample_written += 1
                            if sample_written >= int(sample_pairs_per_step):
                                break
                    if sample_written >= int(sample_pairs_per_step):
                        sample_this_step = False
                values = np.vstack(values_rows) if values_rows else np.asarray([], dtype=np.float64)

            item = summarize_distances(step=int(step), sources=sources, targets=targets, values=values)
            mean_values[local_idx] = item.mean_shortest_delay_ms
            summary_writer.writerow(summary_to_row(item))

            if progress_every > 0 and (local_idx + 1) % int(progress_every) == 0:
                elapsed = time.time() - started_at
                print(
                    f"[topology-workflow] {topology_name}: {local_idx + 1}/{len(steps)} "
                    f"step={step} mean={item.mean_shortest_delay_ms:.4f} ms elapsed={elapsed:.1f}s",
                    flush=True,
                )

    np.save(out_dir / "mean_shortest_delay_ms.npy", mean_values)
    meta = {
        "topology": str(topology_name),
        "constellation": config.name,
        "P": int(config.P),
        "N": int(config.N),
        "total_sats": int(config.total_sats),
        "source_group_id": int(source_group_id),
        "source_group_name": group_name(config, source_group_id),
        "target_group_id": int(target_group_id),
        "target_group_name": group_name(config, target_group_id),
        "num_steps": int(len(steps)),
        "start": int(min(steps)) if steps else None,
        "end": int(max(steps)) if steps else None,
        "num_edges": int(edge_table.num_edges),
        "edges_from_delay_store": int(lookup.topology_edge_indices.size),
        "intra_edges_from_position_cache_fallback": int(lookup.fallback_intra_edge_indices.size),
        "delay_store_dir": str(delay_store.store_dir),
        "position_cache_dir": str(position_store.cache_dir) if position_store is not None else None,
        "engine": engine_name,
        "metric": "shortest_delay",
        "mean_delay_ms_min": finite_or_none(float(np.nanmin(mean_values))) if mean_values.size else None,
        "mean_delay_ms_max": finite_or_none(float(np.nanmax(mean_values))) if mean_values.size else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[topology-workflow] {topology_name}: mean_delay_ms=({meta['mean_delay_ms_min']}, "
        f"{meta['mean_delay_ms_max']}) out_dir={out_dir}",
        flush=True,
    )
    return mean_values
