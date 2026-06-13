from __future__ import annotations

import csv
import heapq
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from src.link_delay.module.edge_options import EdgeTable, write_edges_csv

from .group_states import group_nodes_for_step


@dataclass(frozen=True)
class WeightedPairSpec:
    key: str
    source_group_id: int
    target_group_id: int
    label: str = ""


@dataclass(frozen=True)
class WeightedTopologyAdjacency:
    indptr: np.ndarray
    neighbors: np.ndarray
    edge_ids: np.ndarray


@dataclass(frozen=True)
class WeightedBetweennessSummary:
    source_nodes: int
    target_nodes: int
    reachable_pairs: int
    total_shortest_weight: float
    mean_shortest_weight: float
    min_shortest_weight: float
    max_shortest_weight: float
    max_edge_betweenness: float
    nonzero_edges: int
    edge_value_sum: float


def build_weighted_adjacency(edge_table: EdgeTable, total_nodes: int) -> WeightedTopologyAdjacency:
    total_nodes = int(total_nodes)
    degree = np.zeros(total_nodes, dtype=np.int32)
    src = np.asarray(edge_table.src, dtype=np.int32)
    dst = np.asarray(edge_table.dst, dtype=np.int32)
    np.add.at(degree, src, 1)
    np.add.at(degree, dst, 1)

    indptr = np.empty(total_nodes + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    cursor = indptr[:-1].copy()
    neighbors = np.empty(int(edge_table.num_edges) * 2, dtype=np.int32)
    edge_ids = np.empty(int(edge_table.num_edges) * 2, dtype=np.int32)

    for edge_idx in range(int(edge_table.num_edges)):
        a = int(src[edge_idx])
        b = int(dst[edge_idx])
        if not (0 <= a < total_nodes and 0 <= b < total_nodes):
            raise ValueError(f"edge {edge_idx} endpoint outside total_nodes={total_nodes}: {a}, {b}")
        pos = int(cursor[a])
        neighbors[pos] = b
        edge_ids[pos] = edge_idx
        cursor[a] += 1
        pos = int(cursor[b])
        neighbors[pos] = a
        edge_ids[pos] = edge_idx
        cursor[b] += 1
    return WeightedTopologyAdjacency(indptr=indptr, neighbors=neighbors, edge_ids=edge_ids)


def dijkstra_targets_with_prev_edge(
    *,
    adjacency: WeightedTopologyAdjacency,
    weights: np.ndarray,
    source: int,
    targets: Sequence[int],
    total_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist = np.full(int(total_nodes), np.inf, dtype=np.float64)
    prev_node = np.full(int(total_nodes), -1, dtype=np.int32)
    prev_edge = np.full(int(total_nodes), -1, dtype=np.int32)
    source = int(source)
    dist[source] = 0.0
    remaining = set(int(x) for x in targets if int(x) != source)
    heap: list[tuple[float, int]] = [(0.0, source)]

    indptr = adjacency.indptr
    neighbors = adjacency.neighbors
    edge_ids = adjacency.edge_ids
    while heap and remaining:
        current_dist, node = heapq.heappop(heap)
        if current_dist != float(dist[node]):
            continue
        remaining.discard(int(node))
        for pos in range(int(indptr[node]), int(indptr[node + 1])):
            neighbor = int(neighbors[pos])
            edge_idx = int(edge_ids[pos])
            next_dist = current_dist + float(weights[edge_idx])
            if next_dist < float(dist[neighbor]):
                dist[neighbor] = next_dist
                prev_node[neighbor] = int(node)
                prev_edge[neighbor] = int(edge_idx)
                heapq.heappush(heap, (next_dist, neighbor))
    return dist, prev_node, prev_edge


def reconstruct_path_and_edges(
    *,
    source: int,
    target: int,
    prev_node: np.ndarray,
    prev_edge: np.ndarray,
) -> tuple[list[int], list[int]]:
    source = int(source)
    target = int(target)
    if source == target:
        return [source], []
    path = [target]
    edges: list[int] = []
    current = target
    guard = 0
    while current != source:
        edge_idx = int(prev_edge[current])
        parent = int(prev_node[current])
        if edge_idx < 0 or parent < 0:
            return [], []
        edges.append(edge_idx)
        current = parent
        path.append(current)
        guard += 1
        if guard > int(prev_node.size):
            return [], []
    path.reverse()
    edges.reverse()
    return path, edges


def weighted_edge_betweenness_between_node_sets(
    edge_table: EdgeTable,
    *,
    total_nodes: int,
    source_nodes: Sequence[int],
    target_nodes: Sequence[int],
    weights: np.ndarray,
    adjacency: WeightedTopologyAdjacency | None = None,
    sample_path_limit: int = 0,
) -> tuple[np.ndarray, WeightedBetweennessSummary, list[dict[str, Any]]]:
    """Count one deterministic weighted shortest path for each source-target pair."""

    adjacency = adjacency if adjacency is not None else build_weighted_adjacency(edge_table, int(total_nodes))
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[0] != int(edge_table.num_edges):
        raise ValueError(f"weights length {weights.shape[0]} does not match edges {edge_table.num_edges}")

    sources = tuple(sorted(int(x) for x in source_nodes if 0 <= int(x) < int(total_nodes)))
    targets = tuple(sorted(int(x) for x in target_nodes if 0 <= int(x) < int(total_nodes)))
    edge_values = np.zeros(int(edge_table.num_edges), dtype=np.float64)
    delays: list[float] = []
    samples: list[dict[str, Any]] = []

    for source in sources:
        dist, prev_node, prev_edge = dijkstra_targets_with_prev_edge(
            adjacency=adjacency,
            weights=weights,
            source=int(source),
            targets=targets,
            total_nodes=int(total_nodes),
        )
        for target in targets:
            if int(target) == int(source):
                continue
            value = float(dist[int(target)])
            if not math.isfinite(value):
                continue
            path, edge_indices = reconstruct_path_and_edges(
                source=int(source),
                target=int(target),
                prev_node=prev_node,
                prev_edge=prev_edge,
            )
            if not edge_indices:
                continue
            delays.append(value)
            for edge_idx in edge_indices:
                edge_values[int(edge_idx)] += 1.0
            if len(samples) < int(sample_path_limit):
                samples.append(
                    {
                        "source": int(source),
                        "target": int(target),
                        "shortest_weight": float(value),
                        "path": path,
                        "edge_indices": edge_indices,
                    }
                )

    if delays:
        arr = np.asarray(delays, dtype=np.float64)
        summary = WeightedBetweennessSummary(
            source_nodes=len(sources),
            target_nodes=len(targets),
            reachable_pairs=int(arr.size),
            total_shortest_weight=float(np.sum(arr)),
            mean_shortest_weight=float(np.mean(arr)),
            min_shortest_weight=float(np.min(arr)),
            max_shortest_weight=float(np.max(arr)),
            max_edge_betweenness=float(np.max(edge_values)) if edge_values.size else 0.0,
            nonzero_edges=int(np.count_nonzero(edge_values > 0.0)),
            edge_value_sum=float(np.sum(edge_values)),
        )
    else:
        summary = WeightedBetweennessSummary(
            source_nodes=len(sources),
            target_nodes=len(targets),
            reachable_pairs=0,
            total_shortest_weight=0.0,
            mean_shortest_weight=float("nan"),
            min_shortest_weight=float("nan"),
            max_shortest_weight=float("nan"),
            max_edge_betweenness=0.0,
            nonzero_edges=0,
            edge_value_sum=0.0,
        )
    return edge_values.astype(np.float32), summary, samples


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _summary_row(*, step: int, pair: WeightedPairSpec, summary: WeightedBetweennessSummary) -> dict[str, Any]:
    return {
        "step": int(step),
        "pair_key": str(pair.key),
        "pair_label": pair.label or pair.key,
        "source_group_id": int(pair.source_group_id),
        "target_group_id": int(pair.target_group_id),
        "source_nodes": int(summary.source_nodes),
        "target_nodes": int(summary.target_nodes),
        "reachable_pairs": int(summary.reachable_pairs),
        "total_shortest_weight": _finite_or_none(summary.total_shortest_weight),
        "mean_shortest_weight": _finite_or_none(summary.mean_shortest_weight),
        "min_shortest_weight": _finite_or_none(summary.min_shortest_weight),
        "max_shortest_weight": _finite_or_none(summary.max_shortest_weight),
        "max_edge_betweenness": float(summary.max_edge_betweenness),
        "nonzero_edges": int(summary.nonzero_edges),
        "edge_value_sum": float(summary.edge_value_sum),
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_weighted_edge_betweenness_timeseries(
    *,
    topology_name: str,
    edge_table: EdgeTable,
    total_nodes: int,
    group_data: Mapping,
    steps: Sequence[int],
    weights_by_step: np.ndarray,
    pair_specs: Sequence[WeightedPairSpec],
    out_dir: str | Path,
    sample_steps: int = 0,
    sample_pairs_per_step: int = 0,
    progress_every: int = 100,
    force: bool = False,
) -> dict[str, Any]:
    """Compute weighted shortest-path edge usage for several region pairs over time."""

    if not pair_specs:
        raise ValueError("pair_specs is empty")
    steps = [int(step) for step in steps]
    weights_by_step = np.asarray(weights_by_step, dtype=np.float32)
    if weights_by_step.shape != (len(steps), int(edge_table.num_edges)):
        raise ValueError(
            f"weights_by_step shape {weights_by_step.shape} does not match "
            f"({len(steps)}, {edge_table.num_edges})"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(edge_table, out_dir / "edges.csv")
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))

    adjacency = build_weighted_adjacency(edge_table, int(total_nodes))
    combined_sum = open_memmap(
        out_dir / "combined_sum_edge_betweenness.npy",
        mode="w+" if bool(force) else "w+",
        dtype=np.float32,
        shape=(len(steps), int(edge_table.num_edges)),
    )
    combined_sum[:] = 0.0
    pair_mmaps: dict[str, np.memmap] = {}
    for pair in pair_specs:
        pair_dir = out_dir / str(pair.key)
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_mmaps[str(pair.key)] = open_memmap(
            pair_dir / "edge_betweenness.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(steps), int(edge_table.num_edges)),
        )

    summary_by_pair: dict[str, list[dict[str, Any]]] = {str(pair.key): [] for pair in pair_specs}
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step_idx, step in enumerate(steps):
        weights = np.asarray(weights_by_step[step_idx], dtype=np.float32)
        for pair in pair_specs:
            sources = group_nodes_for_step(group_data, int(step), int(pair.source_group_id))
            targets = group_nodes_for_step(group_data, int(step), int(pair.target_group_id))
            sample_limit = int(sample_pairs_per_step) if step_idx < int(sample_steps) else 0
            values, summary, step_samples = weighted_edge_betweenness_between_node_sets(
                edge_table,
                total_nodes=int(total_nodes),
                source_nodes=sources,
                target_nodes=targets,
                weights=weights,
                adjacency=adjacency,
                sample_path_limit=sample_limit,
            )
            pair_mmaps[str(pair.key)][step_idx, :] = values
            combined_sum[step_idx, :] += values
            summary_by_pair[str(pair.key)].append(_summary_row(step=int(step), pair=pair, summary=summary))
            for sample in step_samples:
                samples.append(
                    {
                        "step": int(step),
                        "pair_key": str(pair.key),
                        "pair_label": pair.label or pair.key,
                        "source": int(sample["source"]),
                        "target": int(sample["target"]),
                        "shortest_weight": float(sample["shortest_weight"]),
                        "path": "->".join(str(x) for x in sample["path"]),
                        "edge_indices": " ".join(str(x) for x in sample["edge_indices"]),
                    }
                )
        if int(progress_every) > 0 and (step_idx + 1 == len(steps) or (step_idx + 1) % int(progress_every) == 0):
            elapsed = time.perf_counter() - started
            print(
                f"[topology-metrics] weighted betweenness {step_idx + 1}/{len(steps)} "
                f"step={step} elapsed={elapsed:.1f}s",
                flush=True,
            )

    combined_sum.flush()
    combined_max = np.asarray(combined_sum, dtype=np.float32).max(axis=0)
    np.save(out_dir / "combined_max_over_time.npy", combined_max.astype(np.float32, copy=False))
    for pair in pair_specs:
        key = str(pair.key)
        pair_mmaps[key].flush()
        _write_rows(out_dir / key / "step_summary.csv", summary_by_pair[key])
    if samples:
        _write_rows(out_dir / "path_samples.csv", samples)

    meta = {
        "topology": str(topology_name),
        "metric": "weighted_shortest_path_edge_betweenness",
        "num_steps": len(steps),
        "num_edges": int(edge_table.num_edges),
        "total_nodes": int(total_nodes),
        "pairs": [
            {
                "key": str(pair.key),
                "label": pair.label or pair.key,
                "source_group_id": int(pair.source_group_id),
                "target_group_id": int(pair.target_group_id),
            }
            for pair in pair_specs
        ],
        "storage": {
            "per_pair": "<out_dir>/<pair_key>/edge_betweenness.npy",
            "combined_sum": "combined_sum_edge_betweenness.npy",
            "combined_max_over_time": "combined_max_over_time.npy",
        },
    }
    (out_dir / "weighted_edge_betweenness_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta
