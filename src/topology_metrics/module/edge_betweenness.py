from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.link_delay.module.edge_options import EdgeTable


@dataclass(frozen=True)
class EdgeBetweennessSummary:
    source_nodes: int
    target_nodes: int
    reachable_pairs: int
    total_shortest_distance_hops: float
    mean_shortest_distance_hops: float
    max_edge_betweenness: float
    nonzero_edges: int
    edge_value_sum: float


def build_undirected_adjacency(edge_table: EdgeTable, total_nodes: int) -> list[list[tuple[int, int]]]:
    """Build adjacency entries as `(neighbor_node, edge_idx)` pairs."""

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(int(total_nodes))]
    for edge_idx in range(int(edge_table.num_edges)):
        src = int(edge_table.src[edge_idx])
        dst = int(edge_table.dst[edge_idx])
        if not (0 <= src < int(total_nodes) and 0 <= dst < int(total_nodes)):
            raise ValueError(f"edge {edge_idx} endpoint outside total_nodes={total_nodes}: {src}, {dst}")
        adjacency[src].append((dst, edge_idx))
        adjacency[dst].append((src, edge_idx))

    for neighbors in adjacency:
        neighbors.sort(key=lambda item: (item[0], item[1]))
    return adjacency


def _bfs_shortest_path_dag(
    adjacency: list[list[tuple[int, int]]],
    source: int,
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int]]], list[int]]:
    total_nodes = len(adjacency)
    dist = np.full(total_nodes, -1, dtype=np.int32)
    sigma = np.zeros(total_nodes, dtype=np.float64)
    predecessors: list[list[tuple[int, int]]] = [[] for _ in range(total_nodes)]
    order: list[int] = []
    queue: deque[int] = deque([int(source)])
    dist[int(source)] = 0
    sigma[int(source)] = 1.0

    while queue:
        node = queue.popleft()
        order.append(node)
        next_dist = int(dist[node]) + 1
        for neighbor, edge_idx in adjacency[node]:
            if dist[neighbor] < 0:
                dist[neighbor] = next_dist
                queue.append(neighbor)
            if dist[neighbor] == next_dist:
                sigma[neighbor] += sigma[node]
                predecessors[neighbor].append((node, edge_idx))

    return dist, sigma, predecessors, order


def representative_shortest_path(
    predecessors: list[list[tuple[int, int]]],
    *,
    source: int,
    target: int,
) -> list[int]:
    """Return one deterministic shortest path from a predecessor DAG."""

    if int(source) == int(target):
        return [int(source)]
    path = [int(target)]
    current = int(target)
    seen = {current}
    while current != int(source):
        preds = predecessors[current]
        if not preds:
            return []
        current = int(preds[0][0])
        if current in seen:
            return []
        seen.add(current)
        path.append(current)
    path.reverse()
    return path


def edge_betweenness_between_node_sets(
    edge_table: EdgeTable,
    *,
    total_nodes: int,
    source_nodes: Iterable[int],
    target_nodes: Iterable[int],
    adjacency: list[list[tuple[int, int]]] | None = None,
    sample_path_limit: int = 0,
) -> tuple[np.ndarray, EdgeBetweennessSummary, list[dict]]:
    """Count edge usage over unweighted shortest paths between two node sets.

    Equal-length shortest paths are split fractionally, so each reachable
    source-target pair contributes one unit of demand distributed across all
    shortest paths.
    """

    adjacency = adjacency if adjacency is not None else build_undirected_adjacency(edge_table, int(total_nodes))
    sources = sorted(int(x) for x in source_nodes if 0 <= int(x) < int(total_nodes))
    targets = sorted(int(x) for x in target_nodes if 0 <= int(x) < int(total_nodes))
    edge_values = np.zeros(int(edge_table.num_edges), dtype=np.float64)
    reachable_pairs = 0
    total_distance = 0.0
    samples: list[dict] = []

    for source in sources:
        dist, sigma, predecessors, order = _bfs_shortest_path_dag(adjacency, int(source))
        delta = np.zeros(int(total_nodes), dtype=np.float64)

        for target in targets:
            if target == source or dist[target] < 0 or sigma[target] <= 0:
                continue
            delta[target] += 1.0
            reachable_pairs += 1
            total_distance += float(dist[target])

            if len(samples) < int(sample_path_limit):
                samples.append(
                    {
                        "source": int(source),
                        "target": int(target),
                        "distance_hops": int(dist[target]),
                        "num_shortest_paths": float(sigma[target]),
                        "representative_path": representative_shortest_path(
                            predecessors,
                            source=int(source),
                            target=int(target),
                        ),
                    }
                )

        for node in reversed(order):
            if delta[node] == 0.0 or sigma[node] <= 0.0:
                continue
            for pred, edge_idx in predecessors[node]:
                contribution = (sigma[pred] / sigma[node]) * delta[node]
                edge_values[int(edge_idx)] += contribution
                delta[pred] += contribution

    summary = EdgeBetweennessSummary(
        source_nodes=len(sources),
        target_nodes=len(targets),
        reachable_pairs=int(reachable_pairs),
        total_shortest_distance_hops=float(total_distance),
        mean_shortest_distance_hops=float(total_distance / reachable_pairs) if reachable_pairs else 0.0,
        max_edge_betweenness=float(np.max(edge_values)) if edge_values.size else 0.0,
        nonzero_edges=int(np.count_nonzero(edge_values > 0.0)),
        edge_value_sum=float(np.sum(edge_values)),
    )
    return edge_values.astype(np.float32), summary, samples

