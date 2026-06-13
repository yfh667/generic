from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.satellite_topology_viewer.module.topology_edges import build_undirected_adjacency


@dataclass(frozen=True)
class StepBetweennessSummary:
    step: int
    source_group_id: int
    target_group_id: int
    source_nodes: int
    target_nodes: int
    reachable_pairs: int
    total_shortest_distance_hops: float
    mean_shortest_distance_hops: float
    max_edge_betweenness: float
    nonzero_edges: int


def _bfs_shortest_path_dag(
    adjacency: list[list[tuple[int, int]]],
    source: int,
) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, int]]], list[int]]:
    total_nodes = len(adjacency)
    dist = np.full(total_nodes, -1, dtype=np.int32)
    sigma = np.zeros(total_nodes, dtype=np.float64)
    predecessors: list[list[tuple[int, int]]] = [[] for _ in range(total_nodes)]
    order: list[int] = []

    dist[int(source)] = 0
    sigma[int(source)] = 1.0
    queue: deque[int] = deque([int(source)])

    while queue:
        v = queue.popleft()
        order.append(v)
        next_dist = int(dist[v]) + 1
        for w, edge_idx in adjacency[v]:
            if dist[w] < 0:
                dist[w] = next_dist
                queue.append(w)
            if dist[w] == next_dist:
                sigma[w] += sigma[v]
                predecessors[w].append((v, edge_idx))

    return dist, sigma, predecessors, order


def representative_shortest_path(
    predecessors: list[list[tuple[int, int]]],
    *,
    source: int,
    target: int,
) -> list[int]:
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
    source_nodes: set[int],
    target_nodes: set[int],
    adjacency: list[list[tuple[int, int]]] | None = None,
    sample_path_limit: int = 60,
) -> tuple[np.ndarray, dict, list[dict]]:
    """Count edge usage over shortest paths from source_nodes to target_nodes.

    If a source-target pair has multiple equal-length shortest paths, the pair's
    contribution is split fractionally across those paths. Thus each reachable pair
    contributes exactly its shortest-path hop length to the total edge-count sum.
    """

    adjacency = adjacency if adjacency is not None else build_undirected_adjacency(edge_table, total_nodes)
    edge_values = np.zeros(int(edge_table.num_edges), dtype=np.float64)
    sources = sorted(int(x) for x in source_nodes if 0 <= int(x) < int(total_nodes))
    targets_all = sorted(int(x) for x in target_nodes if 0 <= int(x) < int(total_nodes))

    reachable_pairs = 0
    total_distance = 0.0
    path_samples: list[dict] = []

    for source in sources:
        dist, sigma, predecessors, order = _bfs_shortest_path_dag(adjacency, source)
        delta = np.zeros(int(total_nodes), dtype=np.float64)

        for target in targets_all:
            if target == source:
                continue
            if dist[target] < 0 or sigma[target] <= 0:
                continue
            delta[target] += 1.0
            reachable_pairs += 1
            total_distance += float(dist[target])

            if len(path_samples) < int(sample_path_limit):
                path = representative_shortest_path(predecessors, source=source, target=target)
                path_samples.append(
                    {
                        "source": int(source),
                        "target": int(target),
                        "distance_hops": int(dist[target]),
                        "num_shortest_paths": float(sigma[target]),
                        "representative_path": path,
                    }
                )

        for w in reversed(order):
            if delta[w] == 0.0 or sigma[w] <= 0.0:
                continue
            for v, edge_idx in predecessors[w]:
                contribution = (sigma[v] / sigma[w]) * delta[w]
                edge_values[int(edge_idx)] += contribution
                delta[v] += contribution

    summary = {
        "source_nodes": len(sources),
        "target_nodes": len(targets_all),
        "reachable_pairs": int(reachable_pairs),
        "total_shortest_distance_hops": float(total_distance),
        "mean_shortest_distance_hops": float(total_distance / reachable_pairs) if reachable_pairs else 0.0,
        "max_edge_betweenness": float(np.max(edge_values)) if edge_values.size else 0.0,
        "nonzero_edges": int(np.count_nonzero(edge_values > 0.0)),
        "edge_value_sum": float(np.sum(edge_values)),
    }
    return edge_values.astype(np.float32), summary, path_samples


def _group_nodes_for_step(group_data: dict, step: int, group_id: int) -> set[int]:
    current = group_data.get(int(step), {}) if group_data else {}
    groups = current.get("groups", {}) if isinstance(current, dict) else {}
    return set(int(x) for x in groups.get(int(group_id), set()) or set())


def compute_group_pair_edge_betweenness_by_step(
    *,
    edge_table: EdgeTable,
    total_nodes: int,
    steps: list[int],
    group_data: dict,
    source_group_id: int,
    target_group_id: int,
    sample_path_limit_per_step: int = 60,
) -> tuple[np.ndarray, list[StepBetweennessSummary], list[dict]]:
    adjacency = build_undirected_adjacency(edge_table, total_nodes)
    values = np.zeros((len(steps), int(edge_table.num_edges)), dtype=np.float32)
    summaries: list[StepBetweennessSummary] = []
    path_samples: list[dict] = []

    for row, step in enumerate(int(x) for x in steps):
        source_nodes = _group_nodes_for_step(group_data, step, int(source_group_id))
        target_nodes = _group_nodes_for_step(group_data, step, int(target_group_id))
        row_values, summary, samples = edge_betweenness_between_node_sets(
            edge_table,
            total_nodes=int(total_nodes),
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            adjacency=adjacency,
            sample_path_limit=int(sample_path_limit_per_step),
        )
        values[row, :] = row_values
        summaries.append(
            StepBetweennessSummary(
                step=int(step),
                source_group_id=int(source_group_id),
                target_group_id=int(target_group_id),
                source_nodes=int(summary["source_nodes"]),
                target_nodes=int(summary["target_nodes"]),
                reachable_pairs=int(summary["reachable_pairs"]),
                total_shortest_distance_hops=float(summary["total_shortest_distance_hops"]),
                mean_shortest_distance_hops=float(summary["mean_shortest_distance_hops"]),
                max_edge_betweenness=float(summary["max_edge_betweenness"]),
                nonzero_edges=int(summary["nonzero_edges"]),
            )
        )
        for item in samples:
            item = dict(item)
            item["step"] = int(step)
            path_samples.append(item)

    return values, summaries, path_samples


def write_group_betweenness_outputs(
    *,
    out_dir: str | Path,
    edge_table: EdgeTable,
    steps: list[int],
    values: np.ndarray,
    summaries: list[StepBetweennessSummary],
    path_samples: list[dict],
    source_group_name: str,
    target_group_name: str,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "edge_betweenness.npy", np.asarray(values, dtype=np.float32))
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))
    write_edges_csv(edge_table, out_dir / "edges.csv")

    with (out_dir / "step_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(StepBetweennessSummary.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow(item.__dict__)

    with (out_dir / "path_samples.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["step", "source", "target", "distance_hops", "num_shortest_paths", "representative_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in path_samples:
            row = dict(item)
            row["representative_path"] = " ".join(str(x) for x in row.get("representative_path", []))
            writer.writerow(row)

    meta = {
        "source_group_name": source_group_name,
        "target_group_name": target_group_name,
        "steps": [int(x) for x in steps],
        "num_edges": int(edge_table.num_edges),
        "value_unit": "fractional_shortest_path_count",
        "value_min": float(np.nanmin(values)) if values.size else 0.0,
        "value_max": float(np.nanmax(values)) if values.size else 0.0,
        "counting_rule": "Each reachable source-target pair contributes 1 split fractionally across all equal-length shortest paths.",
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
