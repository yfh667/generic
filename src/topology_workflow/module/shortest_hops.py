from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.topology_metrics.module.group_states import build_group_state_index, group_nodes_for_step
from src.topology_metrics.module.stores import write_state_definitions

from .config import group_name


@dataclass(frozen=True)
class HopSummary:
    state_id: int
    source_nodes: int
    target_nodes: int
    reachable_pairs: int
    total_shortest_hops: float
    mean_shortest_hops: float
    min_shortest_hops: float
    max_shortest_hops: float


def node_xy(node: int, config: ViewerConfig) -> tuple[int, int]:
    node = int(node)
    return node // int(config.N), node % int(config.N)


def path_xy_text(path: list[int], config: ViewerConfig) -> str:
    return "->".join(f"({p},{y})" for p, y in (node_xy(node, config) for node in path))


def build_adjacency(edge_table: EdgeTable, total_nodes: int) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(int(total_nodes))]
    for idx in range(edge_table.num_edges):
        src = int(edge_table.src[idx])
        dst = int(edge_table.dst[idx])
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    for neighbors in adjacency:
        neighbors.sort()
    return adjacency


def precompute_hop_and_next(edge_table: EdgeTable, total_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    adjacency = build_adjacency(edge_table, total_nodes)
    n = int(total_nodes)
    dist = np.full((n, n), -1, dtype=np.int16)
    next_hop = np.full((n, n), -1, dtype=np.int32)

    for source in range(n):
        dist[source, source] = 0
        next_hop[source, source] = source
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            next_dist = int(dist[source, node]) + 1
            for neighbor in adjacency[node]:
                if dist[source, neighbor] >= 0:
                    continue
                dist[source, neighbor] = next_dist
                next_hop[source, neighbor] = neighbor if node == source else int(next_hop[source, node])
                queue.append(neighbor)
    return dist, next_hop


def reconstruct_path(next_hop: np.ndarray, source: int, target: int) -> list[int]:
    source = int(source)
    target = int(target)
    if int(next_hop[source, target]) < 0:
        return []
    path = [source]
    current = source
    guard = 0
    while current != target:
        current = int(next_hop[current, target])
        if current < 0:
            return []
        path.append(current)
        guard += 1
        if guard > next_hop.shape[0]:
            return []
    return path


def summarize_state(dist: np.ndarray, source_nodes: tuple[int, ...], target_nodes: tuple[int, ...], state_id: int) -> HopSummary:
    source = np.asarray(source_nodes, dtype=np.int32)
    target = np.asarray(target_nodes, dtype=np.int32)
    if source.size == 0 or target.size == 0:
        return HopSummary(int(state_id), int(source.size), int(target.size), 0, 0.0, float("nan"), float("nan"), float("nan"))

    values = dist[np.ix_(source, target)]
    reachable = values >= 0
    if not bool(np.any(reachable)):
        return HopSummary(int(state_id), int(source.size), int(target.size), 0, 0.0, float("nan"), float("nan"), float("nan"))

    reachable_values = values[reachable].astype(np.float64)
    total = float(np.sum(reachable_values))
    count = int(reachable_values.size)
    return HopSummary(
        state_id=int(state_id),
        source_nodes=int(source.size),
        target_nodes=int(target.size),
        reachable_pairs=count,
        total_shortest_hops=total,
        mean_shortest_hops=float(total / count),
        min_shortest_hops=float(np.min(reachable_values)),
        max_shortest_hops=float(np.max(reachable_values)),
    )


def compute_shortest_hops_timeseries(
    *,
    topology_name: str,
    edge_table: EdgeTable,
    config: ViewerConfig,
    group_data: dict,
    steps: list[int],
    source_group_id: int,
    target_group_id: int,
    out_dir: str | Path,
    sample_steps: int = 3,
    sample_pairs_per_step: int = 60,
) -> np.ndarray:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(edge_table, out_dir / "edges.csv")
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))

    print(f"[topology-workflow] shortest_hops topology={topology_name} edges={edge_table.num_edges}", flush=True)
    dist, next_hop = precompute_hop_and_next(edge_table, int(config.total_sats))
    state_index = build_group_state_index(
        group_data=group_data,
        steps=steps,
        source_group_id=int(source_group_id),
        target_group_id=int(target_group_id),
    )
    np.save(out_dir / "state_ids.npy", state_index.state_ids)
    write_state_definitions(out_dir / "state_definitions.json", state_index.unique_states)

    summaries = [
        summarize_state(dist, source_nodes, target_nodes, state_id)
        for state_id, (source_nodes, target_nodes) in enumerate(state_index.unique_states)
    ]
    by_state = {int(item.state_id): item for item in summaries}

    with (out_dir / "state_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(HopSummary.__dataclass_fields__.keys()))
        writer.writeheader()
        for item in summaries:
            writer.writerow(item.__dict__)

    mean_values = np.empty(len(steps), dtype=np.float32)
    with (out_dir / "step_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "step",
            "source_nodes",
            "target_nodes",
            "reachable_pairs",
            "total_shortest_hops",
            "mean_shortest_hops",
            "min_shortest_hops",
            "max_shortest_hops",
            "state_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, step in enumerate(steps):
            item = by_state[int(state_index.state_ids[row])]
            mean_values[row] = item.mean_shortest_hops
            writer.writerow(
                {
                    "step": int(step),
                    "source_nodes": int(item.source_nodes),
                    "target_nodes": int(item.target_nodes),
                    "reachable_pairs": int(item.reachable_pairs),
                    "total_shortest_hops": float(item.total_shortest_hops),
                    "mean_shortest_hops": float(item.mean_shortest_hops),
                    "min_shortest_hops": float(item.min_shortest_hops),
                    "max_shortest_hops": float(item.max_shortest_hops),
                    "state_id": int(item.state_id),
                }
            )

    with (out_dir / "path_samples.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["step", "source", "source_xy", "target", "target_xy", "hops", "path", "path_xy", "path_indexed"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        written_steps = 0
        for step in steps:
            if written_steps >= int(sample_steps):
                break
            sources = group_nodes_for_step(group_data, int(step), int(source_group_id))
            targets = group_nodes_for_step(group_data, int(step), int(target_group_id))
            written_pairs = 0
            for source in sources:
                for target in targets:
                    hops = int(dist[int(source), int(target)])
                    if hops < 0:
                        continue
                    path = reconstruct_path(next_hop, int(source), int(target))
                    writer.writerow(
                        {
                            "step": int(step),
                            "source": int(source),
                            "source_xy": str(node_xy(source, config)),
                            "target": int(target),
                            "target_xy": str(node_xy(target, config)),
                            "hops": int(hops),
                            "path": "->".join(str(x) for x in path),
                            "path_xy": path_xy_text(path, config),
                            "path_indexed": ",".join(f"{idx + 1}:{node}" for idx, node in enumerate(path)),
                        }
                    )
                    written_pairs += 1
                    if written_pairs >= int(sample_pairs_per_step):
                        break
                if written_pairs >= int(sample_pairs_per_step):
                    break
            written_steps += 1

    np.save(out_dir / "mean_shortest_hops.npy", mean_values)
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
        "num_unique_group_states": int(state_index.num_states),
        "num_edges": int(edge_table.num_edges),
        "metric": "shortest_hops",
        "mean_hops_min": float(np.nanmin(mean_values)) if mean_values.size else None,
        "mean_hops_max": float(np.nanmax(mean_values)) if mean_values.size else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[topology-workflow] {topology_name}: mean_hops=({meta['mean_hops_min']}, "
        f"{meta['mean_hops_max']}) out_dir={out_dir}",
        flush=True,
    )
    return mean_values
