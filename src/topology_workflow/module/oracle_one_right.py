from __future__ import annotations

import csv
import heapq
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.link_delay.module.edge_options import EdgeTable

from .edge_tables import INTRA_OPTION, make_edge_table_from_records


@dataclass(frozen=True)
class ChosenRightEdge:
    src: int
    dst: int
    option: int
    weight: float
    usage_count: int
    selection_reason: str


@dataclass
class _FlowEdge:
    to: int
    rev: int
    cap: int
    cost: int
    payload: int | None = None


def normalized_pair(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def build_edge_pair_index(edge_table: EdgeTable) -> dict[tuple[int, int], int]:
    return {
        normalized_pair(int(src), int(dst)): int(idx)
        for idx, (src, dst) in enumerate(zip(edge_table.src, edge_table.dst))
    }


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    u: int,
    v: int,
    cap: int,
    cost: int,
    payload: int | None = None,
) -> None:
    graph[u].append(_FlowEdge(to=v, rev=len(graph[v]), cap=int(cap), cost=int(cost), payload=payload))
    graph[v].append(_FlowEdge(to=u, rev=len(graph[u]) - 1, cap=0, cost=-int(cost), payload=None))


def min_cost_matching_payloads(
    *,
    source_count: int,
    target_count: int,
    candidate_edges: Sequence[tuple[int, int, int, int]],
) -> list[int]:
    """Solve a unit-capacity min-cost matching and return selected payload ids."""

    node_count = 2 + int(source_count) + int(target_count)
    source_node = 0
    left_base = 1
    right_base = left_base + int(source_count)
    sink_node = node_count - 1
    graph: list[list[_FlowEdge]] = [[] for _ in range(node_count)]

    for row in range(int(source_count)):
        _add_flow_edge(graph, source_node, left_base + row, 1, 0)
    for col in range(int(target_count)):
        _add_flow_edge(graph, right_base + col, sink_node, 1, 0)
    for row, col, cost, payload in candidate_edges:
        _add_flow_edge(graph, left_base + int(row), right_base + int(col), 1, int(cost), int(payload))

    flow = 0
    potentials = [0] * node_count
    inf = 10**30
    while flow < int(source_count):
        dist = [inf] * node_count
        prev: list[tuple[int, int] | None] = [None] * node_count
        dist[source_node] = 0
        heap: list[tuple[int, int]] = [(0, source_node)]
        while heap:
            current_dist, u = heapq.heappop(heap)
            if current_dist != dist[u]:
                continue
            for edge_pos, edge in enumerate(graph[u]):
                if edge.cap <= 0:
                    continue
                nd = current_dist + edge.cost + potentials[u] - potentials[edge.to]
                if nd < dist[edge.to]:
                    dist[edge.to] = nd
                    prev[edge.to] = (u, edge_pos)
                    heapq.heappush(heap, (nd, edge.to))
        if prev[sink_node] is None:
            raise RuntimeError("Cannot build a full one-right/one-left matching from candidate edges.")

        for node, value in enumerate(dist):
            if value < inf:
                potentials[node] += int(value)

        v = sink_node
        while v != source_node:
            item = prev[v]
            if item is None:
                raise RuntimeError("Broken residual path while solving matching.")
            u, edge_pos = item
            edge = graph[u][edge_pos]
            edge.cap -= 1
            graph[v][edge.rev].cap += 1
            v = u
        flow += 1

    selected_payloads: list[int] = []
    for row in range(int(source_count)):
        node = left_base + row
        for edge in graph[node]:
            if edge.payload is not None and edge.cap == 0:
                selected_payloads.append(int(edge.payload))
                break
    if len(selected_payloads) != int(source_count):
        raise RuntimeError(f"Matching selected {len(selected_payloads)} edges, expected {source_count}.")
    return selected_payloads


def choose_one_right_one_left_matching(
    *,
    edge_table: EdgeTable,
    weights: np.ndarray,
    usage: Mapping[int, int] | np.ndarray,
    intra_option: int = INTRA_OPTION,
    use_priority: int = 1_000_000,
    weight_scale: int = 1000,
) -> list[ChosenRightEdge]:
    """Choose inter edges with max usage and one-right/one-left constraints.

    Usage is primary.  Edge weight is used as a deterministic tie breaker, so
    among equal full-link path usage counts the lower-weight edge is preferred.
    """

    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[0] != int(edge_table.num_edges):
        raise ValueError(f"weights length {weights.shape[0]} does not match edges {edge_table.num_edges}")
    usage_counter = Counter()
    if isinstance(usage, np.ndarray):
        for idx, value in enumerate(np.asarray(usage).ravel()):
            if float(value) > 0:
                usage_counter[int(idx)] = int(round(float(value)))
    else:
        usage_counter.update({int(k): int(v) for k, v in usage.items()})

    candidate_edge_indices = [
        edge_idx for edge_idx in range(int(edge_table.num_edges)) if int(edge_table.option[edge_idx]) != int(intra_option)
    ]
    source_nodes = sorted({int(edge_table.src[edge_idx]) for edge_idx in candidate_edge_indices})
    target_nodes = sorted({int(edge_table.dst[edge_idx]) for edge_idx in candidate_edge_indices})
    if len(source_nodes) != len(target_nodes):
        raise RuntimeError(f"source node count {len(source_nodes)} != target node count {len(target_nodes)}")

    source_to_row = {node: row for row, node in enumerate(source_nodes)}
    target_to_col = {node: col for col, node in enumerate(target_nodes)}
    max_use = max((int(usage_counter.get(edge_idx, 0)) for edge_idx in candidate_edge_indices), default=0)

    flow_candidates: list[tuple[int, int, int, int]] = []
    for edge_idx in candidate_edge_indices:
        use_count = int(usage_counter.get(edge_idx, 0))
        weight_cost = int(round(float(weights[edge_idx]) * int(weight_scale)))
        cost = int(max_use - use_count) * int(use_priority) + weight_cost
        flow_candidates.append(
            (
                source_to_row[int(edge_table.src[edge_idx])],
                target_to_col[int(edge_table.dst[edge_idx])],
                int(cost),
                int(edge_idx),
            )
        )

    selected_indices = min_cost_matching_payloads(
        source_count=len(source_nodes),
        target_count=len(target_nodes),
        candidate_edges=flow_candidates,
    )
    chosen: list[ChosenRightEdge] = []
    for selected_idx in sorted(selected_indices, key=lambda idx: int(edge_table.src[idx])):
        use_count = int(usage_counter.get(selected_idx, 0))
        chosen.append(
            ChosenRightEdge(
                src=int(edge_table.src[selected_idx]),
                dst=int(edge_table.dst[selected_idx]),
                option=int(edge_table.option[selected_idx]),
                weight=float(weights[selected_idx]),
                usage_count=use_count,
                selection_reason="max_usage_one_right_one_left_matching" if use_count > 0 else "matching_fallback_min_weight",
            )
        )
    return chosen


def verify_one_right_one_left(chosen: Sequence[ChosenRightEdge]) -> dict[str, int]:
    out_counts = Counter(int(edge.src) for edge in chosen)
    in_counts = Counter(int(edge.dst) for edge in chosen)
    max_out = max(out_counts.values(), default=0)
    max_in = max(in_counts.values(), default=0)
    if max_out > 1 or max_in > 1:
        raise RuntimeError(f"Invalid topology: max_right_out_degree={max_out}, max_left_in_degree={max_in}")
    return {
        "max_right_out_degree": int(max_out),
        "max_left_in_degree": int(max_in),
        "right_source_nodes": int(len(out_counts)),
        "left_target_nodes": int(len(in_counts)),
    }


def edge_table_from_chosen_right_edges(
    *,
    chosen: Sequence[ChosenRightEdge],
    base_edge_table: EdgeTable,
    p: int,
    n: int,
    include_intra: bool = True,
    intra_option: int = INTRA_OPTION,
) -> EdgeTable:
    records: list[tuple[int, int, int, int, int]] = []
    for edge in chosen:
        records.append(
            (
                int(edge.src) // int(n),
                int(edge.src) % int(n),
                int(edge.dst) // int(n),
                int(edge.dst) % int(n),
                int(edge.option),
            )
        )
    if bool(include_intra):
        for idx in range(int(base_edge_table.num_edges)):
            if int(base_edge_table.option[idx]) != int(intra_option):
                continue
            src = int(base_edge_table.src[idx])
            dst = int(base_edge_table.dst[idx])
            records.append((src // int(n), src % int(n), dst // int(n), dst % int(n), int(intra_option)))
    return make_edge_table_from_records(records=records, p=int(p), n=int(n))


def write_chosen_right_edges_csv(chosen: Sequence[ChosenRightEdge], path: str | Path, *, n: int) -> None:
    fields = [
        "src_node",
        "src_plane",
        "src_y",
        "dst_node",
        "dst_plane",
        "dst_y",
        "option",
        "weight",
        "usage_count",
        "selection_reason",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for edge in chosen:
            writer.writerow(
                {
                    "src_node": int(edge.src),
                    "src_plane": int(edge.src) // int(n),
                    "src_y": int(edge.src) % int(n),
                    "dst_node": int(edge.dst),
                    "dst_plane": int(edge.dst) // int(n),
                    "dst_y": int(edge.dst) % int(n),
                    "option": int(edge.option),
                    "weight": float(edge.weight),
                    "usage_count": int(edge.usage_count),
                    "selection_reason": str(edge.selection_reason),
                }
            )
