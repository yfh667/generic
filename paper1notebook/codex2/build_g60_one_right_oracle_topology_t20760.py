from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data

from build_g60_motif_gridplus_shortest_delay_timeseries_parallel import (
    DEFAULT_XML,
    dijkstra_targets,
    edge_table_from_delay_store,
    group_nodes_for_step,
    load_steps_and_rows,
    make_worker_spec,
    reconstruct_from_prev,
)


DEFAULT_DELAY_STORE = PROJECT_ROOT / "data" / "linshi" / "G60_full_options_plus_intra_t0_86164_stride1"
DEFAULT_GROUP_CACHE = PROJECT_ROOT / "data" / "linshi" / "cache" / "group_data_cache"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_oracle_one_right_topology_t20760_china_europe"

OPTION_LABEL = {
    -1: "intra_y",
    0: "A/right_same",
    1: "B/right_down",
    2: "D/skip_plane",
    4: "C/right_up",
}
@dataclass(frozen=True)
class ChosenEdge:
    src: int
    dst: int
    option: int
    delay_ms: float
    full_path_use_count: int
    selection_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one 20760s oracle topology: every satellite keeps at most one right neighbor. "
            "For each source satellite, choose the candidate right edge most used by full_link "
            "China-Europe shortest paths at this same step; unused sources fall back to min-delay."
        )
    )
    parser.add_argument("--step", type=int, default=20760)
    parser.add_argument("--source-group", type=int, default=2)
    parser.add_argument("--target-group", type=int, default=3)
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def node_xy(node: int) -> tuple[int, int]:
    node = int(node)
    return node // int(G60_CONFIG.N), node % int(G60_CONFIG.N)


def normalized_pair(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def build_edge_pair_index(edge_table: EdgeTable) -> dict[tuple[int, int], int]:
    return {
        normalized_pair(int(src), int(dst)): idx
        for idx, (src, dst) in enumerate(zip(edge_table.src, edge_table.dst))
    }


def full_link_path_usage(
    *,
    step: int,
    delay_row: int,
    edge_table: EdgeTable,
    spec,
    delay_ms_array: np.ndarray,
    pair_to_edge_idx: dict[tuple[int, int], int],
    sources: tuple[int, ...],
    targets: tuple[int, ...],
) -> tuple[Counter[int], list[float]]:
    weights = np.asarray(delay_ms_array[int(delay_row), spec.store_edge_indices], dtype=np.float32)
    usage: Counter[int] = Counter()
    pair_delays: list[float] = []
    for source in sources:
        dist, prev = dijkstra_targets(
            spec=spec,
            weights=weights,
            source=int(source),
            targets=targets,
            want_prev=True,
        )
        for target in targets:
            value = float(dist[int(target)])
            if not math.isfinite(value):
                continue
            pair_delays.append(value)
            path = reconstruct_from_prev(prev, int(source), int(target))
            for u, v in zip(path, path[1:]):
                usage[pair_to_edge_idx[normalized_pair(u, v)]] += 1
    return usage, pair_delays


def edge_table_from_chosen(chosen: list[ChosenEdge], full_edge_table: EdgeTable) -> EdgeTable:
    src_values = [edge.src for edge in chosen]
    dst_values = [edge.dst for edge in chosen]
    option_values = [edge.option for edge in chosen]
    for idx in range(full_edge_table.num_edges):
        if int(full_edge_table.option[idx]) == -1:
            src_values.append(int(full_edge_table.src[idx]))
            dst_values.append(int(full_edge_table.dst[idx]))
            option_values.append(-1)

    src_arr = np.asarray(src_values, dtype=np.int32)
    dst_arr = np.asarray(dst_values, dtype=np.int32)
    return EdgeTable(
        src=src_arr,
        dst=dst_arr,
        option=np.asarray(option_values, dtype=np.int16),
        src_plane=(src_arr // int(G60_CONFIG.N)).astype(np.int16),
        src_y=(src_arr % int(G60_CONFIG.N)).astype(np.int16),
        dst_plane=(dst_arr // int(G60_CONFIG.N)).astype(np.int16),
        dst_y=(dst_arr % int(G60_CONFIG.N)).astype(np.int16),
        sat_ids=[str(i + 1) for i in range(int(G60_CONFIG.total_sats))],
    )


def compute_mean_delay_for_topology(
    *,
    edge_table: EdgeTable,
    delay_store,
    delay_row: int,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
) -> dict[str, float | int]:
    spec = make_worker_spec(edge_table, delay_store, "oracle_one_right")
    weights = np.asarray(delay_store.delay_ms_array[int(delay_row), spec.store_edge_indices], dtype=np.float32)
    values: list[float] = []
    for source in sources:
        dist, _ = dijkstra_targets(
            spec=spec,
            weights=weights,
            source=int(source),
            targets=targets,
            want_prev=False,
        )
        for target in targets:
            value = float(dist[int(target)])
            if math.isfinite(value):
                values.append(value)
    expected = int(len(sources) * len(targets))
    return {
        "reachable_pairs": int(len(values)),
        "expected_pairs": int(expected),
        "mean_delay_ms": float(np.mean(values)) if values else math.nan,
        "min_delay_ms": float(np.min(values)) if values else math.nan,
        "max_delay_ms": float(np.max(values)) if values else math.nan,
    }


def write_chosen_edges_csv(chosen: list[ChosenEdge], path: Path) -> None:
    fields = [
        "src",
        "src_plane",
        "src_y",
        "dst",
        "dst_plane",
        "dst_y",
        "option",
        "option_label",
        "delay_ms",
        "full_path_use_count",
        "selection_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for edge in chosen:
            sp, sy = node_xy(edge.src)
            dp, dy = node_xy(edge.dst)
            writer.writerow(
                {
                    "src": int(edge.src),
                    "src_plane": sp,
                    "src_y": sy,
                    "dst": int(edge.dst),
                    "dst_plane": dp,
                    "dst_y": dy,
                    "option": int(edge.option),
                    "option_label": OPTION_LABEL.get(int(edge.option), str(edge.option)),
                    "delay_ms": float(edge.delay_ms),
                    "full_path_use_count": int(edge.full_path_use_count),
                    "selection_reason": str(edge.selection_reason),
                }
            )


@dataclass
class _FlowEdge:
    to: int
    rev: int
    cap: int
    cost: int
    payload: int | None = None


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


def _min_cost_flow_selected_payloads(
    *,
    source_count: int,
    target_count: int,
    candidate_edges: list[tuple[int, int, int, int]],
) -> list[int]:
    """Return payload ids for a minimum-cost full matching.

    candidate_edges entries are (source_row, target_col, cost, payload).
    """

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
    full_edge_table: EdgeTable,
    weights: np.ndarray,
    usage: Counter[int],
) -> list[ChosenEdge]:
    candidate_edge_indices = [
        edge_idx for edge_idx in range(int(full_edge_table.num_edges)) if int(full_edge_table.option[edge_idx]) != -1
    ]
    source_nodes = sorted({int(full_edge_table.src[edge_idx]) for edge_idx in candidate_edge_indices})
    target_nodes = sorted({int(full_edge_table.dst[edge_idx]) for edge_idx in candidate_edge_indices})
    if len(source_nodes) != len(target_nodes):
        raise RuntimeError(f"source node count {len(source_nodes)} != target node count {len(target_nodes)}")

    source_to_row = {node: row for row, node in enumerate(source_nodes)}
    target_to_col = {node: col for col, node in enumerate(target_nodes)}
    max_use = max((int(usage.get(edge_idx, 0)) for edge_idx in candidate_edge_indices), default=0)
    use_priority = 1_000_000
    delay_scale = 1000

    flow_candidates: list[tuple[int, int, int, int]] = []
    for edge_idx in candidate_edge_indices:
        use_count = int(usage.get(edge_idx, 0))
        delay_cost = int(round(float(weights[edge_idx]) * delay_scale))
        cost = int(max_use - use_count) * use_priority + delay_cost
        flow_candidates.append(
            (
                source_to_row[int(full_edge_table.src[edge_idx])],
                target_to_col[int(full_edge_table.dst[edge_idx])],
                cost,
                int(edge_idx),
            )
        )

    selected_indices = _min_cost_flow_selected_payloads(
        source_count=len(source_nodes),
        target_count=len(target_nodes),
        candidate_edges=flow_candidates,
    )
    chosen: list[ChosenEdge] = []
    for selected_idx in sorted(selected_indices, key=lambda idx: int(full_edge_table.src[idx])):
        use_count = int(usage.get(selected_idx, 0))
        chosen.append(
            ChosenEdge(
                src=int(full_edge_table.src[selected_idx]),
                dst=int(full_edge_table.dst[selected_idx]),
                option=int(full_edge_table.option[selected_idx]),
                delay_ms=float(weights[selected_idx]),
                full_path_use_count=use_count,
                selection_reason="max_weight_one_right_one_left_matching" if use_count > 0 else "matching_fallback_min_delay",
            )
        )
    return chosen


def verify_one_right_one_left(chosen: list[ChosenEdge]) -> dict[str, int]:
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


def write_viewer_screenshot(
    *,
    edge_table: EdgeTable,
    chosen: list[ChosenEdge],
    group_data: dict,
    step: int,
    screenshot_path: Path,
    title: str,
) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    edge_values = np.zeros((1, int(edge_table.num_edges)), dtype=np.float32)
    for idx, edge in enumerate(chosen):
        edge_values[0, idx] = float(edge.full_path_use_count)

    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=[int(step)],
        edge_table=edge_table,
        edge_values=edge_values,
        value_min=0.0,
        value_max=float(np.nanmax(edge_values)) if edge_values.size else 0.0,
        edge_value_label="full_link_path_use_count",
        scale_edge_width_by_value=True,
        value_width_min=0.008,
        value_width_max=0.08,
        value_color_mode="red_alpha",
        value_solid_color="#C1121F",
        value_alpha_min=35,
        value_alpha_max=235,
        zero_value_edges_visible=False,
        zero_value_threshold=0.0,
        show_topology_under_edge_values=True,
        topology_edge_color="#000000",
        topology_edge_alpha=150,
        topology_edge_width=0.014,
        hide_y_wrap_edges=True,
        window_title=title,
        group_data=group_data,
        show_groups=True,
    )
    run_viewer_widget(
        viewer,
        width=1400,
        height=900,
        check_only=False,
        offscreen=True,
        screenshot=screenshot_path,
    )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delay_store, steps, delay_rows = load_steps_and_rows(
        Path(args.delay_store_dir),
        int(args.step),
        int(args.step),
        stride=1,
    )
    step = int(steps[0])
    delay_row = int(delay_rows[0])

    group_data = load_or_build_group_data(
        xml_file=DEFAULT_XML,
        group_cache_dir=Path(args.group_cache_dir),
        steps=[step],
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=1,
        enabled=True,
        force=False,
    )
    sources = group_nodes_for_step(group_data, step, int(args.source_group))
    targets = group_nodes_for_step(group_data, step, int(args.target_group))

    full_edge_table = edge_table_from_delay_store(delay_store)
    full_spec = make_worker_spec(full_edge_table, delay_store, "full_link")
    pair_to_edge_idx = build_edge_pair_index(full_edge_table)
    usage, full_pair_delays = full_link_path_usage(
        step=step,
        delay_row=delay_row,
        edge_table=full_edge_table,
        spec=full_spec,
        delay_ms_array=delay_store.delay_ms_array,
        pair_to_edge_idx=pair_to_edge_idx,
        sources=sources,
        targets=targets,
    )

    weights = np.asarray(delay_store.delay_ms_array[delay_row, full_spec.store_edge_indices], dtype=np.float32)
    chosen = choose_one_right_one_left_matching(
        full_edge_table=full_edge_table,
        weights=weights,
        usage=usage,
    )
    constraint_check = verify_one_right_one_left(chosen)

    oracle_edge_table = edge_table_from_chosen(chosen, full_edge_table)
    oracle_metric = compute_mean_delay_for_topology(
        edge_table=oracle_edge_table,
        delay_store=delay_store,
        delay_row=delay_row,
        sources=sources,
        targets=targets,
    )

    full_total_uses = int(sum(usage.values()))
    full_inter_uses = int(sum(count for edge_idx, count in usage.items() if int(full_edge_table.option[edge_idx]) != -1))
    full_intra_uses = int(sum(count for edge_idx, count in usage.items() if int(full_edge_table.option[edge_idx]) == -1))
    selected_inter_uses = int(
        sum(edge.full_path_use_count for edge in chosen if int(edge.option) != -1)
    )
    selected_edges_used = int(sum(1 for edge in chosen if edge.full_path_use_count > 0))
    selected_edges_fallback = int(sum(1 for edge in chosen if edge.full_path_use_count <= 0))

    write_chosen_edges_csv(chosen, out_dir / "oracle_one_right_inter_edges.csv")
    write_edges_csv(oracle_edge_table, out_dir / "oracle_one_right_with_intra_edges.csv")
    screenshot_path = out_dir / "oracle_one_right_one_left_topology_t20760_viewer.png"
    write_viewer_screenshot(
        edge_table=oracle_edge_table,
        chosen=chosen,
        group_data=group_data,
        step=step,
        screenshot_path=screenshot_path,
        title=(
            "G60 China-Europe one-right/one-left oracle topology at 20760s "
            f"(inter coverage={selected_inter_uses / full_inter_uses:.3f})"
        ),
    )

    option_counts = Counter(edge.option for edge in chosen)
    used_option_counts = Counter(edge.option for edge in chosen if edge.full_path_use_count > 0)
    summary = {
        "step": int(step),
        "source_group": int(args.source_group),
        "target_group": int(args.target_group),
        "source_nodes": int(len(sources)),
        "target_nodes": int(len(targets)),
        "full_link_mean_delay_ms": float(np.mean(full_pair_delays)),
        "full_link_min_delay_ms": float(np.min(full_pair_delays)),
        "full_link_max_delay_ms": float(np.max(full_pair_delays)),
        "oracle_one_right_metric": oracle_metric,
        "coverage_definition": (
            "Build a maximum-weight bipartite matching over all candidate right-neighbor "
            "inter edges. Each source satellite can select at most one right neighbor and "
            "each destination satellite can receive at most one left-neighbor link. Edge "
            "weight is the use count among full_link China-Europe shortest paths at this "
            "step; delay is used only as a tie-breaker/fallback among equal use counts. "
            "The reported coverage is selected full-link path inter-edge uses divided by "
            "all full-link path inter-edge uses."
        ),
        "constraint_check": constraint_check,
        "full_link_path_edge_uses": full_total_uses,
        "full_link_path_inter_edge_uses": full_inter_uses,
        "full_link_path_intra_edge_uses": full_intra_uses,
        "selected_inter_edge_uses": selected_inter_uses,
        "selected_inter_coverage": float(selected_inter_uses / full_inter_uses) if full_inter_uses else None,
        "selected_total_coverage_with_all_intra": float((selected_inter_uses + full_intra_uses) / full_total_uses)
        if full_total_uses
        else None,
        "chosen_inter_edges": int(len(chosen)),
        "chosen_edges_used_by_full_link_paths": selected_edges_used,
        "chosen_edges_fallback_min_delay": selected_edges_fallback,
        "chosen_option_counts": {str(k): int(v) for k, v in sorted(option_counts.items())},
        "used_chosen_option_counts": {str(k): int(v) for k, v in sorted(used_option_counts.items())},
        "outputs": {
            "inter_edges_csv": str(out_dir / "oracle_one_right_inter_edges.csv"),
            "full_edges_csv": str(out_dir / "oracle_one_right_with_intra_edges.csv"),
            "viewer_screenshot": str(screenshot_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
