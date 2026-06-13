from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.delay_store import LIGHT_SPEED_KM_S
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.link_delay.module.position_cache import PositionCacheStore, open_position_cache_for_interval
from src.link_delay.module.query import FullLinkDelayStore, open_delay_store_for_interval
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data

from build_g60_motif_gridplus_shortest_path_timeseries import (
    build_legacy_gridplus_edge_table,
    build_support_motif_edge_table,
    group_name,
    group_nodes_for_step,
)


DEFAULT_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE = THIS_DIR / "cache" / "group_data_cache"
DEFAULT_MOTIF_CONFIG = GENERIC_ROOT / "src" / "motif_generator" / "examples" / "configs" / "dad_cxx_support.yaml"
DEFAULT_GRIDPLUS_CONFIG = PROJECT_ROOT / "data" / "topology_design" / "grid+" / "config" / "motif.json"
DEFAULT_DELAY_STORE = (
    PROJECT_ROOT
    / "data"
    / "linshi"
    / "G60_full_options_plus_intra_t0_86164_stride1"
)
DEFAULT_POSITION_CACHE = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "_position_cache"
    / "cache_0_86164_1s"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_motif_gridplus_shortest_delay_t0_86164"
INTRA_OPTION = -1
TOPOLOGY_SUPPORT = "support_motif_DAD_Cxx"
TOPOLOGY_GRIDPLUS = "gridplus"


@dataclass(frozen=True)
class WeightedSummary:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute true weighted China-Europe shortest propagation-delay time series "
            "for G60 support-motif and grid+ topologies."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100, help="Inclusive end step. Use 86164 for the full G60 run.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--source-group", type=int, default=2, help="Default G60 group 2 is China.")
    parser.add_argument("--target-group", type=int, default=3, help="Default G60 group 3 is Europe.")
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--motif-config", type=Path, default=DEFAULT_MOTIF_CONFIG)
    parser.add_argument("--gridplus-config", type=Path, default=DEFAULT_GRIDPLUS_CONFIG)
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE)
    parser.add_argument(
        "--position-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional fallback for old inter-only delay stores. "
            "Not needed when --delay-store-dir points to a plus-intra delay store."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--topologies",
        nargs="+",
        choices=[TOPOLOGY_SUPPORT, TOPOLOGY_GRIDPLUS],
        default=[TOPOLOGY_SUPPORT, TOPOLOGY_GRIDPLUS],
    )
    parser.add_argument("--engine", choices=["auto", "scipy", "heapq"], default="auto")
    parser.add_argument("--sample-steps", type=int, default=3)
    parser.add_argument("--sample-pairs-per-step", type=int, default=60)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def finite_or_none(value: float) -> float | None:
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def node_xy(node: int) -> tuple[int, int]:
    node = int(node)
    return node // int(G60_CONFIG.N), node % int(G60_CONFIG.N)


def path_xy_text(path: list[int]) -> str:
    return "->".join(f"({p},{y})" for p, y in (node_xy(node) for node in path))


def build_weight_lookup(edge_table: EdgeTable, delay_store: FullLinkDelayStore, *, allow_intra_fallback: bool) -> WeightLookup:
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
            sp, sy = node_xy(src)
            dp, dy = node_xy(dst)
            missing.append(f"edge_idx={idx} {src}({sp},{sy})-{dst}({dp},{dy}) option={option}")
            continue
        topology_edge_indices.append(idx)
        store_edge_indices.append(int(delay_idx))

    if missing:
        preview = "\n".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise KeyError(
            "Some inter-plane topology edges are not present in the full-option delay store:\n"
            f"{preview}{more}"
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
            raise RuntimeError(
                "The delay store does not contain some intra links. "
                "Pass --position-cache-dir to enable the old position-cache fallback, "
                "or build the plus-intra delay store first."
            )
        positions = np.asarray(position_store.positions_km[int(position_row)], dtype=np.float32)
        src_pos = positions[lookup.fallback_intra_src_nodes]
        dst_pos = positions[lookup.fallback_intra_dst_nodes]
        dist_km = np.linalg.norm(src_pos - dst_pos, axis=1)
        weights[lookup.fallback_intra_edge_indices] = (dist_km / float(LIGHT_SPEED_KM_S) * 1000.0).astype(np.float32)
    return weights


def build_bidirectional_sparse_parts(edge_table: EdgeTable) -> tuple[np.ndarray, np.ndarray]:
    src = np.asarray(edge_table.src, dtype=np.int32)
    dst = np.asarray(edge_table.dst, dtype=np.int32)
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    return row, col


def build_heapq_adjacency(edge_table: EdgeTable, total_nodes: int) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(int(total_nodes))]
    for edge_idx in range(edge_table.num_edges):
        src = int(edge_table.src[edge_idx])
        dst = int(edge_table.dst[edge_idx])
        adjacency[src].append((dst, edge_idx))
        adjacency[dst].append((src, edge_idx))
    return adjacency


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


def summarize_distances(step: int, sources: tuple[int, ...], targets: tuple[int, ...], values: np.ndarray) -> WeightedSummary:
    reachable = np.isfinite(values)
    if sources and targets and bool(np.any(reachable)):
        reachable_values = np.asarray(values[reachable], dtype=np.float64)
        total = float(np.sum(reachable_values))
        count = int(reachable_values.size)
        return WeightedSummary(
            step=int(step),
            source_nodes=len(sources),
            target_nodes=len(targets),
            reachable_pairs=count,
            total_shortest_delay_ms=total,
            mean_shortest_delay_ms=float(total / count),
            min_shortest_delay_ms=float(np.min(reachable_values)),
            max_shortest_delay_ms=float(np.max(reachable_values)),
        )

    return WeightedSummary(
        step=int(step),
        source_nodes=len(sources),
        target_nodes=len(targets),
        reachable_pairs=0,
        total_shortest_delay_ms=0.0,
        mean_shortest_delay_ms=float("nan"),
        min_shortest_delay_ms=float("nan"),
        max_shortest_delay_ms=float("nan"),
    )


def choose_engine(requested: str):
    if requested in ("auto", "scipy"):
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

            return "scipy", csr_matrix, scipy_dijkstra
        except Exception as exc:
            if requested == "scipy":
                raise RuntimeError(f"--engine scipy was requested, but scipy is unavailable: {exc}") from exc
    return "heapq", None, None


def write_summary_header(writer: csv.DictWriter) -> None:
    writer.writeheader()


def summary_to_row(item: WeightedSummary) -> dict:
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
) -> None:
    writer.writerow(
        {
            "step": int(step),
            "source": int(source),
            "source_xy": str(node_xy(source)),
            "target": int(target),
            "target_xy": str(node_xy(target)),
            "delay_ms": float(delay_ms),
            "path": "->".join(str(x) for x in path),
            "path_xy": path_xy_text(path),
            "path_indexed": ",".join(f"{idx + 1}:{node}" for idx, node in enumerate(path)),
        }
    )


def reconstruct_scipy_path(predecessors: np.ndarray, source: int, target: int) -> list[int]:
    return reconstruct_from_prev(np.asarray(predecessors, dtype=np.int32), int(source), int(target))


def compute_one_topology_delay(
    *,
    topology_name: str,
    edge_table: EdgeTable,
    group_data: dict,
    steps: list[int],
    delay_rows: np.ndarray,
    position_rows: np.ndarray | None,
    delay_store: FullLinkDelayStore,
    position_store: PositionCacheStore | None,
    source_group: int,
    target_group: int,
    out_dir: Path,
    engine_name: str,
    csr_matrix,
    scipy_dijkstra,
    sample_steps: int,
    sample_pairs_per_step: int,
    progress_every: int,
) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(edge_table, out_dir / "edges.csv")
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))

    lookup = build_weight_lookup(edge_table, delay_store, allow_intra_fallback=position_store is not None)
    row_index, col_index = build_bidirectional_sparse_parts(edge_table)
    heapq_adjacency = build_heapq_adjacency(edge_table, int(G60_CONFIG.total_sats)) if engine_name == "heapq" else None
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
        f"[shortest-delay] {topology_name}: edges={edge_table.num_edges} "
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
        write_summary_header(summary_writer)
        sample_writer.writeheader()

        for local_idx, step in enumerate(steps):
            sources = group_nodes_for_step(group_data, int(step), int(source_group))
            targets = group_nodes_for_step(group_data, int(step), int(target_group))
            sample_this_step = local_idx < max(0, int(sample_steps)) and int(sample_pairs_per_step) > 0
            sample_written = 0

            if not sources or not targets:
                item = summarize_distances(int(step), sources, targets, np.asarray([], dtype=np.float64))
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
                    shape=(int(G60_CONFIG.total_sats), int(G60_CONFIG.total_sats)),
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
                            path = reconstruct_scipy_path(predecessors[source_i], int(source), int(target))
                            write_path_sample_row(
                                sample_writer,
                                step=int(step),
                                source=int(source),
                                target=int(target),
                                delay_ms=delay_ms,
                                path=path,
                            )
                            sample_written += 1
                            if sample_written >= int(sample_pairs_per_step):
                                break
                        if sample_written >= int(sample_pairs_per_step):
                            break

            else:
                values_rows: list[np.ndarray] = []
                target_set = set(int(x) for x in targets)
                for source in sources:
                    dist, prev = dijkstra_heapq(
                        adjacency=heapq_adjacency,
                        weights=weights,
                        source=int(source),
                        targets=target_set,
                    )
                    values_rows.append(dist[target_array])
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
                            )
                            sample_written += 1
                            if sample_written >= int(sample_pairs_per_step):
                                break
                    if sample_written >= int(sample_pairs_per_step):
                        sample_this_step = False
                values = np.vstack(values_rows) if values_rows else np.asarray([], dtype=np.float64)

            item = summarize_distances(int(step), sources, targets, values)
            mean_values[local_idx] = item.mean_shortest_delay_ms
            summary_writer.writerow(summary_to_row(item))

            if progress_every > 0 and (local_idx + 1) % int(progress_every) == 0:
                elapsed = time.time() - started_at
                print(
                    f"[shortest-delay] {topology_name}: {local_idx + 1}/{len(steps)} "
                    f"step={step} mean={item.mean_shortest_delay_ms:.4f} ms elapsed={elapsed:.1f}s",
                    flush=True,
                )

    np.save(out_dir / "mean_shortest_delay_ms.npy", mean_values)
    meta = {
        "topology": topology_name,
        "constellation": G60_CONFIG.name,
        "P": int(G60_CONFIG.P),
        "N": int(G60_CONFIG.N),
        "total_sats": int(G60_CONFIG.total_sats),
        "source_group_id": int(source_group),
        "source_group_name": group_name(source_group),
        "target_group_id": int(target_group),
        "target_group_name": group_name(target_group),
        "num_steps": int(len(steps)),
        "start": int(min(steps)) if steps else None,
        "end": int(max(steps)) if steps else None,
        "num_edges": int(edge_table.num_edges),
        "edges_from_delay_store": int(lookup.topology_edge_indices.size),
        "intra_edges_from_position_cache_fallback": int(lookup.fallback_intra_edge_indices.size),
        "delay_store_dir": str(delay_store.store_dir),
        "position_cache_dir": str(position_store.cache_dir) if position_store is not None else None,
        "engine": engine_name,
        "method": (
            "For each time step, inter-plane edge weights are read from the full-option delay store; "
            "intra-plane y-ring edge weights are computed from the position cache; "
            "then weighted shortest paths are computed from source group nodes to target group nodes."
        ),
        "mean_delay_ms_min": finite_or_none(float(np.nanmin(mean_values))) if mean_values.size else None,
        "mean_delay_ms_max": finite_or_none(float(np.nanmax(mean_values))) if mean_values.size else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[shortest-delay] {topology_name}: mean_delay_ms=({meta['mean_delay_ms_min']}, "
        f"{meta['mean_delay_ms_max']}) out_dir={out_dir}",
        flush=True,
    )
    return mean_values


def read_step_delay_means(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("mean_shortest_delay_ms")
            out[int(row["step"])] = float(raw) if raw not in (None, "") else float("nan")
    return out


def write_comparison(out_dir: Path, topology_dirs: dict[str, Path], steps: list[int]) -> None:
    means_by_topology = {name: read_step_delay_means(path / "step_summary.csv") for name, path in topology_dirs.items()}
    names = list(topology_dirs)
    comparison_path = out_dir / "compare_step_summary.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["step"] + [f"{name}_mean_delay_ms" for name in names]
        if len(names) == 2:
            fieldnames.append(f"{names[0]}_minus_{names[1]}_ms")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in steps:
            row = {"step": int(step)}
            for name in names:
                row[f"{name}_mean_delay_ms"] = means_by_topology[name].get(int(step), float("nan"))
            if len(names) == 2:
                row[f"{names[0]}_minus_{names[1]}_ms"] = (
                    float(row[f"{names[0]}_mean_delay_ms"]) - float(row[f"{names[1]}_mean_delay_ms"])
                )
            writer.writerow(row)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4.8))
        x = np.asarray(steps, dtype=np.int64)
        for name in names:
            y = np.asarray([means_by_topology[name].get(int(step), np.nan) for step in steps], dtype=np.float32)
            ax.plot(x, y, linewidth=1.0, label=name)
        ax.set_xlabel("time step (s)")
        ax.set_ylabel("China-Europe mean shortest delay (ms)")
        ax.grid(alpha=0.28, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "mean_shortest_delay_timeseries.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[shortest-delay] plot skipped: {exc}", flush=True)


def main() -> int:
    args = parse_args()
    if int(args.end) < int(args.start):
        raise ValueError("--end must be >= --start")
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")

    delay_store = open_delay_store_for_interval(
        int(args.start),
        int(args.end),
        stride=int(args.stride),
        store_dir=args.delay_store_dir,
    )
    delay_rows = delay_store.rows_for_interval(int(args.start), int(args.end), int(args.stride))
    steps = [int(x) for x in np.asarray(delay_store.time_indices[delay_rows], dtype=np.int64)]
    position_store: PositionCacheStore | None = None
    position_rows: np.ndarray | None = None
    if args.position_cache_dir is not None:
        position_store = open_position_cache_for_interval(
            int(args.start),
            int(args.end),
            stride=int(args.stride),
            full_cache_dir=args.position_cache_dir,
        )
        position_rows = position_store.rows_for_interval(int(args.start), int(args.end), int(args.stride))
        position_steps = [int(x) for x in np.asarray(position_store.times_s[position_rows], dtype=np.int64)]
        if steps != position_steps:
            raise ValueError("delay store time steps and position cache time steps are not aligned")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_data = load_or_build_group_data(
        xml_file=args.xml_file,
        group_cache_dir=args.group_cache_dir,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )
    engine_name, csr_matrix, scipy_dijkstra = choose_engine(str(args.engine))
    print(
        f"[shortest-delay] steps={len(steps)} range={steps[0]}..{steps[-1]} "
        f"delay_store={delay_store.store_dir} "
        f"position_cache={position_store.cache_dir if position_store is not None else 'not used'}",
        flush=True,
    )
    print(f"[shortest-delay] groups={len(group_data)} engine={engine_name} out_dir={out_dir}", flush=True)

    topology_tables: dict[str, EdgeTable] = {}
    if TOPOLOGY_SUPPORT in args.topologies:
        topology_tables[TOPOLOGY_SUPPORT] = build_support_motif_edge_table(
            motif_config=Path(args.motif_config),
            p=int(G60_CONFIG.P),
            n=int(G60_CONFIG.N),
        )
    if TOPOLOGY_GRIDPLUS in args.topologies:
        topology_tables[TOPOLOGY_GRIDPLUS] = build_legacy_gridplus_edge_table(motif_json=Path(args.gridplus_config))

    topology_dirs: dict[str, Path] = {}
    for name, edge_table in topology_tables.items():
        topology_dirs[name] = out_dir / name
        compute_one_topology_delay(
            topology_name=name,
            edge_table=edge_table,
            group_data=group_data,
            steps=steps,
            delay_rows=delay_rows,
            position_rows=position_rows,
            delay_store=delay_store,
            position_store=position_store,
            source_group=int(args.source_group),
            target_group=int(args.target_group),
            out_dir=topology_dirs[name],
            engine_name=engine_name,
            csr_matrix=csr_matrix,
            scipy_dijkstra=scipy_dijkstra,
            sample_steps=int(args.sample_steps),
            sample_pairs_per_step=int(args.sample_pairs_per_step),
            progress_every=int(args.progress_every),
        )

    write_comparison(out_dir, topology_dirs, steps)
    meta = {
        "constellation": G60_CONFIG.name,
        "source_group_id": int(args.source_group),
        "source_group_name": group_name(args.source_group),
        "target_group_id": int(args.target_group),
        "target_group_name": group_name(args.target_group),
        "start": int(args.start),
        "end": int(args.end),
        "end_semantics": "inclusive",
        "stride": int(args.stride),
        "num_steps": int(len(steps)),
        "delay_store_dir": str(delay_store.store_dir),
        "position_cache_dir": str(position_store.cache_dir) if position_store is not None else None,
        "motif_config": str(Path(args.motif_config)),
        "gridplus_config": str(Path(args.gridplus_config)),
        "topologies": list(topology_tables),
        "outputs": {name: str(path) for name, path in topology_dirs.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[shortest-delay] done | comparison={out_dir / 'compare_step_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
