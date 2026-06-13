from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.link_delay.module.query import FullLinkDelayStore, open_delay_store_for_interval


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
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_motif_gridplus_shortest_delay_parallel_t0_86164"

TOPOLOGY_SUPPORT = "support_motif_DAD_Cxx"
TOPOLOGY_GRIDPLUS = "gridplus"
TOPOLOGY_FULL_LINK = "full_link"

_WORKER: dict[str, Any] = {}


@dataclass(frozen=True)
class TopologyWorkerSpec:
    name: str
    num_edges: int
    store_edge_indices: np.ndarray
    indptr: np.ndarray
    neighbors: np.ndarray
    edge_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel weighted China-Europe shortest propagation-delay time series "
            "for G60 support motif DAD_Cxx and grid+."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=99, help="Inclusive end step. Use 86164 for the full run.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--source-group", type=int, default=2, help="Default G60 group 2 is China.")
    parser.add_argument("--target-group", type=int, default=3, help="Default G60 group 3 is Europe.")
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--motif-config", type=Path, default=DEFAULT_MOTIF_CONFIG)
    parser.add_argument("--gridplus-config", type=Path, default=DEFAULT_GRIDPLUS_CONFIG)
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--topologies",
        nargs="+",
        choices=[TOPOLOGY_SUPPORT, TOPOLOGY_GRIDPLUS, TOPOLOGY_FULL_LINK],
        default=[TOPOLOGY_SUPPORT, TOPOLOGY_GRIDPLUS],
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="0 means auto: min(8, cpu_count - 1), at least 1.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Steps per worker task. 0 means auto: small chunks for small tests, capped at 200 for long runs.",
    )
    parser.add_argument("--sample-steps", type=int, default=3)
    parser.add_argument("--sample-pairs-per-step", type=int, default=60)
    parser.add_argument("--progress-every-futures", type=int, default=10)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def finite_or_none(value: float) -> float | None:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def node_xy(node: int) -> tuple[int, int]:
    node = int(node)
    return node // int(G60_CONFIG.N), node % int(G60_CONFIG.N)


def path_xy_text(path: list[int]) -> str:
    return "->".join(f"({p},{y})" for p, y in (node_xy(node) for node in path))


def group_name(group_id: int) -> str:
    return str(G60_CONFIG.station_groups.get(int(group_id), {}).get("name", f"Group {group_id}"))


def group_nodes_for_step(group_data: dict, step: int, group_id: int) -> tuple[int, ...]:
    current = group_data.get(int(step), {}) if group_data else {}
    groups = current.get("groups", {}) if isinstance(current, dict) else {}
    return tuple(sorted(int(x) for x in groups.get(int(group_id), set()) or set()))


def make_worker_spec(edge_table: EdgeTable, delay_store: FullLinkDelayStore, topology_name: str) -> TopologyWorkerSpec:
    store_edge_indices = np.empty(edge_table.num_edges, dtype=np.int64)
    missing: list[str] = []
    for idx in range(edge_table.num_edges):
        src = int(edge_table.src[idx])
        dst = int(edge_table.dst[idx])
        try:
            store_edge_indices[idx] = int(delay_store.edge_idx(src, dst))
        except KeyError as exc:
            sp, sy = node_xy(src)
            dp, dy = node_xy(dst)
            missing.append(
                f"edge_idx={idx} {src}({sp},{sy})-{dst}({dp},{dy}) "
                f"option={int(edge_table.option[idx])}: {exc}"
            )

    if missing:
        preview = "\n".join(missing[:20])
        more = "" if len(missing) <= 20 else f"\n... and {len(missing) - 20} more"
        raise KeyError(
            f"{topology_name} has edges missing from the plus-intra delay store. "
            "This parallel script intentionally has no position-cache fallback.\n"
            f"{preview}{more}"
        )

    total_nodes = int(G60_CONFIG.total_sats)
    degree = np.zeros(total_nodes, dtype=np.int32)
    src = np.asarray(edge_table.src, dtype=np.int32)
    dst = np.asarray(edge_table.dst, dtype=np.int32)
    np.add.at(degree, src, 1)
    np.add.at(degree, dst, 1)

    indptr = np.empty(total_nodes + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    cursor = indptr[:-1].copy()
    neighbors = np.empty(edge_table.num_edges * 2, dtype=np.int32)
    edge_ids = np.empty(edge_table.num_edges * 2, dtype=np.int32)

    for edge_idx in range(edge_table.num_edges):
        a = int(src[edge_idx])
        b = int(dst[edge_idx])
        pos = int(cursor[a])
        neighbors[pos] = b
        edge_ids[pos] = edge_idx
        cursor[a] += 1

        pos = int(cursor[b])
        neighbors[pos] = a
        edge_ids[pos] = edge_idx
        cursor[b] += 1

    return TopologyWorkerSpec(
        name=str(topology_name),
        num_edges=int(edge_table.num_edges),
        store_edge_indices=store_edge_indices,
        indptr=indptr,
        neighbors=neighbors,
        edge_ids=edge_ids,
    )


def edge_table_from_delay_store(delay_store: FullLinkDelayStore) -> EdgeTable:
    """Use every edge in the plus-intra delay store as one full-link topology."""

    records = delay_store.edges
    total_sats = int(G60_CONFIG.total_sats)
    sat_ids = [str(i + 1) for i in range(total_sats)]
    for record in records:
        src = int(record.src_node)
        dst = int(record.dst_node)
        if 0 <= src < total_sats:
            sat_ids[src] = str(record.src_sat_id)
        if 0 <= dst < total_sats:
            sat_ids[dst] = str(record.dst_sat_id)

    src_values = np.asarray([int(record.src_node) for record in records], dtype=np.int32)
    dst_values = np.asarray([int(record.dst_node) for record in records], dtype=np.int32)
    option_values = np.asarray([int(record.option) for record in records], dtype=np.int16)
    n_count = int(G60_CONFIG.N)

    return EdgeTable(
        src=src_values,
        dst=dst_values,
        option=option_values,
        src_plane=(src_values // n_count).astype(np.int16),
        src_y=(src_values % n_count).astype(np.int16),
        dst_plane=(dst_values // n_count).astype(np.int16),
        dst_y=(dst_values % n_count).astype(np.int16),
        sat_ids=sat_ids,
    )


def dijkstra_targets(
    *,
    spec: TopologyWorkerSpec,
    weights: np.ndarray,
    source: int,
    targets: tuple[int, ...],
    want_prev: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    total_nodes = int(G60_CONFIG.total_sats)
    dist = np.full(total_nodes, np.inf, dtype=np.float64)
    prev = np.full(total_nodes, -1, dtype=np.int32) if want_prev else None
    dist[int(source)] = 0.0
    remaining = set(int(x) for x in targets)
    heap: list[tuple[float, int]] = [(0.0, int(source))]

    indptr = spec.indptr
    neighbors = spec.neighbors
    edge_ids = spec.edge_ids

    while heap and remaining:
        current_dist, node = heapq.heappop(heap)
        if current_dist != float(dist[node]):
            continue
        remaining.discard(node)
        start = int(indptr[node])
        end = int(indptr[node + 1])
        for pos in range(start, end):
            neighbor = int(neighbors[pos])
            edge_idx = int(edge_ids[pos])
            next_dist = current_dist + float(weights[edge_idx])
            if next_dist < float(dist[neighbor]):
                dist[neighbor] = next_dist
                if prev is not None:
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
        if guard > int(prev.size):
            return []
    path.reverse()
    return path


def summary_row(
    *,
    step: int,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
    values: list[float],
) -> dict[str, Any]:
    if values:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "step": int(step),
            "source_nodes": int(len(sources)),
            "target_nodes": int(len(targets)),
            "reachable_pairs": int(arr.size),
            "total_shortest_delay_ms": finite_or_none(float(np.sum(arr))),
            "mean_shortest_delay_ms": finite_or_none(float(np.mean(arr))),
            "min_shortest_delay_ms": finite_or_none(float(np.min(arr))),
            "max_shortest_delay_ms": finite_or_none(float(np.max(arr))),
        }
    return {
        "step": int(step),
        "source_nodes": int(len(sources)),
        "target_nodes": int(len(targets)),
        "reachable_pairs": 0,
        "total_shortest_delay_ms": 0.0,
        "mean_shortest_delay_ms": None,
        "min_shortest_delay_ms": None,
        "max_shortest_delay_ms": None,
    }


def path_sample_row(
    *,
    step: int,
    source: int,
    target: int,
    delay_ms: float,
    path: list[int],
) -> dict[str, Any]:
    return {
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


def init_worker(payload: dict[str, Any]) -> None:
    delay_store_dir = str(payload["delay_store_dir"])
    steps = np.load(payload["steps_path"], mmap_mode="r")
    delay_rows = np.load(payload["delay_rows_path"], mmap_mode="r")
    source_indptr = np.load(payload["source_indptr_path"], mmap_mode="r")
    source_nodes = np.load(payload["source_nodes_path"], mmap_mode="r")
    target_indptr = np.load(payload["target_indptr_path"], mmap_mode="r")
    target_nodes = np.load(payload["target_nodes_path"], mmap_mode="r")
    topology_specs = payload["topology_specs"]
    delay_store = FullLinkDelayStore(delay_store_dir)
    _WORKER.clear()
    _WORKER.update(
        {
            "delay_ms_array": delay_store.delay_ms_array,
            "steps": steps,
            "delay_rows": delay_rows,
            "source_indptr": source_indptr,
            "source_nodes": source_nodes,
            "target_indptr": target_indptr,
            "target_nodes": target_nodes,
            "topology_specs": topology_specs,
        }
    )


def nodes_for_index(indptr: np.ndarray, nodes: np.ndarray, idx: int) -> tuple[int, ...]:
    start = int(indptr[int(idx)])
    end = int(indptr[int(idx) + 1])
    return tuple(int(x) for x in np.asarray(nodes[start:end], dtype=np.int32))


def compute_chunk(task: tuple[str, int, int, int, int]) -> dict[str, Any]:
    topology_name, start_idx, end_idx, sample_steps, sample_pairs_per_step = task
    spec: TopologyWorkerSpec = _WORKER["topology_specs"][topology_name]
    delay_ms_array = _WORKER["delay_ms_array"]
    steps = _WORKER["steps"]
    delay_rows = _WORKER["delay_rows"]
    source_indptr = _WORKER["source_indptr"]
    source_nodes_flat = _WORKER["source_nodes"]
    target_indptr = _WORKER["target_indptr"]
    target_nodes_flat = _WORKER["target_nodes"]

    summaries: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    means = np.full(int(end_idx) - int(start_idx), np.nan, dtype=np.float32)

    for offset, local_idx in enumerate(range(int(start_idx), int(end_idx))):
        step = int(steps[local_idx])
        sources = nodes_for_index(source_indptr, source_nodes_flat, local_idx)
        targets = nodes_for_index(target_indptr, target_nodes_flat, local_idx)
        if not sources or not targets:
            row = summary_row(step=step, sources=sources, targets=targets, values=[])
            summaries.append(row)
            continue

        delay_row = int(delay_rows[local_idx])
        weights = np.asarray(delay_ms_array[delay_row, spec.store_edge_indices], dtype=np.float32)
        want_sample = local_idx < int(sample_steps) and int(sample_pairs_per_step) > 0
        sample_written = 0
        values: list[float] = []

        for source in sources:
            dist, prev = dijkstra_targets(
                spec=spec,
                weights=weights,
                source=int(source),
                targets=targets,
                want_prev=want_sample and sample_written < int(sample_pairs_per_step),
            )
            for target in targets:
                value = float(dist[int(target)])
                if not math.isfinite(value):
                    continue
                values.append(value)
                if want_sample and prev is not None and sample_written < int(sample_pairs_per_step):
                    path = reconstruct_from_prev(prev, int(source), int(target))
                    samples.append(
                        path_sample_row(
                            step=step,
                            source=int(source),
                            target=int(target),
                            delay_ms=value,
                            path=path,
                        )
                    )
                    sample_written += 1

        row = summary_row(step=step, sources=sources, targets=targets, values=values)
        summaries.append(row)
        mean_value = row["mean_shortest_delay_ms"]
        if mean_value is not None:
            means[offset] = float(mean_value)

    return {
        "topology": topology_name,
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "summaries": summaries,
        "samples": samples,
        "means": means,
    }


def auto_workers(requested: int) -> int:
    if int(requested) > 0:
        return int(requested)
    cpu = os.cpu_count() or 2
    return max(1, min(8, int(cpu) - 1))


def auto_chunk_size(requested: int, num_steps: int, workers: int) -> int:
    if int(requested) > 0:
        return int(requested)
    target = max(10, int(num_steps) // max(1, int(workers) * 4))
    return max(1, min(200, target))


def load_steps_and_rows(delay_store_dir: Path, start: int, end: int, stride: int) -> tuple[FullLinkDelayStore, list[int], list[int]]:
    delay_store = open_delay_store_for_interval(
        int(start),
        int(end),
        stride=int(stride),
        store_dir=delay_store_dir,
    )
    delay_rows = delay_store.rows_for_interval(int(start), int(end), int(stride))
    steps = [int(x) for x in np.asarray(delay_store.time_indices[delay_rows], dtype=np.int64)]
    return delay_store, steps, [int(x) for x in np.asarray(delay_rows, dtype=np.int64)]


def flatten_node_groups(groups_by_index: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.empty(len(groups_by_index) + 1, dtype=np.int64)
    flat: list[int] = []
    indptr[0] = 0
    for idx, nodes in enumerate(groups_by_index):
        flat.extend(int(x) for x in nodes)
        indptr[idx + 1] = len(flat)
    return indptr, np.asarray(flat, dtype=np.int32)


def write_worker_input_arrays(
    *,
    out_dir: Path,
    steps: list[int],
    delay_rows: list[int],
    source_nodes_by_index: list[tuple[int, ...]],
    target_nodes_by_index: list[tuple[int, ...]],
) -> dict[str, str]:
    input_dir = Path(out_dir) / "_parallel_worker_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_indptr, source_nodes = flatten_node_groups(source_nodes_by_index)
    target_indptr, target_nodes = flatten_node_groups(target_nodes_by_index)

    paths = {
        "steps_path": input_dir / "steps.npy",
        "delay_rows_path": input_dir / "delay_rows.npy",
        "source_indptr_path": input_dir / "source_indptr.npy",
        "source_nodes_path": input_dir / "source_nodes.npy",
        "target_indptr_path": input_dir / "target_indptr.npy",
        "target_nodes_path": input_dir / "target_nodes.npy",
    }
    np.save(paths["steps_path"], np.asarray(steps, dtype=np.int64))
    np.save(paths["delay_rows_path"], np.asarray(delay_rows, dtype=np.int64))
    np.save(paths["source_indptr_path"], source_indptr)
    np.save(paths["source_nodes_path"], source_nodes)
    np.save(paths["target_indptr_path"], target_indptr)
    np.save(paths["target_nodes_path"], target_nodes)
    return {key: str(path) for key, path in paths.items()}


def write_topology_outputs(
    *,
    out_dir: Path,
    topology_name: str,
    edge_table: EdgeTable,
    steps: list[int],
    summaries: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    means: np.ndarray,
    worker_spec: TopologyWorkerSpec,
    args: argparse.Namespace,
    elapsed_s: float,
) -> None:
    topology_dir = out_dir / topology_name
    topology_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(edge_table, topology_dir / "edges.csv")
    np.save(topology_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))
    np.save(topology_dir / "mean_shortest_delay_ms.npy", np.asarray(means, dtype=np.float32))

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
    with (topology_dir / "step_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    sample_fields = ["step", "source", "source_xy", "target", "target_xy", "delay_ms", "path", "path_xy", "path_indexed"]
    samples = sorted(samples, key=lambda row: (int(row["step"]), int(row["source"]), int(row["target"])))
    with (topology_dir / "path_samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields)
        writer.writeheader()
        for row in samples:
            writer.writerow(row)

    finite = np.asarray(means[np.isfinite(means)], dtype=np.float64)
    meta = {
        "topology": topology_name,
        "constellation": G60_CONFIG.name,
        "P": int(G60_CONFIG.P),
        "N": int(G60_CONFIG.N),
        "total_sats": int(G60_CONFIG.total_sats),
        "source_group_id": int(args.source_group),
        "source_group_name": group_name(args.source_group),
        "target_group_id": int(args.target_group),
        "target_group_name": group_name(args.target_group),
        "num_steps": int(len(steps)),
        "start": int(steps[0]) if steps else None,
        "end": int(steps[-1]) if steps else None,
        "num_edges": int(edge_table.num_edges),
        "edges_from_delay_store": int(worker_spec.store_edge_indices.size),
        "intra_position_fallback": 0,
        "delay_store_dir": str(Path(args.delay_store_dir)),
        "method": (
            "All topology edge weights, including intra option -1 y-ring links, "
            "are read from the plus-intra delay store. Each time step then runs "
            "weighted Dijkstra from source group nodes to target group nodes."
        ),
        "parallel": {
            "max_workers": int(args.max_workers_resolved),
            "chunk_size": int(args.chunk_size_resolved),
            "elapsed_s": float(elapsed_s),
        },
        "mean_delay_ms_min": finite_or_none(float(np.min(finite))) if finite.size else None,
        "mean_delay_ms_max": finite_or_none(float(np.max(finite))) if finite.size else None,
    }
    (topology_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def write_comparison(out_dir: Path, topology_names: list[str], steps: list[int], means_by_topology: dict[str, np.ndarray]) -> None:
    comparison_path = out_dir / "compare_step_summary.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["step"] + [f"{name}_mean_delay_ms" for name in topology_names]
        if len(topology_names) == 2:
            fieldnames.append(f"{topology_names[0]}_minus_{topology_names[1]}_ms")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(steps):
            row: dict[str, Any] = {"step": int(step)}
            for name in topology_names:
                value = float(means_by_topology[name][idx])
                row[f"{name}_mean_delay_ms"] = finite_or_none(value)
            if len(topology_names) == 2:
                a = row[f"{topology_names[0]}_mean_delay_ms"]
                b = row[f"{topology_names[1]}_mean_delay_ms"]
                row[f"{topology_names[0]}_minus_{topology_names[1]}_ms"] = (
                    None if a is None or b is None else float(a) - float(b)
                )
            writer.writerow(row)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4.8))
        x = np.asarray(steps, dtype=np.int64)
        for name in topology_names:
            ax.plot(x, means_by_topology[name], linewidth=1.0, label=name)
        ax.set_xlabel("time step (s)")
        ax.set_ylabel("China-Europe mean shortest delay (ms)")
        ax.grid(alpha=0.28, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "mean_shortest_delay_timeseries.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[parallel-shortest-delay] plot skipped: {exc}", flush=True)


def main() -> int:
    from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data

    from build_g60_motif_gridplus_shortest_path_timeseries import (
        build_legacy_gridplus_edge_table,
        build_support_motif_edge_table,
    )

    args = parse_args()
    if int(args.end) < int(args.start):
        raise ValueError("--end must be >= --start")
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")

    started_at = time.time()
    delay_store, steps, delay_rows = load_steps_and_rows(
        Path(args.delay_store_dir),
        int(args.start),
        int(args.end),
        int(args.stride),
    )
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
    source_nodes_by_index = [
        group_nodes_for_step(group_data, int(step), int(args.source_group))
        for step in steps
    ]
    target_nodes_by_index = [
        group_nodes_for_step(group_data, int(step), int(args.target_group))
        for step in steps
    ]
    worker_array_paths = write_worker_input_arrays(
        out_dir=out_dir,
        steps=steps,
        delay_rows=delay_rows,
        source_nodes_by_index=source_nodes_by_index,
        target_nodes_by_index=target_nodes_by_index,
    )
    del source_nodes_by_index
    del target_nodes_by_index

    edge_tables: dict[str, EdgeTable] = {}
    if TOPOLOGY_SUPPORT in args.topologies:
        edge_tables[TOPOLOGY_SUPPORT] = build_support_motif_edge_table(
            motif_config=Path(args.motif_config),
            p=int(G60_CONFIG.P),
            n=int(G60_CONFIG.N),
        )
    if TOPOLOGY_GRIDPLUS in args.topologies:
        edge_tables[TOPOLOGY_GRIDPLUS] = build_legacy_gridplus_edge_table(motif_json=Path(args.gridplus_config))
    if TOPOLOGY_FULL_LINK in args.topologies:
        edge_tables[TOPOLOGY_FULL_LINK] = edge_table_from_delay_store(delay_store)

    worker_specs = {
        name: make_worker_spec(edge_table, delay_store, name)
        for name, edge_table in edge_tables.items()
    }
    topology_names = list(edge_tables)
    max_workers = auto_workers(int(args.max_workers))
    chunk_size = auto_chunk_size(int(args.chunk_size), len(steps), max_workers)
    args.max_workers_resolved = max_workers
    args.chunk_size_resolved = chunk_size

    tasks: list[tuple[str, int, int, int, int]] = []
    for name in topology_names:
        for start_idx in range(0, len(steps), chunk_size):
            end_idx = min(len(steps), start_idx + chunk_size)
            tasks.append((name, start_idx, end_idx, int(args.sample_steps), int(args.sample_pairs_per_step)))

    print(
        f"[parallel-shortest-delay] steps={len(steps)} range={steps[0]}..{steps[-1]} "
        f"topologies={topology_names} workers={max_workers} chunk_size={chunk_size} tasks={len(tasks)}",
        flush=True,
    )
    print(f"[parallel-shortest-delay] delay_store={delay_store.store_dir}", flush=True)
    for name, spec in worker_specs.items():
        print(f"[parallel-shortest-delay] {name}: edges={spec.num_edges} cache_edges={spec.store_edge_indices.size}", flush=True)

    summaries_by_topology: dict[str, list[dict[str, Any] | None]] = {
        name: [None] * len(steps)
        for name in topology_names
    }
    means_by_topology: dict[str, np.ndarray] = {
        name: np.full(len(steps), np.nan, dtype=np.float32)
        for name in topology_names
    }
    samples_by_topology: dict[str, list[dict[str, Any]]] = {name: [] for name in topology_names}

    worker_init_kwargs = {
        "delay_store_dir": str(delay_store.store_dir),
        "topology_specs": worker_specs,
    }
    worker_init_kwargs.update(worker_array_paths)

    completed = 0
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=get_context("spawn"),
        initializer=init_worker,
        initargs=(worker_init_kwargs,),
    ) as executor:
        futures = [executor.submit(compute_chunk, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            name = str(result["topology"])
            start_idx = int(result["start_idx"])
            end_idx = int(result["end_idx"])
            rows = result["summaries"]
            summaries_by_topology[name][start_idx:end_idx] = rows
            means_by_topology[name][start_idx:end_idx] = np.asarray(result["means"], dtype=np.float32)
            samples_by_topology[name].extend(result["samples"])
            completed += 1
            if int(args.progress_every_futures) > 0 and (
                completed == len(tasks) or completed % int(args.progress_every_futures) == 0
            ):
                elapsed = time.time() - started_at
                print(
                    f"[parallel-shortest-delay] completed {completed}/{len(tasks)} futures "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    elapsed_s = time.time() - started_at
    for name in topology_names:
        summaries = summaries_by_topology[name]
        if any(row is None for row in summaries):
            missing = [idx for idx, row in enumerate(summaries) if row is None][:10]
            raise RuntimeError(f"{name} has missing chunk results at indices {missing}")
        write_topology_outputs(
            out_dir=out_dir,
            topology_name=name,
            edge_table=edge_tables[name],
            steps=steps,
            summaries=[row for row in summaries if row is not None],
            samples=samples_by_topology[name],
            means=means_by_topology[name],
            worker_spec=worker_specs[name],
            args=args,
            elapsed_s=elapsed_s,
        )

    write_comparison(out_dir, topology_names, steps, means_by_topology)
    root_meta = {
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
        "motif_config": str(Path(args.motif_config)),
        "gridplus_config": str(Path(args.gridplus_config)),
        "topologies": topology_names,
        "max_workers": int(max_workers),
        "chunk_size": int(chunk_size),
        "elapsed_s": float(elapsed_s),
        "outputs": {name: str(out_dir / name) for name in topology_names},
        "all_edge_weights_from_plus_intra_delay_cache": True,
    }
    (out_dir / "meta.json").write_text(json.dumps(root_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[parallel-shortest-delay] done elapsed={elapsed_s:.1f}s out_dir={out_dir}", flush=True)
    print(f"[parallel-shortest-delay] comparison={out_dir / 'compare_step_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
