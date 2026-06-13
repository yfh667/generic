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
from numpy.lib.format import open_memmap


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.link_delay.module.query import FullLinkDelayStore
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data


DEFAULT_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE = THIS_DIR / "cache" / "group_data_cache"
DEFAULT_DELAY_STORE = PROJECT_ROOT / "data" / "linshi" / "G60_full_options_plus_intra_t0_86164_stride1"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "data" / "linshi"
DEFAULT_PAIRS = (
    ("china_europe", 2, 3),
    ("china_america", 2, 0),
    ("china_africa", 2, 1),
)

_WORKER: dict[str, Any] = {}


@dataclass(frozen=True)
class PairSpec:
    name: str
    source_group: int
    target_group: int


@dataclass(frozen=True)
class TopologySpec:
    indptr: np.ndarray
    neighbors: np.ndarray
    edge_ids: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute weighted shortest-delay edge betweenness on the G60 full-link + intra topology "
            "for China-Europe, China-America, and China-Africa."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86164, help="Inclusive end step.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--sample-steps", type=int, default=3)
    parser.add_argument("--sample-pairs-per-step", type=int, default=20)
    parser.add_argument("--progress-every-futures", type=int, default=10)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def default_out_dir(output_base: Path, start: int, end: int, stride: int) -> Path:
    return Path(output_base) / f"g60_full_link_multi_region_weighted_betweenness_t{start}_{end}_stride{stride}"


def group_name(group_id: int) -> str:
    return str(G60_CONFIG.station_groups.get(int(group_id), {}).get("name", f"Group {group_id}"))


def pair_label(pair: PairSpec) -> str:
    return f"{group_name(pair.source_group)}-{group_name(pair.target_group)}"


def node_xy(node: int) -> tuple[int, int]:
    node = int(node)
    return node // int(G60_CONFIG.N), node % int(G60_CONFIG.N)


def path_xy_text(path: list[int]) -> str:
    return "->".join(f"({p},{y})" for p, y in (node_xy(node) for node in path))


def edge_table_from_delay_store(delay_store: FullLinkDelayStore) -> EdgeTable:
    total_sats = int(G60_CONFIG.total_sats)
    sat_ids = [str(i + 1) for i in range(total_sats)]
    records = sorted(delay_store.edges, key=lambda item: int(item.edge_idx))
    for record in records:
        src = int(record.src_node)
        dst = int(record.dst_node)
        if 0 <= src < total_sats:
            sat_ids[src] = str(record.src_sat_id)
        if 0 <= dst < total_sats:
            sat_ids[dst] = str(record.dst_sat_id)

    src = np.asarray([int(record.src_node) for record in records], dtype=np.int32)
    dst = np.asarray([int(record.dst_node) for record in records], dtype=np.int32)
    option = np.asarray([int(record.option) for record in records], dtype=np.int16)
    n_count = int(G60_CONFIG.N)
    return EdgeTable(
        src=src,
        dst=dst,
        option=option,
        src_plane=(src // n_count).astype(np.int16),
        src_y=(src % n_count).astype(np.int16),
        dst_plane=(dst // n_count).astype(np.int16),
        dst_y=(dst % n_count).astype(np.int16),
        sat_ids=sat_ids,
    )


def build_topology_spec(edge_table: EdgeTable) -> TopologySpec:
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

    for edge_idx in range(int(edge_table.num_edges)):
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

    return TopologySpec(indptr=indptr, neighbors=neighbors, edge_ids=edge_ids)


def group_nodes_for_step(group_data: dict, step: int, group_id: int) -> tuple[int, ...]:
    current = group_data.get(int(step), {}) if group_data else {}
    groups = current.get("groups", {}) if isinstance(current, dict) else {}
    return tuple(sorted(int(x) for x in groups.get(int(group_id), set()) or set()))


def write_group_node_cache(
    *,
    out_dir: Path,
    group_data: dict,
    steps: list[int],
    group_ids: list[int],
) -> dict[int, tuple[Path, Path]]:
    worker_dir = out_dir / "_worker_inputs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, tuple[Path, Path]] = {}
    for group_id in group_ids:
        indptr = np.zeros(len(steps) + 1, dtype=np.int64)
        chunks: list[np.ndarray] = []
        total = 0
        for row, step in enumerate(steps):
            nodes = np.asarray(group_nodes_for_step(group_data, step, int(group_id)), dtype=np.int32)
            chunks.append(nodes)
            total += int(nodes.size)
            indptr[row + 1] = int(total)
        flat = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
        indptr_path = worker_dir / f"group_{int(group_id)}_indptr.npy"
        nodes_path = worker_dir / f"group_{int(group_id)}_nodes.npy"
        np.save(indptr_path, indptr)
        np.save(nodes_path, flat)
        result[int(group_id)] = (indptr_path, nodes_path)
    return result


def nodes_for_row(group_id: int, row: int) -> tuple[int, ...]:
    indptr, nodes = _WORKER["group_nodes"][int(group_id)]
    start = int(indptr[int(row)])
    end = int(indptr[int(row) + 1])
    return tuple(int(x) for x in np.asarray(nodes[start:end], dtype=np.int32))


def dijkstra_targets_with_prev_edge(
    *,
    spec: TopologySpec,
    weights: np.ndarray,
    source: int,
    targets: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_nodes = int(G60_CONFIG.total_sats)
    dist = np.full(total_nodes, np.inf, dtype=np.float64)
    prev_node = np.full(total_nodes, -1, dtype=np.int32)
    prev_edge = np.full(total_nodes, -1, dtype=np.int32)
    source = int(source)
    dist[source] = 0.0
    remaining = set(int(x) for x in targets if int(x) != source)
    heap: list[tuple[float, int]] = [(0.0, source)]

    indptr = spec.indptr
    neighbors = spec.neighbors
    edge_ids = spec.edge_ids
    while heap and remaining:
        current_dist, node = heapq.heappop(heap)
        if current_dist != float(dist[node]):
            continue
        remaining.discard(int(node))
        start = int(indptr[node])
        end = int(indptr[node + 1])
        for pos in range(start, end):
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


def finite_or_none(value: float) -> float | None:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def summarize_pair(
    *,
    step: int,
    pair: PairSpec,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
    delays: list[float],
    edge_values: np.ndarray,
) -> dict[str, Any]:
    if delays:
        arr = np.asarray(delays, dtype=np.float64)
        return {
            "step": int(step),
            "pair_name": pair.name,
            "source_group_id": int(pair.source_group),
            "source_group_name": group_name(pair.source_group),
            "target_group_id": int(pair.target_group),
            "target_group_name": group_name(pair.target_group),
            "source_nodes": int(len(sources)),
            "target_nodes": int(len(targets)),
            "reachable_pairs": int(arr.size),
            "total_shortest_delay_ms": float(np.sum(arr)),
            "mean_shortest_delay_ms": float(np.mean(arr)),
            "min_shortest_delay_ms": float(np.min(arr)),
            "max_shortest_delay_ms": float(np.max(arr)),
            "max_edge_betweenness": float(np.max(edge_values)) if edge_values.size else 0.0,
            "nonzero_edges": int(np.count_nonzero(edge_values > 0.0)),
            "edge_value_sum": float(np.sum(edge_values)),
        }
    return {
        "step": int(step),
        "pair_name": pair.name,
        "source_group_id": int(pair.source_group),
        "source_group_name": group_name(pair.source_group),
        "target_group_id": int(pair.target_group),
        "target_group_name": group_name(pair.target_group),
        "source_nodes": int(len(sources)),
        "target_nodes": int(len(targets)),
        "reachable_pairs": 0,
        "total_shortest_delay_ms": 0.0,
        "mean_shortest_delay_ms": None,
        "min_shortest_delay_ms": None,
        "max_shortest_delay_ms": None,
        "max_edge_betweenness": 0.0,
        "nonzero_edges": 0,
        "edge_value_sum": 0.0,
    }


def path_sample_row(
    *,
    step: int,
    pair: PairSpec,
    source: int,
    target: int,
    delay_ms: float,
    path: list[int],
    edge_indices: list[int],
) -> dict[str, Any]:
    return {
        "step": int(step),
        "pair_name": pair.name,
        "pair_label": pair_label(pair),
        "source": int(source),
        "source_xy": str(node_xy(source)),
        "target": int(target),
        "target_xy": str(node_xy(target)),
        "delay_ms": float(delay_ms),
        "path": "->".join(str(x) for x in path),
        "path_xy": path_xy_text(path),
        "edge_indices": " ".join(str(x) for x in edge_indices),
    }


def init_worker(payload: dict[str, Any]) -> None:
    delay_store = FullLinkDelayStore(payload["delay_store_dir"])
    group_nodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for raw_group_id, raw_paths in payload["group_node_paths"].items():
        group_id = int(raw_group_id)
        indptr_path, nodes_path = raw_paths
        group_nodes[group_id] = (
            np.load(indptr_path, mmap_mode="r"),
            np.load(nodes_path, mmap_mode="r"),
        )
    _WORKER.clear()
    _WORKER.update(
        {
            "delay_ms_array": delay_store.delay_ms_array,
            "delay_rows": np.load(payload["delay_rows_path"], mmap_mode="r"),
            "steps": np.load(payload["steps_path"], mmap_mode="r"),
            "spec": payload["spec"],
            "pairs": payload["pairs"],
            "group_nodes": group_nodes,
        }
    )


def compute_chunk(task: tuple[int, int, int, int]) -> dict[str, Any]:
    start_idx, end_idx, sample_steps, sample_pairs_per_step = task
    delay_ms_array = _WORKER["delay_ms_array"]
    delay_rows = _WORKER["delay_rows"]
    steps = _WORKER["steps"]
    spec: TopologySpec = _WORKER["spec"]
    pairs: list[PairSpec] = _WORKER["pairs"]
    num_edges = int(spec.edge_ids.max()) + 1

    pair_values = np.zeros((len(pairs), int(end_idx) - int(start_idx), num_edges), dtype=np.float32)
    summaries_by_pair: dict[str, list[dict[str, Any]]] = {pair.name: [] for pair in pairs}
    samples: list[dict[str, Any]] = []

    for offset, local_idx in enumerate(range(int(start_idx), int(end_idx))):
        step = int(steps[local_idx])
        weights = np.asarray(delay_ms_array[int(delay_rows[local_idx]), :], dtype=np.float32)
        sample_this_step = local_idx < int(sample_steps)

        for pair_idx, pair in enumerate(pairs):
            sources = nodes_for_row(pair.source_group, local_idx)
            targets = nodes_for_row(pair.target_group, local_idx)
            edge_values = pair_values[pair_idx, offset, :]
            delays: list[float] = []
            sample_written = 0

            if sources and targets:
                for source in sources:
                    dist, prev_node, prev_edge = dijkstra_targets_with_prev_edge(
                        spec=spec,
                        weights=weights,
                        source=int(source),
                        targets=targets,
                    )
                    for target in targets:
                        if int(target) == int(source):
                            continue
                        delay_value = float(dist[int(target)])
                        if not math.isfinite(delay_value):
                            continue
                        path, edge_indices = reconstruct_path_and_edges(
                            source=int(source),
                            target=int(target),
                            prev_node=prev_node,
                            prev_edge=prev_edge,
                        )
                        if not edge_indices:
                            continue
                        delays.append(delay_value)
                        for edge_idx in edge_indices:
                            edge_values[int(edge_idx)] += 1.0
                        if sample_this_step and sample_written < int(sample_pairs_per_step):
                            samples.append(
                                path_sample_row(
                                    step=step,
                                    pair=pair,
                                    source=int(source),
                                    target=int(target),
                                    delay_ms=delay_value,
                                    path=path,
                                    edge_indices=edge_indices,
                                )
                            )
                            sample_written += 1

            summaries_by_pair[pair.name].append(
                summarize_pair(
                    step=step,
                    pair=pair,
                    sources=sources,
                    targets=targets,
                    delays=delays,
                    edge_values=edge_values,
                )
            )

    return {
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "pair_values": pair_values,
        "summaries_by_pair": summaries_by_pair,
        "samples": samples,
    }


def auto_workers(requested: int) -> int:
    if int(requested) > 0:
        return int(requested)
    cpu = os.cpu_count() or 2
    return max(1, min(8, int(cpu) - 1))


def auto_chunk_size(requested: int, num_steps: int, workers: int) -> int:
    if int(requested) > 0:
        return int(requested)
    if int(num_steps) <= 100:
        return max(1, math.ceil(int(num_steps) / max(1, int(workers))))
    return 32


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_meta(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    steps: list[int],
    edge_table: EdgeTable,
    pairs: list[PairSpec],
    combined_sum: np.ndarray,
    combined_max: np.ndarray,
) -> None:
    payload = {
        "constellation": G60_CONFIG.name,
        "P": int(G60_CONFIG.P),
        "N": int(G60_CONFIG.N),
        "total_sats": int(G60_CONFIG.total_sats),
        "delay_store_dir": str(Path(args.delay_store_dir).resolve()),
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "num_steps": int(len(steps)),
        "num_edges": int(edge_table.num_edges),
        "pairs": [
            {
                "name": pair.name,
                "source_group_id": int(pair.source_group),
                "source_group_name": group_name(pair.source_group),
                "target_group_id": int(pair.target_group),
                "target_group_name": group_name(pair.target_group),
            }
            for pair in pairs
        ],
        "edge_value_unit": "shortest_delay_path_count",
        "shortest_path_weight": "propagation_delay_ms",
        "counting_rule": (
            "For each source-target satellite pair, the deterministic shortest-delay path contributes 1 "
            "to every edge on that path. combined_sum is the sum over all region pairs."
        ),
        "files": {
            "combined_sum": "edge_betweenness_sum.npy",
            "combined_max": "edge_betweenness_max.npy",
            "pair_matrix": "<pair_name>/edge_betweenness.npy",
        },
        "value_min": float(np.nanmin(combined_sum)) if combined_sum.size else 0.0,
        "value_max": float(np.nanmax(combined_sum)) if combined_sum.size else 0.0,
        "combined_max_value_min": float(np.nanmin(combined_max)) if combined_max.size else 0.0,
        "combined_max_value_max": float(np.nanmax(combined_max)) if combined_max.size else 0.0,
    }
    (out_dir / "meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.stride) <= 0:
        raise ValueError("stride must be positive")
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty step range")
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_out_dir(
        Path(args.output_base), int(args.start), int(args.end), int(args.stride)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [PairSpec(name=name, source_group=src, target_group=dst) for name, src, dst in DEFAULT_PAIRS]
    delay_store = FullLinkDelayStore(args.delay_store_dir)
    delay_rows = delay_store.rows_for_interval(int(args.start), int(args.end), int(args.stride))
    edge_table = edge_table_from_delay_store(delay_store)
    spec = build_topology_spec(edge_table)

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
    group_ids = sorted({pair.source_group for pair in pairs} | {pair.target_group for pair in pairs})
    group_node_paths = write_group_node_cache(
        out_dir=out_dir,
        group_data=group_data,
        steps=steps,
        group_ids=group_ids,
    )

    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))
    np.save(out_dir / "_worker_inputs" / "delay_rows.npy", np.asarray(delay_rows, dtype=np.int64))
    write_edges_csv(edge_table, out_dir / "edges.csv")

    expected_shape = (len(steps), int(edge_table.num_edges))
    pair_mmaps: dict[str, np.ndarray] = {}
    for pair in pairs:
        pair_dir = out_dir / pair.name
        pair_dir.mkdir(parents=True, exist_ok=True)
        pair_mmaps[pair.name] = open_memmap(
            pair_dir / "edge_betweenness.npy",
            mode="w+" if args.force or not (pair_dir / "edge_betweenness.npy").exists() else "r+",
            dtype=np.float32,
            shape=expected_shape,
        )
    combined_sum = open_memmap(out_dir / "edge_betweenness_sum.npy", mode="w+", dtype=np.float32, shape=expected_shape)
    combined_max = open_memmap(out_dir / "edge_betweenness_max.npy", mode="w+", dtype=np.float32, shape=expected_shape)

    workers = auto_workers(int(args.max_workers))
    chunk_size = auto_chunk_size(int(args.chunk_size), len(steps), workers)
    tasks = [
        (start, min(start + chunk_size, len(steps)), int(args.sample_steps), int(args.sample_pairs_per_step))
        for start in range(0, len(steps), chunk_size)
    ]
    payload = {
        "delay_store_dir": str(Path(args.delay_store_dir)),
        "delay_rows_path": str(out_dir / "_worker_inputs" / "delay_rows.npy"),
        "steps_path": str(out_dir / "time_indices.npy"),
        "group_node_paths": {int(k): (str(v[0]), str(v[1])) for k, v in group_node_paths.items()},
        "spec": spec,
        "pairs": pairs,
    }

    print(
        f"[multi-region-weighted-betweenness] steps={len(steps)} edges={edge_table.num_edges} "
        f"pairs={[pair.name for pair in pairs]} workers={workers} chunk_size={chunk_size} out_dir={out_dir}",
        flush=True,
    )

    summaries_by_pair: dict[str, list[dict[str, Any]]] = {pair.name: [] for pair in pairs}
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    completed = 0
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker, initargs=(payload,)) as executor:
        futures = {executor.submit(compute_chunk, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            start_idx = int(result["start_idx"])
            end_idx = int(result["end_idx"])
            pair_values = np.asarray(result["pair_values"], dtype=np.float32)

            sum_block = np.zeros((end_idx - start_idx, int(edge_table.num_edges)), dtype=np.float32)
            max_block = np.zeros_like(sum_block)
            for pair_idx, pair in enumerate(pairs):
                values = pair_values[pair_idx]
                pair_mmaps[pair.name][start_idx:end_idx, :] = values
                sum_block += values
                np.maximum(max_block, values, out=max_block)
                summaries_by_pair[pair.name].extend(result["summaries_by_pair"][pair.name])
            combined_sum[start_idx:end_idx, :] = sum_block
            combined_max[start_idx:end_idx, :] = max_block
            samples.extend(result["samples"])

            completed += 1
            if completed % max(1, int(args.progress_every_futures)) == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                rate = completed / max(1e-9, elapsed)
                eta = (len(tasks) - completed) / max(1e-9, rate)
                print(
                    f"[multi-region-weighted-betweenness] futures {completed}/{len(tasks)} "
                    f"rows={end_idx}/{len(steps)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

    for arr in pair_mmaps.values():
        arr.flush()
    combined_sum.flush()
    combined_max.flush()

    for pair in pairs:
        rows = sorted(summaries_by_pair[pair.name], key=lambda row: int(row["step"]))
        write_csv_rows(out_dir / pair.name / "step_summary.csv", rows)
        np.save(out_dir / pair.name / "time_indices.npy", np.asarray(steps, dtype=np.int64))
        write_edges_csv(edge_table, out_dir / pair.name / "edges.csv")
    write_csv_rows(out_dir / "path_samples.csv", sorted(samples, key=lambda row: (int(row["step"]), row["pair_name"])))
    write_meta(
        out_dir=out_dir,
        args=args,
        steps=steps,
        edge_table=edge_table,
        pairs=pairs,
        combined_sum=combined_sum,
        combined_max=combined_max,
    )
    elapsed = time.perf_counter() - started
    print(
        f"[multi-region-weighted-betweenness] done elapsed={elapsed:.1f}s "
        f"combined_value=({float(np.nanmin(combined_sum)):.4f}, {float(np.nanmax(combined_sum)):.4f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
