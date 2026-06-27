from __future__ import annotations

import csv
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path

from src.link_delay.module.edge_options import EdgeTable
from src.topology_metrics.module.group_states import group_nodes_for_step


@dataclass(frozen=True)
class GroupPairNodeArrays:
    """Padded source/target node arrays for a group pair over a time axis."""

    steps: np.ndarray
    source_nodes: np.ndarray
    source_counts: np.ndarray
    target_nodes: np.ndarray
    target_counts: np.ndarray


@dataclass(frozen=True)
class GroupPairShortestTimeseries:
    """Per-step shortest-hop and optional shortest-delay summaries."""

    steps: np.ndarray
    source_counts: np.ndarray
    target_counts: np.ndarray
    active_edges: np.ndarray
    hop_reachable_pairs: np.ndarray
    mean_shortest_hops: np.ndarray
    min_shortest_hops: np.ndarray
    max_shortest_hops: np.ndarray
    delay_reachable_pairs: np.ndarray
    mean_shortest_delay_ms: np.ndarray
    min_shortest_delay_ms: np.ndarray
    max_shortest_delay_ms: np.ndarray

    @property
    def has_delay(self) -> bool:
        return bool(np.any(np.isfinite(self.mean_shortest_delay_ms)))


def read_edge_table_csv(edges_csv: str | Path, *, total_nodes: int) -> EdgeTable:
    """Read an ``EdgeTable`` written by ``write_edges_csv`` without importing GUI modules."""

    src: list[int] = []
    dst: list[int] = []
    option: list[int] = []
    src_plane: list[int] = []
    src_y: list[int] = []
    dst_plane: list[int] = []
    dst_y: list[int] = []
    with Path(edges_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            src.append(int(row["src_node"]))
            dst.append(int(row["dst_node"]))
            option.append(int(row["option"]))
            src_plane.append(int(row["src_plane"]))
            src_y.append(int(row["src_y"]))
            dst_plane.append(int(row["dst_plane"]))
            dst_y.append(int(row["dst_y"]))

    return EdgeTable(
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        option=np.asarray(option, dtype=np.int16),
        src_plane=np.asarray(src_plane, dtype=np.int16),
        src_y=np.asarray(src_y, dtype=np.int16),
        dst_plane=np.asarray(dst_plane, dtype=np.int16),
        dst_y=np.asarray(dst_y, dtype=np.int16),
        sat_ids=[str(idx + 1) for idx in range(int(total_nodes))],
    )


def _as_steps(steps: Sequence[int] | np.ndarray) -> np.ndarray:
    out = np.asarray(steps, dtype=np.int64)
    if out.ndim != 1 or out.size == 0:
        raise ValueError("steps must be a non-empty 1-D sequence")
    return out


def _as_time_edge_array(
    values: np.ndarray | Sequence,
    *,
    rows: int,
    edges: int,
    name: str,
    dtype,
) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 1:
        if arr.shape != (int(edges),):
            raise ValueError(f"{name} shape {arr.shape} != ({edges},)")
        return np.broadcast_to(arr.reshape(1, int(edges)), (int(rows), int(edges)))
    if arr.ndim == 2:
        if arr.shape != (int(rows), int(edges)):
            raise ValueError(f"{name} shape {arr.shape} != ({rows}, {edges})")
        return arr
    raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")


def build_group_pair_node_arrays(
    *,
    group_data: Mapping,
    steps: Sequence[int] | np.ndarray,
    source_group_id: int,
    target_group_id: int,
) -> GroupPairNodeArrays:
    """Convert ``group_data`` to compact padded arrays for repeated shortest-path runs."""

    steps_array = _as_steps(steps)
    source_lists: list[tuple[int, ...]] = []
    target_lists: list[tuple[int, ...]] = []
    max_source = 0
    max_target = 0
    for step in steps_array:
        sources = group_nodes_for_step(group_data, int(step), int(source_group_id))
        targets = group_nodes_for_step(group_data, int(step), int(target_group_id))
        source_lists.append(sources)
        target_lists.append(targets)
        max_source = max(max_source, len(sources))
        max_target = max(max_target, len(targets))

    source_nodes = np.full((steps_array.size, max_source), -1, dtype=np.int32)
    target_nodes = np.full((steps_array.size, max_target), -1, dtype=np.int32)
    source_counts = np.zeros(steps_array.size, dtype=np.int16)
    target_counts = np.zeros(steps_array.size, dtype=np.int16)
    for row, nodes in enumerate(source_lists):
        source_counts[row] = int(len(nodes))
        if nodes:
            source_nodes[row, : len(nodes)] = np.asarray(nodes, dtype=np.int32)
    for row, nodes in enumerate(target_lists):
        target_counts[row] = int(len(nodes))
        if nodes:
            target_nodes[row, : len(nodes)] = np.asarray(nodes, dtype=np.int32)

    return GroupPairNodeArrays(
        steps=steps_array,
        source_nodes=source_nodes,
        source_counts=source_counts,
        target_nodes=target_nodes,
        target_counts=target_counts,
    )


def _finite_summary(values: np.ndarray) -> tuple[float, int, float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), 0, float("nan"), float("nan")
    return float(np.mean(finite)), int(finite.size), float(np.min(finite)), float(np.max(finite))


def compute_group_pair_shortest_step(
    *,
    edge_table: EdgeTable,
    total_nodes: int,
    active_mask: np.ndarray,
    source_nodes: Sequence[int],
    target_nodes: Sequence[int],
    edge_weights_ms: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Compute one step's mean shortest hops and optional delay for one group pair."""

    mask = np.asarray(active_mask, dtype=bool)
    if mask.shape != (int(edge_table.num_edges),):
        raise ValueError(f"active_mask shape {mask.shape} != ({edge_table.num_edges},)")

    sources = np.asarray(source_nodes, dtype=np.int32)
    targets = np.asarray(target_nodes, dtype=np.int32)
    active_src = np.asarray(edge_table.src, dtype=np.int32)[mask]
    active_dst = np.asarray(edge_table.dst, dtype=np.int32)[mask]
    active_edges = int(active_src.size)

    empty = {
        "source_nodes": int(sources.size),
        "target_nodes": int(targets.size),
        "active_edges": active_edges,
        "hop_reachable_pairs": 0,
        "mean_shortest_hops": float("nan"),
        "min_shortest_hops": float("nan"),
        "max_shortest_hops": float("nan"),
        "delay_reachable_pairs": 0,
        "mean_shortest_delay_ms": float("nan"),
        "min_shortest_delay_ms": float("nan"),
        "max_shortest_delay_ms": float("nan"),
    }
    if sources.size == 0 or targets.size == 0 or active_edges == 0:
        return empty

    row_index = np.concatenate([active_src, active_dst])
    col_index = np.concatenate([active_dst, active_src])
    hop_graph = csr_matrix(
        (np.ones(active_edges * 2, dtype=np.float32), (row_index, col_index)),
        shape=(int(total_nodes), int(total_nodes)),
    )
    hop_dist = shortest_path(hop_graph, directed=False, unweighted=True, indices=sources)
    hop_values = np.atleast_2d(hop_dist)[:, targets]
    hop_mean, hop_count, hop_min, hop_max = _finite_summary(hop_values)

    out = dict(empty)
    out.update(
        {
            "hop_reachable_pairs": int(hop_count),
            "mean_shortest_hops": float(hop_mean),
            "min_shortest_hops": float(hop_min),
            "max_shortest_hops": float(hop_max),
        }
    )

    if edge_weights_ms is None:
        return out

    weights = np.asarray(edge_weights_ms, dtype=np.float32)
    if weights.shape != (int(edge_table.num_edges),):
        raise ValueError(f"edge_weights_ms shape {weights.shape} != ({edge_table.num_edges},)")
    active_weights = weights[mask]
    delay_graph = csr_matrix(
        (np.concatenate([active_weights, active_weights]), (row_index, col_index)),
        shape=(int(total_nodes), int(total_nodes)),
    )
    delay_dist = dijkstra(delay_graph, directed=False, indices=sources)
    delay_values = np.atleast_2d(delay_dist)[:, targets]
    delay_mean, delay_count, delay_min, delay_max = _finite_summary(delay_values)
    out.update(
        {
            "delay_reachable_pairs": int(delay_count),
            "mean_shortest_delay_ms": float(delay_mean),
            "min_shortest_delay_ms": float(delay_min),
            "max_shortest_delay_ms": float(delay_max),
        }
    )
    return out


def _blank_result(steps: np.ndarray) -> dict[str, np.ndarray]:
    count = int(steps.size)
    return {
        "source_counts": np.zeros(count, dtype=np.int16),
        "target_counts": np.zeros(count, dtype=np.int16),
        "active_edges": np.zeros(count, dtype=np.int32),
        "hop_reachable_pairs": np.zeros(count, dtype=np.int32),
        "mean_shortest_hops": np.full(count, np.nan, dtype=np.float32),
        "min_shortest_hops": np.full(count, np.nan, dtype=np.float32),
        "max_shortest_hops": np.full(count, np.nan, dtype=np.float32),
        "delay_reachable_pairs": np.zeros(count, dtype=np.int32),
        "mean_shortest_delay_ms": np.full(count, np.nan, dtype=np.float32),
        "min_shortest_delay_ms": np.full(count, np.nan, dtype=np.float32),
        "max_shortest_delay_ms": np.full(count, np.nan, dtype=np.float32),
    }


def _node_slice(nodes: np.ndarray, counts: np.ndarray, row: int) -> np.ndarray:
    return np.asarray(nodes[int(row), : int(counts[int(row)])], dtype=np.int32)


def _compute_rows_serial(
    *,
    edge_table: EdgeTable,
    total_nodes: int,
    active_masks: np.ndarray,
    edge_weights_ms: np.ndarray | None,
    group_nodes: GroupPairNodeArrays,
    rows: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    local_steps = group_nodes.steps[rows]
    out = _blank_result(local_steps)
    for local_i, global_i in enumerate(rows):
        global_i = int(global_i)
        sources = _node_slice(group_nodes.source_nodes, group_nodes.source_counts, global_i)
        targets = _node_slice(group_nodes.target_nodes, group_nodes.target_counts, global_i)
        item = compute_group_pair_shortest_step(
            edge_table=edge_table,
            total_nodes=int(total_nodes),
            active_mask=active_masks[global_i],
            source_nodes=sources,
            target_nodes=targets,
            edge_weights_ms=None if edge_weights_ms is None else edge_weights_ms[global_i],
        )
        out["source_counts"][local_i] = int(item["source_nodes"])
        out["target_counts"][local_i] = int(item["target_nodes"])
        out["active_edges"][local_i] = int(item["active_edges"])
        out["hop_reachable_pairs"][local_i] = int(item["hop_reachable_pairs"])
        out["mean_shortest_hops"][local_i] = float(item["mean_shortest_hops"])
        out["min_shortest_hops"][local_i] = float(item["min_shortest_hops"])
        out["max_shortest_hops"][local_i] = float(item["max_shortest_hops"])
        out["delay_reachable_pairs"][local_i] = int(item["delay_reachable_pairs"])
        out["mean_shortest_delay_ms"][local_i] = float(item["mean_shortest_delay_ms"])
        out["min_shortest_delay_ms"][local_i] = float(item["min_shortest_delay_ms"])
        out["max_shortest_delay_ms"][local_i] = float(item["max_shortest_delay_ms"])
    return rows.astype(np.int64), out


def _compute_chunk(payload: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    global_rows = np.asarray(payload["global_rows"], dtype=np.int64)
    local_payload = {key: value for key, value in payload.items() if key != "global_rows"}
    _local_rows, chunk_out = _compute_rows_serial(**local_payload)
    return global_rows, chunk_out


def compute_group_pair_shortest_timeseries(
    *,
    edge_table: EdgeTable,
    total_nodes: int,
    steps: Sequence[int] | np.ndarray,
    edge_active_mask: np.ndarray | Sequence,
    group_data: Mapping | None = None,
    group_nodes: GroupPairNodeArrays | None = None,
    source_group_id: int | None = None,
    target_group_id: int | None = None,
    edge_weights_ms: np.ndarray | Sequence | None = None,
    workers: int = 1,
    chunk_size: int = 500,
    progress_every_chunks: int = 0,
) -> GroupPairShortestTimeseries:
    """Compute mean shortest hops and optional delay for a dynamic edge-mask series.

    ``edge_active_mask`` may be either ``(edge,)`` for a static topology or
    ``(time, edge)`` for a dynamic topology. ``edge_weights_ms`` follows the
    same convention and is optional; when omitted, delay fields are ``NaN``.
    """

    steps_array = _as_steps(steps)
    active = _as_time_edge_array(
        edge_active_mask,
        rows=int(steps_array.size),
        edges=int(edge_table.num_edges),
        name="edge_active_mask",
        dtype=bool,
    )
    weights = None
    if edge_weights_ms is not None:
        weights = _as_time_edge_array(
            edge_weights_ms,
            rows=int(steps_array.size),
            edges=int(edge_table.num_edges),
            name="edge_weights_ms",
            dtype=np.float32,
        )

    if group_nodes is None:
        if group_data is None or source_group_id is None or target_group_id is None:
            raise ValueError("Provide group_nodes or group_data with source_group_id and target_group_id")
        group_nodes = build_group_pair_node_arrays(
            group_data=group_data,
            steps=steps_array,
            source_group_id=int(source_group_id),
            target_group_id=int(target_group_id),
        )
    if not np.array_equal(np.asarray(group_nodes.steps, dtype=np.int64), steps_array):
        raise ValueError("group_nodes.steps must match steps")

    result_arrays = _blank_result(steps_array)
    all_rows = np.arange(int(steps_array.size), dtype=np.int64)
    chunks = [all_rows[i : i + int(chunk_size)] for i in range(0, int(all_rows.size), int(chunk_size))]
    started = time.time()

    if int(workers) <= 1 or len(chunks) <= 1:
        for chunk_idx, rows in enumerate(chunks, start=1):
            row_ids, chunk_out = _compute_rows_serial(
                edge_table=edge_table,
                total_nodes=int(total_nodes),
                active_masks=active,
                edge_weights_ms=weights,
                group_nodes=group_nodes,
                rows=rows,
            )
            for key, values in chunk_out.items():
                result_arrays[key][row_ids] = values
            if progress_every_chunks and chunk_idx % int(progress_every_chunks) == 0:
                print(
                    f"[topology-metrics] shortest {int(row_ids[-1]) + 1}/{steps_array.size} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    else:
        payloads = []
        for rows in chunks:
            local_rows = np.arange(int(rows.size), dtype=np.int64)
            chunk_nodes = GroupPairNodeArrays(
                steps=steps_array[rows],
                source_nodes=np.asarray(group_nodes.source_nodes[rows], dtype=np.int32),
                source_counts=np.asarray(group_nodes.source_counts[rows], dtype=np.int16),
                target_nodes=np.asarray(group_nodes.target_nodes[rows], dtype=np.int32),
                target_counts=np.asarray(group_nodes.target_counts[rows], dtype=np.int16),
            )
            payloads.append(
                {
                    "global_rows": rows,
                    "edge_table": edge_table,
                    "total_nodes": int(total_nodes),
                    "active_masks": np.asarray(active[rows], dtype=bool),
                    "edge_weights_ms": None if weights is None else np.asarray(weights[rows], dtype=np.float32),
                    "group_nodes": chunk_nodes,
                    "rows": local_rows,
                }
            )
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = [pool.submit(_compute_chunk, payload) for payload in payloads]
            for future in as_completed(futures):
                row_ids, chunk_out = future.result()
                for key, values in chunk_out.items():
                    result_arrays[key][row_ids] = values
                done_chunks += 1
                if progress_every_chunks and done_chunks % int(progress_every_chunks) == 0:
                    print(
                        f"[topology-metrics] shortest chunks={done_chunks}/{len(chunks)} "
                        f"rows_done~{done_chunks * int(chunk_size)}/{steps_array.size} "
                        f"elapsed={time.time() - started:.1f}s",
                        flush=True,
                    )

    return GroupPairShortestTimeseries(steps=steps_array, **result_arrays)


def finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    return float(np.mean(finite)) if finite.size else float("nan")


def finite_min(values: np.ndarray) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    return float(np.min(finite)) if finite.size else float("nan")


def finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    return float(np.max(finite)) if finite.size else float("nan")


def result_summary(result: GroupPairShortestTimeseries) -> dict:
    return {
        "steps": int(result.steps.size),
        "start": int(result.steps[0]) if result.steps.size else None,
        "end": int(result.steps[-1]) if result.steps.size else None,
        "mean_hops": finite_mean(result.mean_shortest_hops),
        "min_hops": finite_min(result.mean_shortest_hops),
        "max_hops": finite_max(result.mean_shortest_hops),
        "mean_shortest_delay_ms": finite_mean(result.mean_shortest_delay_ms),
        "min_delay_ms": finite_min(result.mean_shortest_delay_ms),
        "max_delay_ms": finite_max(result.mean_shortest_delay_ms),
    }


def _json_float(value: float) -> float | None:
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def write_group_pair_shortest_timeseries(
    result: GroupPairShortestTimeseries,
    out_dir: str | Path,
    *,
    meta: Mapping | None = None,
) -> dict:
    """Write CSV, NPY arrays, and summary JSON for a shortest-metric series."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "time_indices.npy", result.steps.astype(np.int64))
    np.save(out / "mean_shortest_hops.npy", result.mean_shortest_hops.astype(np.float32))
    np.save(out / "mean_shortest_delay_ms.npy", result.mean_shortest_delay_ms.astype(np.float32))

    with (out / "timeseries.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "step",
            "hour",
            "source_nodes",
            "target_nodes",
            "active_edges",
            "hop_reachable_pairs",
            "mean_shortest_hops",
            "min_shortest_hops",
            "max_shortest_hops",
            "delay_reachable_pairs",
            "mean_shortest_delay_ms",
            "min_shortest_delay_ms",
            "max_shortest_delay_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, step in enumerate(result.steps):
            writer.writerow(
                {
                    "step": int(step),
                    "hour": f"{int(step) / 3600.0:.9f}",
                    "source_nodes": int(result.source_counts[idx]),
                    "target_nodes": int(result.target_counts[idx]),
                    "active_edges": int(result.active_edges[idx]),
                    "hop_reachable_pairs": int(result.hop_reachable_pairs[idx]),
                    "mean_shortest_hops": _json_float(float(result.mean_shortest_hops[idx])),
                    "min_shortest_hops": _json_float(float(result.min_shortest_hops[idx])),
                    "max_shortest_hops": _json_float(float(result.max_shortest_hops[idx])),
                    "delay_reachable_pairs": int(result.delay_reachable_pairs[idx]),
                    "mean_shortest_delay_ms": _json_float(float(result.mean_shortest_delay_ms[idx])),
                    "min_shortest_delay_ms": _json_float(float(result.min_shortest_delay_ms[idx])),
                    "max_shortest_delay_ms": _json_float(float(result.max_shortest_delay_ms[idx])),
                }
            )

    summary = result_summary(result)
    summary = {key: _json_float(value) if isinstance(value, float) else value for key, value in summary.items()}
    payload = {"summary": summary, "meta": dict(meta or {})}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "out_dir": str(out),
        "timeseries": str(out / "timeseries.csv"),
        "time_indices": str(out / "time_indices.npy"),
        "mean_shortest_hops": str(out / "mean_shortest_hops.npy"),
        "mean_shortest_delay_ms": str(out / "mean_shortest_delay_ms.npy"),
        "summary": str(out / "summary.json"),
    }


def plot_group_pair_shortest_timeseries(
    series: Mapping[str, GroupPairShortestTimeseries],
    out_dir: str | Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    colors: Mapping[str, str] | None = None,
    linewidths: Mapping[str, float] | None = None,
) -> str:
    """Plot one metric from several aligned shortest-metric series."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16.2, 6.8), dpi=180)
    for name, result in series.items():
        if not hasattr(result, metric):
            raise AttributeError(f"GroupPairShortestTimeseries has no metric {metric!r}")
        values = np.asarray(getattr(result, metric), dtype=np.float32)
        x = np.asarray(result.steps, dtype=np.float64) / 3600.0
        mean = finite_mean(values)
        ax.plot(
            x,
            values,
            label=f"{name} | mean={mean:.3f}",
            color=None if colors is None else colors.get(name),
            linewidth=1.2 if linewidths is None else float(linewidths.get(name, 1.2)),
        )
    ax.set_title(title)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    path = out / filename
    fig.savefig(path)
    plt.close(fig)
    return str(path)
