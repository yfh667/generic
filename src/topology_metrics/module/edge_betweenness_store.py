from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from src.link_delay.module.edge_options import EdgeTable, write_edges_csv

from .edge_betweenness import build_undirected_adjacency, edge_betweenness_between_node_sets, edge_usage_share_from_counts
from .group_states import GroupState, build_group_state_index
from .stores import MetricStoreLayout, expand_unique_state_values, write_meta, write_state_definitions


_WORKER_EDGE_TABLE: EdgeTable | None = None
_WORKER_ADJACENCY = None
_WORKER_TOTAL_NODES: int | None = None


def _init_worker(edge_table: EdgeTable, total_nodes: int) -> None:
    global _WORKER_EDGE_TABLE, _WORKER_ADJACENCY, _WORKER_TOTAL_NODES
    _WORKER_EDGE_TABLE = edge_table
    _WORKER_TOTAL_NODES = int(total_nodes)
    _WORKER_ADJACENCY = build_undirected_adjacency(edge_table, int(total_nodes))


def _compute_state_value(task: tuple[int, GroupState, int]) -> tuple[int, np.ndarray, dict[str, Any], list[dict]]:
    if _WORKER_EDGE_TABLE is None or _WORKER_ADJACENCY is None or _WORKER_TOTAL_NODES is None:
        raise RuntimeError("worker is not initialized")
    state_id, state, sample_path_limit = task
    source_nodes, target_nodes = state
    values, summary, samples = edge_betweenness_between_node_sets(
        _WORKER_EDGE_TABLE,
        total_nodes=int(_WORKER_TOTAL_NODES),
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        adjacency=_WORKER_ADJACENCY,
        sample_path_limit=int(sample_path_limit),
    )
    return int(state_id), values, summary.__dict__, samples


def _state_values_complete(path: Path, *, num_states: int, num_edges: int) -> bool:
    if not path.exists():
        return False
    arr = np.load(path, mmap_mode="r")
    return tuple(arr.shape) == (int(num_states), int(num_edges))


def _write_state_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "state_id",
        "source_nodes",
        "target_nodes",
        "reachable_pairs",
        "total_shortest_distance_hops",
        "mean_shortest_distance_hops",
        "max_edge_betweenness",
        "max_edge_usage_share",
        "nonzero_edges",
        "edge_value_sum",
        "edge_usage_share_sum",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_state_summary(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_unique_usage_share_values(
    path: Path,
    *,
    unique_values: np.ndarray,
    state_summaries: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    values = np.asarray(unique_values, dtype=np.float32)
    shares = open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=values.shape,
    )
    for state_id in range(values.shape[0]):
        reachable_pairs = float(state_summaries[state_id].get("reachable_pairs", 0) or 0)
        shares[state_id, :] = edge_usage_share_from_counts(values[state_id, :], reachable_pairs)
    shares.flush()
    return shares


def _with_usage_share_fields(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        reachable_pairs = float(item.get("reachable_pairs", 0) or 0)
        max_value = float(item.get("max_edge_betweenness", 0.0) or 0.0)
        value_sum = float(item.get("edge_value_sum", 0.0) or 0.0)
        item["max_edge_usage_share"] = float(max_value / reachable_pairs) if reachable_pairs > 0.0 else 0.0
        item["edge_usage_share_sum"] = float(value_sum / reachable_pairs) if reachable_pairs > 0.0 else 0.0
        out.append(item)
    return out


def _write_path_samples(path: Path, samples_by_state: Mapping[int, Sequence[Mapping[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "state_id",
            "source",
            "target",
            "distance_hops",
            "num_shortest_paths",
            "representative_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for state_id, samples in samples_by_state.items():
            for sample in samples:
                writer.writerow(
                    {
                        "state_id": int(state_id),
                        "source": int(sample.get("source", -1)),
                        "target": int(sample.get("target", -1)),
                        "distance_hops": int(sample.get("distance_hops", -1)),
                        "num_shortest_paths": float(sample.get("num_shortest_paths", 0.0)),
                        "representative_path": "->".join(str(x) for x in sample.get("representative_path", [])),
                    }
                )


def _write_step_summary(
    path: Path,
    *,
    steps: Sequence[int],
    state_ids: np.ndarray,
    unique_state_values: np.ndarray,
    state_summaries: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "step",
            "state_id",
            "reachable_pairs",
            "max_edge_betweenness",
            "max_edge_usage_share",
            "nonzero_edges",
            "edge_value_sum",
            "edge_usage_share_sum",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, step in enumerate(steps):
            state_id = int(state_ids[row])
            values = np.asarray(unique_state_values[state_id, :], dtype=np.float32)
            summary = state_summaries[state_id]
            reachable_pairs = int(float(summary.get("reachable_pairs", 0) or 0))
            denom = float(reachable_pairs)
            max_value = float(np.max(values)) if values.size else 0.0
            value_sum = float(np.sum(values))
            writer.writerow(
                {
                    "step": int(step),
                    "state_id": state_id,
                    "reachable_pairs": reachable_pairs,
                    "max_edge_betweenness": max_value,
                    "max_edge_usage_share": float(max_value / denom) if denom > 0.0 else 0.0,
                    "nonzero_edges": int(np.count_nonzero(values > 0.0)),
                    "edge_value_sum": value_sum,
                    "edge_usage_share_sum": float(value_sum / denom) if denom > 0.0 else 0.0,
                }
            )


def build_unique_edge_betweenness_values(
    *,
    edge_table: EdgeTable,
    total_nodes: int,
    unique_states: Sequence[GroupState],
    out_dir: str | Path,
    workers: int = 1,
    progress_every: int = 50,
    force: bool = False,
    sample_path_limit: int = 0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compute edge betweenness once per unique source/target group state."""

    layout = MetricStoreLayout(Path(out_dir))
    layout.root.mkdir(parents=True, exist_ok=True)
    num_states = int(len(unique_states))
    num_edges = int(edge_table.num_edges)

    if (
        not bool(force)
        and _state_values_complete(layout.unique_state_values_npy, num_states=num_states, num_edges=num_edges)
        and layout.state_summary_csv.exists()
    ):
        print(f"[topology-metrics] reusing unique edge betweenness: {layout.unique_state_values_npy}", flush=True)
        old_rows = _read_state_summary(layout.state_summary_csv)
        summary_rows = _with_usage_share_fields(old_rows)
        if summary_rows and ("max_edge_usage_share" not in old_rows[0] or "edge_usage_share_sum" not in old_rows[0]):
            _write_state_summary(layout.state_summary_csv, summary_rows)
        if not layout.unique_state_usage_share_npy.exists():
            _write_unique_usage_share_values(
                layout.unique_state_usage_share_npy,
                unique_values=np.load(layout.unique_state_values_npy, mmap_mode="r"),
                state_summaries=summary_rows,
            )
        return np.load(layout.unique_state_values_npy, mmap_mode="r"), summary_rows

    values = open_memmap(
        layout.unique_state_values_npy,
        mode="w+",
        dtype=np.float32,
        shape=(num_states, num_edges),
    )
    tasks = [(idx, state, int(sample_path_limit)) for idx, state in enumerate(unique_states)]
    summaries: list[dict[str, Any] | None] = [None] * num_states
    samples_by_state: dict[int, list[dict]] = {}

    started_at = time.perf_counter()
    completed = 0
    workers = max(1, int(workers))
    print(
        f"[topology-metrics] computing unique edge betweenness states={num_states} "
        f"edges={num_edges} workers={workers}",
        flush=True,
    )
    if workers == 1:
        _init_worker(edge_table, int(total_nodes))
        for task in tasks:
            state_id, row_values, summary, samples = _compute_state_value(task)
            values[state_id, :] = row_values
            reachable_pairs = float(summary.get("reachable_pairs", 0) or 0)
            max_value = float(summary.get("max_edge_betweenness", 0.0) or 0.0)
            value_sum = float(summary.get("edge_value_sum", 0.0) or 0.0)
            summaries[state_id] = {
                "state_id": int(state_id),
                **summary,
                "max_edge_usage_share": float(max_value / reachable_pairs) if reachable_pairs > 0.0 else 0.0,
                "edge_usage_share_sum": float(value_sum / reachable_pairs) if reachable_pairs > 0.0 else 0.0,
            }
            if samples:
                samples_by_state[int(state_id)] = samples
            completed += 1
            if int(progress_every) > 0 and (completed == num_states or completed % int(progress_every) == 0):
                print(f"[topology-metrics] states {completed}/{num_states}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(edge_table, int(total_nodes))) as executor:
            futures = [executor.submit(_compute_state_value, task) for task in tasks]
            for future in as_completed(futures):
                state_id, row_values, summary, samples = future.result()
                values[state_id, :] = row_values
                reachable_pairs = float(summary.get("reachable_pairs", 0) or 0)
                max_value = float(summary.get("max_edge_betweenness", 0.0) or 0.0)
                value_sum = float(summary.get("edge_value_sum", 0.0) or 0.0)
                summaries[state_id] = {
                    "state_id": int(state_id),
                    **summary,
                    "max_edge_usage_share": float(max_value / reachable_pairs) if reachable_pairs > 0.0 else 0.0,
                    "edge_usage_share_sum": float(value_sum / reachable_pairs) if reachable_pairs > 0.0 else 0.0,
                }
                if samples:
                    samples_by_state[int(state_id)] = samples
                completed += 1
                if int(progress_every) > 0 and (completed == num_states or completed % int(progress_every) == 0):
                    elapsed = time.perf_counter() - started_at
                    print(f"[topology-metrics] states {completed}/{num_states} elapsed={elapsed:.1f}s", flush=True)

    values.flush()
    final_summaries = []
    for row in summaries:
        if row is None:
            raise RuntimeError("missing edge betweenness state summary")
        final_summaries.append(row)
    _write_state_summary(layout.state_summary_csv, final_summaries)
    _write_unique_usage_share_values(
        layout.unique_state_usage_share_npy,
        unique_values=np.asarray(values),
        state_summaries=final_summaries,
    )
    if int(sample_path_limit) > 0:
        _write_path_samples(layout.root / "path_samples.csv", samples_by_state)
    return values, final_summaries


def compute_edge_betweenness_store(
    *,
    topology_name: str,
    edge_table: EdgeTable,
    total_nodes: int,
    group_data: Mapping,
    steps: Sequence[int],
    source_group_id: int,
    target_group_id: int,
    out_dir: str | Path,
    workers: int = 1,
    progress_every: int = 50,
    force: bool = False,
    expand_full_matrix: bool = False,
    sample_path_limit: int = 0,
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reusable unweighted edge-betweenness metric store.

    The compact store is ``unique_state_values.npy`` plus ``state_ids.npy``.
    Set ``expand_full_matrix=True`` only when a downstream consumer really
    needs a full ``(step, edge)`` matrix.
    """

    steps = [int(step) for step in steps]
    layout = MetricStoreLayout(Path(out_dir))
    layout.root.mkdir(parents=True, exist_ok=True)
    write_edges_csv(edge_table, layout.edges_csv)
    np.save(layout.time_indices_npy, np.asarray(steps, dtype=np.int64))

    state_index = build_group_state_index(
        group_data=group_data,
        steps=steps,
        source_group_id=int(source_group_id),
        target_group_id=int(target_group_id),
    )
    np.save(layout.state_ids_npy, state_index.state_ids)
    write_state_definitions(layout.state_definitions_json, state_index.unique_states)

    unique_values, state_summaries = build_unique_edge_betweenness_values(
        edge_table=edge_table,
        total_nodes=int(total_nodes),
        unique_states=state_index.unique_states,
        out_dir=layout.root,
        workers=int(workers),
        progress_every=int(progress_every),
        force=bool(force),
        sample_path_limit=int(sample_path_limit),
    )
    _write_step_summary(
        layout.step_summary_csv,
        steps=steps,
        state_ids=state_index.state_ids,
        unique_state_values=unique_values,
        state_summaries=state_summaries,
    )

    if bool(expand_full_matrix):
        full_values = expand_unique_state_values(
            unique_state_values=np.asarray(unique_values),
            state_ids=state_index.state_ids,
        )
        np.save(layout.metric_values_npy, full_values.astype(np.float32, copy=False))
        unique_share_values = np.load(layout.unique_state_usage_share_npy, mmap_mode="r")
        full_share_values = expand_unique_state_values(
            unique_state_values=np.asarray(unique_share_values),
            state_ids=state_index.state_ids,
        )
        np.save(layout.metric_usage_share_npy, full_share_values.astype(np.float32, copy=False))

    meta = {
        "topology": str(topology_name),
        "metric": "unweighted_edge_betweenness_between_group_sets",
        "storage": "unique_state_values + unique_state_usage_share + state_ids",
        "counting_rule": "edge_betweenness is shortest-hop path usage count; edge_usage_share = count / reachable_pairs",
        "num_steps": int(len(steps)),
        "num_unique_group_states": int(state_index.num_states),
        "num_edges": int(edge_table.num_edges),
        "source_group_id": int(source_group_id),
        "target_group_id": int(target_group_id),
        "expanded_full_matrix": bool(expand_full_matrix),
        "expanded_full_share_matrix": bool(expand_full_matrix),
        "max_edge_betweenness": max((float(row["max_edge_betweenness"]) for row in state_summaries), default=0.0),
        "max_edge_usage_share": max((float(row.get("max_edge_usage_share", 0.0)) for row in state_summaries), default=0.0),
        "extra": dict(extra_meta or {}),
    }
    write_meta(layout.meta_json, meta)
    print(f"[topology-metrics] edge betweenness store written: {layout.root}", flush=True)
    return meta
