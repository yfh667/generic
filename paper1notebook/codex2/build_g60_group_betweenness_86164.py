from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from group_edge_betweenness import (
    StepBetweennessSummary,
    edge_betweenness_between_node_sets,
    write_group_betweenness_outputs,
)
from topology_edges import build_full_option_plus_intra_edges, build_undirected_adjacency
from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import write_edges_csv
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data


DEFAULT_XML = PROJECT_ROOT / "data" / "basic_file" / "G60" / "satellitesposition" / "station_visible_satellites_20250106.xml"
DEFAULT_GROUP_CACHE = THIS_DIR / "cache" / "group_data_cache"
DEFAULT_OUT_DIR = THIS_DIR / "outputs" / "g60_group_betweenness_t0_86164"

_WORKER_EDGE_TABLE = None
_WORKER_ADJACENCY = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full-length G60 China-Europe edge betweenness for the full-option + intra topology."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86164)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--source-group", type=int, default=2, help="Default G60 group 2 is China.")
    parser.add_argument("--target-group", type=int, default=3, help="Default G60 group 3 is Europe.")
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute state values and full matrix even if files exist.")
    return parser.parse_args(argv)


def group_name(group_id: int) -> str:
    return str(G60_CONFIG.station_groups.get(int(group_id), {}).get("name", f"Group {group_id}"))


def state_key_for_step(group_data: dict, step: int, source_group: int, target_group: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    current = group_data.get(int(step), {}) if group_data else {}
    groups = current.get("groups", {}) if isinstance(current, dict) else {}
    source_nodes = tuple(sorted(int(x) for x in groups.get(int(source_group), set()) or set()))
    target_nodes = tuple(sorted(int(x) for x in groups.get(int(target_group), set()) or set()))
    return source_nodes, target_nodes


def build_state_index(
    *,
    group_data: dict,
    steps: list[int],
    source_group: int,
    target_group: int,
) -> tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], np.ndarray]:
    state_to_id: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    unique_states: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    state_ids = np.empty(len(steps), dtype=np.int32)
    for row, step in enumerate(steps):
        key = state_key_for_step(
            group_data,
            step,
            source_group=int(source_group),
            target_group=int(target_group),
        )
        state_id = state_to_id.get(key)
        if state_id is None:
            state_id = len(unique_states)
            state_to_id[key] = state_id
            unique_states.append(key)
        state_ids[row] = int(state_id)
    return unique_states, state_ids


def _init_worker(edge_table):
    global _WORKER_EDGE_TABLE, _WORKER_ADJACENCY
    _WORKER_EDGE_TABLE = edge_table
    _WORKER_ADJACENCY = build_undirected_adjacency(edge_table, G60_CONFIG.total_sats)


def _compute_state_value(task: tuple[int, tuple[int, ...], tuple[int, ...]]):
    state_id, source_nodes, target_nodes = task
    values, summary, _samples = edge_betweenness_between_node_sets(
        _WORKER_EDGE_TABLE,
        total_nodes=G60_CONFIG.total_sats,
        source_nodes=set(int(x) for x in source_nodes),
        target_nodes=set(int(x) for x in target_nodes),
        adjacency=_WORKER_ADJACENCY,
        sample_path_limit=0,
    )
    return int(state_id), values, summary


def state_values_complete(out_dir: Path, *, num_states: int, num_edges: int) -> bool:
    path = out_dir / "unique_state_values.npy"
    summary_path = out_dir / "state_summary.csv"
    if not path.exists() or not summary_path.exists():
        return False
    arr = np.load(path, mmap_mode="r")
    return tuple(arr.shape) == (int(num_states), int(num_edges))


def write_state_definitions(path: Path, unique_states: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> None:
    payload = [
        {
            "state_id": int(idx),
            "source_nodes": [int(x) for x in source_nodes],
            "target_nodes": [int(x) for x in target_nodes],
        }
        for idx, (source_nodes, target_nodes) in enumerate(unique_states)
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_unique_state_values(
    *,
    out_dir: Path,
    edge_table,
    unique_states: list[tuple[tuple[int, ...], tuple[int, ...]]],
    workers: int,
    progress_every: int,
    force: bool,
) -> tuple[np.ndarray, list[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    state_values_path = out_dir / "unique_state_values.npy"
    state_summary_path = out_dir / "state_summary.csv"
    num_states = len(unique_states)
    num_edges = int(edge_table.num_edges)

    if not force and state_values_complete(out_dir, num_states=num_states, num_edges=num_edges):
        print(f"[full-betweenness] Reusing state values: {state_values_path}", flush=True)
        state_values = np.load(state_values_path, mmap_mode="r")
        with state_summary_path.open("r", encoding="utf-8", newline="") as f:
            state_summaries = list(csv.DictReader(f))
        return state_values, state_summaries

    state_values = open_memmap(
        state_values_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(num_states), int(num_edges)),
    )
    tasks = [(idx, source_nodes, target_nodes) for idx, (source_nodes, target_nodes) in enumerate(unique_states)]
    state_summaries: list[dict | None] = [None] * int(num_states)

    started = time.perf_counter()
    completed = 0
    print(
        f"[full-betweenness] Computing {num_states} unique group states with {workers} workers...",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_init_worker,
        initargs=(edge_table,),
    ) as executor:
        futures = {executor.submit(_compute_state_value, task): int(task[0]) for task in tasks}
        for future in as_completed(futures):
            state_id, values, summary = future.result()
            state_values[int(state_id), :] = values
            state_summaries[int(state_id)] = {
                "state_id": int(state_id),
                "source_nodes": int(summary["source_nodes"]),
                "target_nodes": int(summary["target_nodes"]),
                "reachable_pairs": int(summary["reachable_pairs"]),
                "total_shortest_distance_hops": float(summary["total_shortest_distance_hops"]),
                "mean_shortest_distance_hops": float(summary["mean_shortest_distance_hops"]),
                "max_edge_betweenness": float(summary["max_edge_betweenness"]),
                "nonzero_edges": int(summary["nonzero_edges"]),
                "edge_value_sum": float(summary["edge_value_sum"]),
            }
            completed += 1
            if completed % max(1, int(progress_every)) == 0 or completed == num_states:
                elapsed = time.perf_counter() - started
                rate = completed / max(1e-9, elapsed)
                eta = (num_states - completed) / max(1e-9, rate)
                print(
                    f"[full-betweenness] states {completed}/{num_states} | "
                    f"elapsed={elapsed:.1f}s | eta={eta:.1f}s",
                    flush=True,
                )

    state_values.flush()
    with state_summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "state_id",
            "source_nodes",
            "target_nodes",
            "reachable_pairs",
            "total_shortest_distance_hops",
            "mean_shortest_distance_hops",
            "max_edge_betweenness",
            "nonzero_edges",
            "edge_value_sum",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in state_summaries:
            if row is None:
                raise RuntimeError("missing state summary")
            writer.writerow(row)

    return state_values, [row for row in state_summaries if row is not None]


def write_full_time_matrix(
    *,
    out_dir: Path,
    state_values: np.ndarray,
    state_ids: np.ndarray,
    force: bool,
) -> np.ndarray:
    path = out_dir / "edge_betweenness.npy"
    expected_shape = (int(state_ids.size), int(state_values.shape[1]))
    if not force and path.exists():
        existing = np.load(path, mmap_mode="r")
        if tuple(existing.shape) == expected_shape:
            print(f"[full-betweenness] Reusing full matrix: {path}", flush=True)
            return existing

    values = open_memmap(path, mode="w+", dtype=np.float32, shape=expected_shape)
    started = time.perf_counter()
    for start in range(0, int(state_ids.size), 2048):
        end = min(start + 2048, int(state_ids.size))
        values[start:end, :] = state_values[state_ids[start:end], :]
        if end % 16384 == 0 or end == int(state_ids.size):
            elapsed = time.perf_counter() - started
            print(f"[full-betweenness] wrote rows {end}/{state_ids.size} | elapsed={elapsed:.1f}s", flush=True)
    values.flush()
    return values


def write_step_summary(
    *,
    path: Path,
    steps: list[int],
    state_ids: np.ndarray,
    state_summaries: list[dict],
    source_group: int,
    target_group: int,
) -> None:
    summary_by_state = {int(row["state_id"]): row for row in state_summaries}
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(StepBetweennessSummary.__dataclass_fields__.keys()) + ["state_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, step in enumerate(steps):
            state_id = int(state_ids[row])
            state_summary = summary_by_state[state_id]
            writer.writerow(
                {
                    "step": int(step),
                    "source_group_id": int(source_group),
                    "target_group_id": int(target_group),
                    "source_nodes": int(state_summary["source_nodes"]),
                    "target_nodes": int(state_summary["target_nodes"]),
                    "reachable_pairs": int(state_summary["reachable_pairs"]),
                    "total_shortest_distance_hops": float(state_summary["total_shortest_distance_hops"]),
                    "mean_shortest_distance_hops": float(state_summary["mean_shortest_distance_hops"]),
                    "max_edge_betweenness": float(state_summary["max_edge_betweenness"]),
                    "nonzero_edges": int(state_summary["nonzero_edges"]),
                    "state_id": state_id,
                }
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_table = build_full_option_plus_intra_edges(G60_CONFIG)
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

    unique_states, state_ids = build_state_index(
        group_data=group_data,
        steps=steps,
        source_group=int(args.source_group),
        target_group=int(args.target_group),
    )
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))
    np.save(out_dir / "state_ids.npy", state_ids)
    write_edges_csv(edge_table, out_dir / "edges.csv")
    write_state_definitions(out_dir / "state_definitions.json", unique_states)
    print(
        f"[full-betweenness] steps={len(steps)} edges={edge_table.num_edges} "
        f"unique_states={len(unique_states)} out_dir={out_dir}",
        flush=True,
    )

    state_values, state_summaries = build_unique_state_values(
        out_dir=out_dir,
        edge_table=edge_table,
        unique_states=unique_states,
        workers=int(args.workers),
        progress_every=int(args.progress_every),
        force=bool(args.force),
    )
    values = write_full_time_matrix(
        out_dir=out_dir,
        state_values=state_values,
        state_ids=state_ids,
        force=bool(args.force),
    )
    write_step_summary(
        path=out_dir / "step_summary.csv",
        steps=steps,
        state_ids=state_ids,
        state_summaries=state_summaries,
        source_group=int(args.source_group),
        target_group=int(args.target_group),
    )

    sample_steps = steps[: min(3, len(steps))]
    sample_values = np.asarray(values[: len(sample_steps), :], dtype=np.float32)
    write_group_betweenness_outputs(
        out_dir=out_dir / "sample_first_steps",
        edge_table=edge_table,
        steps=sample_steps,
        values=sample_values,
        summaries=[
            StepBetweennessSummary(
                step=int(step),
                source_group_id=int(args.source_group),
                target_group_id=int(args.target_group),
                source_nodes=int(state_summaries[int(state_ids[row])]["source_nodes"]),
                target_nodes=int(state_summaries[int(state_ids[row])]["target_nodes"]),
                reachable_pairs=int(state_summaries[int(state_ids[row])]["reachable_pairs"]),
                total_shortest_distance_hops=float(state_summaries[int(state_ids[row])]["total_shortest_distance_hops"]),
                mean_shortest_distance_hops=float(state_summaries[int(state_ids[row])]["mean_shortest_distance_hops"]),
                max_edge_betweenness=float(state_summaries[int(state_ids[row])]["max_edge_betweenness"]),
                nonzero_edges=int(state_summaries[int(state_ids[row])]["nonzero_edges"]),
            )
            for row, step in enumerate(sample_steps)
        ],
        path_samples=[],
        source_group_name=group_name(args.source_group),
        target_group_name=group_name(args.target_group),
    )

    meta = {
        "constellation": G60_CONFIG.name,
        "P": int(G60_CONFIG.P),
        "N": int(G60_CONFIG.N),
        "total_sats": int(G60_CONFIG.total_sats),
        "source_group_id": int(args.source_group),
        "source_group_name": group_name(args.source_group),
        "target_group_id": int(args.target_group),
        "target_group_name": group_name(args.target_group),
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "num_steps": int(len(steps)),
        "num_edges": int(edge_table.num_edges),
        "unique_states": int(len(unique_states)),
        "edge_betweenness_shape": [int(x) for x in values.shape],
        "value_min": float(np.nanmin(values)),
        "value_max": float(np.nanmax(values)),
        "counting_rule": "Each reachable source-target pair contributes 1 split fractionally across all equal-length shortest paths.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[full-betweenness] done | matrix_shape={values.shape} "
        f"value=({meta['value_min']:.4f}, {meta['value_max']:.4f})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
