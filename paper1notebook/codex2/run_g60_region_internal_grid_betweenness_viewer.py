from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from group_edge_betweenness import edge_betweenness_between_node_sets
from run_g60_china_europe_internal_grid_dynamic_viewer import (
    DEFAULT_GROUP_CACHE,
    DEFAULT_XML,
    default_out_dir as default_topology_out_dir,
    group_name,
    group_signature,
)
from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from topology_edges import build_full_option_plus_intra_edges, build_undirected_adjacency


DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "linshi"
DEFAULT_SOURCE_GROUP = 2
DEFAULT_TARGET_GROUP = 3

_WORKER: dict[str, Any] = {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute/open China-Europe shortest-path edge betweenness on the dynamic "
            "side-aware region-internal +grid topology."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86164, help="Inclusive end step.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--topology-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--constrained-groups", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--source-group", type=int, default=DEFAULT_SOURCE_GROUP)
    parser.add_argument("--target-group", type=int, default=DEFAULT_TARGET_GROUP)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--sample-path-limit-per-state", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--hide-groups", action="store_true")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--edge-width-max", type=float, default=0.095)
    parser.add_argument("--value-alpha-min", type=int, default=18)
    parser.add_argument("--value-alpha-max", type=int, default=235)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args(argv)


def default_out_dir(start: int, end: int, stride: int, source_group: int, target_group: int) -> Path:
    return (
        DEFAULT_OUT_ROOT
        / (
            f"g60_region_internal_grid_betweenness_"
            f"g{int(source_group)}_g{int(target_group)}_t{int(start)}_{int(end)}_stride{int(stride)}"
        )
    )


def auto_workers(requested: int) -> int:
    if int(requested) > 0:
        return int(requested)
    count = os.cpu_count() or 4
    return max(1, min(8, int(count) - 2))


def edge_table_subset(edge_table: EdgeTable, indices: np.ndarray) -> EdgeTable:
    indices = np.asarray(indices, dtype=np.int64)
    return EdgeTable(
        src=np.asarray(edge_table.src[indices], dtype=np.int32),
        dst=np.asarray(edge_table.dst[indices], dtype=np.int32),
        option=np.asarray(edge_table.option[indices], dtype=np.int16),
        src_plane=np.asarray(edge_table.src_plane[indices], dtype=np.int16),
        src_y=np.asarray(edge_table.src_y[indices], dtype=np.int16),
        dst_plane=np.asarray(edge_table.dst_plane[indices], dtype=np.int16),
        dst_y=np.asarray(edge_table.dst_y[indices], dtype=np.int16),
        sat_ids=list(edge_table.sat_ids),
    )


def state_key_for_step(
    *,
    group_data: dict,
    step: int,
    constrained_groups: list[int],
) -> tuple[tuple[int, ...], ...]:
    return group_signature(group_data, int(step), constrained_groups)


def source_target_nodes_from_signature(
    signature: tuple[tuple[int, ...], ...],
    *,
    constrained_groups: list[int],
    source_group: int,
    target_group: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    group_to_nodes = {int(gid): tuple(signature[pos]) for pos, gid in enumerate(constrained_groups)}
    return tuple(group_to_nodes.get(int(source_group), ())), tuple(group_to_nodes.get(int(target_group), ()))


def build_state_index(
    *,
    steps: list[int],
    group_data: dict,
    constrained_groups: list[int],
    source_group: int,
    target_group: int,
) -> tuple[list[dict], np.ndarray]:
    state_to_id: dict[tuple[tuple[int, ...], ...], int] = {}
    states: list[dict] = []
    state_ids = np.empty(len(steps), dtype=np.int32)
    for row, step in enumerate(steps):
        signature = state_key_for_step(
            group_data=group_data,
            step=int(step),
            constrained_groups=constrained_groups,
        )
        state_id = state_to_id.get(signature)
        if state_id is None:
            source_nodes, target_nodes = source_target_nodes_from_signature(
                signature,
                constrained_groups=constrained_groups,
                source_group=int(source_group),
                target_group=int(target_group),
            )
            state_id = len(states)
            state_to_id[signature] = state_id
            states.append(
                {
                    "state_id": int(state_id),
                    "first_row": int(row),
                    "source_nodes": source_nodes,
                    "target_nodes": target_nodes,
                    "group_node_counts": ";".join(str(len(nodes)) for nodes in signature),
                }
            )
        state_ids[row] = int(state_id)
    return states, state_ids


def init_worker(edge_table: EdgeTable, active_mask_path: str, total_nodes: int) -> None:
    _WORKER.clear()
    _WORKER["edge_table"] = edge_table
    _WORKER["active_mask"] = np.load(active_mask_path, mmap_mode="r")
    _WORKER["total_nodes"] = int(total_nodes)


def compute_state_chunk(task: tuple[int, list[dict], int]) -> dict:
    chunk_id, states, sample_path_limit = task
    edge_table: EdgeTable = _WORKER["edge_table"]
    active_mask = _WORKER["active_mask"]
    total_nodes = int(_WORKER["total_nodes"])
    num_edges = int(edge_table.num_edges)

    values = np.zeros((len(states), num_edges), dtype=np.float32)
    summaries: list[dict] = []
    samples: list[dict] = []

    for local_row, state in enumerate(states):
        state_id = int(state["state_id"])
        first_row = int(state["first_row"])
        active_indices = np.flatnonzero(np.asarray(active_mask[first_row], dtype=bool)).astype(np.int32)
        state_edge_table = edge_table_subset(edge_table, active_indices)
        adjacency = build_undirected_adjacency(state_edge_table, total_nodes)
        row_values, summary, path_samples = edge_betweenness_between_node_sets(
            state_edge_table,
            total_nodes=total_nodes,
            source_nodes=set(int(x) for x in state["source_nodes"]),
            target_nodes=set(int(x) for x in state["target_nodes"]),
            adjacency=adjacency,
            sample_path_limit=int(sample_path_limit),
        )
        values[local_row, active_indices] = row_values
        summaries.append(
            {
                "state_id": int(state_id),
                "first_row": int(first_row),
                "source_nodes": int(summary["source_nodes"]),
                "target_nodes": int(summary["target_nodes"]),
                "reachable_pairs": int(summary["reachable_pairs"]),
                "total_shortest_distance_hops": float(summary["total_shortest_distance_hops"]),
                "mean_shortest_distance_hops": float(summary["mean_shortest_distance_hops"]),
                "max_edge_betweenness": float(summary["max_edge_betweenness"]),
                "nonzero_edges": int(summary["nonzero_edges"]),
                "edge_value_sum": float(summary["edge_value_sum"]),
                "active_edges": int(active_indices.size),
                "group_node_counts": str(state.get("group_node_counts", "")),
            }
        )
        for item in path_samples:
            item = dict(item)
            item["state_id"] = int(state_id)
            item["first_row"] = int(first_row)
            samples.append(item)

    return {
        "chunk_id": int(chunk_id),
        "states": [int(state["state_id"]) for state in states],
        "values": values,
        "summaries": summaries,
        "samples": samples,
    }


def chunk_states(states: list[dict], chunk_size: int) -> list[tuple[int, list[dict], int]]:
    size = max(1, int(chunk_size))
    return [(chunk_id, states[start : start + size], 0) for chunk_id, start in enumerate(range(0, len(states), size))]


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cached_outputs_valid(out_dir: Path, *, expected_shape: tuple[int, int], expected_state_count: int) -> bool:
    values_path = out_dir / "edge_betweenness.npy"
    unique_path = out_dir / "unique_state_values.npy"
    state_ids_path = out_dir / "state_ids.npy"
    meta_path = out_dir / "meta.json"
    if not (values_path.exists() and unique_path.exists() and state_ids_path.exists() and meta_path.exists()):
        return False
    try:
        values = np.load(values_path, mmap_mode="r")
        unique = np.load(unique_path, mmap_mode="r")
        state_ids = np.load(state_ids_path, mmap_mode="r")
        return (
            tuple(values.shape) == tuple(expected_shape)
            and int(unique.shape[0]) == int(expected_state_count)
            and int(unique.shape[1]) == int(expected_shape[1])
            and int(state_ids.shape[0]) == int(expected_shape[0])
        )
    except Exception:
        return False


def expand_values_by_state(
    *,
    unique_values: np.ndarray,
    state_ids: np.ndarray,
    out_path: Path,
    force: bool,
) -> np.ndarray:
    expected_shape = (int(state_ids.shape[0]), int(unique_values.shape[1]))
    if not force and out_path.exists():
        existing = np.load(out_path, mmap_mode="r")
        if tuple(existing.shape) == expected_shape:
            return existing

    values = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=expected_shape)
    for start in range(0, int(state_ids.shape[0]), 1024):
        end = min(start + 1024, int(state_ids.shape[0]))
        values[start:end, :] = unique_values[np.asarray(state_ids[start:end], dtype=np.int64), :]
    values.flush()
    return np.load(out_path, mmap_mode="r")


def compute_or_load_betweenness(
    *,
    out_dir: Path,
    topology_dir: Path,
    steps: list[int],
    edge_table: EdgeTable,
    active_mask_path: Path,
    group_data: dict,
    constrained_groups: list[int],
    source_group: int,
    target_group: int,
    max_workers: int,
    chunk_size: int,
    sample_path_limit_per_state: int,
    progress_every: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    states, state_ids = build_state_index(
        steps=steps,
        group_data=group_data,
        constrained_groups=constrained_groups,
        source_group=int(source_group),
        target_group=int(target_group),
    )
    expected_shape = (len(steps), int(edge_table.num_edges))
    if not force and cached_outputs_valid(out_dir, expected_shape=expected_shape, expected_state_count=len(states)):
        values = np.load(out_dir / "edge_betweenness.npy", mmap_mode="r")
        unique_values = np.load(out_dir / "unique_state_values.npy", mmap_mode="r")
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        print(f"[region-grid-betweenness] Reusing cache: {out_dir}", flush=True)
        return values, unique_values, meta

    out_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(edge_table, out_dir / "edges.csv")
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))
    np.save(out_dir / "state_ids.npy", state_ids)

    unique_values = np.lib.format.open_memmap(
        out_dir / "unique_state_values.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(states), int(edge_table.num_edges)),
    )
    unique_values[:, :] = 0.0

    workers = auto_workers(int(max_workers))
    tasks = chunk_states(states, int(chunk_size))
    tasks = [(chunk_id, chunk, int(sample_path_limit_per_state)) for chunk_id, chunk, _sample in tasks]
    summaries: list[dict] = []
    samples: list[dict] = []
    started = time.perf_counter()
    completed = 0

    print(
        f"[region-grid-betweenness] states={len(states)} rows={len(steps)} edges={edge_table.num_edges} "
        f"workers={workers} chunk_size={int(chunk_size)} source={group_name(source_group)} "
        f"target={group_name(target_group)}",
        flush=True,
    )

    ctx = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(edge_table, str(active_mask_path), int(G60_CONFIG.total_sats)),
    ) as executor:
        futures = [executor.submit(compute_state_chunk, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            state_ids_in_chunk = np.asarray(result["states"], dtype=np.int64)
            unique_values[state_ids_in_chunk, :] = np.asarray(result["values"], dtype=np.float32)
            summaries.extend(result["summaries"])
            samples.extend(result["samples"])
            completed += 1
            if completed % max(1, int(progress_every)) == 0 or completed == len(futures):
                elapsed = time.perf_counter() - started
                rate = completed / max(1e-9, elapsed)
                eta = (len(futures) - completed) / max(1e-9, rate)
                done_states = min(completed * int(chunk_size), len(states))
                print(
                    f"[region-grid-betweenness] chunks {completed}/{len(futures)} "
                    f"states~{done_states}/{len(states)} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

    unique_values.flush()
    summaries.sort(key=lambda row: int(row["state_id"]))
    write_csv_rows(out_dir / "state_summary.csv", summaries)
    if samples:
        for row in samples:
            row["representative_path"] = " ".join(str(x) for x in row.get("representative_path", []))
        write_csv_rows(out_dir / "path_samples.csv", samples)

    step_rows: list[dict] = []
    summary_by_state = {int(row["state_id"]): row for row in summaries}
    for row, step in enumerate(steps):
        summary = summary_by_state[int(state_ids[row])]
        if row < 200 or row == len(steps) - 1:
            step_rows.append(
                {
                    "row": int(row),
                    "step": int(step),
                    "state_id": int(state_ids[row]),
                    "source_nodes": int(summary["source_nodes"]),
                    "target_nodes": int(summary["target_nodes"]),
                    "reachable_pairs": int(summary["reachable_pairs"]),
                    "mean_shortest_distance_hops": float(summary["mean_shortest_distance_hops"]),
                    "max_edge_betweenness": float(summary["max_edge_betweenness"]),
                    "nonzero_edges": int(summary["nonzero_edges"]),
                    "active_edges": int(summary["active_edges"]),
                }
            )
    write_csv_rows(out_dir / "sample_step_summary.csv", step_rows)

    unique_values_ro = np.load(out_dir / "unique_state_values.npy", mmap_mode="r")
    values = expand_values_by_state(
        unique_values=unique_values_ro,
        state_ids=state_ids,
        out_path=out_dir / "edge_betweenness.npy",
        force=True,
    )
    value_max = float(np.nanmax(unique_values_ro)) if unique_values_ro.size else 0.0
    nonzero_unique = int(np.count_nonzero(unique_values_ro > 0.0))
    meta = {
        "topology_dir": str(topology_dir),
        "active_mask": str(active_mask_path),
        "constellation": G60_CONFIG.name,
        "start": int(steps[0]),
        "end": int(steps[-1]),
        "stride": int(steps[1] - steps[0]) if len(steps) > 1 else 1,
        "num_steps": int(len(steps)),
        "num_edges": int(edge_table.num_edges),
        "unique_states": int(len(states)),
        "source_group": int(source_group),
        "target_group": int(target_group),
        "source_group_name": group_name(int(source_group)),
        "target_group_name": group_name(int(target_group)),
        "edge_betweenness_shape": [int(x) for x in values.shape],
        "unique_state_values_shape": [int(x) for x in unique_values_ro.shape],
        "value_unit": "fractional_shortest_path_count",
        "value_min": 0.0,
        "value_max": value_max,
        "nonzero_unique_state_values": int(nonzero_unique),
        "counting_rule": (
            "For each reachable China-Europe source-target pair, contribution is split fractionally "
            "across all equal-hop shortest paths."
        ),
        "outputs": {
            "edge_betweenness": str(out_dir / "edge_betweenness.npy"),
            "unique_state_values": str(out_dir / "unique_state_values.npy"),
            "state_ids": str(out_dir / "state_ids.npy"),
            "time_indices": str(out_dir / "time_indices.npy"),
            "edges_csv": str(out_dir / "edges.csv"),
            "state_summary": str(out_dir / "state_summary.csv"),
            "sample_step_summary": str(out_dir / "sample_step_summary.csv"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return values, unique_values_ro, meta


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.stride) <= 0:
        raise ValueError("stride must be positive")
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty step range")

    constrained_groups = [int(x) for x in args.constrained_groups]
    topology_dir = Path(args.topology_dir) if args.topology_dir else default_topology_out_dir(
        int(args.start),
        int(args.end),
        int(args.stride),
        constrained_groups,
    )
    active_mask_path = topology_dir / "edge_active_mask.npy"
    if not active_mask_path.exists():
        raise FileNotFoundError(
            f"Missing dynamic topology active mask: {active_mask_path}. "
            "Run run_g60_china_europe_internal_grid_dynamic_viewer.py first."
        )

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir(
        int(args.start),
        int(args.end),
        int(args.stride),
        int(args.source_group),
        int(args.target_group),
    )

    edge_table = build_full_option_plus_intra_edges(G60_CONFIG)
    active_mask = np.load(active_mask_path, mmap_mode="r")
    expected_active_shape = (len(steps), int(edge_table.num_edges))
    if tuple(active_mask.shape) != expected_active_shape:
        raise ValueError(f"active mask shape {active_mask.shape} != expected {expected_active_shape}")

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

    values, _unique_values, meta = compute_or_load_betweenness(
        out_dir=out_dir,
        topology_dir=topology_dir,
        steps=steps,
        edge_table=edge_table,
        active_mask_path=active_mask_path,
        group_data=group_data,
        constrained_groups=constrained_groups,
        source_group=int(args.source_group),
        target_group=int(args.target_group),
        max_workers=int(args.max_workers),
        chunk_size=int(args.chunk_size),
        sample_path_limit_per_state=int(args.sample_path_limit_per_state),
        progress_every=int(args.progress_every),
        force=bool(args.force),
    )
    value_max = float(meta.get("value_max", 0.0))
    print(
        f"[region-grid-betweenness] ready rows={len(steps)} edges={edge_table.num_edges} "
        f"unique_states={meta['unique_states']} value_max={value_max:.3f} out={out_dir}",
        flush=True,
    )

    if args.build_only or args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=active_mask,
        edge_values=values,
        value_min=0.0,
        value_max=value_max,
        edge_value_label="edge_betweenness",
        scale_edge_width_by_value=True,
        value_width_min=0.006,
        value_width_max=float(args.edge_width_max),
        value_color_mode="red_alpha",
        value_solid_color="#C1121F",
        value_alpha_min=int(args.value_alpha_min),
        value_alpha_max=int(args.value_alpha_max),
        zero_value_edges_visible=False,
        zero_value_threshold=0.0,
        show_topology_under_edge_values=True,
        topology_edge_color="#000000",
        topology_edge_alpha=155,
        topology_edge_width=0.014,
        window_title=(
            f"G60 region-internal +grid edge betweenness: "
            f"{group_name(args.source_group)}-{group_name(args.target_group)}"
        ),
        group_data={} if args.hide_groups else group_data,
        show_groups=not bool(args.hide_groups),
    )
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
