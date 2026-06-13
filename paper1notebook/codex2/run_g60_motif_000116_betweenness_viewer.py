from __future__ import annotations

import argparse
import csv
import json
import time
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_g60_selected_motifs_shortest_delay_parallel import edge_table_for_motif
from group_edge_betweenness import (
    edge_betweenness_between_node_sets,
)
from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import write_edges_csv
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from topology_edges import build_undirected_adjacency


MOTIF_ID = 116
MOTIF_TEXT = "AAA | DBD | --B"
DEFAULT_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE = PROJECT_ROOT / "data" / "linshi" / "cache" / "group_data_cache"

_WORKER_EDGE_TABLE = None
_WORKER_ADJACENCY = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize motif_000116 China-Europe edge betweenness with the original 2D viewer."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--source-group", type=int, default=2, help="Default G60 group 2 is China.")
    parser.add_argument("--target-group", type=int, default=3, help="Default G60 group 3 is Europe.")
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: E:/paper11/data/linshi/motif_000116_betweenness_viewer_t{start}_{end}_stride{stride}",
    )
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--edge-width-max", type=float, default=0.095)
    parser.add_argument("--value-alpha-min", type=int, default=18)
    parser.add_argument("--value-alpha-max", type=int, default=235)
    parser.add_argument("--sample-path-limit-per-step", type=int, default=40)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="Recompute even if the cache already exists.")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--hide-groups", action="store_true")
    parser.add_argument(
        "--show-background-grid",
        action="store_true",
        help="Show the viewer's pale background grid. By default it is hidden so grid lines are not mistaken for links.",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args(argv)


def default_out_dir(start: int, end: int, stride: int) -> Path:
    return (
        PROJECT_ROOT
        / "data"
        / "linshi"
        / f"motif_000116_betweenness_viewer_t{int(start)}_{int(end)}_stride{int(stride)}"
    )


def group_name(group_id: int) -> str:
    return str(G60_CONFIG.station_groups.get(int(group_id), {}).get("name", f"Group {group_id}"))


def set_background_grid_visible(viewer: SatelliteTopology2DViewer, visible: bool) -> None:
    """Hide only the QGraphicsLineItem background grid/axis lines; edge paths stay visible."""

    for item in viewer.scene.items():
        if isinstance(item, QtWidgets.QGraphicsLineItem):
            item.setVisible(bool(visible))


def load_meta(out_dir: Path) -> dict:
    meta_path = Path(out_dir) / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


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
            int(step),
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
    _WORKER_ADJACENCY = build_undirected_adjacency(edge_table, int(G60_CONFIG.total_sats))


def _compute_state_value(task: tuple[int, tuple[int, ...], tuple[int, ...]]):
    state_id, source_nodes, target_nodes = task
    values, summary, _samples = edge_betweenness_between_node_sets(
        _WORKER_EDGE_TABLE,
        total_nodes=int(G60_CONFIG.total_sats),
        source_nodes=set(int(x) for x in source_nodes),
        target_nodes=set(int(x) for x in target_nodes),
        adjacency=_WORKER_ADJACENCY,
        sample_path_limit=0,
    )
    return int(state_id), values, summary


def state_values_complete(out_dir: Path, *, num_states: int, num_edges: int) -> bool:
    values_path = Path(out_dir) / "unique_state_values.npy"
    summary_path = Path(out_dir) / "state_summary.csv"
    if not values_path.exists() or not summary_path.exists():
        return False
    values = np.load(values_path, mmap_mode="r")
    return tuple(values.shape) == (int(num_states), int(num_edges))


def write_state_definitions(path: Path, unique_states: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> None:
    payload = [
        {
            "state_id": int(idx),
            "source_nodes": [int(x) for x in source_nodes],
            "target_nodes": [int(x) for x in target_nodes],
        }
        for idx, (source_nodes, target_nodes) in enumerate(unique_states)
    ]
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_unique_state_values(
    *,
    out_dir: Path,
    edge_table,
    unique_states: list[tuple[tuple[int, ...], tuple[int, ...]]],
    workers: int,
    progress_every: int,
    force: bool,
) -> tuple[np.ndarray, list[dict]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values_path = out_dir / "unique_state_values.npy"
    summary_path = out_dir / "state_summary.csv"
    num_states = len(unique_states)
    num_edges = int(edge_table.num_edges)

    if not bool(force) and state_values_complete(out_dir, num_states=num_states, num_edges=num_edges):
        print(f"[motif-000116-betweenness] Reusing unique-state values: {values_path}", flush=True)
        values = np.load(values_path, mmap_mode="r")
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            summaries = list(csv.DictReader(f))
        return values, summaries

    values = open_memmap(values_path, mode="w+", dtype=np.float32, shape=(int(num_states), int(num_edges)))
    summaries: list[dict | None] = [None] * int(num_states)
    tasks = [(idx, source_nodes, target_nodes) for idx, (source_nodes, target_nodes) in enumerate(unique_states)]

    started = time.perf_counter()
    completed = 0
    print(
        f"[motif-000116-betweenness] Computing {num_states} unique group states "
        f"with {int(workers)} workers...",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_init_worker,
        initargs=(edge_table,),
    ) as executor:
        futures = [executor.submit(_compute_state_value, task) for task in tasks]
        for future in as_completed(futures):
            state_id, row_values, summary = future.result()
            values[int(state_id), :] = row_values
            summaries[int(state_id)] = {
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
            if completed == len(tasks) or completed % max(1, int(progress_every)) == 0:
                elapsed = time.perf_counter() - started
                rate = completed / max(1e-9, elapsed)
                eta = (len(tasks) - completed) / max(1e-9, rate)
                print(
                    f"[motif-000116-betweenness] states {completed}/{len(tasks)} | "
                    f"elapsed={elapsed:.1f}s | eta={eta:.1f}s",
                    flush=True,
                )

    values.flush()
    with summary_path.open("w", encoding="utf-8", newline="") as f:
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
        for row in summaries:
            if row is None:
                raise RuntimeError("missing state summary")
            writer.writerow(row)

    return values, [row for row in summaries if row is not None]


def write_full_time_matrix(
    *,
    out_dir: Path,
    state_values: np.ndarray,
    state_ids: np.ndarray,
    force: bool,
) -> np.ndarray:
    path = Path(out_dir) / "edge_betweenness.npy"
    expected_shape = (int(state_ids.size), int(state_values.shape[1]))
    if not bool(force) and path.exists():
        existing = np.load(path, mmap_mode="r")
        if tuple(existing.shape) == expected_shape:
            print(f"[motif-000116-betweenness] Reusing full matrix: {path}", flush=True)
            return existing

    values = open_memmap(path, mode="w+", dtype=np.float32, shape=expected_shape)
    started = time.perf_counter()
    for start in range(0, int(state_ids.size), 2048):
        end = min(start + 2048, int(state_ids.size))
        values[start:end, :] = state_values[state_ids[start:end], :]
        if end % 16384 == 0 or end == int(state_ids.size):
            elapsed = time.perf_counter() - started
            print(
                f"[motif-000116-betweenness] wrote rows {end}/{state_ids.size} | elapsed={elapsed:.1f}s",
                flush=True,
            )
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
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "step",
            "source_group_id",
            "target_group_id",
            "source_nodes",
            "target_nodes",
            "reachable_pairs",
            "total_shortest_distance_hops",
            "mean_shortest_distance_hops",
            "max_edge_betweenness",
            "nonzero_edges",
            "state_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, step in enumerate(steps):
            state_id = int(state_ids[row])
            summary = summary_by_state[state_id]
            writer.writerow(
                {
                    "step": int(step),
                    "source_group_id": int(source_group),
                    "target_group_id": int(target_group),
                    "source_nodes": int(summary["source_nodes"]),
                    "target_nodes": int(summary["target_nodes"]),
                    "reachable_pairs": int(summary["reachable_pairs"]),
                    "total_shortest_distance_hops": float(summary["total_shortest_distance_hops"]),
                    "mean_shortest_distance_hops": float(summary["mean_shortest_distance_hops"]),
                    "max_edge_betweenness": float(summary["max_edge_betweenness"]),
                    "nonzero_edges": int(summary["nonzero_edges"]),
                    "state_id": int(state_id),
                }
            )


def load_or_compute_values(args: argparse.Namespace, steps: list[int], edge_table):
    out_dir = Path(args.out_dir)
    values_path = out_dir / "edge_betweenness.npy"
    steps_path = out_dir / "time_indices.npy"
    expected_shape = (len(steps), int(edge_table.num_edges))

    if not bool(args.force) and values_path.exists() and steps_path.exists():
        cached_steps = [int(x) for x in np.load(steps_path)]
        cached_values = np.load(values_path, mmap_mode="r")
        if cached_steps == [int(x) for x in steps] and tuple(cached_values.shape) == expected_shape:
            print(f"[motif-000116-betweenness] Reusing values: {values_path}", flush=True)
            return cached_values, None, None, load_meta(out_dir)

    group_data = load_or_build_group_data(
        xml_file=Path(args.xml_file),
        group_cache_dir=Path(args.group_cache_dir),
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
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))
    np.save(out_dir / "state_ids.npy", state_ids)
    write_edges_csv(edge_table, out_dir / "edges.csv")
    write_state_definitions(out_dir / "state_definitions.json", unique_states)
    print(
        f"[motif-000116-betweenness] steps={len(steps)} edges={edge_table.num_edges} "
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
    meta = {
        "constellation": G60_CONFIG.name,
        "motif_id": MOTIF_ID,
        "motif": MOTIF_TEXT,
        "P": int(G60_CONFIG.P),
        "N": int(G60_CONFIG.N),
        "total_sats": int(G60_CONFIG.total_sats),
        "source_group_id": int(args.source_group),
        "source_group_name": group_name(args.source_group),
        "target_group_id": int(args.target_group),
        "target_group_name": group_name(args.target_group),
        "start": int(steps[0]),
        "end": int(steps[-1]),
        "stride": int(args.stride),
        "num_steps": int(len(steps)),
        "num_edges": int(edge_table.num_edges),
        "unique_states": int(len(unique_states)),
        "edge_betweenness_shape": [int(x) for x in values.shape],
        "value_min": float(np.nanmin(state_values)),
        "value_max": float(np.nanmax(state_values)),
        "counting_rule": "Each reachable source-target pair contributes 1 split fractionally across all equal-length shortest paths.",
        "cache_rule": "Only unique China-Europe group states are computed; per-step matrix is rebuilt by state_ids.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return values, group_data, None, meta


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty step range")
    if args.out_dir is None:
        args.out_dir = default_out_dir(int(args.start), int(args.end), int(args.stride))

    edge_table = edge_table_for_motif(MOTIF_TEXT)
    values, group_data, summaries, meta = load_or_compute_values(args, steps, edge_table)
    if group_data is None and not bool(args.hide_groups):
        group_data = load_or_build_group_data(
            xml_file=Path(args.xml_file),
            group_cache_dir=Path(args.group_cache_dir),
            steps=steps,
            station_groups=G60_CONFIG.station_groups,
            total_sats=G60_CONFIG.total_sats,
            constellation_name=G60_CONFIG.name,
            stride=int(args.stride),
            enabled=True,
            force=False,
        )
    group_data = {} if bool(args.hide_groups) else (group_data or {})

    value_max = float(meta.get("value_max", np.nanmax(values) if np.asarray(values).size else 0.0))
    nonzero = int(meta.get("nonzero_values", -1))
    nonzero_text = str(nonzero) if nonzero >= 0 else "not_scanned"
    print(
        f"[motif-000116-betweenness] motif={MOTIF_TEXT!r} steps={len(steps)} "
        f"range={steps[0]}..{steps[-1]} edges={edge_table.num_edges} "
        f"value=(0.0000,{value_max:.4f}) nonzero_values={nonzero_text} "
        f"group_steps={len(group_data)} out_dir={args.out_dir}",
        flush=True,
    )
    if summaries:
        for summary in summaries[: min(3, len(summaries))]:
            print(
                f"[motif-000116-betweenness] step={summary.step} "
                f"pairs={summary.reachable_pairs} mean_hops={summary.mean_shortest_distance_hops:.3f} "
                f"max_edge={summary.max_edge_betweenness:.3f} nonzero_edges={summary.nonzero_edges}",
                flush=True,
            )

    if args.check_only:
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
            f"G60 motif_000116 {group_name(args.source_group)}-{group_name(args.target_group)} "
            f"edge betweenness {steps[0]}..{steps[-1]}s"
        ),
        group_data=group_data,
        show_groups=not bool(args.hide_groups),
    )
    set_background_grid_visible(viewer, bool(args.show_background_grid))
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
