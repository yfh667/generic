from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from group_edge_betweenness import StepBetweennessSummary, edge_betweenness_between_node_sets
from run_g60_dynamic_schedule_topology_viewer_120min import (
    DEFAULT_GROUP_CACHE,
    DEFAULT_SCHEDULE_DIR,
    DEFAULT_SELECTED_MOTIFS,
    DEFAULT_XML,
    DynamicScheduleTopologyViewer,
    build_active_mask,
    edge_key,
    load_motif_info,
    make_union_edge_table,
    set_background_grid_visible,
    write_topology_by_step,
)
from build_g60_selected_motifs_shortest_delay_parallel import edge_table_for_motif
from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from topology_edges import build_undirected_adjacency


DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_dynamic_schedule_120min_betweenness_viewer"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the G60 120-minute dynamic topology schedule with China-Europe edge betweenness overlay."
    )
    parser.add_argument("--schedule-dir", type=Path, default=DEFAULT_SCHEDULE_DIR)
    parser.add_argument("--by-step", type=Path, default=None)
    parser.add_argument("--segments", type=Path, default=None)
    parser.add_argument("--selected-motifs", type=Path, default=DEFAULT_SELECTED_MOTIFS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--source-group", type=int, default=2, help="Default G60 group 2 is China.")
    parser.add_argument("--target-group", type=int, default=3, help="Default G60 group 3 is Europe.")
    parser.add_argument("--sample-path-limit-per-state", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--edge-width-max", type=float, default=0.095)
    parser.add_argument("--value-alpha-min", type=int, default=18)
    parser.add_argument("--value-alpha-max", type=int, default=235)
    parser.add_argument("--hide-groups", action="store_true")
    parser.add_argument("--show-background-grid", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args(argv)


def group_name(group_id: int) -> str:
    return str(G60_CONFIG.station_groups.get(int(group_id), {}).get("name", f"Group {group_id}"))


def group_nodes_for_step(group_data: dict, step: int, group_id: int) -> tuple[int, ...]:
    current = group_data.get(int(step), {}) if group_data else {}
    groups = current.get("groups", {}) if isinstance(current, dict) else {}
    return tuple(sorted(int(x) for x in groups.get(int(group_id), set()) or set()))


def build_state_index(
    *,
    group_data: dict,
    steps: list[int],
    topology_by_row: list[str],
    source_group: int,
    target_group: int,
) -> tuple[list[tuple[str, tuple[int, ...], tuple[int, ...]]], np.ndarray]:
    state_to_id: dict[tuple[str, tuple[int, ...], tuple[int, ...]], int] = {}
    states: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    state_ids = np.empty(len(steps), dtype=np.int32)
    for row, step in enumerate(steps):
        key = (
            str(topology_by_row[row]),
            group_nodes_for_step(group_data, int(step), int(source_group)),
            group_nodes_for_step(group_data, int(step), int(target_group)),
        )
        state_id = state_to_id.get(key)
        if state_id is None:
            state_id = len(states)
            state_to_id[key] = state_id
            states.append(key)
        state_ids[row] = int(state_id)
    return states, state_ids


def union_index_map(union_edge_table: EdgeTable) -> dict[tuple[int, int], int]:
    return {
        edge_key(int(union_edge_table.src[idx]), int(union_edge_table.dst[idx])): int(idx)
        for idx in range(union_edge_table.num_edges)
    }


def topology_edge_to_union_indices(edge_table: EdgeTable, union_map: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.empty(int(edge_table.num_edges), dtype=np.int32)
    for idx in range(edge_table.num_edges):
        out[idx] = int(union_map[edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]))])
    return out


def load_cached_outputs(out_dir: Path, *, expected_shape: tuple[int, int]) -> tuple[np.ndarray, list[dict]] | None:
    values_path = out_dir / "edge_betweenness.npy"
    summary_path = out_dir / "step_summary.csv"
    if not values_path.exists() or not summary_path.exists():
        return None
    values = np.load(values_path, mmap_mode="r")
    if tuple(values.shape) != tuple(expected_shape):
        return None
    summaries = list(csv.DictReader(summary_path.open("r", encoding="utf-8-sig", newline="")))
    return np.asarray(values, dtype=np.float32), summaries


def write_step_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_or_load_betweenness(
    *,
    out_dir: Path,
    steps: list[int],
    topology_by_row: list[str],
    edge_tables: dict[str, EdgeTable],
    union_edge_table: EdgeTable,
    group_data: dict,
    source_group: int,
    target_group: int,
    force: bool,
    sample_path_limit_per_state: int,
    progress_every: int,
) -> tuple[np.ndarray, list[dict]]:
    expected_shape = (len(steps), int(union_edge_table.num_edges))
    if not force:
        cached = load_cached_outputs(out_dir, expected_shape=expected_shape)
        if cached is not None:
            print(f"[dynamic-betweenness] Reusing edge_betweenness.npy from {out_dir}", flush=True)
            return cached

    states, state_ids = build_state_index(
        group_data=group_data,
        steps=steps,
        topology_by_row=topology_by_row,
        source_group=int(source_group),
        target_group=int(target_group),
    )
    print(
        f"[dynamic-betweenness] unique states={len(states)} rows={len(steps)} "
        f"source={group_name(source_group)} target={group_name(target_group)}",
        flush=True,
    )

    union_map = union_index_map(union_edge_table)
    topology_to_union = {
        name: topology_edge_to_union_indices(table, union_map)
        for name, table in edge_tables.items()
    }
    topology_adjacency = {
        name: build_undirected_adjacency(table, int(G60_CONFIG.total_sats))
        for name, table in edge_tables.items()
    }

    state_values = np.zeros((len(states), int(union_edge_table.num_edges)), dtype=np.float32)
    state_summaries: list[dict] = []
    path_samples: list[dict] = []
    started = time.time()

    for state_id, (topology, source_nodes, target_nodes) in enumerate(states):
        edge_table = edge_tables[topology]
        row_values, summary, samples = edge_betweenness_between_node_sets(
            edge_table,
            total_nodes=int(G60_CONFIG.total_sats),
            source_nodes=set(source_nodes),
            target_nodes=set(target_nodes),
            adjacency=topology_adjacency[topology],
            sample_path_limit=int(sample_path_limit_per_state),
        )
        state_values[state_id, topology_to_union[topology]] = row_values
        state_summaries.append(
            {
                "state_id": int(state_id),
                "topology": topology,
                "source_nodes": int(summary["source_nodes"]),
                "target_nodes": int(summary["target_nodes"]),
                "reachable_pairs": int(summary["reachable_pairs"]),
                "total_shortest_distance_hops": float(summary["total_shortest_distance_hops"]),
                "mean_shortest_distance_hops": float(summary["mean_shortest_distance_hops"]),
                "max_edge_betweenness": float(summary["max_edge_betweenness"]),
                "nonzero_edges": int(summary["nonzero_edges"]),
                "edge_value_sum": float(summary["edge_value_sum"]),
            }
        )
        for item in samples:
            item = dict(item)
            item["state_id"] = int(state_id)
            item["topology"] = topology
            path_samples.append(item)

        if progress_every > 0 and ((state_id + 1) % int(progress_every) == 0 or state_id + 1 == len(states)):
            elapsed = time.time() - started
            print(
                f"[dynamic-betweenness] states {state_id + 1}/{len(states)} "
                f"elapsed={elapsed:.1f}s max_edge={summary['max_edge_betweenness']:.3f}",
                flush=True,
            )

    values = np.asarray(state_values[state_ids], dtype=np.float32)
    summary_by_step: list[dict] = []
    for row, step in enumerate(steps):
        state_summary = state_summaries[int(state_ids[row])]
        summary_by_step.append(
            {
                "step": int(step),
                "row": int(row),
                "state_id": int(state_ids[row]),
                "topology": topology_by_row[row],
                "source_group_id": int(source_group),
                "target_group_id": int(target_group),
                "source_nodes": int(state_summary["source_nodes"]),
                "target_nodes": int(state_summary["target_nodes"]),
                "reachable_pairs": int(state_summary["reachable_pairs"]),
                "total_shortest_distance_hops": float(state_summary["total_shortest_distance_hops"]),
                "mean_shortest_distance_hops": float(state_summary["mean_shortest_distance_hops"]),
                "max_edge_betweenness": float(state_summary["max_edge_betweenness"]),
                "nonzero_edges": int(state_summary["nonzero_edges"]),
                "edge_value_sum": float(state_summary["edge_value_sum"]),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "edge_betweenness.npy", values)
    np.save(out_dir / "unique_state_values.npy", state_values)
    np.save(out_dir / "state_ids.npy", state_ids)
    write_step_summary(out_dir / "step_summary.csv", summary_by_step)
    write_step_summary(out_dir / "state_summary.csv", state_summaries)

    if path_samples:
        with (out_dir / "path_samples.csv").open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["state_id", "topology", "source", "target", "distance_hops", "num_shortest_paths", "representative_path"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in path_samples:
                row = dict(item)
                row["representative_path"] = " ".join(str(x) for x in row.get("representative_path", []))
                writer.writerow(row)

    return values, summary_by_step


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schedule_dir = Path(args.schedule_dir)
    by_step_path = Path(args.by_step) if args.by_step else schedule_dir / "min_dwell_120min_by_step.csv"
    segments_path = Path(args.segments) if args.segments else schedule_dir / "min_dwell_120min_segments.csv"
    if not by_step_path.exists():
        raise FileNotFoundError(f"Missing schedule by-step CSV: {by_step_path}")
    if not segments_path.exists():
        raise FileNotFoundError(f"Missing schedule segment CSV: {segments_path}")

    by_step = pd.read_csv(by_step_path)
    steps = [int(x) for x in by_step["step"].tolist()]
    step_stride = int(steps[1] - steps[0]) if len(steps) > 1 else 1
    topology_by_row = [str(x) for x in by_step["topology"].tolist()]
    unique_topologies = list(dict.fromkeys(topology_by_row))
    motif_info = load_motif_info(Path(args.selected_motifs), segments_path)
    missing = [name for name in unique_topologies if name not in motif_info or not motif_info[name].get("motif")]
    if missing:
        raise ValueError(f"Missing motif definitions for: {missing}")

    edge_tables = {name: edge_table_for_motif(motif_info[name]["motif"]) for name in unique_topologies}
    union_edge_table, topology_keys = make_union_edge_table(edge_tables)
    active_mask = build_active_mask(
        union_edge_table=union_edge_table,
        topology_keys=topology_keys,
        topology_by_row=topology_by_row,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(union_edge_table, out_dir / "union_edges.csv")
    np.save(out_dir / "edge_active_mask.npy", active_mask)
    write_topology_by_step(out_dir / "topology_by_step.csv", by_step)
    pd.read_csv(segments_path).to_csv(out_dir / "topology_segments.csv", index=False)

    group_data = load_or_build_group_data(
        xml_file=Path(args.xml_file),
        group_cache_dir=Path(args.group_cache_dir),
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=int(G60_CONFIG.total_sats),
        constellation_name=G60_CONFIG.name,
        stride=int(step_stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )

    values, summaries = compute_or_load_betweenness(
        out_dir=out_dir,
        steps=steps,
        topology_by_row=topology_by_row,
        edge_tables=edge_tables,
        union_edge_table=union_edge_table,
        group_data=group_data,
        source_group=int(args.source_group),
        target_group=int(args.target_group),
        force=bool(args.force),
        sample_path_limit_per_state=int(args.sample_path_limit_per_state),
        progress_every=int(args.progress_every),
    )
    value_max = float(np.nanmax(values)) if values.size else 0.0

    meta = {
        "by_step": str(by_step_path),
        "segments": str(segments_path),
        "selected_motifs": str(args.selected_motifs),
        "steps": len(steps),
        "start_step": int(steps[0]),
        "end_step": int(steps[-1]),
        "step_stride": int(step_stride),
        "source_group": int(args.source_group),
        "target_group": int(args.target_group),
        "source_group_name": group_name(int(args.source_group)),
        "target_group_name": group_name(int(args.target_group)),
        "unique_topologies": unique_topologies,
        "union_edges": int(union_edge_table.num_edges),
        "active_edges_min": int(active_mask.sum(axis=1).min()),
        "active_edges_max": int(active_mask.sum(axis=1).max()),
        "edge_betweenness_shape": [int(x) for x in values.shape],
        "edge_betweenness_max": value_max,
        "edge_betweenness_nonzero_values": int(np.count_nonzero(values > 0.0)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[dynamic-betweenness] steps={len(steps)} topologies={unique_topologies} "
        f"union_edges={union_edge_table.num_edges} value_max={value_max:.3f} out_dir={out_dir}",
        flush=True,
    )

    if args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = DynamicScheduleTopologyViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=union_edge_table,
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
        topology_by_row=topology_by_row,
        motif_info=motif_info,
        window_title=(
            f"G60 dynamic topology schedule edge betweenness: "
            f"{group_name(args.source_group)}-{group_name(args.target_group)}"
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
