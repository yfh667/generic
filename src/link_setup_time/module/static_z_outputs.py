from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.link_delay.module.edge_options import EdgeTable, write_edges_csv

from .static_z_switch import ChainPlan, NodeTransition, chain_summary, plan_row_to_dict, transition_to_dict


def edge_key(src: int, dst: int) -> tuple[int, int]:
    a, b = int(src), int(dst)
    return (a, b) if a <= b else (b, a)


def build_edge_key_index(edge_table: EdgeTable) -> dict[tuple[int, int], int]:
    return {
        edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])): int(idx)
        for idx in range(int(edge_table.num_edges))
    }


def write_dict_rows_csv(path: str | Path, rows: Sequence[Mapping]) -> None:
    """Write heterogeneous dict rows while preserving first-seen field order."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_edge_state_masks_from_transitions(
    *,
    steps: Sequence[int] | np.ndarray,
    edge_table: EdgeTable,
    initial_active_edges: Iterable[tuple[int, int]],
    transitions: Sequence[NodeTransition],
    always_active_edges: Iterable[tuple[int, int]] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize active/building edge masks from static-z node transitions."""

    steps_array = np.asarray(steps, dtype=np.int64)
    if steps_array.ndim != 1 or steps_array.size == 0:
        raise ValueError("steps must be a non-empty 1-D sequence")

    key_to_idx = build_edge_key_index(edge_table)
    rows = int(steps_array.size)
    edges = int(edge_table.num_edges)
    active = np.zeros((rows, edges), dtype=bool)
    building = np.zeros((rows, edges), dtype=bool)

    for src, dst in always_active_edges:
        col = key_to_idx.get(edge_key(int(src), int(dst)))
        if col is not None:
            active[:, int(col)] = True
    for src, dst in initial_active_edges:
        col = key_to_idx.get(edge_key(int(src), int(dst)))
        if col is not None:
            active[:, int(col)] = True

    ordered = sorted(
        transitions,
        key=lambda item: (
            10**30 if item.start is None else int(item.start),
            int(item.switch_index),
            int(item.owner),
        ),
    )
    for transition in ordered:
        if transition.start is None or transition.ready is None:
            continue
        start_row = int(np.searchsorted(steps_array, int(transition.start), side="left"))
        ready_row = int(np.searchsorted(steps_array, int(transition.ready), side="left"))
        old_release_time = transition.old_release if transition.old_release is not None else transition.start
        release_row = int(np.searchsorted(steps_array, int(old_release_time), side="left"))
        start_row = max(0, min(rows, start_row))
        ready_row = max(0, min(rows, ready_row))
        release_row = max(0, min(rows, release_row))
        release_row = min(release_row, start_row)

        if transition.old_right is not None:
            old_col = key_to_idx.get(edge_key(int(transition.owner), int(transition.old_right)))
            if old_col is not None:
                active[release_row:, int(old_col)] = False

        if transition.new_right is None:
            continue

        new_col = key_to_idx.get(edge_key(int(transition.owner), int(transition.new_right)))
        if new_col is None:
            continue
        if ready_row > start_row:
            building[start_row:ready_row, int(new_col)] = True
        active[ready_row:, int(new_col)] = True
        building[ready_row:, int(new_col)] = False

    return active, building


def write_static_z_chain_plan(
    *,
    out_dir: str | Path,
    chain: ChainPlan,
    steps: Sequence[int] | np.ndarray,
    edge_table: EdgeTable | None = None,
    edge_active_mask: np.ndarray | None = None,
    edge_building_mask: np.ndarray | None = None,
    edge_values: np.ndarray | None = None,
    summary_extra: Mapping | None = None,
) -> dict:
    """Write common static-z chain artifacts and return their paths."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps_array = np.asarray(steps, dtype=np.int64)
    np.save(out / "steps.npy", steps_array)

    files: dict[str, str] = {"steps": "steps.npy"}
    if edge_table is not None:
        write_edges_csv(edge_table, out / "union_edges.csv")
        files["union_edges"] = "union_edges.csv"
    if edge_active_mask is not None:
        np.save(out / "edge_active_mask.npy", np.asarray(edge_active_mask, dtype=bool))
        files["edge_active_mask"] = "edge_active_mask.npy"
    if edge_building_mask is not None:
        np.save(out / "edge_building_mask.npy", np.asarray(edge_building_mask, dtype=bool))
        files["edge_building_mask"] = "edge_building_mask.npy"
    if edge_values is not None:
        np.save(out / "edge_usage_values.npy", np.asarray(edge_values, dtype=np.float32))
        files["edge_usage_values"] = "edge_usage_values.npy"

    for switch in chain.switches:
        rows = [plan_row_to_dict(row) for row in switch.rows]
        for row in rows:
            row["switch_index"] = int(switch.index)
            row["from"] = switch.from_name
            row["to"] = switch.to_name
        filename = f"switch_{int(switch.index):03d}_{switch.from_name}_to_{switch.to_name}_plan.csv"
        write_dict_rows_csv(out / filename, rows)
        files[f"switch_{int(switch.index):03d}_plan"] = filename

    write_dict_rows_csv(out / "transitions.csv", [transition_to_dict(item) for item in chain.transitions])
    files["transitions"] = "transitions.csv"

    if edge_building_mask is not None:
        building_counts = np.sum(np.asarray(edge_building_mask, dtype=bool), axis=1).astype(np.int32)
        np.save(out / "building_concurrency.npy", building_counts)
        with (out / "building_concurrency_by_step.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "hour", "building_count"])
            for step, count in zip(steps_array, building_counts):
                writer.writerow([int(step), f"{int(step) / 3600.0:.6f}", int(count)])
        files["building_concurrency"] = "building_concurrency_by_step.csv"

    summary = {
        "method": "static_z_chain",
        "initial": chain.initial,
        "final": chain.final,
        "steps": {
            "start": int(steps_array[0]) if steps_array.size else None,
            "end": int(steps_array[-1]) if steps_array.size else None,
            "count": int(steps_array.size),
        },
        "switches": chain_summary(chain),
        "warnings": list(chain.warnings),
        "files": files,
    }
    if summary_extra:
        summary.update(dict(summary_extra))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: str(out / value) for key, value in files.items()} | {"summary": str(out / "summary.json")}
