from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.link_delay.module.position_cache import open_position_cache_for_interval
from src.link_delay.module.query import open_delay_store_for_interval
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_metrics.module.group_states import group_nodes_for_step
from src.topology_workflow.module.edge_tables import build_motif_text_edge_table, make_edge_table_from_records
from src.topology_workflow.module.hybrid_edges import build_y_band_hybrid_edge_table, cyclic_rows
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step

from .config_io import load_yaml_dict, optional_path
from .full_link_gap_selector import _read_topology_library, _viewer_config_from_raw


@dataclass(frozen=True)
class DynamicScheduleSeries:
    steps: np.ndarray
    topology_names: tuple[str, ...]
    selected_action: np.ndarray
    selected_topology_names: tuple[str, ...]
    edge_table: EdgeTable
    target_mask: np.ndarray
    setup_command_mask: np.ndarray
    active_mask: np.ndarray
    building_mask: np.ndarray


def _edge_key(src: int, dst: int) -> tuple[int, int]:
    return (int(src), int(dst)) if int(src) <= int(dst) else (int(dst), int(src))


def _records_from_edge_table(edge_table: EdgeTable) -> list[tuple[int, int, int, int, int]]:
    return [
        (
            int(edge_table.src_plane[idx]),
            int(edge_table.src_y[idx]),
            int(edge_table.dst_plane[idx]),
            int(edge_table.dst_y[idx]),
            int(edge_table.option[idx]),
        )
        for idx in range(int(edge_table.num_edges))
    ]


def _edge_table_for_topology_name(
    *,
    name: str,
    config: ViewerConfig,
    topology_library: Mapping[str, Mapping[str, Any]],
    extra_topology_tables: Mapping[str, EdgeTable] | None = None,
) -> EdgeTable:
    if extra_topology_tables and str(name) in extra_topology_tables:
        return extra_topology_tables[str(name)]
    if str(name) == "gridplus":
        return build_motif_text_edge_table(motif_text="A", config=config, add_intra_ring=True)
    row = topology_library.get(str(name))
    if row is None:
        raise KeyError(f"topology {name!r} not found in topology library")
    motif = str(row.get("motif", ""))
    if not motif:
        raise ValueError(f"topology {name!r} has no motif text")
    return build_motif_text_edge_table(motif_text=motif, config=config, add_intra_ring=True)


def _edge_table_from_hybrid_meta(
    *,
    name: str,
    config: ViewerConfig,
    topology_library: Mapping[str, Mapping[str, Any]],
    candidate_meta: Mapping[str, Any],
) -> EdgeTable:
    """Rebuild a y-band hybrid edge table from recorded candidate metadata."""

    base_name = str(candidate_meta.get("base", ""))
    patch_name = str(candidate_meta.get("patch", ""))
    if not base_name or not patch_name:
        raise ValueError(f"hybrid topology {name!r} meta must contain base and patch")
    base_table = _edge_table_for_topology_name(
        name=base_name,
        config=config,
        topology_library=topology_library,
    )
    patch_table = _edge_table_for_topology_name(
        name=patch_name,
        config=config,
        topology_library=topology_library,
    )
    raw_rows = candidate_meta.get("band_rows")
    if raw_rows not in (None, ""):
        rows = [int(float(x)) for x in str(raw_rows).split()]
    else:
        start = int(candidate_meta["band_start"])
        end = int(candidate_meta["band_end"])
        rows = list(cyclic_rows(start, end, n=int(config.N)))
    result = build_y_band_hybrid_edge_table(
        base_edge_table=base_table,
        patch_edge_table=patch_table,
        p=int(config.P),
        n=int(config.N),
        rows=rows,
        patch_options=None,
        remove_base_conflicts=True,
        fail_on_degree_violation=True,
    )
    return result.edge_table


def edge_tables_from_hybrid_meta(
    *,
    config: ViewerConfig,
    topology_library: Mapping[str, Mapping[str, Any]],
    hybrid_meta_by_name: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, EdgeTable]:
    """Build extra topology tables, typically for local hybrid schedules."""

    out: dict[str, EdgeTable] = {}
    for name, meta in (hybrid_meta_by_name or {}).items():
        out[str(name)] = _edge_table_from_hybrid_meta(
            name=str(name),
            config=config,
            topology_library=topology_library,
            candidate_meta=meta,
        )
    return out


def build_union_edge_table_for_schedule(
    *,
    config: ViewerConfig,
    topology_names: Sequence[str],
    selected_action: np.ndarray,
    topology_library_csv: str | Path,
    extra_topology_tables: Mapping[str, EdgeTable] | None = None,
) -> tuple[EdgeTable, dict[int, EdgeTable]]:
    topology_library = _read_topology_library(topology_library_csv)
    unique_actions = sorted(set(int(x) for x in np.asarray(selected_action, dtype=np.int32).tolist()))
    tables_by_action: dict[int, EdgeTable] = {}
    records: list[tuple[int, int, int, int, int]] = []
    for action in unique_actions:
        name = str(topology_names[int(action)])
        edge_table = _edge_table_for_topology_name(
            name=name,
            config=config,
            topology_library=topology_library,
            extra_topology_tables=extra_topology_tables,
        )
        tables_by_action[int(action)] = edge_table
        records.extend(_records_from_edge_table(edge_table))
    union_table = make_edge_table_from_records(p=int(config.P), n=int(config.N), records=records)
    return union_table, tables_by_action


def _read_by_step_topology_sequence(
    path: str | Path,
    *,
    step_column: str = "step",
    topology_column: str = "topology",
) -> tuple[np.ndarray, tuple[str, ...]]:
    rows: list[tuple[int, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if step_column not in row or topology_column not in row:
                raise ValueError(f"{path} must contain {step_column!r} and {topology_column!r} columns")
            rows.append((int(float(row[step_column])), str(row[topology_column])))
    if not rows:
        raise ValueError(f"empty by-step topology schedule: {path}")
    return (
        np.asarray([item[0] for item in rows], dtype=np.int64),
        tuple(item[1] for item in rows),
    )


def load_schedule_series_from_by_step_csv(
    *,
    by_step_csv: str | Path,
    selector_dir: str | Path,
    setup_time_seconds: int | float,
    setup_mode: str = "break_before_make",
    setup_timing: str = "reactive",
    hybrid_meta_by_name: Mapping[str, Mapping[str, Any]] | None = None,
) -> DynamicScheduleSeries:
    """Load a topology schedule from a CSV with ``step`` and ``topology``.

    This is the generic entry for local replacement or learned policies that do
    not exist as an action in an original selector array. Region groups are not
    transformed here; they are used later only by metric evaluation.
    """

    selector_dir = Path(selector_dir)
    raw_config = load_yaml_dict(selector_dir / "selector_config.yaml")
    config = _viewer_config_from_raw(raw_config)
    topology_library_csv = optional_path(raw_config.get("inputs", {}).get("topology_library_csv"))
    if topology_library_csv is None:
        raise ValueError("selector_config.yaml inputs.topology_library_csv is required")
    topology_library = _read_topology_library(topology_library_csv)
    extra_tables = edge_tables_from_hybrid_meta(
        config=config,
        topology_library=topology_library,
        hybrid_meta_by_name=hybrid_meta_by_name,
    )

    steps, selected_names = _read_by_step_topology_sequence(by_step_csv)
    topology_names = tuple(dict.fromkeys(selected_names))
    name_to_action = {name: idx for idx, name in enumerate(topology_names)}
    selected_action = np.asarray([name_to_action[name] for name in selected_names], dtype=np.int32)
    union_edge_table, tables_by_action = build_union_edge_table_for_schedule(
        config=config,
        topology_names=topology_names,
        selected_action=selected_action,
        topology_library_csv=topology_library_csv,
        extra_topology_tables=extra_tables,
    )
    target_mask = build_target_mask(
        union_edge_table=union_edge_table,
        tables_by_action=tables_by_action,
        selected_action=selected_action,
    )
    active_mask, building_mask, setup_command_mask = build_link_setup_masks_from_target_mask(
        steps=steps,
        target_mask=target_mask,
        setup_time_seconds=float(setup_time_seconds),
        warm_start=True,
        setup_mode=str(setup_mode),
        setup_timing=str(setup_timing),
    )
    return DynamicScheduleSeries(
        steps=steps,
        topology_names=topology_names,
        selected_action=selected_action,
        selected_topology_names=selected_names,
        edge_table=union_edge_table,
        target_mask=target_mask,
        setup_command_mask=setup_command_mask,
        active_mask=active_mask,
        building_mask=building_mask,
    )


def build_target_mask(
    *,
    union_edge_table: EdgeTable,
    tables_by_action: Mapping[int, EdgeTable],
    selected_action: np.ndarray,
) -> np.ndarray:
    key_to_col = {
        _edge_key(int(union_edge_table.src[idx]), int(union_edge_table.dst[idx])): int(idx)
        for idx in range(int(union_edge_table.num_edges))
    }
    action_cols: dict[int, np.ndarray] = {}
    for action, edge_table in tables_by_action.items():
        cols = [
            key_to_col[_edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]))]
            for idx in range(int(edge_table.num_edges))
        ]
        action_cols[int(action)] = np.asarray(cols, dtype=np.int32)
    target_mask = np.zeros((int(selected_action.size), int(union_edge_table.num_edges)), dtype=bool)
    for row, action in enumerate(np.asarray(selected_action, dtype=np.int32)):
        target_mask[int(row), action_cols[int(action)]] = True
    return target_mask


def setup_command_counts_from_target_mask(target_mask: np.ndarray, *, warm_start: bool = True) -> np.ndarray:
    """Count newly requested edges at each sampled step.

    With a warm start, edges present in the first row are treated as already
    established and therefore do not count as setup commands at step 0.
    """

    target = np.asarray(target_mask, dtype=bool)
    if target.ndim != 2:
        raise ValueError("target_mask must be 2D")
    counts = np.zeros(target.shape[0], dtype=np.int32)
    if target.shape[0] == 0:
        return counts
    if not bool(warm_start):
        counts[0] = int(np.count_nonzero(target[0]))
    if target.shape[0] > 1:
        counts[1:] = np.count_nonzero(target[1:] & ~target[:-1], axis=1).astype(np.int32, copy=False)
    return counts


def setup_command_counts_from_command_mask(command_mask: np.ndarray) -> np.ndarray:
    """Count explicit setup commands from a per-step command mask."""

    command = np.asarray(command_mask, dtype=bool)
    if command.ndim != 2:
        raise ValueError("command_mask must be 2D")
    return np.count_nonzero(command, axis=1).astype(np.int32, copy=False)


def _validate_setup_inputs(
    *,
    steps: Sequence[int],
    target_mask: np.ndarray,
    setup_time_seconds: int | float,
) -> tuple[np.ndarray, np.ndarray, float]:
    steps_arr = np.asarray(steps, dtype=np.int64)
    target = np.asarray(target_mask, dtype=bool)
    if target.ndim != 2:
        raise ValueError("target_mask must be 2D")
    if steps_arr.shape[0] != target.shape[0]:
        raise ValueError("steps and target_mask row count differ")
    if np.any(np.diff(steps_arr) < 0):
        raise ValueError("steps must be sorted")
    setup = float(setup_time_seconds)
    if setup < 0:
        raise ValueError("setup_time_seconds must be >= 0")
    return steps_arr, target, setup


def _reactive_link_setup_masks(
    *,
    steps_arr: np.ndarray,
    target: np.ndarray,
    setup: float,
    warm_start: bool,
    setup_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Command-driven semantics: new target edges first enter building state.

    In this mode ``target`` is a setup-command/intent sequence rather than an
    already routable desired topology. When an edge first appears, a setup
    command is issued. The edge is marked ``building`` and is not routable until
    it has waited for ``setup_time_seconds``.
    """

    active = np.zeros_like(target, dtype=bool)
    building = np.zeros_like(target, dtype=bool)
    command = np.zeros_like(target, dtype=bool)
    start_time = np.full(target.shape[1], np.nan, dtype=np.float64)
    end_time = np.full(target.shape[1], np.nan, dtype=np.float64)

    for row, step in enumerate(steps_arr):
        current = target[row]
        if row == 0:
            if bool(warm_start):
                start_time[current] = float(step) - setup
            else:
                start_time[current] = float(step)
                command[row, current] = True
        else:
            started = current & ~target[row - 1]
            ended = ~current & target[row - 1]
            command[row, started] = True
            start_time[started] = float(step)
            end_time[started] = np.nan
            start_time[ended] = np.nan
            end_time[ended] = float(step)

        elapsed = float(step) - start_time
        ready = current & np.isfinite(start_time) & (elapsed >= setup)
        active[row, ready] = True
        building[row, current & ~ready] = True
        if setup_mode == "make_before_break":
            linger_elapsed = float(step) - end_time
            linger = ~current & np.isfinite(end_time) & (linger_elapsed < setup)
            active[row, linger] = True
    return active, building, command


def _advance_link_setup_masks(
    *,
    steps_arr: np.ndarray,
    target: np.ndarray,
    setup: float,
    warm_start: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offline desired-active semantics: issue commands LST before target onset.

    ``target`` is interpreted as the topology that should be routable at each
    sampled time. For every edge run that starts at time ``t``, the setup
    command time is ``t - setup``. The edge is marked ``building`` at sampled
    rows between the command time and ``t``, and it is routable according to
    ``target``. This is useful for an offline/upper-bound study where future
    switch times are known before the command is issued. Runs already active in
    row 0 are treated as warm-started by default.
    """

    active = target.copy()
    building = np.zeros_like(target, dtype=bool)
    command = np.zeros_like(target, dtype=bool)
    n_rows, n_edges = target.shape
    if n_rows == 0:
        return active, building, command

    for edge_idx in range(n_edges):
        values = target[:, edge_idx]
        if not np.any(values):
            continue
        previous = False
        for row in range(n_rows):
            current = bool(values[row])
            starts_run = current and not previous
            if starts_run:
                if row == 0 and bool(warm_start):
                    previous = current
                    continue
                start_step = float(steps_arr[row])
                command_time = start_step - setup
                if row == 0:
                    command[0, edge_idx] = True
                    if not bool(warm_start):
                        ready_time = float(steps_arr[0]) + setup
                        active[:, edge_idx] = values & (steps_arr.astype(np.float64) >= ready_time)
                        building[:, edge_idx] |= values & (steps_arr.astype(np.float64) < ready_time)
                else:
                    command_row = int(np.searchsorted(steps_arr, command_time, side="left"))
                    build_row = max(0, command_row)
                    if command_row < n_rows:
                        command[command_row, edge_idx] = True
                    if build_row < row:
                        building[build_row:row, edge_idx] = True
            previous = current
    return active, building, command


def build_link_setup_masks_from_target_mask(
    *,
    steps: Sequence[int],
    target_mask: np.ndarray,
    setup_time_seconds: int | float,
    warm_start: bool = True,
    setup_mode: str = "break_before_make",
    setup_timing: str = "reactive",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build active, building, and setup-command masks.

    ``setup_timing`` defines what the input ``target_mask`` means:

    - ``reactive``: command-driven behavior. When a new edge first appears in
      the target mask, that row issues the setup command. The edge is
      ``building`` until it has waited one LST interval, and only then becomes
      active/routable.
    - ``advance``: offline desired-active behavior. The target mask is the
      topology desired to be active at each sampled row, so setup commands are
      placed LST seconds before each edge run starts. Use this only for
      idealised known-future comparisons.

    Region groups are not constraints in either mode; they are only endpoint
    sets used by metric evaluation.
    """

    steps_arr, target, setup = _validate_setup_inputs(
        steps=steps,
        target_mask=target_mask,
        setup_time_seconds=setup_time_seconds,
    )
    mode = str(setup_mode or "break_before_make").lower()
    if mode not in {"break_before_make", "make_before_break"}:
        raise ValueError("setup_mode must be 'break_before_make' or 'make_before_break'")
    timing = str(setup_timing or "reactive").lower()
    if timing not in {"reactive", "advance"}:
        raise ValueError("setup_timing must be 'reactive' or 'advance'")
    if timing == "advance":
        return _advance_link_setup_masks(
            steps_arr=steps_arr,
            target=target,
            setup=setup,
            warm_start=bool(warm_start),
        )
    return _reactive_link_setup_masks(
        steps_arr=steps_arr,
        target=target,
        setup=setup,
        warm_start=bool(warm_start),
        setup_mode=mode,
    )


def apply_link_setup_time_to_target_mask(
    *,
    steps: Sequence[int],
    target_mask: np.ndarray,
    setup_time_seconds: int | float,
    warm_start: bool = True,
    setup_mode: str = "break_before_make",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a generic edge-level link-setup-time state machine.

    ``target_mask[row, edge]`` means a setup command has been issued for that
    edge because the desired topology at this sampled time contains it. A newly
    requested edge is marked ``building`` immediately, but it is not routable
    until it has remained requested for at least ``setup_time_seconds``. Once
    ready, it is marked ``active`` and participates in shortest-path metrics.

    If ``warm_start`` is true, edges already present in the first sampled target
    topology are treated as already built at step 0. Region groups are not
    topology constraints here; they are only endpoint sets used later when
    evaluating pair metrics.

    ``setup_mode`` controls what happens to old links while replacement links
    are waiting for LST:

    - ``break_before_make``: edges no longer in the target topology disappear
      immediately. This preserves the original evaluator behavior.
    - ``make_before_break``: old target edges remain active for one LST window
      after they leave the target topology, while newly requested edges are
      building. This models issuing a setup command and keeping the old link
      usable until the new link can become active.
    """

    active, building, _command = build_link_setup_masks_from_target_mask(
        steps=steps,
        target_mask=target_mask,
        setup_time_seconds=setup_time_seconds,
        warm_start=bool(warm_start),
        setup_mode=str(setup_mode),
        setup_timing="reactive",
    )
    return active, building


def load_schedule_series_from_selector(
    *,
    selector_dir: str | Path,
    schedule_name: str,
    setup_time_seconds: int | float,
    setup_mode: str = "break_before_make",
    setup_timing: str = "reactive",
) -> DynamicScheduleSeries:
    selector_dir = Path(selector_dir)
    arrays = np.load(selector_dir / "selector_arrays.npz", allow_pickle=False)
    raw_config = load_yaml_dict(selector_dir / "selector_config.yaml")
    config = _viewer_config_from_raw(raw_config)
    topology_library_csv = optional_path(raw_config.get("inputs", {}).get("topology_library_csv"))
    if topology_library_csv is None:
        raise ValueError("selector_config.yaml inputs.topology_library_csv is required")

    key = f"schedule_{schedule_name}"
    if key not in arrays.files:
        raise KeyError(f"schedule {schedule_name!r} not found in {selector_dir / 'selector_arrays.npz'}")
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    topology_names = tuple(str(x) for x in np.asarray(arrays["topology_names"]))
    selected_action = np.asarray(arrays[key], dtype=np.int32)
    union_edge_table, tables_by_action = build_union_edge_table_for_schedule(
        config=config,
        topology_names=topology_names,
        selected_action=selected_action,
        topology_library_csv=topology_library_csv,
    )
    target_mask = build_target_mask(
        union_edge_table=union_edge_table,
        tables_by_action=tables_by_action,
        selected_action=selected_action,
    )
    active_mask, building_mask, setup_command_mask = build_link_setup_masks_from_target_mask(
        steps=steps,
        target_mask=target_mask,
        setup_time_seconds=float(setup_time_seconds),
        warm_start=True,
        setup_mode=str(setup_mode),
        setup_timing=str(setup_timing),
    )
    selected_names = tuple(str(topology_names[int(action)]) for action in selected_action)
    return DynamicScheduleSeries(
        steps=steps,
        topology_names=topology_names,
        selected_action=selected_action,
        selected_topology_names=selected_names,
        edge_table=union_edge_table,
        target_mask=target_mask,
        setup_command_mask=setup_command_mask,
        active_mask=active_mask,
        building_mask=building_mask,
    )


def _mean_shortest_hops_for_graph(
    *,
    n_nodes: int,
    src_edges: np.ndarray,
    dst_edges: np.ndarray,
    sources: Sequence[int],
    targets: Sequence[int],
) -> tuple[float, int]:
    if not sources or not targets:
        return math.nan, 0
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    rows = np.concatenate([src_edges, dst_edges]).astype(np.int32, copy=False)
    cols = np.concatenate([dst_edges, src_edges]).astype(np.int32, copy=False)
    data = np.ones(rows.shape[0], dtype=np.float32)
    graph = csr_matrix((data, (rows, cols)), shape=(int(n_nodes), int(n_nodes)))
    source_arr = np.asarray(sources, dtype=np.int32)
    target_arr = np.asarray(targets, dtype=np.int32)
    dist = shortest_path(graph, directed=False, unweighted=True, indices=source_arr)
    dist = np.atleast_2d(np.asarray(dist, dtype=np.float64))
    values = dist[:, target_arr]
    finite = np.isfinite(values)
    if not np.any(finite):
        return math.nan, 0
    return float(np.mean(values[finite])), int(np.count_nonzero(finite))


def _mean_shortest_delay_for_graph(
    *,
    n_nodes: int,
    src_edges: np.ndarray,
    dst_edges: np.ndarray,
    weights: np.ndarray,
    sources: Sequence[int],
    targets: Sequence[int],
) -> tuple[float, int]:
    if not sources or not targets:
        return math.nan, 0
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    rows = np.concatenate([src_edges, dst_edges]).astype(np.int32, copy=False)
    cols = np.concatenate([dst_edges, src_edges]).astype(np.int32, copy=False)
    data = np.concatenate([weights, weights]).astype(np.float32, copy=False)
    graph = csr_matrix((data, (rows, cols)), shape=(int(n_nodes), int(n_nodes)))
    source_arr = np.asarray(sources, dtype=np.int32)
    target_arr = np.asarray(targets, dtype=np.int32)
    dist = dijkstra(graph, directed=False, indices=source_arr)
    dist = np.atleast_2d(np.asarray(dist, dtype=np.float64))
    values = dist[:, target_arr]
    finite = np.isfinite(values)
    if not np.any(finite):
        return math.nan, 0
    return float(np.mean(values[finite])), int(np.count_nonzero(finite))


def evaluate_dynamic_schedule_after_lst(
    *,
    series: DynamicScheduleSeries,
    config: ViewerConfig,
    group_data: Mapping,
    source_group_id: int,
    target_group_id: int,
    delay_store_dir: str | Path,
    position_cache_dir: str | Path | None,
    out_dir: str | Path,
    schedule_name: str,
    setup_time_seconds: int | float,
    setup_mode: str = "break_before_make",
    setup_timing: str = "reactive",
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(series.edge_table, out_dir / "union_edges.csv")
    np.save(out_dir / "steps.npy", series.steps.astype(np.int64))
    np.save(out_dir / "target_mask.npy", series.target_mask.astype(bool))
    np.save(out_dir / "setup_command_mask.npy", series.setup_command_mask.astype(bool))
    np.save(out_dir / "edge_active_mask.npy", series.active_mask.astype(bool))
    np.save(out_dir / "edge_building_mask.npy", series.building_mask.astype(bool))
    setup_command_counts = setup_command_counts_from_command_mask(series.setup_command_mask)
    np.save(out_dir / "setup_command_counts.npy", setup_command_counts.astype(np.int32, copy=False))

    delay_store = open_delay_store_for_interval(
        int(series.steps[0]),
        int(series.steps[-1]),
        stride=int(series.steps[1] - series.steps[0]) if series.steps.size > 1 else 1,
        store_dir=Path(delay_store_dir),
        constellation_name=config.name,
    )
    delay_rows = delay_store.rows_for_interval(
        int(series.steps[0]),
        int(series.steps[-1]),
        int(series.steps[1] - series.steps[0]) if series.steps.size > 1 else 1,
    )
    position_store = None
    position_rows = None
    if position_cache_dir not in (None, "", False):
        position_store = open_position_cache_for_interval(
            int(series.steps[0]),
            int(series.steps[-1]),
            stride=int(series.steps[1] - series.steps[0]) if series.steps.size > 1 else 1,
            cache_dir=Path(position_cache_dir),
        )
        position_rows = position_store.rows_for_interval(
            int(series.steps[0]),
            int(series.steps[-1]),
            int(series.steps[1] - series.steps[0]) if series.steps.size > 1 else 1,
        )

    lookup = build_weight_lookup(
        series.edge_table,
        delay_store,
        config=config,
        allow_intra_fallback=position_store is not None,
    )
    all_weights_by_step = [
        edge_weights_for_step(
            edge_table=series.edge_table,
            lookup=lookup,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=int(delay_rows[row]),
            position_row=int(position_rows[row]) if position_rows is not None else None,
        )
        for row in range(int(series.steps.size))
    ]

    hop_values = np.full(series.steps.shape[0], np.nan, dtype=np.float32)
    delay_values = np.full(series.steps.shape[0], np.nan, dtype=np.float32)
    strict_hop_values = np.full(series.steps.shape[0], np.nan, dtype=np.float32)
    strict_delay_values = np.full(series.steps.shape[0], np.nan, dtype=np.float32)
    pair_reachability = np.full(series.steps.shape[0], np.nan, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    src_all = np.asarray(series.edge_table.src, dtype=np.int32)
    dst_all = np.asarray(series.edge_table.dst, dtype=np.int32)

    for row, step in enumerate(series.steps):
        active_cols = np.flatnonzero(series.active_mask[row])
        src_edges = src_all[active_cols]
        dst_edges = dst_all[active_cols]
        sources = group_nodes_for_step(group_data, int(step), int(source_group_id))
        targets = group_nodes_for_step(group_data, int(step), int(target_group_id))
        hop_mean, hop_reachable = _mean_shortest_hops_for_graph(
            n_nodes=int(config.total_sats),
            src_edges=src_edges,
            dst_edges=dst_edges,
            sources=sources,
            targets=targets,
        )
        weights = np.asarray(all_weights_by_step[row], dtype=np.float32)[active_cols]
        delay_mean, delay_reachable = _mean_shortest_delay_for_graph(
            n_nodes=int(config.total_sats),
            src_edges=src_edges,
            dst_edges=dst_edges,
            weights=weights,
            sources=sources,
            targets=targets,
        )
        all_pairs = int(len(sources) * len(targets))
        reach = min(int(hop_reachable), int(delay_reachable))
        if all_pairs > 0:
            pair_reachability[row] = float(reach) / float(all_pairs)
        if all_pairs > 0 and int(hop_reachable) == all_pairs:
            strict_hop_values[row] = hop_mean
        if all_pairs > 0 and int(delay_reachable) == all_pairs:
            strict_delay_values[row] = delay_mean
        hop_values[row] = hop_mean
        delay_values[row] = delay_mean
        rows.append(
            {
                "step": int(step),
                "topology": series.selected_topology_names[row],
                "action_idx": int(series.selected_action[row]),
                "target_edges": int(np.count_nonzero(series.target_mask[row])),
                "active_edges": int(active_cols.size),
                "building_edges": int(np.count_nonzero(series.building_mask[row])),
                "setup_commands": int(setup_command_counts[row]),
                "source_nodes": int(len(sources)),
                "target_nodes": int(len(targets)),
                "all_pairs": int(all_pairs),
                "pair_reachability": None
                if not math.isfinite(float(pair_reachability[row]))
                else float(pair_reachability[row]),
                "reachable_pairs_hops": int(hop_reachable),
                "mean_shortest_hops": None if not math.isfinite(float(hop_mean)) else float(hop_mean),
                "mean_shortest_hops_strict_all_pairs": None
                if not math.isfinite(float(strict_hop_values[row]))
                else float(strict_hop_values[row]),
                "reachable_pairs_delay": int(delay_reachable),
                "mean_shortest_delay_ms": None if not math.isfinite(float(delay_mean)) else float(delay_mean),
                "mean_shortest_delay_ms_strict_all_pairs": None
                if not math.isfinite(float(strict_delay_values[row]))
                else float(strict_delay_values[row]),
            }
        )

    np.save(out_dir / "mean_shortest_hops.npy", hop_values)
    np.save(out_dir / "mean_shortest_delay_ms.npy", delay_values)
    np.save(out_dir / "mean_shortest_hops_strict_all_pairs.npy", strict_hop_values)
    np.save(out_dir / "mean_shortest_delay_ms_strict_all_pairs.npy", strict_delay_values)
    np.save(out_dir / "pair_reachability.npy", pair_reachability)
    with (out_dir / "step_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    finite_hop = hop_values[np.isfinite(hop_values)]
    finite_delay = delay_values[np.isfinite(delay_values)]
    finite_strict_hop = strict_hop_values[np.isfinite(strict_hop_values)]
    finite_strict_delay = strict_delay_values[np.isfinite(strict_delay_values)]
    strict_all_rows = bool(
        series.steps.size > 0
        and finite_strict_hop.size == series.steps.size
        and finite_strict_delay.size == series.steps.size
    )
    building_counts = np.count_nonzero(series.building_mask, axis=1)
    switches = int(np.count_nonzero(series.selected_action[1:] != series.selected_action[:-1])) if series.selected_action.size > 1 else 0
    meta = {
        "schedule_name": str(schedule_name),
        "setup_time_seconds": float(setup_time_seconds),
        "setup_mode": str(setup_mode),
        "setup_timing": str(setup_timing),
        "num_steps": int(series.steps.size),
        "num_union_edges": int(series.edge_table.num_edges),
        "num_switches": switches,
        "mean_building_edges": float(np.mean(building_counts)),
        "max_building_edges": int(np.max(building_counts)) if building_counts.size else 0,
        "building_edge_seconds": float(np.sum(building_counts) * (float(series.steps[1] - series.steps[0]) if series.steps.size > 1 else 1.0)),
        "total_setup_commands": int(np.sum(setup_command_counts, dtype=np.int64)),
        "max_setup_commands_per_step": int(np.max(setup_command_counts)) if setup_command_counts.size else 0,
        "mean_setup_commands_per_step": float(np.mean(setup_command_counts)) if setup_command_counts.size else 0.0,
        "strict_all_pairs_rows": int(min(finite_strict_hop.size, finite_strict_delay.size)),
        "disconnected_rows": int(series.steps.size - min(finite_strict_hop.size, finite_strict_delay.size)),
        "strict_all_pairs_complete": strict_all_rows,
        "mean_pair_reachability": float(np.nanmean(pair_reachability)) if pair_reachability.size else None,
        "mean_shortest_hops": float(np.mean(finite_hop)) if finite_hop.size else None,
        "mean_shortest_delay_ms": float(np.mean(finite_delay)) if finite_delay.size else None,
        "mean_shortest_hops_strict_all_pairs": float(np.mean(finite_strict_hop))
        if finite_strict_hop.size
        else None,
        "mean_shortest_delay_ms_strict_all_pairs": float(np.mean(finite_strict_delay))
        if finite_strict_delay.size
        else None,
        "mean_shortest_hops_strict_all_rows": float(np.mean(finite_strict_hop))
        if strict_all_rows
        else None,
        "mean_shortest_delay_ms_strict_all_rows": float(np.mean(finite_strict_delay))
        if strict_all_rows
        else None,
        "outputs": {
            "step_summary_csv": str(out_dir / "step_summary.csv"),
            "active_mask": str(out_dir / "edge_active_mask.npy"),
            "building_mask": str(out_dir / "edge_building_mask.npy"),
            "setup_command_mask": str(out_dir / "setup_command_mask.npy"),
            "setup_command_counts": str(out_dir / "setup_command_counts.npy"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def run_lst_evaluation_from_selector(
    *,
    selector_dir: str | Path,
    schedule_name: str,
    setup_time_seconds: int | float,
    delay_store_dir: str | Path,
    position_cache_dir: str | Path | None,
    group_xml: str | Path,
    group_cache_dir: str | Path,
    source_group_id: int,
    target_group_id: int,
    out_dir: str | Path,
    setup_mode: str = "break_before_make",
    setup_timing: str = "reactive",
    force_group_cache: bool = False,
) -> Path:
    selector_dir = Path(selector_dir)
    raw_config = load_yaml_dict(selector_dir / "selector_config.yaml")
    config = _viewer_config_from_raw(raw_config)
    if not config.station_groups and config.name == "G60" and int(config.P) == 18 and int(config.N) == 36:
        from src.config.viewer_config import G60_CONFIG

        config = G60_CONFIG
    series = load_schedule_series_from_selector(
        selector_dir=selector_dir,
        schedule_name=schedule_name,
        setup_time_seconds=float(setup_time_seconds),
        setup_mode=str(setup_mode),
        setup_timing=str(setup_timing),
    )
    group_data = load_or_build_group_data(
        xml_file=Path(group_xml),
        group_cache_dir=Path(group_cache_dir),
        steps=[int(x) for x in series.steps],
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(series.steps[1] - series.steps[0]) if series.steps.size > 1 else 1,
        enabled=True,
        force=bool(force_group_cache),
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selector_dir / "selector_config.yaml", out_dir / "selector_config.yaml")
    meta = evaluate_dynamic_schedule_after_lst(
        series=series,
        config=config,
        group_data=group_data,
        source_group_id=int(source_group_id),
        target_group_id=int(target_group_id),
        delay_store_dir=delay_store_dir,
        position_cache_dir=position_cache_dir,
        out_dir=out_dir,
        schedule_name=schedule_name,
        setup_time_seconds=float(setup_time_seconds),
        setup_mode=str(setup_mode),
        setup_timing=str(setup_timing),
    )
    print(json.dumps({"out_dir": str(out_dir), **meta}, ensure_ascii=False, indent=2), flush=True)
    return out_dir


def run_lst_evaluation_from_by_step_csv(
    *,
    by_step_csv: str | Path,
    selector_dir: str | Path,
    schedule_name: str,
    setup_time_seconds: int | float,
    delay_store_dir: str | Path,
    position_cache_dir: str | Path | None,
    group_xml: str | Path,
    group_cache_dir: str | Path,
    source_group_id: int,
    target_group_id: int,
    out_dir: str | Path,
    setup_mode: str = "break_before_make",
    setup_timing: str = "reactive",
    hybrid_meta_by_name: Mapping[str, Mapping[str, Any]] | None = None,
    force_group_cache: bool = False,
) -> Path:
    selector_dir = Path(selector_dir)
    raw_config = load_yaml_dict(selector_dir / "selector_config.yaml")
    config = _viewer_config_from_raw(raw_config)
    if not config.station_groups and config.name == "G60" and int(config.P) == 18 and int(config.N) == 36:
        from src.config.viewer_config import G60_CONFIG

        config = G60_CONFIG
    series = load_schedule_series_from_by_step_csv(
        by_step_csv=by_step_csv,
        selector_dir=selector_dir,
        setup_time_seconds=float(setup_time_seconds),
        setup_mode=str(setup_mode),
        setup_timing=str(setup_timing),
        hybrid_meta_by_name=hybrid_meta_by_name,
    )
    group_data = load_or_build_group_data(
        xml_file=Path(group_xml),
        group_cache_dir=Path(group_cache_dir),
        steps=[int(x) for x in series.steps],
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(series.steps[1] - series.steps[0]) if series.steps.size > 1 else 1,
        enabled=True,
        force=bool(force_group_cache),
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selector_dir / "selector_config.yaml", out_dir / "selector_config.yaml")
    meta = evaluate_dynamic_schedule_after_lst(
        series=series,
        config=config,
        group_data=group_data,
        source_group_id=int(source_group_id),
        target_group_id=int(target_group_id),
        delay_store_dir=delay_store_dir,
        position_cache_dir=position_cache_dir,
        out_dir=out_dir,
        schedule_name=schedule_name,
        setup_time_seconds=float(setup_time_seconds),
        setup_mode=str(setup_mode),
        setup_timing=str(setup_timing),
    )
    payload = {"out_dir": str(out_dir), "by_step_csv": str(by_step_csv), **meta}
    (out_dir / "by_step_lst_eval_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return out_dir
