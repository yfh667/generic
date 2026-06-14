from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable, build_full_option_edges
from src.topology_metrics.module.group_states import group_nodes_for_step
from src.topology_workflow.module.edge_tables import make_edge_table_from_records

from .core import (
    P_GRID_ACTIVE,
    P_GRID_BUILDING,
    P_MOTIF_ACTIVE,
    P_MOTIF_BUILDING,
    _compute_active_and_internal_keys_for_state,
    _edge_key,
    _edge_ports,
    _records_by_key,
)


@dataclass(frozen=True)
class LinkSetupTimeTargetSeries:
    """Precomputed target-topology runs used by fast LST sweeps."""

    edge_table: EdgeTable
    all_records_by_key: dict[int, tuple[int, int, int, int, int]]
    key_to_col: dict[int, int]
    col_to_key: np.ndarray
    state_payloads: tuple[dict[str, object], ...]
    state_ids: np.ndarray
    target_runs: tuple[tuple[tuple[int, int, bool], ...], ...]
    grid_runs: tuple[tuple[tuple[int, int], ...], ...]
    inter_cols: np.ndarray
    intra_count: int
    ports_by_col: tuple[tuple[int, int] | None, ...]
    incident_cols_by_port: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class LinkSetupTimeSweepRow:
    lst_s: int
    mean_building_edges: float
    max_building_edges: int
    first_max_step_s: int
    nonzero_steps: int
    p95_building_edges: float
    total_building_edge_seconds: int
    mean_active_edges: float
    timeseries_csv: str = ""


@dataclass(frozen=True)
class LinkSetupTimeSweepResult:
    steps: tuple[int, ...]
    rows: tuple[LinkSetupTimeSweepRow, ...]
    active_by_lst: dict[int, np.ndarray]
    building_by_lst: dict[int, np.ndarray]


def group_state_sequence_from_group_data(
    *,
    group_data: Mapping,
    steps: Sequence[int],
    constrained_groups: Sequence[int],
) -> list[tuple[tuple[int, ...], ...]]:
    """Convert viewer-style group data into hashable group states."""

    constrained = tuple(int(x) for x in constrained_groups)
    return [
        tuple(
            tuple(int(node) for node in group_nodes_for_step(group_data, int(step), int(group_id)))
            for group_id in constrained
        )
        for step in steps
    ]


def build_region_internal_target_series(
    *,
    config: ViewerConfig,
    base_edge_table: EdgeTable,
    state_sequence: Sequence[tuple[tuple[int, ...], ...]],
    forced_option: int = 0,
    wrap_planes: bool = False,
) -> LinkSetupTimeTargetSeries:
    """Precompute target topology states and edge runs for LST sweeps.

    The target topology is the topology after applying the region-internal
    forced-option rule at each sampled step, but before applying LST.
    """

    total_nodes = int(config.total_sats)
    forced_option_edges = build_full_option_edges(
        config,
        options=(int(forced_option),),
        sat_ids=list(base_edge_table.sat_ids),
        wrap_planes=bool(wrap_planes),
    )
    base_records_by_key = _records_by_key(base_edge_table, total_nodes)
    forced_records_by_key = _records_by_key(forced_option_edges, total_nodes)
    all_records_by_key = dict(base_records_by_key)
    all_records_by_key.update(forced_records_by_key)

    edge_table = make_edge_table_from_records(
        p=int(config.P),
        n=int(config.N),
        records=list(all_records_by_key.values()),
        sat_ids=list(base_edge_table.sat_ids),
    )
    key_to_col = {
        _edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]), total_nodes): int(idx)
        for idx in range(int(edge_table.num_edges))
    }
    col_to_key = np.empty(int(edge_table.num_edges), dtype=np.int64)
    for key, col in key_to_col.items():
        col_to_key[int(col)] = int(key)

    state_to_id: dict[tuple[tuple[int, ...], ...], int] = {}
    payloads: list[dict[str, object]] = []
    state_ids = np.empty(len(state_sequence), dtype=np.int32)
    for row, state_key in enumerate(state_sequence):
        state_id = state_to_id.get(state_key)
        if state_id is None:
            active_keys, internal_keys = _compute_active_and_internal_keys_for_state(
                state_key=state_key,
                total_nodes=total_nodes,
                n=int(config.N),
                p=int(config.P),
                forced_option=int(forced_option),
                wrap_planes=bool(wrap_planes),
                base_records_by_key=base_records_by_key,
                forced_records_by_key=forced_records_by_key,
            )
            state_id = len(payloads)
            state_to_id[state_key] = state_id
            payloads.append(
                {
                    "active_keys": set(int(key) for key in active_keys),
                    "internal_keys": set(int(key) for key in internal_keys),
                    "active_cols": np.asarray([key_to_col[int(key)] for key in active_keys], dtype=np.int32),
                    "internal_cols": np.asarray([key_to_col[int(key)] for key in internal_keys], dtype=np.int32),
                }
            )
        state_ids[row] = int(state_id)

    target_runs, grid_runs = build_edge_presence_runs(
        n_edges=int(edge_table.num_edges),
        state_payloads=payloads,
        state_ids=state_ids,
    )
    inter_cols = np.asarray(np.where(np.asarray(edge_table.option) != -1)[0], dtype=np.int32)
    intra_count = int(np.count_nonzero(np.asarray(edge_table.option) == -1))
    ports_by_col, incident_cols_by_port = build_port_index(
        config=config,
        edge_table=edge_table,
        all_records_by_key=all_records_by_key,
        col_to_key=col_to_key,
        inter_cols=inter_cols,
        wrap_planes=bool(wrap_planes),
    )

    return LinkSetupTimeTargetSeries(
        edge_table=edge_table,
        all_records_by_key=all_records_by_key,
        key_to_col=key_to_col,
        col_to_key=col_to_key,
        state_payloads=tuple(payloads),
        state_ids=state_ids,
        target_runs=tuple(tuple(run) for run in target_runs),
        grid_runs=tuple(tuple(run) for run in grid_runs),
        inter_cols=inter_cols,
        intra_count=intra_count,
        ports_by_col=tuple(ports_by_col),
        incident_cols_by_port=tuple(incident_cols_by_port),
    )


def build_edge_presence_runs(
    *,
    n_edges: int,
    state_payloads: Sequence[Mapping[str, object]],
    state_ids: np.ndarray,
) -> tuple[list[list[tuple[int, int, bool]]], list[list[tuple[int, int]]]]:
    """Compress target and region-internal edge presence into row intervals."""

    n_steps = int(state_ids.size)
    target_runs: list[list[tuple[int, int, bool]]] = [[] for _ in range(int(n_edges))]
    grid_runs: list[list[tuple[int, int]]] = [[] for _ in range(int(n_edges))]

    last_target = np.zeros(int(n_edges), dtype=bool)
    last_grid = np.zeros(int(n_edges), dtype=bool)
    target_start = np.full(int(n_edges), -1, dtype=np.int32)
    grid_start = np.full(int(n_edges), -1, dtype=np.int32)
    target_entered_grid = np.zeros(int(n_edges), dtype=bool)

    for row in range(n_steps):
        payload = state_payloads[int(state_ids[row])]
        current_target = np.zeros(int(n_edges), dtype=bool)
        current_grid = np.zeros(int(n_edges), dtype=bool)
        current_target[np.asarray(payload["active_cols"], dtype=np.int32)] = True
        current_grid[np.asarray(payload["internal_cols"], dtype=np.int32)] = True

        ended = np.flatnonzero(last_target & ~current_target)
        for col in ended:
            target_runs[int(col)].append((int(target_start[col]), int(row), bool(target_entered_grid[col])))
            target_start[int(col)] = -1
        started = np.flatnonzero(current_target & ~last_target)
        target_start[started] = int(row)
        target_entered_grid[started] = current_grid[started]

        grid_ended = np.flatnonzero(last_grid & ~current_grid)
        for col in grid_ended:
            grid_runs[int(col)].append((int(grid_start[col]), int(row)))
            grid_start[int(col)] = -1
        grid_started = np.flatnonzero(current_grid & ~last_grid)
        grid_start[grid_started] = int(row)

        last_target = current_target
        last_grid = current_grid

    ended = np.flatnonzero(last_target)
    for col in ended:
        target_runs[int(col)].append((int(target_start[col]), int(n_steps), bool(target_entered_grid[col])))
    grid_ended = np.flatnonzero(last_grid)
    for col in grid_ended:
        grid_runs[int(col)].append((int(grid_start[col]), int(n_steps)))
    return target_runs, grid_runs


def build_port_index(
    *,
    config: ViewerConfig,
    edge_table: EdgeTable,
    all_records_by_key: Mapping[int, tuple[int, int, int, int, int]],
    col_to_key: np.ndarray,
    inter_cols: np.ndarray,
    wrap_planes: bool = False,
) -> tuple[list[tuple[int, int] | None], list[np.ndarray]]:
    n_ports = int(config.total_sats) * 2
    ports_by_col: list[tuple[int, int] | None] = [None] * int(edge_table.num_edges)
    incident: list[list[int]] = [[] for _ in range(n_ports)]
    for col in inter_cols:
        col = int(col)
        key = int(col_to_key[col])
        ports = _edge_ports(
            all_records_by_key[key],
            n=int(config.N),
            p=int(config.P),
            wrap_planes=bool(wrap_planes),
        )
        if len(ports) != 2:
            continue
        port_ids = tuple(int(node) * 2 + int(side) for node, side in ports)
        ports_by_col[col] = (int(port_ids[0]), int(port_ids[1]))
        incident[int(port_ids[0])].append(col)
        incident[int(port_ids[1])].append(col)
    incident_cols_by_port = [np.asarray(sorted(cols), dtype=np.int32) for cols in incident]
    return ports_by_col, incident_cols_by_port


def _fill_interval(
    array: np.ndarray,
    chunk_start: int,
    chunk_end: int,
    start: int,
    end: int,
    col: int,
    value,
) -> None:
    left = max(int(start), int(chunk_start))
    right = min(int(end), int(chunk_end))
    if left < right:
        array[left - int(chunk_start) : right - int(chunk_start), int(col)] = value


def compute_link_setup_counts_for_lst(
    *,
    target_series: LinkSetupTimeTargetSeries,
    steps: Sequence[int],
    setup_time_seconds: int | float,
    chunk_size: int = 6000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return active/building edge counts for one LST without materializing masks."""

    times = np.asarray([int(x) for x in steps], dtype=np.int64)
    n_steps = int(times.size)
    n_edges = int(target_series.edge_table.num_edges)
    active_counts = np.zeros(n_steps, dtype=np.int32)
    building_counts = np.zeros(n_steps, dtype=np.int32)
    lst = float(setup_time_seconds)
    if lst < 0:
        raise ValueError("setup_time_seconds must be >= 0")

    inter_cols = np.asarray(target_series.inter_cols, dtype=np.int32)
    ports_by_col = target_series.ports_by_col
    incident_cols_by_port = target_series.incident_cols_by_port

    for chunk_start in range(0, n_steps, int(chunk_size)):
        chunk_end = min(n_steps, int(chunk_start) + int(chunk_size))
        rows_count = int(chunk_end) - int(chunk_start)
        priority = np.zeros((rows_count, n_edges), dtype=np.int8)
        active_claim = np.zeros((rows_count, n_edges), dtype=bool)

        for raw_col in inter_cols:
            col = int(raw_col)
            for start, end, entered_grid in target_series.target_runs[col]:
                anticipatory_start = int(start)
                if start > 0 and entered_grid:
                    anticipatory_start = int(np.searchsorted(times, int(times[start]) - lst, side="left"))
                    anticipatory_start = max(1, anticipatory_start)
                if end <= chunk_start or (start >= chunk_end and anticipatory_start >= chunk_end):
                    continue

                if start > 0 and entered_grid:
                    _fill_interval(priority, chunk_start, chunk_end, anticipatory_start, start, col, P_GRID_BUILDING)

                if start == 0 or entered_grid:
                    active_start = int(start)
                else:
                    active_start = int(np.searchsorted(times, int(times[start]) + lst, side="left"))
                    active_start = max(int(start), min(active_start, int(end)))
                    _fill_interval(priority, chunk_start, chunk_end, start, active_start, col, P_MOTIF_BUILDING)

                if active_start < end:
                    _fill_interval(priority, chunk_start, chunk_end, active_start, end, col, P_MOTIF_ACTIVE)
                    _fill_interval(active_claim, chunk_start, chunk_end, active_start, end, col, True)
                    for grid_start, grid_end in target_series.grid_runs[col]:
                        overlay_start = max(int(active_start), int(grid_start))
                        overlay_end = min(int(end), int(grid_end))
                        if overlay_end <= chunk_start or overlay_start >= chunk_end:
                            continue
                        _fill_interval(priority, chunk_start, chunk_end, overlay_start, overlay_end, col, P_GRID_ACTIVE)
                        _fill_interval(active_claim, chunk_start, chunk_end, overlay_start, overlay_end, col, True)

        n_ports = len(incident_cols_by_port)
        port_winner = np.full((rows_count, n_ports), -1, dtype=np.int32)
        port_winner_active = np.zeros((rows_count, n_ports), dtype=bool)
        row_ids = np.arange(rows_count)
        for port, cols in enumerate(incident_cols_by_port):
            if int(cols.size) == 0:
                continue
            port_priority = priority[:, cols]
            max_priority = port_priority.max(axis=1)
            has_owner = max_priority > 0
            if not np.any(has_owner):
                continue
            local_winner = port_priority.argmax(axis=1)
            winners = cols[local_winner]
            port_winner[has_owner, int(port)] = winners[has_owner]
            port_winner_active[has_owner, int(port)] = active_claim[row_ids[has_owner], winners[has_owner]]

        active_chunk = np.full(rows_count, int(target_series.intra_count), dtype=np.int32)
        building_chunk = np.zeros(rows_count, dtype=np.int32)
        for raw_col in inter_cols:
            col = int(raw_col)
            ports = ports_by_col[col]
            if ports is None:
                continue
            p0, p1 = ports
            owns = (port_winner[:, p0] == col) & (port_winner[:, p1] == col)
            if not np.any(owns):
                continue
            both_active = owns & port_winner_active[:, p0] & port_winner_active[:, p1]
            both_building = owns & (~port_winner_active[:, p0]) & (~port_winner_active[:, p1])
            active_chunk += both_active.astype(np.int32)
            building_chunk += both_building.astype(np.int32)

        active_counts[int(chunk_start) : int(chunk_end)] = active_chunk
        building_counts[int(chunk_start) : int(chunk_end)] = building_chunk

    return active_counts, building_counts


def sweep_link_setup_times(
    *,
    target_series: LinkSetupTimeTargetSeries,
    steps: Sequence[int],
    lst_values: Sequence[int | float],
    chunk_size: int = 6000,
) -> LinkSetupTimeSweepResult:
    steps_tuple = tuple(int(x) for x in steps)
    times = np.asarray(steps_tuple, dtype=np.int64)
    rows: list[LinkSetupTimeSweepRow] = []
    active_by_lst: dict[int, np.ndarray] = {}
    building_by_lst: dict[int, np.ndarray] = {}

    for value in lst_values:
        lst = int(value)
        active, building = compute_link_setup_counts_for_lst(
            target_series=target_series,
            steps=steps_tuple,
            setup_time_seconds=float(value),
            chunk_size=int(chunk_size),
        )
        active_by_lst[lst] = active
        building_by_lst[lst] = building
        max_value = int(np.max(building))
        max_idx = int(np.flatnonzero(building == max_value)[0])
        rows.append(
            LinkSetupTimeSweepRow(
                lst_s=lst,
                mean_building_edges=float(np.mean(building)),
                max_building_edges=max_value,
                first_max_step_s=int(times[max_idx]),
                nonzero_steps=int(np.count_nonzero(building)),
                p95_building_edges=float(np.percentile(building, 95)),
                total_building_edge_seconds=int(np.sum(building)),
                mean_active_edges=float(np.mean(active)),
            )
        )
    return LinkSetupTimeSweepResult(
        steps=steps_tuple,
        rows=tuple(rows),
        active_by_lst=active_by_lst,
        building_by_lst=building_by_lst,
    )


def write_timeseries_csv(path: str | Path, *, steps: Sequence[int], active: np.ndarray, building: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step_s", "time_h", "active_edges", "building_edges"])
        writer.writeheader()
        for step, active_count, building_count in zip(steps, active, building):
            writer.writerow(
                {
                    "step_s": int(step),
                    "time_h": f"{float(step) / 3600.0:.6f}",
                    "active_edges": int(active_count),
                    "building_edges": int(building_count),
                }
            )


def write_sweep_summary_csv(path: str | Path, rows: Sequence[LinkSetupTimeSweepRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "lst_s",
        "mean_building_edges",
        "max_building_edges",
        "first_max_step_s",
        "nonzero_steps",
        "p95_building_edges",
        "total_building_edge_seconds",
        "mean_active_edges",
        "timeseries_csv",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_sweep_outputs(
    *,
    out_dir: str | Path,
    result: LinkSetupTimeSweepResult,
) -> LinkSetupTimeSweepResult:
    """Write per-LST time series and summary CSV, returning rows with CSV paths."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[LinkSetupTimeSweepRow] = []
    for row in result.rows:
        ts_path = out_dir / f"building_link_timeseries_lst{int(row.lst_s):03d}.csv"
        write_timeseries_csv(
            ts_path,
            steps=result.steps,
            active=result.active_by_lst[int(row.lst_s)],
            building=result.building_by_lst[int(row.lst_s)],
        )
        rows.append(LinkSetupTimeSweepRow(**{**asdict(row), "timeseries_csv": str(ts_path)}))
    final = LinkSetupTimeSweepResult(
        steps=result.steps,
        rows=tuple(rows),
        active_by_lst=result.active_by_lst,
        building_by_lst=result.building_by_lst,
    )
    write_sweep_summary_csv(out_dir / "lst_sweep_summary.csv", final.rows)
    return final


def plot_sweep_timeseries(
    *,
    path: str | Path,
    result: LinkSetupTimeSweepResult,
    title: str = "Building-link count under LST sweep",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_hours = np.asarray(result.steps, dtype=np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(16, 7), dpi=180)
    cmap = plt.get_cmap("turbo")
    rows = list(result.rows)
    for idx, row in enumerate(rows):
        ax.plot(
            x_hours,
            result.building_by_lst[int(row.lst_s)],
            linewidth=0.85,
            alpha=0.82,
            color=cmap(idx / max(1, len(rows) - 1)),
            label=f"{int(row.lst_s)}s",
        )
    ax.set_title(str(title))
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("building links")
    ax.grid(True, alpha=0.25)
    ax.legend(title="LST", ncol=2, fontsize=8, title_fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sweep_summary(
    *,
    path: str | Path,
    rows: Sequence[LinkSetupTimeSweepRow],
    title: str = "Building-link count summary vs LST",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lst_arr = np.asarray([row.lst_s for row in rows], dtype=np.float64)
    mean_arr = np.asarray([row.mean_building_edges for row in rows], dtype=np.float64)
    max_arr = np.asarray([row.max_building_edges for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    ax.plot(lst_arr, mean_arr, marker="o", label="mean building links", color="#C1121F")
    ax.plot(lst_arr, max_arr, marker="s", label="max building links", color="#2F6FED")
    ax.set_xlabel("LST (s)")
    ax.set_ylabel("building links")
    ax.set_title(str(title))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
