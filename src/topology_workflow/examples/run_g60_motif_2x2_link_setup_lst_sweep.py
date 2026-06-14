from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import build_full_option_edges
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.examples.run_g60_motif_2x2_link_setup_viewer import (
    DEFAULT_GROUP_CACHE,
    DEFAULT_XML,
    MOTIF_2X2_CB,
    build_base_edge_table,
)
from src.topology_workflow.module.edge_tables import make_edge_table_from_records
from src.topology_workflow.module.link_setup_time import (
    P_GRID_ACTIVE,
    P_GRID_BUILDING,
    P_MOTIF_ACTIVE,
    P_MOTIF_BUILDING,
    _compute_active_and_internal_keys_for_state,
    _edge_key,
    _edge_ports,
    _group_state_key_for_step,
    _records_by_key,
)


DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "runs"
    / "paper1"
    / "G60"
    / "link_setup_time"
    / "motif_2x2_CB_china_europe"
    / "lst_sweep_10_140"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep LST values for G60 motif_2x2_CB China/Europe link-setup statistics."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86400)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--lst-values", type=int, nargs="*", default=list(range(10, 141, 10)))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--constrained-groups", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--grid-option", type=int, default=0)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def build_target_payloads(*, group_data: Mapping, steps: Sequence[int], constrained_groups: tuple[int, ...]):
    base_edge_table = build_base_edge_table()
    forced_option_edges = build_full_option_edges(
        G60_CONFIG,
        options=(0,),
        sat_ids=list(base_edge_table.sat_ids),
        wrap_planes=False,
    )
    total_nodes = int(G60_CONFIG.total_sats)
    base_records_by_key = _records_by_key(base_edge_table, total_nodes)
    forced_records_by_key = _records_by_key(forced_option_edges, total_nodes)
    all_records_by_key = dict(base_records_by_key)
    all_records_by_key.update(forced_records_by_key)

    union_edge_table = make_edge_table_from_records(
        p=int(G60_CONFIG.P),
        n=int(G60_CONFIG.N),
        records=list(all_records_by_key.values()),
        sat_ids=list(base_edge_table.sat_ids),
    )
    key_to_col = {
        _edge_key(int(union_edge_table.src[idx]), int(union_edge_table.dst[idx]), total_nodes): int(idx)
        for idx in range(int(union_edge_table.num_edges))
    }
    col_to_key = np.empty(int(union_edge_table.num_edges), dtype=np.int64)
    for key, col in key_to_col.items():
        col_to_key[int(col)] = int(key)

    state_to_id: dict[tuple[tuple[int, ...], ...], int] = {}
    state_payloads: list[dict[str, object]] = []
    state_ids = np.empty(len(steps), dtype=np.int32)
    for row, step in enumerate(steps):
        state_key = _group_state_key_for_step(group_data, int(step), constrained_groups)
        state_id = state_to_id.get(state_key)
        if state_id is None:
            active_keys, internal_keys = _compute_active_and_internal_keys_for_state(
                state_key=state_key,
                total_nodes=total_nodes,
                n=int(G60_CONFIG.N),
                p=int(G60_CONFIG.P),
                forced_option=0,
                wrap_planes=False,
                base_records_by_key=base_records_by_key,
                forced_records_by_key=forced_records_by_key,
            )
            state_id = len(state_payloads)
            state_to_id[state_key] = state_id
            state_payloads.append(
                {
                    "active_cols": np.asarray(
                        [key_to_col[int(key)] for key in active_keys if int(key) in key_to_col],
                        dtype=np.int32,
                    ),
                    "internal_cols": np.asarray(
                        [key_to_col[int(key)] for key in internal_keys if int(key) in key_to_col],
                        dtype=np.int32,
                    ),
                }
            )
        state_ids[row] = int(state_id)

    return union_edge_table, all_records_by_key, key_to_col, col_to_key, state_payloads, state_ids


def build_target_masks(*, steps: Sequence[int], union_edge_table, state_payloads, state_ids):
    n_steps = len(steps)
    n_edges = int(union_edge_table.num_edges)
    target = np.zeros((n_steps, n_edges), dtype=bool)
    grid = np.zeros((n_steps, n_edges), dtype=bool)
    for row in range(n_steps):
        payload = state_payloads[int(state_ids[row])]
        target[row, payload["active_cols"]] = True
        grid[row, payload["internal_cols"]] = True
    inter_cols = np.asarray(np.where(np.asarray(union_edge_table.option) != -1)[0], dtype=np.int32)
    intra_cols = np.asarray(np.where(np.asarray(union_edge_table.option) == -1)[0], dtype=np.int32)
    return target, grid, inter_cols, intra_cols


def interval_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not np.any(mask):
        return []
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]


def compute_counts_for_lst(
    *,
    lst: int,
    times: np.ndarray,
    target: np.ndarray,
    grid: np.ndarray,
    inter_cols: np.ndarray,
    intra_cols: np.ndarray,
    ports_by_col: list[tuple[int, int] | None],
    incident_cols_by_port: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    n_steps, n_edges = target.shape
    target_inter = target[:, inter_cols]
    grid_inter = grid[:, inter_cols]

    priority = np.zeros((n_steps, n_edges), dtype=np.int8)
    is_active_claim = np.zeros((n_steps, n_edges), dtype=bool)

    for local_idx, col in enumerate(inter_cols):
        col = int(col)
        tmask = target_inter[:, local_idx]
        gmask = grid_inter[:, local_idx]
        for start, end in interval_runs(tmask):
            entered_grid = bool(gmask[start])
            if start == 0 or entered_grid:
                rows = slice(start, end)
                priority[rows, col] = np.where(gmask[rows], P_GRID_ACTIVE, P_MOTIF_ACTIVE)
                is_active_claim[rows, col] = True
            else:
                active_start = int(np.searchsorted(times, int(times[start]) + int(lst), side="left"))
                active_start = max(start, min(active_start, end))
                if start < active_start:
                    priority[start:active_start, col] = P_MOTIF_BUILDING
                if active_start < end:
                    rows = slice(active_start, end)
                    priority[rows, col] = np.where(gmask[rows], P_GRID_ACTIVE, P_MOTIF_ACTIVE)
                    is_active_claim[rows, col] = True

            if start > 0 and entered_grid:
                lookback_start = int(np.searchsorted(times, int(times[start]) - int(lst), side="left"))
                lookback_start = max(1, lookback_start)
                if lookback_start < start:
                    rows = np.arange(lookback_start, start, dtype=np.int32)
                    rows = rows[~tmask[rows]]
                    if rows.size:
                        priority[rows, col] = np.maximum(priority[rows, col], P_GRID_BUILDING)

    n_ports = len(incident_cols_by_port)
    port_winner = np.full((n_steps, n_ports), -1, dtype=np.int16)
    port_winner_active = np.zeros((n_steps, n_ports), dtype=bool)
    for port, cols in enumerate(incident_cols_by_port):
        if cols.size == 0:
            continue
        port_priority = priority[:, cols]
        max_priority = port_priority.max(axis=1)
        has_owner = max_priority > 0
        if not np.any(has_owner):
            continue
        local_winner = port_priority.argmax(axis=1)
        winners = cols[local_winner]
        port_winner[has_owner, port] = winners[has_owner]
        port_winner_active[has_owner, port] = is_active_claim[np.flatnonzero(has_owner), winners[has_owner]]

    active_counts = np.zeros(n_steps, dtype=np.int32)
    building_counts = np.zeros(n_steps, dtype=np.int32)
    if intra_cols.size:
        active_counts += np.count_nonzero(target[:, intra_cols], axis=1).astype(np.int32)

    for col in inter_cols:
        col = int(col)
        ports = ports_by_col[col]
        if ports is None:
            continue
        p0, p1 = ports
        owns = (port_winner[:, p0] == col) & (port_winner[:, p1] == col)
        if not np.any(owns):
            continue
        both_active = owns & port_winner_active[:, p0] & port_winner_active[:, p1]
        both_building = owns & (~port_winner_active[:, p0]) & (~port_winner_active[:, p1])
        active_counts += both_active.astype(np.int32)
        building_counts += both_building.astype(np.int32)

    del priority, is_active_claim, port_winner, port_winner_active
    return active_counts, building_counts


def write_timeseries(path: Path, *, steps: Sequence[int], active: np.ndarray, building: np.ndarray) -> None:
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


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty time range")
    lst_values = [int(x) for x in args.lst_values]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    constrained_groups = tuple(int(x) for x in args.constrained_groups)
    union_edge_table, all_records_by_key, key_to_col, col_to_key, state_payloads, state_ids = build_target_payloads(
        group_data=group_data,
        steps=steps,
        constrained_groups=constrained_groups,
    )
    target, grid, inter_cols, intra_cols = build_target_masks(
        steps=steps,
        union_edge_table=union_edge_table,
        state_payloads=state_payloads,
        state_ids=state_ids,
    )

    n_ports = int(G60_CONFIG.total_sats) * 2
    ports_by_col: list[tuple[int, int] | None] = [None] * int(union_edge_table.num_edges)
    incident: list[list[int]] = [[] for _ in range(n_ports)]
    for col in inter_cols:
        key = int(col_to_key[int(col)])
        record = all_records_by_key[key]
        ports = _edge_ports(record, n=int(G60_CONFIG.N), p=int(G60_CONFIG.P), wrap_planes=False)
        if len(ports) != 2:
            continue
        port_ids = tuple(int(node) * 2 + int(side) for node, side in ports)
        ports_by_col[int(col)] = (port_ids[0], port_ids[1])
        incident[port_ids[0]].append(int(col))
        incident[port_ids[1]].append(int(col))
    incident_cols_by_port = [np.asarray(sorted(cols), dtype=np.int32) for cols in incident]

    times = np.asarray(steps, dtype=np.int64)
    summary_rows = []
    plot_series: dict[int, np.ndarray] = {}
    for lst in lst_values:
        active_counts, building_counts = compute_counts_for_lst(
            lst=int(lst),
            times=times,
            target=target,
            grid=grid,
            inter_cols=inter_cols,
            intra_cols=intra_cols,
            ports_by_col=ports_by_col,
            incident_cols_by_port=incident_cols_by_port,
        )
        plot_series[int(lst)] = building_counts
        timeseries_path = out_dir / f"building_link_timeseries_lst{int(lst):03d}.csv"
        write_timeseries(timeseries_path, steps=steps, active=active_counts, building=building_counts)
        max_value = int(building_counts.max()) if building_counts.size else 0
        max_positions = np.flatnonzero(building_counts == max_value)
        first_max_step = int(times[int(max_positions[0])]) if max_positions.size else None
        summary_rows.append(
            {
                "lst_s": int(lst),
                "mean_building_edges": float(np.mean(building_counts)),
                "max_building_edges": max_value,
                "first_max_step_s": first_max_step,
                "nonzero_steps": int(np.count_nonzero(building_counts)),
                "p95_building_edges": float(np.percentile(building_counts, 95)),
                "total_building_edge_seconds": int(np.sum(building_counts)),
                "mean_active_edges": float(np.mean(active_counts)),
                "timeseries_csv": str(timeseries_path),
            }
        )
        print(f"[lst-sweep] LST={lst}s mean={summary_rows[-1]['mean_building_edges']:.3f} max={max_value}", flush=True)
        del active_counts
        gc.collect()

    summary_path = out_dir / "lst_sweep_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    x_hours = times.astype(np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(16, 7), dpi=180)
    cmap = plt.get_cmap("turbo")
    for idx, lst in enumerate(lst_values):
        color = cmap(idx / max(1, len(lst_values) - 1))
        ax.plot(x_hours, plot_series[int(lst)], linewidth=0.85, alpha=0.82, color=color, label=f"{int(lst)}s")
    ax.set_title("G60 motif_2x2_CB China/Europe building-link count under LST sweep")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("building links")
    ax.grid(True, alpha=0.25)
    ax.legend(title="LST", ncol=2, fontsize=8, title_fontsize=9, loc="upper right")
    fig.tight_layout()
    plot_path = out_dir / "lst_sweep_building_link_timeseries.png"
    fig.savefig(plot_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)
    lst_arr = np.asarray([row["lst_s"] for row in summary_rows], dtype=np.float64)
    mean_arr = np.asarray([row["mean_building_edges"] for row in summary_rows], dtype=np.float64)
    max_arr = np.asarray([row["max_building_edges"] for row in summary_rows], dtype=np.float64)
    ax.plot(lst_arr, mean_arr, marker="o", label="mean building links", color="#C1121F")
    ax.plot(lst_arr, max_arr, marker="s", label="max building links", color="#2F6FED")
    ax.set_xlabel("LST (s)")
    ax.set_ylabel("building links")
    ax.set_title("Building-link count summary vs LST")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    summary_plot_path = out_dir / "lst_sweep_summary.png"
    fig.savefig(summary_plot_path)
    plt.close(fig)

    print(
        {
            "out_dir": str(out_dir),
            "summary_csv": str(summary_path),
            "timeseries_plot": str(plot_path),
            "summary_plot": str(summary_plot_path),
            "lst_values": lst_values,
            "steps": len(steps),
            "union_edges": int(union_edge_table.num_edges),
            "motif": MOTIF_2X2_CB["name"],
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
