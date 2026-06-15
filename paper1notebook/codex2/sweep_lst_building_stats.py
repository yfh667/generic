from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from apply_lst_to_dynamic_splice import edge_ports, read_edge_table_csv  # noqa: E402
from run_m56_local_patch_hybrid_topology import DEFAULT_CONFIG  # noqa: E402
from src.topology_workflow.module.config import load_workflow_yaml, viewer_config_from_workflow  # noqa: E402


DEFAULT_DYNAMIC_DIR = (
    Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3\shortest_hops_t0_86160_stride60")
    / "dynamic_splice_topologies"
    / "dyn_m056_m040_ca_b6-12-18_c_t0_86100_s1_sel-transition-dp_pen0p001"
)
DEFAULT_OUT_DIR = Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60\lst_sweep_dp_c_pen001")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep LST values and summarize per-step building-edge counts without "
            "materializing full active/building masks for every LST."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dynamic-dir", type=Path, default=DEFAULT_DYNAMIC_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lst-values", nargs="+", type=int, default=tuple(range(10, 141, 10)))
    parser.add_argument("--progress-every", type=int, default=20000)
    return parser.parse_args()


def read_steps(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [int(row["step"]) for row in csv.DictReader(f)]


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "min": None, "max": None, "p50": None, "p95": None}
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def build_setup_events(
    *,
    steps: np.ndarray,
    target: np.ndarray,
    max_lst: int,
) -> list[list[tuple[int, int, int]]]:
    n_steps, n_edges = target.shape
    events_by_row: list[list[tuple[int, int, int]]] = [[] for _ in range(int(n_steps))]
    for edge_idx in range(int(n_edges)):
        col = np.asarray(target[:, edge_idx], dtype=bool)
        starts = np.flatnonzero(col & np.concatenate(([True], ~col[:-1])))
        for start_raw in starts.tolist():
            start = int(start_raw)
            if start == 0:
                continue
            activation_time = int(steps[start])
            row = start - 1
            while row >= 0 and activation_time - int(steps[row]) <= int(max_lst):
                if not bool(col[row]):
                    delta = int(activation_time - int(steps[row]))
                    events_by_row[row].append((int(edge_idx), int(start), int(delta)))
                row -= 1
    return events_by_row


def compute_for_lst(
    *,
    lst: int,
    steps: np.ndarray,
    target: np.ndarray,
    target_active_counts: np.ndarray,
    events_by_row: list[list[tuple[int, int, int]]],
    ports_by_edge: list[tuple[tuple[int, int], ...]],
    port_to_edges: dict[tuple[int, int], list[int]],
    progress_every: int,
) -> dict[str, Any]:
    n_steps = int(steps.size)
    building_counts = np.zeros(n_steps, dtype=np.int32)
    active_dropped_counts = np.zeros(n_steps, dtype=np.int32)
    building_dropped_counts = np.zeros(n_steps, dtype=np.int32)
    active_counts = np.asarray(target_active_counts, dtype=np.int32).copy()

    started = time.time()
    for row_idx, events in enumerate(events_by_row):
        if not events:
            continue
        candidates = [
            (int(edge_idx), int(deadline))
            for edge_idx, deadline, delta in events
            if int(delta) <= int(lst)
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[1], item[0]))

        port_owner: dict[tuple[int, int], int] = {}
        building_dropped = 0
        for edge_idx, _deadline in candidates:
            ports = ports_by_edge[int(edge_idx)]
            if not ports:
                continue
            if any(port in port_owner for port in ports):
                building_dropped += 1
                continue
            for port in ports:
                port_owner[port] = int(edge_idx)
            building_counts[row_idx] += 1

        if port_owner:
            dropped_edges: set[int] = set()
            row_target = target[row_idx]
            for port, owner in port_owner.items():
                for edge_idx in port_to_edges.get(port, ()):
                    if int(edge_idx) == int(owner):
                        continue
                    if bool(row_target[int(edge_idx)]):
                        dropped_edges.add(int(edge_idx))
            active_dropped_counts[row_idx] = len(dropped_edges)
            active_counts[row_idx] = int(target_active_counts[row_idx]) - len(dropped_edges)
        building_dropped_counts[row_idx] = int(building_dropped)

        if progress_every and (row_idx + 1) % int(progress_every) == 0:
            print(
                f"[lst-sweep] lst={lst} row={row_idx + 1}/{n_steps} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    return {
        "lst": int(lst),
        "building_counts": building_counts,
        "active_dropped_counts": active_dropped_counts,
        "building_dropped_counts": building_dropped_counts,
        "active_counts": active_counts,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "lst",
        "mean_building_edges",
        "max_building_edges",
        "p50_building_edges",
        "p95_building_edges",
        "building_nonzero_steps",
        "building_edge_seconds",
        "mean_active_dropped",
        "max_active_dropped",
        "active_dropped_nonzero_steps",
        "active_dropped_edge_seconds",
        "mean_active_edges",
        "min_active_edges",
        "building_dropped_by_conflict_sum",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_timeseries_csv(
    path: Path,
    *,
    steps: np.ndarray,
    by_lst: dict[int, dict[str, Any]],
) -> None:
    lst_values = sorted(by_lst)
    fieldnames = ["step"]
    for lst in lst_values:
        fieldnames.append(f"building_edges_lst{lst}")
    for lst in lst_values:
        fieldnames.append(f"active_dropped_lst{lst}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, step in enumerate(steps.tolist()):
            row: dict[str, Any] = {"step": int(step)}
            for lst in lst_values:
                row[f"building_edges_lst{lst}"] = int(by_lst[lst]["building_counts"][row_idx])
            for lst in lst_values:
                row[f"active_dropped_lst{lst}"] = int(by_lst[lst]["active_dropped_counts"][row_idx])
            writer.writerow(row)


def plot_outputs(
    *,
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    steps: np.ndarray,
    by_lst: dict[int, dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lst_values = np.asarray([int(row["lst"]) for row in summary_rows], dtype=np.int32)
    mean_building = np.asarray([float(row["mean_building_edges"]) for row in summary_rows], dtype=np.float64)
    max_building = np.asarray([float(row["max_building_edges"]) for row in summary_rows], dtype=np.float64)
    p95_building = np.asarray([float(row["p95_building_edges"]) for row in summary_rows], dtype=np.float64)
    nonzero = np.asarray([int(row["building_nonzero_steps"]) for row in summary_rows], dtype=np.int32)

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9), dpi=170, sharex=True)
    axes[0].plot(lst_values, mean_building, marker="o", color="#2563eb", label="mean")
    axes[0].plot(lst_values, p95_building, marker="o", color="#7c3aed", label="p95")
    axes[0].set_ylabel("building edges")
    axes[0].legend()
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.55)

    axes[1].plot(lst_values, max_building, marker="o", color="#dc2626")
    axes[1].set_ylabel("max building")
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.55)

    axes[2].plot(lst_values, nonzero, marker="o", color="#111827")
    axes[2].set_ylabel("nonzero steps")
    axes[2].set_xlabel("LST (s)")
    axes[2].grid(alpha=0.25, linestyle="--", linewidth=0.55)
    fig.suptitle("Building-edge statistics vs LST")
    fig.tight_layout()
    fig.savefig(out_dir / "lst_building_summary.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15.5, 4.8), dpi=170)
    x = steps.astype(np.float64) / 3600.0
    for lst in sorted(by_lst):
        ax.plot(x, by_lst[lst]["building_counts"], linewidth=0.8, alpha=0.55, label=f"LST={lst}s")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("building edges")
    ax.set_title("Per-step building edges under different LST values")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.55)
    ax.legend(ncol=7, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "lst_building_timeseries.png")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    lst_values = sorted({int(x) for x in args.lst_values})
    if not lst_values:
        raise ValueError("empty --lst-values")
    if min(lst_values) < 0:
        raise ValueError("--lst-values must be non-negative")

    workflow = load_workflow_yaml(args.config)
    config = viewer_config_from_workflow(workflow)
    dynamic_dir = Path(args.dynamic_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = np.asarray(read_steps(dynamic_dir / "dynamic_schedule.csv"), dtype=np.int64)
    target = np.load(dynamic_dir / "edge_active_mask.npy", mmap_mode="r")
    if int(target.shape[0]) != int(steps.size):
        raise ValueError(f"steps and target mask disagree: {steps.size} vs {target.shape[0]}")
    edge_table = read_edge_table_csv(dynamic_dir / "union_edges.csv", total_sats=int(config.total_sats))
    ports_by_edge = [edge_ports(edge_table, idx, p=int(config.P), n=int(config.N)) for idx in range(edge_table.num_edges)]
    port_to_edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for edge_idx, ports in enumerate(ports_by_edge):
        for port in ports:
            port_to_edges[port].append(int(edge_idx))

    print(
        f"[lst-sweep] steps={steps.size} edges={target.shape[1]} lst={lst_values} dynamic_dir={dynamic_dir}",
        flush=True,
    )
    target_active_counts = np.asarray(np.sum(target, axis=1), dtype=np.int32)
    events_by_row = build_setup_events(
        steps=steps,
        target=target,
        max_lst=max(lst_values),
    )
    event_seconds = sum(len(row) for row in events_by_row)
    print(f"[lst-sweep] setup event edge-seconds up to max LST: {event_seconds}", flush=True)

    by_lst: dict[int, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for lst in lst_values:
        started = time.time()
        result = compute_for_lst(
            lst=int(lst),
            steps=steps,
            target=target,
            target_active_counts=target_active_counts,
            events_by_row=events_by_row,
            ports_by_edge=ports_by_edge,
            port_to_edges=port_to_edges,
            progress_every=int(args.progress_every),
        )
        by_lst[int(lst)] = result
        building = result["building_counts"]
        active_dropped = result["active_dropped_counts"]
        active_counts = result["active_counts"]
        building_dropped = result["building_dropped_counts"]
        building_stats = finite_stats(building)
        active_dropped_stats = finite_stats(active_dropped)
        active_stats = finite_stats(active_counts)
        row = {
            "lst": int(lst),
            "mean_building_edges": building_stats["mean"],
            "max_building_edges": building_stats["max"],
            "p50_building_edges": building_stats["p50"],
            "p95_building_edges": building_stats["p95"],
            "building_nonzero_steps": int(np.count_nonzero(building)),
            "building_edge_seconds": int(np.sum(building, dtype=np.int64)),
            "mean_active_dropped": active_dropped_stats["mean"],
            "max_active_dropped": active_dropped_stats["max"],
            "active_dropped_nonzero_steps": int(np.count_nonzero(active_dropped)),
            "active_dropped_edge_seconds": int(np.sum(active_dropped, dtype=np.int64)),
            "mean_active_edges": active_stats["mean"],
            "min_active_edges": active_stats["min"],
            "building_dropped_by_conflict_sum": int(np.sum(building_dropped, dtype=np.int64)),
        }
        summary_rows.append(row)
        print(
            f"[lst-sweep] LST={lst}s done in {time.time() - started:.1f}s "
            f"mean={row['mean_building_edges']:.3f} max={row['max_building_edges']} "
            f"nonzero={row['building_nonzero_steps']}",
            flush=True,
        )

    write_summary_csv(out_dir / "lst_building_summary.csv", summary_rows)
    write_timeseries_csv(out_dir / "lst_building_timeseries_wide.csv", steps=steps, by_lst=by_lst)
    plot_outputs(out_dir=out_dir, summary_rows=summary_rows, steps=steps, by_lst=by_lst)

    meta = {
        "dynamic_dir": str(dynamic_dir),
        "config": str(args.config),
        "lst_values": lst_values,
        "steps": {
            "start": int(steps[0]),
            "end": int(steps[-1]),
            "count": int(steps.size),
            "stride": int(steps[1] - steps[0]) if steps.size > 1 else None,
        },
        "target_shape": [int(x) for x in target.shape],
        "max_lst_event_edge_seconds": int(event_seconds),
        "outputs": {
            "summary_csv": str(out_dir / "lst_building_summary.csv"),
            "timeseries_csv": str(out_dir / "lst_building_timeseries_wide.csv"),
            "summary_plot": str(out_dir / "lst_building_summary.png"),
            "timeseries_plot": str(out_dir / "lst_building_timeseries.png"),
        },
    }
    (out_dir / "lst_sweep_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
