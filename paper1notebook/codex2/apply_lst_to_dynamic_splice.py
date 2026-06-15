from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_m56_local_patch_hybrid_topology import DEFAULT_CONFIG, path_from, region_pairs_from_workflow  # noqa: E402
from run_m56_m40_dynamic_splice_86100 import read_edge_table_csv  # noqa: E402
from weighted_base_viewer import EdgeUsageTopology2DViewer  # noqa: E402

from src.link_delay.module.edge_options import EdgeTable, write_edges_csv  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.edge_betweenness import (  # noqa: E402
    build_undirected_adjacency,
    edge_betweenness_between_node_sets,
)
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.config import (  # noqa: E402
    load_workflow_yaml,
    viewer_config_from_workflow,
)


DEFAULT_DYNAMIC_DIR = (
    Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3")
    / "shortest_hops_t0_86160_stride60"
    / "dynamic_splice_topologies"
    / "dyn_m056_m040_ca_b6-12-18_c-cb-all_t0_86100_s60"
)


@dataclass(frozen=True)
class LstStats:
    step: int
    target_active_edges: int
    active_edges: int
    building_edges: int
    active_dropped_by_building: int
    building_dropped_by_conflict: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply link-setup-time constraints to an existing dynamic splice active-mask. "
            "Future active links are reserved backward in time; conflicting active links are dropped."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DYNAMIC_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--link-setup-time", type=float, default=120.0, help="LST in seconds.")
    parser.add_argument("--metric-pairs", nargs="+", default=("china_europe", "china_america", "china_africa"))
    parser.add_argument("--usage-pairs", nargs="+", default=("china_america",))
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--topology-only",
        action="store_true",
        help="Only compute active/building masks and step stats; skip shortest-path metrics and edge usage.",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def default_out_dir(input_dir: Path, lst_s: float) -> Path:
    return Path(input_dir) / f"lst{float(lst_s):g}s_backward"


def load_steps(schedule_path: Path) -> list[int]:
    with schedule_path.open("r", encoding="utf-8-sig", newline="") as f:
        return [int(row["step"]) for row in csv.DictReader(f)]


def edge_record(edge_table: EdgeTable, idx: int) -> tuple[int, int, int, int, int]:
    return (
        int(edge_table.src_plane[idx]),
        int(edge_table.src_y[idx]),
        int(edge_table.dst_plane[idx]),
        int(edge_table.dst_y[idx]),
        int(edge_table.option[idx]),
    )


def record_nodes(record: tuple[int, int, int, int, int], n: int) -> tuple[int, int]:
    src_plane, src_y, dst_plane, dst_y, _option = record
    return int(src_plane) * int(n) + int(src_y), int(dst_plane) * int(n) + int(dst_y)


def option_side_for_node(
    record: tuple[int, int, int, int, int],
    node: int,
    *,
    n: int,
    p: int,
    wrap_planes: bool = False,
) -> int | None:
    from src.link_delay.module.edge_options import OPTION_DELTAS

    src_plane, src_y, dst_plane, dst_y, option = record
    if int(option) == -1:
        return None
    src, dst = record_nodes(record, int(n))
    if int(node) not in (src, dst):
        return None

    if int(option) in OPTION_DELTAS:
        dp, _dy = OPTION_DELTAS[int(option)]
        if bool(wrap_planes):
            src_to_dst = (int(src_plane) + int(dp)) % int(p) == int(dst_plane)
            dst_to_src = (int(dst_plane) + int(dp)) % int(p) == int(src_plane)
        else:
            src_to_dst = int(src_plane) + int(dp) == int(dst_plane)
            dst_to_src = int(dst_plane) + int(dp) == int(src_plane)
        if src_to_dst and not dst_to_src:
            return 1 if int(node) == src else 0
        if dst_to_src and not src_to_dst:
            return 1 if int(node) == dst else 0

    if int(node) == src:
        return 1 if int(dst_plane) > int(src_plane) else 0
    return 0 if int(src_plane) < int(dst_plane) else 1


def edge_ports(edge_table: EdgeTable, idx: int, *, p: int, n: int) -> tuple[tuple[int, int], ...]:
    record = edge_record(edge_table, int(idx))
    if int(record[4]) == -1:
        return tuple()
    src, dst = record_nodes(record, int(n))
    ports: list[tuple[int, int]] = []
    for node in (src, dst):
        side = option_side_for_node(record, int(node), n=int(n), p=int(p), wrap_planes=False)
        if side is not None:
            ports.append((int(node), int(side)))
    return tuple(ports)


def build_building_requests(
    *,
    steps: list[int],
    target_active_mask: np.ndarray,
    setup_time_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build backward-looking setup requests.

    If edge e first becomes target-active at row r, then rows before r whose
    sampled time lies within LST request building(e). Row 0 is treated as a warm
    start because links already active at the first observed step may have been
    established before the time window.
    """

    steps_arr = np.asarray(steps, dtype=np.int64)
    target = np.asarray(target_active_mask, dtype=bool)
    n_steps, n_edges = target.shape
    requested = np.zeros((n_steps, n_edges), dtype=bool)
    deadline = np.full((n_steps, n_edges), n_steps + 1, dtype=np.int32)
    lst = float(setup_time_seconds)
    if lst < 0:
        raise ValueError("--link-setup-time must be >= 0")
    if lst == 0:
        return requested, deadline

    for edge_idx in range(n_edges):
        col = target[:, edge_idx]
        starts = np.flatnonzero(col & np.concatenate(([True], ~col[:-1])))
        for start in starts:
            start = int(start)
            if start == 0:
                continue
            activation_time = int(steps_arr[start])
            row = start - 1
            while row >= 0 and activation_time - int(steps_arr[row]) <= lst:
                if not bool(target[row, edge_idx]):
                    requested[row, edge_idx] = True
                    if start < int(deadline[row, edge_idx]):
                        deadline[row, edge_idx] = start
                row -= 1
    return requested, deadline


def apply_backward_lst(
    *,
    steps: list[int],
    edge_table: EdgeTable,
    target_active_mask: np.ndarray,
    setup_time_seconds: float,
    p: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray, tuple[LstStats, ...]]:
    target = np.asarray(target_active_mask, dtype=bool)
    requested_building, deadline = build_building_requests(
        steps=steps,
        target_active_mask=target,
        setup_time_seconds=float(setup_time_seconds),
    )
    n_steps, n_edges = target.shape
    active = np.zeros_like(target, dtype=bool)
    building = np.zeros_like(target, dtype=bool)
    ports_by_edge = [edge_ports(edge_table, idx, p=int(p), n=int(n)) for idx in range(n_edges)]
    stats: list[LstStats] = []

    for row in range(n_steps):
        port_owner: dict[tuple[int, int], int] = {}
        building_candidates = np.flatnonzero(requested_building[row])
        building_candidates = sorted(
            (int(idx) for idx in building_candidates),
            key=lambda idx: (int(deadline[row, idx]), int(idx)),
        )
        building_dropped = 0
        for edge_idx in building_candidates:
            ports = ports_by_edge[int(edge_idx)]
            if not ports:
                continue
            if any(port in port_owner for port in ports):
                building_dropped += 1
                continue
            for port in ports:
                port_owner[port] = int(edge_idx)
            building[row, int(edge_idx)] = True

        active_dropped = 0
        for edge_idx in np.flatnonzero(target[row]):
            edge_idx = int(edge_idx)
            ports = ports_by_edge[edge_idx]
            if not ports:
                active[row, edge_idx] = True
                continue
            conflict = False
            for port in ports:
                owner = port_owner.get(port)
                if owner is not None and int(owner) != edge_idx:
                    conflict = True
                    break
            if conflict:
                active_dropped += 1
                continue
            active[row, edge_idx] = True

        stats.append(
            LstStats(
                step=int(steps[row]),
                target_active_edges=int(np.count_nonzero(target[row])),
                active_edges=int(np.count_nonzero(active[row])),
                building_edges=int(np.count_nonzero(building[row])),
                active_dropped_by_building=int(active_dropped),
                building_dropped_by_conflict=int(building_dropped),
            )
        )
    return active, building, tuple(stats)


def subset_edge_table(edge_table: EdgeTable, cols: np.ndarray) -> EdgeTable:
    cols = np.asarray(cols, dtype=np.int32)
    return EdgeTable(
        src=np.asarray(edge_table.src[cols], dtype=np.int32),
        dst=np.asarray(edge_table.dst[cols], dtype=np.int32),
        option=np.asarray(edge_table.option[cols], dtype=np.int16),
        src_plane=np.asarray(edge_table.src_plane[cols], dtype=np.int16),
        src_y=np.asarray(edge_table.src_y[cols], dtype=np.int16),
        dst_plane=np.asarray(edge_table.dst_plane[cols], dtype=np.int16),
        dst_y=np.asarray(edge_table.dst_y[cols], dtype=np.int16),
        sat_ids=list(edge_table.sat_ids),
    )


def compute_metrics_and_usage(
    *,
    steps: list[int],
    edge_table: EdgeTable,
    active_mask: np.ndarray,
    config,
    group_data: dict,
    pair_by_key: dict,
    metric_pairs: list[str],
    usage_pairs: list[str],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    usage_values = np.zeros((len(steps), int(edge_table.num_edges)), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for row_idx, step in enumerate(steps):
        cols = np.flatnonzero(active_mask[row_idx])
        active_table = subset_edge_table(edge_table, cols)
        adjacency = build_undirected_adjacency(active_table, int(config.total_sats))
        for pair_key in metric_pairs:
            pair = pair_by_key[pair_key]
            values, summary, _samples = edge_betweenness_between_node_sets(
                active_table,
                total_nodes=int(config.total_sats),
                source_nodes=group_nodes_for_step(group_data, int(step), int(pair.source_group_id)),
                target_nodes=group_nodes_for_step(group_data, int(step), int(pair.target_group_id)),
                adjacency=adjacency,
                sample_path_limit=0,
            )
            rows.append(
                {
                    "step": int(step),
                    "pair": str(pair_key),
                    "pair_label": str(pair.label),
                    "source_nodes": int(summary.source_nodes),
                    "target_nodes": int(summary.target_nodes),
                    "reachable_pairs": int(summary.reachable_pairs),
                    "mean_hops": float(summary.mean_shortest_distance_hops),
                    "total_hops": float(summary.total_shortest_distance_hops),
                    "max_edge_usage": float(summary.max_edge_betweenness),
                    "nonzero_edges": int(summary.nonzero_edges),
                    "edge_usage_sum": float(summary.edge_value_sum),
                }
            )
            if pair_key in set(usage_pairs):
                usage_values[row_idx, cols] += values
        if (row_idx + 1) % 100 == 0 or row_idx + 1 == len(steps):
            print(f"[lst-dynamic-splice] metrics/usage {row_idx + 1}/{len(steps)} step={step}", flush=True)
    return rows, usage_values


def write_stats(path: Path, stats: Iterable[LstStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "target_active_edges",
                "active_edges",
                "building_edges",
                "active_dropped_by_building",
                "building_dropped_by_conflict",
            ],
        )
        writer.writeheader()
        for row in stats:
            writer.writerow(row.__dict__)


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "step",
            "pair",
            "pair_label",
            "source_nodes",
            "target_nodes",
            "reachable_pairs",
            "mean_hops",
            "total_hops",
            "max_edge_usage",
            "nonzero_edges",
            "edge_usage_sum",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_lst_summary(path: Path, *, stats: tuple[LstStats, ...], metrics: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([row.step for row in stats], dtype=np.float64)
    x_hours = steps / 3600.0
    active = np.asarray([row.active_edges for row in stats], dtype=np.float64)
    target = np.asarray([row.target_active_edges for row in stats], dtype=np.float64)
    building = np.asarray([row.building_edges for row in stats], dtype=np.float64)
    dropped = np.asarray([row.active_dropped_by_building for row in stats], dtype=np.float64)

    pair_to_values: dict[str, list[tuple[int, float]]] = {}
    for row in metrics:
        pair_to_values.setdefault(str(row["pair"]), []).append((int(row["step"]), float(row["mean_hops"])))

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 8.2), dpi=165, sharex=True)
    axes[0].plot(x_hours, target, color="#64748b", linewidth=1.4, label="target active edges")
    axes[0].plot(x_hours, active, color="#111827", linewidth=1.4, label="LST-feasible active edges")
    axes[0].plot(x_hours, building, color="#2F6FED", linewidth=1.2, label="building edges")
    axes[0].plot(x_hours, dropped, color="#C1121F", linewidth=1.0, label="active dropped by building")
    axes[0].set_ylabel("edge count")
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.55)
    axes[0].legend(loc="upper right", fontsize=8)

    colors = ["#2563eb", "#C1121F", "#16a34a", "#7c3aed"]
    for idx, (pair, values) in enumerate(pair_to_values.items()):
        values = sorted(values)
        axes[1].plot(
            np.asarray([step for step, _value in values], dtype=np.float64) / 3600.0,
            np.asarray([value for _step, value in values], dtype=np.float64),
            linewidth=1.4,
            color=colors[idx % len(colors)],
            label=f"{pair} mean hops",
        )
    axes[1].set_xlabel("time (hour)")
    axes[1].set_ylabel("mean shortest hops")
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.55)
    if pair_to_values:
        axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Dynamic splice with backward link setup time constraint", y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def load_or_build(args: argparse.Namespace) -> dict[str, Any]:
    workflow = load_workflow_yaml(args.config)
    config = viewer_config_from_workflow(workflow)
    paths_raw = workflow.get("paths", {})
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else default_out_dir(input_dir, float(args.link_setup_time))

    steps_path = out_dir / "steps.npy"
    if args.reuse and steps_path.exists():
        steps = [int(x) for x in np.load(steps_path)]
        edge_table = read_edge_table_csv(out_dir / "union_edges.csv", total_sats=int(config.total_sats))
        active_mask = np.load(out_dir / "edge_active_mask.npy")
        building_mask = np.load(out_dir / "edge_building_mask.npy")
        usage_values = np.load(out_dir / "edge_usage_values.npy")
    else:
        steps = load_steps(input_dir / "dynamic_schedule.csv")
        edge_table = read_edge_table_csv(input_dir / "union_edges.csv", total_sats=int(config.total_sats))
        target_active_mask = np.load(input_dir / "edge_active_mask.npy")
        if target_active_mask.shape != (len(steps), int(edge_table.num_edges)):
            raise ValueError(
                f"target active mask shape {target_active_mask.shape} does not match "
                f"steps={len(steps)} edges={edge_table.num_edges}"
            )
        active_mask, building_mask, stats = apply_backward_lst(
            steps=steps,
            edge_table=edge_table,
            target_active_mask=target_active_mask,
            setup_time_seconds=float(args.link_setup_time),
            p=int(config.P),
            n=int(config.N),
        )
        metric_pairs = [str(x) for x in args.metric_pairs]
        usage_pairs = [str(x) for x in args.usage_pairs]
        if bool(args.topology_only):
            metrics = []
            usage_values = np.zeros((1, int(edge_table.num_edges)), dtype=np.float32)
            print("[lst-dynamic-splice] topology_only=True, skipped metrics and edge usage", flush=True)
        else:
            group_data_for_metrics = load_or_build_group_data(
                xml_file=path_from(paths_raw, "group_xml"),
                group_cache_dir=path_from(paths_raw, "group_cache_dir"),
                steps=steps,
                station_groups=config.station_groups,
                total_sats=config.total_sats,
                constellation_name=config.name,
                stride=int(steps[1] - steps[0]) if len(steps) > 1 else 1,
                enabled=True,
                force=bool(args.force_group_cache),
            )
            pair_by_key = region_pairs_from_workflow(workflow)
            missing = [key for key in list(dict.fromkeys(metric_pairs + usage_pairs)) if key not in pair_by_key]
            if missing:
                raise ValueError(f"unknown region pair(s): {missing}; available={sorted(pair_by_key)}")
            metrics, usage_values = compute_metrics_and_usage(
                steps=steps,
                edge_table=edge_table,
                active_mask=active_mask,
                config=config,
                group_data=group_data_for_metrics,
                pair_by_key=pair_by_key,
                metric_pairs=metric_pairs,
                usage_pairs=usage_pairs,
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "steps.npy", np.asarray(steps, dtype=np.int64))
        np.save(out_dir / "target_edge_active_mask.npy", target_active_mask.astype(bool))
        np.save(out_dir / "edge_active_mask.npy", active_mask.astype(bool))
        np.save(out_dir / "edge_building_mask.npy", building_mask.astype(bool))
        np.save(out_dir / "edge_usage_values.npy", usage_values.astype(np.float32))
        write_edges_csv(edge_table, out_dir / "union_edges.csv")
        write_stats(out_dir / "lst_step_stats.csv", stats)
        write_metrics(out_dir / "lst_metrics.csv", metrics)
        plot_lst_summary(out_dir / "lst_summary.png", stats=stats, metrics=metrics)
        meta = {
            "input_dir": str(input_dir),
            "link_setup_time_seconds": float(args.link_setup_time),
            "steps": [int(steps[0]), int(steps[-1]), int(steps[1] - steps[0]) if len(steps) > 1 else None],
            "num_steps": int(len(steps)),
            "num_edges": int(edge_table.num_edges),
            "target_active_state_rows": int(np.count_nonzero(target_active_mask)),
            "active_state_rows": int(np.count_nonzero(active_mask)),
            "building_state_rows": int(np.count_nonzero(building_mask)),
            "active_dropped_state_rows": int(np.count_nonzero(target_active_mask & ~active_mask)),
            "metric_pairs": metric_pairs,
            "usage_pairs": usage_pairs,
            "topology_only": bool(args.topology_only),
            "rule": (
                "For every edge that starts a future target-active run, reserve both endpoint ports backward "
                "within the LST window. Building edges are not routable. Active edges sharing a reserved port "
                "with a different building edge are dropped at that sampled time."
            ),
            "outputs": {
                "steps": str(out_dir / "steps.npy"),
                "union_edges": str(out_dir / "union_edges.csv"),
                "target_edge_active_mask": str(out_dir / "target_edge_active_mask.npy"),
                "edge_active_mask": str(out_dir / "edge_active_mask.npy"),
                "edge_building_mask": str(out_dir / "edge_building_mask.npy"),
                "edge_usage_values": str(out_dir / "edge_usage_values.npy"),
                "lst_step_stats": str(out_dir / "lst_step_stats.csv"),
                "lst_metrics": str(out_dir / "lst_metrics.csv"),
                "plot": str(out_dir / "lst_summary.png"),
            },
        }
        (out_dir / "lst_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    group_data = load_or_build_group_data(
        xml_file=path_from(paths_raw, "group_xml"),
        group_cache_dir=path_from(paths_raw, "group_cache_dir"),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(steps[1] - steps[0]) if len(steps) > 1 else 1,
        enabled=True,
        force=False,
    )
    return {
        "workflow": workflow,
        "config": config,
        "steps": steps,
        "out_dir": out_dir,
        "edge_table": edge_table,
        "active_mask": active_mask,
        "building_mask": building_mask,
        "usage_values": usage_values,
        "group_data": group_data,
    }


def main() -> int:
    args = parse_args()
    outputs = load_or_build(args)
    if args.check_only:
        print(f"[lst-dynamic-splice] out_dir={outputs['out_dir']}", flush=True)
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    usage_values = outputs["usage_values"]
    viewer = EdgeUsageTopology2DViewer(
        outputs["config"],
        steps=outputs["steps"],
        edge_table=outputs["edge_table"],
        edge_active_mask=outputs["active_mask"],
        edge_building_mask=outputs["building_mask"],
        edge_usage_values=usage_values,
        value_max=float(np.nanmax(usage_values)) if usage_values.size else 1.0,
        window_title=f"G60 dynamic splice with backward LST={float(args.link_setup_time):g}s",
        group_data=outputs["group_data"],
        show_groups=True,
        edge_value_label="+".join(str(x) for x in args.usage_pairs) + " edge usage",
        value_color_mode="red_alpha",
        value_solid_color="#C1121F",
        value_alpha_min=35,
        value_alpha_max=210,
        zero_value_edges_visible=False,
        zero_value_threshold=1e-9,
        show_topology_under_edge_values=True,
        topology_edge_color="#000000",
        topology_edge_alpha=120,
        topology_edge_width=0.018,
        building_edge_color="#2F6FED",
        building_edge_alpha=180,
        building_edge_width=0.032,
        scale_edge_width_by_value=True,
        value_width_min=0.010,
        value_width_max=0.070,
        hide_y_wrap_edges=True,
        show_grid_lines=False,
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
