from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.link_delay.module.position_cache import PositionCacheStore  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.edge_tables import INTRA_OPTION  # noqa: E402
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step  # noqa: E402


DEFAULT_EXPERIMENT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_0056_0061_common_spike_analysis\full_link_guided_spike_patch"
    r"\t0_86160_stride60"
)
DEFAULT_FULL_LINK_CACHE = Path(r"E:\paper11\data\linshi\g60_multi_region_weighted_betweenness_t0_86164_stride60")
DEFAULT_DELAY_STORE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\full_option_edge_delay\G60_full_options_t0_86164_stride1"
)
DEFAULT_POSITION_CACHE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\_position_cache\cache_0_86164_1s"
)
DEFAULT_GROUP_XML = PROJECT_ROOT / "data" / "basic_file" / "G60" / "satellitesposition" / "station_visible_satellites_20250106.xml"
DEFAULT_GROUP_CACHE_DIR = PROJECT_ROOT / "data" / "satnet_experiments" / "caches" / "G60" / "group_data_cache"

PAIR_KEY = "china_europe"
SOURCE_GROUP_ID = 2
TARGET_GROUP_ID = 3


@dataclass(frozen=True)
class LstSecondStats:
    step: int
    target_active_edges: int
    active_edges: int
    building_edges: int
    active_dropped_by_building: int
    building_dropped_by_conflict: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep link-setup-time influence on the full-link-guided dynamic one-right topology. "
            "The target topology sequence is evaluated on a 1s time axis before LST is applied."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--full-link-cache", type=Path, default=DEFAULT_FULL_LINK_CACHE)
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE_DIR)
    parser.add_argument("--position-cache-dir", type=Path, default=DEFAULT_POSITION_CACHE_DIR)
    parser.add_argument("--group-xml", type=Path, default=DEFAULT_GROUP_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--lst-start", type=int, default=0)
    parser.add_argument("--lst-end", type=int, default=140)
    parser.add_argument("--lst-step", type=int, default=10)
    parser.add_argument("--metric-top-drop-rows", type=int, default=420)
    parser.add_argument("--metric-regular-period", type=int, default=60)
    parser.add_argument("--save-detail-for-lst", nargs="*", type=int, default=[60, 120])
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def read_edge_table_csv(path: Path, *, total_sats: int) -> EdgeTable:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty edge CSV: {path}")
    return EdgeTable(
        src=np.asarray([int(row["src_node"]) for row in rows], dtype=np.int32),
        dst=np.asarray([int(row["dst_node"]) for row in rows], dtype=np.int32),
        option=np.asarray([int(row["option"]) for row in rows], dtype=np.int16),
        src_plane=np.asarray([int(row["src_plane"]) for row in rows], dtype=np.int16),
        src_y=np.asarray([int(row["src_y"]) for row in rows], dtype=np.int16),
        dst_plane=np.asarray([int(row["dst_plane"]) for row in rows], dtype=np.int16),
        dst_y=np.asarray([int(row["dst_y"]) for row in rows], dtype=np.int16),
        sat_ids=[str(i + 1) for i in range(int(total_sats))],
    )


def port_keys_by_edge(edge_table: EdgeTable) -> list[tuple[tuple[int, str], ...]]:
    out: list[tuple[tuple[int, str], ...]] = []
    for idx in range(int(edge_table.num_edges)):
        if int(edge_table.option[idx]) == INTRA_OPTION:
            out.append(tuple())
        else:
            out.append(((int(edge_table.src[idx]), "right"), (int(edge_table.dst[idx]), "left")))
    return out


def target_row_for_second(step: int, coarse_steps: np.ndarray) -> int:
    idx = int(np.searchsorted(coarse_steps, int(step), side="right") - 1)
    return max(0, min(idx, int(coarse_steps.shape[0]) - 1))


def build_target_port_owner_by_row(
    *,
    target_mask_60s: np.ndarray,
    ports_by_edge: list[tuple[tuple[int, str], ...]],
) -> list[dict[tuple[int, str], int]]:
    out: list[dict[tuple[int, str], int]] = []
    for row in range(int(target_mask_60s.shape[0])):
        owner: dict[tuple[int, str], int] = {}
        for edge in np.flatnonzero(target_mask_60s[row]):
            for port in ports_by_edge[int(edge)]:
                owner[port] = int(edge)
        out.append(owner)
    return out


def build_setup_events(
    *,
    target_mask_60s: np.ndarray,
    coarse_steps: np.ndarray,
    lst_s: int,
    total_seconds: int,
) -> list[list[tuple[int, int]]]:
    events: list[list[tuple[int, int]]] = [[] for _ in range(int(total_seconds) + 1)]
    if int(lst_s) <= 0:
        return events
    for coarse_row in range(1, int(target_mask_60s.shape[0])):
        activation_time = int(coarse_steps[coarse_row])
        added = np.flatnonzero(target_mask_60s[coarse_row] & ~target_mask_60s[coarse_row - 1])
        if added.size == 0:
            continue
        start_time = max(0, activation_time - int(lst_s))
        for edge in added:
            events[int(start_time)].append((int(activation_time), int(edge)))
    return events


def simulate_lst_stats_and_masks(
    *,
    edge_table: EdgeTable,
    target_mask_60s: np.ndarray,
    coarse_steps: np.ndarray,
    ports_by_edge: list[tuple[tuple[int, str], ...]],
    target_port_owner_by_row: list[dict[tuple[int, str], int]],
    lst_s: int,
    eval_steps: Iterable[int] = (),
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    start = int(coarse_steps[0])
    end = int(coarse_steps[-1])
    if start != 0:
        raise ValueError("this experiment expects coarse steps to start at 0")
    target_counts = np.count_nonzero(target_mask_60s, axis=1).astype(np.int32)
    setup_start_events = build_setup_events(
        target_mask_60s=target_mask_60s,
        coarse_steps=coarse_steps,
        lst_s=int(lst_s),
        total_seconds=end,
    )
    eval_set = {int(x) for x in eval_steps}
    eval_masks: dict[int, np.ndarray] = {}
    current_events: list[tuple[int, int]] = []
    stats_rows: list[LstSecondStats] = []

    for step in range(start, end + 1):
        if setup_start_events[step]:
            current_events.extend(setup_start_events[step])
        if current_events:
            current_events = [(deadline, edge) for deadline, edge in current_events if int(deadline) > int(step)]

        coarse_row = target_row_for_second(step, coarse_steps)
        target_row = target_mask_60s[coarse_row]
        target_owner = target_port_owner_by_row[coarse_row]

        building_port_owner: dict[tuple[int, str], int] = {}
        accepted_building_edges: set[int] = set()
        building_dropped = 0
        candidates = sorted(
            (
                (int(deadline), int(edge))
                for deadline, edge in current_events
                if not bool(target_row[int(edge)])
            ),
            key=lambda item: (item[0], item[1]),
        )
        for _deadline, edge in candidates:
            ports = ports_by_edge[int(edge)]
            if not ports:
                continue
            if any(port in building_port_owner for port in ports):
                building_dropped += 1
                continue
            for port in ports:
                building_port_owner[port] = int(edge)
            accepted_building_edges.add(int(edge))

        dropped_target_edges: set[int] = set()
        for port, building_edge in building_port_owner.items():
            target_edge = target_owner.get(port)
            if target_edge is not None and int(target_edge) != int(building_edge):
                dropped_target_edges.add(int(target_edge))

        target_count = int(target_counts[coarse_row])
        active_count = int(target_count - len(dropped_target_edges))
        stats_rows.append(
            LstSecondStats(
                step=int(step),
                target_active_edges=target_count,
                active_edges=active_count,
                building_edges=int(len(accepted_building_edges)),
                active_dropped_by_building=int(len(dropped_target_edges)),
                building_dropped_by_conflict=int(building_dropped),
            )
        )

        if int(step) in eval_set:
            mask = np.asarray(target_row, dtype=bool).copy()
            if dropped_target_edges:
                mask[np.asarray(sorted(dropped_target_edges), dtype=np.int32)] = False
            eval_masks[int(step)] = mask

    return pd.DataFrame([row.__dict__ for row in stats_rows]), eval_masks


def choose_eval_steps(
    *,
    stats: pd.DataFrame,
    coarse_steps: np.ndarray,
    metric_regular_period: int,
    metric_top_drop_rows: int,
) -> list[int]:
    regular = set(range(int(coarse_steps[0]), int(coarse_steps[-1]) + 1, max(1, int(metric_regular_period))))
    affected = stats[stats["active_dropped_by_building"] > 0].copy()
    if int(metric_top_drop_rows) > 0 and not affected.empty:
        affected = affected.sort_values(
            ["active_dropped_by_building", "building_edges", "step"],
            ascending=[False, False, True],
        ).head(int(metric_top_drop_rows))
        regular.update(int(x) for x in affected["step"].tolist())
    return sorted(regular)


def mean_metrics_for_mask(
    *,
    mask: np.ndarray,
    weights: np.ndarray,
    edge_table: EdgeTable,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
) -> tuple[float, int, float, int]:
    if not sources or not targets:
        return math.nan, 0, math.nan, 0
    cols = np.flatnonzero(mask)
    if cols.size == 0:
        return math.nan, 0, math.nan, 0
    src = np.asarray(edge_table.src[cols], dtype=np.int32)
    dst = np.asarray(edge_table.dst[cols], dtype=np.int32)
    row_index = np.concatenate([src, dst])
    col_index = np.concatenate([dst, src])
    source_arr = np.asarray(sources, dtype=np.int32)
    target_arr = np.asarray(targets, dtype=np.int32)

    hop_graph = csr_matrix(
        (np.ones(cols.size * 2, dtype=np.float32), (row_index, col_index)),
        shape=(int(G60_CONFIG.total_sats), int(G60_CONFIG.total_sats)),
    )
    hop_matrix = dijkstra(hop_graph, directed=False, unweighted=True, indices=source_arr)
    hop_values = np.atleast_2d(np.asarray(hop_matrix, dtype=np.float64))[:, target_arr]
    hop_finite = hop_values[np.isfinite(hop_values)]

    selected_weights = np.asarray(weights[cols], dtype=np.float32)
    delay_graph = csr_matrix(
        (np.concatenate([selected_weights, selected_weights]), (row_index, col_index)),
        shape=(int(G60_CONFIG.total_sats), int(G60_CONFIG.total_sats)),
    )
    delay_matrix = dijkstra(delay_graph, directed=False, indices=source_arr)
    delay_values = np.atleast_2d(np.asarray(delay_matrix, dtype=np.float64))[:, target_arr]
    delay_finite = delay_values[np.isfinite(delay_values)]

    return (
        float(np.mean(hop_finite)) if hop_finite.size else math.nan,
        int(hop_finite.size),
        float(np.mean(delay_finite)) if delay_finite.size else math.nan,
        int(delay_finite.size),
    )


def evaluate_metric_rows(
    *,
    lst_s: int,
    eval_steps: list[int],
    lst_active_masks: dict[int, np.ndarray],
    target_mask_60s: np.ndarray,
    coarse_steps: np.ndarray,
    edge_table: EdgeTable,
    group_data: dict,
    delay_store: FullLinkDelayStore,
    position_store: PositionCacheStore,
    lookup,
    progress_every: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for idx, step in enumerate(eval_steps):
        target_row = target_mask_60s[target_row_for_second(int(step), coarse_steps)]
        active_mask = lst_active_masks[int(step)]
        weights = edge_weights_for_step(
            edge_table=edge_table,
            lookup=lookup,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=delay_store.row_for_time_step(int(step)),
            position_row=position_store.row_for_time_step(int(step)),
        )
        sources = group_nodes_for_step(group_data, int(step), SOURCE_GROUP_ID)
        targets = group_nodes_for_step(group_data, int(step), TARGET_GROUP_ID)
        target_hops, target_hop_pairs, target_delay, target_delay_pairs = mean_metrics_for_mask(
            mask=target_row,
            weights=weights,
            edge_table=edge_table,
            sources=sources,
            targets=targets,
        )
        active_hops, active_hop_pairs, active_delay, active_delay_pairs = mean_metrics_for_mask(
            mask=active_mask,
            weights=weights,
            edge_table=edge_table,
            sources=sources,
            targets=targets,
        )
        rows.append(
            {
                "lst_s": int(lst_s),
                "step": int(step),
                "hour": float(step) / 3600.0,
                "is_regular_sample": int(step % 60 == 0),
                "target_active_edges": int(np.count_nonzero(target_row)),
                "lst_active_edges": int(np.count_nonzero(active_mask)),
                "active_edge_loss": int(np.count_nonzero(target_row & ~active_mask)),
                "target_mean_hops": target_hops,
                "lst_mean_hops": active_hops,
                "hops_gap": active_hops - target_hops,
                "target_reachable_pairs_hops": target_hop_pairs,
                "lst_reachable_pairs_hops": active_hop_pairs,
                "target_mean_delay_ms": target_delay,
                "lst_mean_delay_ms": active_delay,
                "delay_gap_ms": active_delay - target_delay,
                "target_reachable_pairs_delay": target_delay_pairs,
                "lst_reachable_pairs_delay": active_delay_pairs,
            }
        )
        if int(progress_every) > 0 and ((idx + 1) % int(progress_every) == 0 or idx + 1 == len(eval_steps)):
            print(
                f"[dynamic-one-right-lst] lst={lst_s} metrics {idx + 1}/{len(eval_steps)} "
                f"step={step} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize_one_lst(*, lst_s: int, stats: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Any]:
    target = stats["target_active_edges"].to_numpy(dtype=np.float64)
    dropped = stats["active_dropped_by_building"].to_numpy(dtype=np.float64)
    building = stats["building_edges"].to_numpy(dtype=np.float64)
    affected = dropped > 0

    row: dict[str, Any] = {
        "lst_s": int(lst_s),
        "seconds": int(len(stats)),
        "affected_seconds": int(np.count_nonzero(affected)),
        "affected_fraction": float(np.mean(affected)),
        "building_mean": float(np.mean(building)),
        "building_p95": float(np.percentile(building, 95)),
        "building_max": int(np.max(building)),
        "active_dropped_mean": float(np.mean(dropped)),
        "active_dropped_p95": float(np.percentile(dropped, 95)),
        "active_dropped_max": int(np.max(dropped)),
        "active_loss_fraction_mean": float(np.mean(dropped / np.maximum(target, 1.0))),
        "active_loss_fraction_p95": float(np.percentile(dropped / np.maximum(target, 1.0), 95)),
        "active_loss_fraction_max": float(np.max(dropped / np.maximum(target, 1.0))),
    }
    if not metrics.empty:
        for label, part in (
            ("regular", metrics[metrics["is_regular_sample"] == 1]),
            ("risk", metrics[metrics["active_edge_loss"] > 0]),
            ("all_eval", metrics),
        ):
            if part.empty:
                row[f"{label}_rows"] = 0
                row[f"{label}_delay_gap_mean_ms"] = math.nan
                row[f"{label}_delay_gap_p95_ms"] = math.nan
                row[f"{label}_hops_gap_mean"] = math.nan
                row[f"{label}_hops_gap_p95"] = math.nan
                continue
            row[f"{label}_rows"] = int(len(part))
            row[f"{label}_delay_gap_mean_ms"] = float(np.nanmean(part["delay_gap_ms"].to_numpy(dtype=float)))
            row[f"{label}_delay_gap_p95_ms"] = float(np.nanpercentile(part["delay_gap_ms"].to_numpy(dtype=float), 95))
            row[f"{label}_hops_gap_mean"] = float(np.nanmean(part["hops_gap"].to_numpy(dtype=float)))
            row[f"{label}_hops_gap_p95"] = float(np.nanpercentile(part["hops_gap"].to_numpy(dtype=float), 95))
    return row


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.2), dpi=170)
    x = summary["lst_s"].to_numpy(dtype=float)

    axes[0, 0].plot(x, summary["affected_fraction"], marker="o", color="#C1121F", linewidth=1.6)
    axes[0, 0].set_title("Affected seconds")
    axes[0, 0].set_ylabel("fraction")

    axes[0, 1].plot(x, summary["building_mean"], marker="o", label="mean building", color="#2563eb")
    axes[0, 1].plot(x, summary["building_p95"], marker="o", label="p95 building", color="#0f766e")
    axes[0, 1].plot(x, summary["building_max"], marker="o", label="max building", color="#111827")
    axes[0, 1].set_title("Building concurrency")
    axes[0, 1].set_ylabel("edges")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x, summary["active_loss_fraction_mean"], marker="o", label="mean loss", color="#d97706")
    axes[1, 0].plot(x, summary["active_loss_fraction_p95"], marker="o", label="p95 loss", color="#be123c")
    axes[1, 0].plot(x, summary["active_loss_fraction_max"], marker="o", label="max loss", color="#7c3aed")
    axes[1, 0].set_title("Target active-edge loss")
    axes[1, 0].set_xlabel("LST (s)")
    axes[1, 0].set_ylabel("fraction")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(x, summary["risk_delay_gap_mean_ms"], marker="o", label="risk mean delay gap", color="#C1121F")
    axes[1, 1].plot(x, summary["risk_delay_gap_p95_ms"], marker="o", label="risk p95 delay gap", color="#111827")
    axes[1, 1].plot(x, summary["risk_hops_gap_mean"], marker="o", label="risk mean hops gap", color="#2563eb")
    axes[1, 1].set_title("Metric gap at risk samples")
    axes[1, 1].set_xlabel("LST (s)")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.ravel():
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.55)
    fig.suptitle("Dynamic one-right topology under link setup time", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "lst_boundary_summary.png")
    plt.close(fig)


def write_report(out_dir: Path, summary: pd.DataFrame) -> None:
    rows = summary.to_dict(orient="records")
    first_affected = next((row for row in rows if float(row["affected_fraction"]) > 0.0), None)
    first_mean_loss_1pct = next((row for row in rows if float(row["active_loss_fraction_mean"]) >= 0.01), None)
    first_risk_delay_1ms = next(
        (
            row
            for row in rows
            if row.get("risk_delay_gap_mean_ms") is not None
            and np.isfinite(float(row["risk_delay_gap_mean_ms"]))
            and float(row["risk_delay_gap_mean_ms"]) >= 1.0
        ),
        None,
    )

    def fmt_boundary(row: dict[str, Any] | None, field: str) -> str:
        if row is None:
            return "not reached"
        return f"LST={int(row['lst_s'])}s ({field}={float(row[field]):.4g})"

    lines = [
        "# Dynamic one-right LST boundary",
        "",
        "This sweep evaluates the ideal dynamic one-right topology on a 1s time axis,",
        "then applies a conservative backward LST model: future setup consumes the corresponding right/left",
        "inter-plane ports, building links are not routable, and target-active links sharing those ports are dropped.",
        "",
        "## Boundary markers",
        "",
        f"- First affected seconds: {fmt_boundary(first_affected, 'affected_fraction')}",
        f"- First mean active-edge loss >= 1%: {fmt_boundary(first_mean_loss_1pct, 'active_loss_fraction_mean')}",
        f"- First risk-sample mean delay gap >= 1 ms: {fmt_boundary(first_risk_delay_1ms, 'risk_delay_gap_mean_ms')}",
        "",
        "## Main table",
        "",
        "| LST | affected | mean building | p95 building | max dropped | mean loss | risk delay gap mean | risk hops gap mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['lst_s'])} | {float(row['affected_fraction']):.3f} | "
            f"{float(row['building_mean']):.1f} | {float(row['building_p95']):.1f} | "
            f"{int(row['active_dropped_max'])} | {float(row['active_loss_fraction_mean']):.3f} | "
            f"{float(row.get('risk_delay_gap_mean_ms', float('nan'))):.3f} | "
            f"{float(row.get('risk_hops_gap_mean', float('nan'))):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `lst_boundary_summary.csv`",
            "- `lst_boundary_summary.png`",
            "- `lstXXX_metric_samples.csv` for each LST",
            "- `lstXXX_second_stats.csv` for selected LST values",
        ]
    )
    (out_dir / "lst_boundary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    full_cache = Path(args.full_link_cache)
    out_dir = Path(args.out_dir) if args.out_dir is not None else experiment_dir / "lst_boundary_dynamic_one_right"
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_table = read_edge_table_csv(full_cache / "edges.csv", total_sats=int(G60_CONFIG.total_sats))
    target_steps_path = experiment_dir / "dynamic_one_right_time_indices.npy"
    if target_steps_path.exists():
        coarse_steps = np.asarray(np.load(target_steps_path, mmap_mode="r"), dtype=np.int64)
    else:
        coarse_steps = np.asarray(np.load(full_cache / "time_indices.npy", mmap_mode="r"), dtype=np.int64)
    target_mask_60s = np.asarray(
        np.load(experiment_dir / "dynamic_one_right_active_full_edge_mask.npy", mmap_mode="r"),
        dtype=bool,
    )
    if target_mask_60s.shape != (int(coarse_steps.shape[0]), int(edge_table.num_edges)):
        raise ValueError(
            f"target mask shape {target_mask_60s.shape} != ({coarse_steps.shape[0]}, {edge_table.num_edges})"
        )

    ports = port_keys_by_edge(edge_table)
    target_port_owner = build_target_port_owner_by_row(target_mask_60s=target_mask_60s, ports_by_edge=ports)
    delay_store = FullLinkDelayStore(Path(args.delay_store_dir))
    position_store = PositionCacheStore(Path(args.position_cache_dir))
    lookup = build_weight_lookup(edge_table, delay_store, config=G60_CONFIG, allow_intra_fallback=True)

    group_data = load_or_build_group_data(
        xml_file=Path(args.group_xml),
        group_cache_dir=Path(args.group_cache_dir),
        steps=list(range(int(coarse_steps[0]), int(coarse_steps[-1]) + 1)),
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=1,
        enabled=True,
        force=False,
    )

    summary_rows: list[dict[str, Any]] = []
    save_detail_for = {int(x) for x in args.save_detail_for_lst}
    lst_values = list(range(int(args.lst_start), int(args.lst_end) + 1, int(args.lst_step)))
    for lst_s in lst_values:
        started = time.perf_counter()
        stats, _ = simulate_lst_stats_and_masks(
            edge_table=edge_table,
            target_mask_60s=target_mask_60s,
            coarse_steps=coarse_steps,
            ports_by_edge=ports,
            target_port_owner_by_row=target_port_owner,
            lst_s=int(lst_s),
            eval_steps=[],
        )
        eval_steps = choose_eval_steps(
            stats=stats,
            coarse_steps=coarse_steps,
            metric_regular_period=int(args.metric_regular_period),
            metric_top_drop_rows=int(args.metric_top_drop_rows),
        )
        _stats_again, eval_masks = simulate_lst_stats_and_masks(
            edge_table=edge_table,
            target_mask_60s=target_mask_60s,
            coarse_steps=coarse_steps,
            ports_by_edge=ports,
            target_port_owner_by_row=target_port_owner,
            lst_s=int(lst_s),
            eval_steps=eval_steps,
        )
        metrics = evaluate_metric_rows(
            lst_s=int(lst_s),
            eval_steps=eval_steps,
            lst_active_masks=eval_masks,
            target_mask_60s=target_mask_60s,
            coarse_steps=coarse_steps,
            edge_table=edge_table,
            group_data=group_data,
            delay_store=delay_store,
            position_store=position_store,
            lookup=lookup,
            progress_every=int(args.progress_every),
        )
        metrics.to_csv(out_dir / f"lst{int(lst_s):03d}_metric_samples.csv", index=False, encoding="utf-8-sig")
        if int(lst_s) in save_detail_for:
            stats.to_csv(out_dir / f"lst{int(lst_s):03d}_second_stats.csv", index=False, encoding="utf-8-sig")
        row = summarize_one_lst(lst_s=int(lst_s), stats=stats, metrics=metrics)
        row["elapsed_s"] = float(time.perf_counter() - started)
        summary_rows.append(row)
        print(
            f"[dynamic-one-right-lst] lst={lst_s} affected={row['affected_fraction']:.3f} "
            f"mean_loss={row['active_loss_fraction_mean']:.4f} "
            f"risk_delay_gap={row.get('risk_delay_gap_mean_ms', float('nan')):.3f} "
            f"elapsed={row['elapsed_s']:.1f}s",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "lst_boundary_summary.csv", index=False, encoding="utf-8-sig")
    plot_summary(summary, out_dir)
    write_report(out_dir, summary)
    meta = {
        "experiment_dir": str(experiment_dir),
        "full_link_cache": str(full_cache),
        "lst_values": lst_values,
        "metric_top_drop_rows": int(args.metric_top_drop_rows),
        "metric_regular_period": int(args.metric_regular_period),
        "rule": (
            "Ideal dynamic one-right target is evaluated on a 1s time axis. "
            "Backward LST reserves future setup ports; building edges are not routable; "
            "target-active edges sharing reserved ports are dropped."
        ),
        "outputs": {
            "summary_csv": str(out_dir / "lst_boundary_summary.csv"),
            "summary_plot": str(out_dir / "lst_boundary_summary.png"),
            "report": str(out_dir / "lst_boundary_report.md"),
        },
    }
    (out_dir / "lst_boundary_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}", flush=True)
    print(f"summary={out_dir / 'lst_boundary_summary.csv'}", flush=True)
    print(f"plot={out_dir / 'lst_boundary_summary.png'}", flush=True)
    print(f"report={out_dir / 'lst_boundary_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
