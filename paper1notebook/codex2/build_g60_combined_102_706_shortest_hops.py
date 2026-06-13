from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_g60_motif_gridplus_shortest_path_timeseries import build_legacy_gridplus_edge_table
from build_g60_selected_motifs_shortest_delay_parallel import (
    DEFAULT_XML,
    edge_table_for_motif,
    group_nodes_for_step,
)
from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, build_full_option_edges, write_edges_csv
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data


DEFAULT_SMALL102_CSV = (
    PROJECT_ROOT
    / "data"
    / "linshi"
    / "motif_up_to_w4h3_small102_translation_dedup"
    / "small102_up_to_w4h3_excluding_w4h3.csv"
)
DEFAULT_706_CSV = (
    PROJECT_ROOT
    / "data"
    / "linshi"
    / "motif_w4h3_primitive_translation_dedup"
    / "w4_h3_primitive_translation_dedup.csv"
)
DEFAULT_GRIDPLUS_CONFIG = PROJECT_ROOT / "data" / "topology_design" / "grid+" / "config" / "motif.json"
DEFAULT_GROUP_CACHE = PROJECT_ROOT / "data" / "linshi" / "cache" / "group_data_cache"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_combined_102_706_shortest_hops_t0_86160_stride60"

INTRA_OPTION = -1
PAIR_CONFIGS = {
    "china_europe": (2, 3, "China-Europe"),
    "china_america": (2, 0, "China-America"),
    "china_africa": (2, 1, "China-Africa"),
}


@dataclass(frozen=True)
class TopologySpec:
    name: str
    library: str
    motif_id: int | None
    motif: str
    source_w: int | None
    source_h: int | None
    edge_count: int | None
    support: str
    edges: str
    baseline: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute unweighted shortest-hop time series for G60 small102 + primitive706 motifs."
    )
    parser.add_argument("--small102-csv", type=Path, default=DEFAULT_SMALL102_CSV)
    parser.add_argument("--motifs706-csv", type=Path, default=DEFAULT_706_CSV)
    parser.add_argument("--gridplus-config", type=Path, default=DEFAULT_GRIDPLUS_CONFIG)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--pairs", nargs="*", default=list(PAIR_CONFIGS), choices=sorted(PAIR_CONFIGS))
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--limit-motifs", type=int, default=0, help="Debug only: limit motif topology count.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--write-edges", action="store_true", help="Also write one edges.csv per topology.")
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def auto_workers(requested: int) -> int:
    if int(requested) > 0:
        return int(requested)
    cpu = os.cpu_count() or 2
    return max(1, min(32, int(cpu) - 1))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_topology_specs(small102_csv: Path, motifs706_csv: Path, limit_motifs: int = 0) -> list[TopologySpec]:
    specs: list[TopologySpec] = []
    for row in read_csv_rows(small102_csv):
        motif_id = int(row["motif_id"])
        specs.append(
            TopologySpec(
                name=f"small102_motif_{motif_id:06d}",
                library="small102",
                motif_id=motif_id,
                motif=str(row["motif"]),
                source_w=int(row.get("source_w") or 0),
                source_h=int(row.get("source_h") or 0),
                edge_count=int(row.get("edge_count") or 0),
                support=str(row.get("support", "")),
                edges=str(row.get("edges", "")),
            )
        )
    for row in read_csv_rows(motifs706_csv):
        motif_id = int(row["motif_id"])
        specs.append(
            TopologySpec(
                name=f"motifs706_motif_{motif_id:06d}",
                library="motifs706",
                motif_id=motif_id,
                motif=str(row["motif"]),
                source_w=4,
                source_h=3,
                edge_count=int(row.get("edge_count") or 0),
                support=str(row.get("support", "")),
                edges=str(row.get("edges", "")),
            )
        )
    if int(limit_motifs) > 0:
        specs = specs[: int(limit_motifs)]
    return specs


def add_intra_to_edge_table(inter_edges: EdgeTable, *, p: int, n: int) -> EdgeTable:
    records: list[tuple[int, int, int, int, int]] = []
    for idx in range(inter_edges.num_edges):
        records.append(
            (
                int(inter_edges.src_plane[idx]),
                int(inter_edges.src_y[idx]),
                int(inter_edges.dst_plane[idx]),
                int(inter_edges.dst_y[idx]),
                int(inter_edges.option[idx]),
            )
        )
    for plane in range(int(p)):
        for y in range(int(n)):
            records.append((plane, y, plane, (y + 1) % int(n), INTRA_OPTION))
    return make_edge_table_from_records(records, p=p, n=n)


def make_edge_table_from_records(records: list[tuple[int, int, int, int, int]], *, p: int, n: int) -> EdgeTable:
    unique: dict[tuple[int, int], int] = {}
    for src_plane, src_y, dst_plane, dst_y, option in records:
        src = int(src_plane) * int(n) + int(src_y)
        dst = int(dst_plane) * int(n) + int(dst_y)
        if src == dst:
            continue
        key = (src, dst) if src < dst else (dst, src)
        unique.setdefault(key, int(option))

    src_values: list[int] = []
    dst_values: list[int] = []
    options: list[int] = []
    for (src, dst), option in sorted(unique.items()):
        src_values.append(src)
        dst_values.append(dst)
        options.append(option)

    src_arr = np.asarray(src_values, dtype=np.int32)
    dst_arr = np.asarray(dst_values, dtype=np.int32)
    total_sats = int(p) * int(n)
    return EdgeTable(
        src=src_arr,
        dst=dst_arr,
        option=np.asarray(options, dtype=np.int16),
        src_plane=(src_arr // int(n)).astype(np.int16),
        src_y=(src_arr % int(n)).astype(np.int16),
        dst_plane=(dst_arr // int(n)).astype(np.int16),
        dst_y=(dst_arr % int(n)).astype(np.int16),
        sat_ids=[str(i + 1) for i in range(total_sats)],
    )


def build_full_link_edge_table() -> EdgeTable:
    inter_edges = build_full_option_edges(G60_CONFIG, options=(0, 1, 2, 4))
    return add_intra_to_edge_table(inter_edges, p=int(G60_CONFIG.P), n=int(G60_CONFIG.N))


def build_adjacency_arrays(edge_table: EdgeTable, total_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    degree = np.zeros(int(total_nodes), dtype=np.int32)
    src = np.asarray(edge_table.src, dtype=np.int32)
    dst = np.asarray(edge_table.dst, dtype=np.int32)
    np.add.at(degree, src, 1)
    np.add.at(degree, dst, 1)
    indptr = np.empty(int(total_nodes) + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(degree, out=indptr[1:])
    cursor = indptr[:-1].copy()
    neighbors = np.empty(int(edge_table.num_edges) * 2, dtype=np.int32)
    for edge_idx in range(int(edge_table.num_edges)):
        a = int(src[edge_idx])
        b = int(dst[edge_idx])
        pos = int(cursor[a])
        neighbors[pos] = b
        cursor[a] += 1
        pos = int(cursor[b])
        neighbors[pos] = a
        cursor[b] += 1
    return indptr, neighbors


def all_pairs_hop_dist(edge_table: EdgeTable, total_nodes: int) -> np.ndarray:
    indptr, neighbors = build_adjacency_arrays(edge_table, total_nodes)
    n = int(total_nodes)
    dist = np.full((n, n), -1, dtype=np.int16)
    queue = np.empty(n, dtype=np.int32)
    for source in range(n):
        row = dist[source]
        row[source] = 0
        head = 0
        tail = 1
        queue[0] = source
        while head < tail:
            node = int(queue[head])
            head += 1
            next_dist = int(row[node]) + 1
            start = int(indptr[node])
            end = int(indptr[node + 1])
            for pos in range(start, end):
                neighbor = int(neighbors[pos])
                if int(row[neighbor]) >= 0:
                    continue
                row[neighbor] = next_dist
                queue[tail] = neighbor
                tail += 1
    return dist


def build_pair_states(
    *,
    group_data: dict,
    steps: list[int],
    pairs: list[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pair_key in pairs:
        source_group, target_group, pair_label = PAIR_CONFIGS[pair_key]
        state_lookup: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
        states: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        state_ids = np.empty(len(steps), dtype=np.int32)
        for row_idx, step in enumerate(steps):
            source_nodes = group_nodes_for_step(group_data, int(step), int(source_group))
            target_nodes = group_nodes_for_step(group_data, int(step), int(target_group))
            key = (source_nodes, target_nodes)
            state_id = state_lookup.get(key)
            if state_id is None:
                state_id = len(states)
                state_lookup[key] = state_id
                states.append(key)
            state_ids[row_idx] = int(state_id)
        out[pair_key] = {
            "label": pair_label,
            "source_group": int(source_group),
            "target_group": int(target_group),
            "states": states,
            "state_ids": state_ids,
        }
    return out


def summarize_state(dist: np.ndarray, source_nodes: tuple[int, ...], target_nodes: tuple[int, ...]) -> tuple[float, int, int, int, float, float]:
    required = int(len(source_nodes) * len(target_nodes))
    if required == 0:
        return math.nan, 0, required, 0, math.nan, math.nan
    sources = np.asarray(source_nodes, dtype=np.int32)
    targets = np.asarray(target_nodes, dtype=np.int32)
    values = dist[sources[:, None], targets].astype(np.int16, copy=False)
    reachable = values >= 0
    reachable_count = int(np.sum(reachable))
    if reachable_count == 0:
        return math.nan, 0, required, 0, math.nan, math.nan
    reachable_values = values[reachable].astype(np.float64, copy=False)
    return (
        float(np.mean(reachable_values)),
        reachable_count,
        required,
        int(reachable_count == required),
        float(np.min(reachable_values)),
        float(np.max(reachable_values)),
    )


def edge_table_for_spec(spec: TopologySpec, gridplus_config: str | None = None) -> EdgeTable:
    if spec.name == "gridplus":
        if gridplus_config is None:
            raise ValueError("gridplus_config is required for gridplus")
        return build_legacy_gridplus_edge_table(motif_json=Path(gridplus_config))
    if spec.name == "full_link":
        return build_full_link_edge_table()
    return edge_table_for_motif(spec.motif)


def compute_topology(task: tuple[TopologySpec, dict[str, dict[str, Any]], str | None, bool, str]) -> dict[str, Any]:
    spec, pair_states, gridplus_config, write_edges, edges_root = task
    started_at = time.time()
    edge_table = edge_table_for_spec(spec, gridplus_config)
    if write_edges:
        topology_dir = Path(edges_root) / spec.name
        topology_dir.mkdir(parents=True, exist_ok=True)
        write_edges_csv(edge_table, topology_dir / "edges.csv")
    dist = all_pairs_hop_dist(edge_table, int(G60_CONFIG.total_sats))

    by_pair: dict[str, dict[str, Any]] = {}
    for pair_key, payload in pair_states.items():
        states = payload["states"]
        state_ids = np.asarray(payload["state_ids"], dtype=np.int32)
        state_mean = np.full(len(states), np.nan, dtype=np.float32)
        state_reachable = np.zeros(len(states), dtype=np.int32)
        state_required = np.zeros(len(states), dtype=np.int32)
        state_full = np.zeros(len(states), dtype=np.bool_)
        state_min = np.full(len(states), np.nan, dtype=np.float32)
        state_max = np.full(len(states), np.nan, dtype=np.float32)
        for state_id, (source_nodes, target_nodes) in enumerate(states):
            mean, reachable, required, full, min_value, max_value = summarize_state(dist, source_nodes, target_nodes)
            state_mean[state_id] = mean
            state_reachable[state_id] = reachable
            state_required[state_id] = required
            state_full[state_id] = bool(full)
            state_min[state_id] = min_value
            state_max[state_id] = max_value

        means = state_mean[state_ids].astype(np.float32, copy=False)
        reachable_counts = state_reachable[state_ids].astype(np.int32, copy=False)
        required_counts = state_required[state_ids].astype(np.int32, copy=False)
        full_mask = state_full[state_ids].astype(np.bool_, copy=False)
        min_values = state_min[state_ids].astype(np.float32, copy=False)
        max_values = state_max[state_ids].astype(np.float32, copy=False)
        by_pair[pair_key] = {
            "mean_hops": means,
            "reachable_pairs": reachable_counts,
            "required_pairs": required_counts,
            "full_reachable": full_mask,
            "min_hops": min_values,
            "max_hops": max_values,
        }

    return {
        "name": spec.name,
        "library": spec.library,
        "baseline": bool(spec.baseline),
        "num_edges": int(edge_table.num_edges),
        "elapsed_s": float(time.time() - started_at),
        "pairs": by_pair,
    }


def finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def write_topology_library(specs: list[TopologySpec], path: Path, edge_counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "topology",
            "library",
            "motif_id",
            "motif",
            "source_w",
            "source_h",
            "edge_count_local",
            "edge_count_full_topology",
            "support",
            "edges",
            "baseline",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "topology": spec.name,
                    "library": spec.library,
                    "motif_id": "" if spec.motif_id is None else int(spec.motif_id),
                    "motif": spec.motif,
                    "source_w": "" if spec.source_w is None else int(spec.source_w),
                    "source_h": "" if spec.source_h is None else int(spec.source_h),
                    "edge_count_local": "" if spec.edge_count is None else int(spec.edge_count),
                    "edge_count_full_topology": edge_counts.get(spec.name, ""),
                    "support": spec.support,
                    "edges": spec.edges,
                    "baseline": bool(spec.baseline),
                }
            )


def write_pair_outputs(
    *,
    pair_key: str,
    pair_label: str,
    out_dir: Path,
    steps: list[int],
    motif_specs: list[TopologySpec],
    baseline_specs: list[TopologySpec],
    pair_results: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair_dir = out_dir / pair_key
    pair_dir.mkdir(parents=True, exist_ok=True)
    topology_names = [spec.name for spec in motif_specs]
    baseline_names = [spec.name for spec in baseline_specs]

    strict_values_by_name: dict[str, np.ndarray] = {}
    finite_values_by_name: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    for spec in motif_specs:
        result = pair_results[spec.name]
        finite_values = np.asarray(result["mean_hops"], dtype=np.float32)
        strict_values = finite_values.astype(np.float32, copy=True)
        full_mask = np.asarray(result["full_reachable"], dtype=bool)
        strict_values[~full_mask] = np.nan
        finite_values_by_name[spec.name] = finite_values
        strict_values_by_name[spec.name] = strict_values
        finite = finite_values[np.isfinite(finite_values)]
        strict_finite = strict_values[np.isfinite(strict_values)]
        fully_all = bool(np.all(full_mask))
        summary_rows.append(
            {
                "topology": spec.name,
                "library": spec.library,
                "motif_id": "" if spec.motif_id is None else int(spec.motif_id),
                "motif": spec.motif,
                "source_w": "" if spec.source_w is None else int(spec.source_w),
                "source_h": "" if spec.source_h is None else int(spec.source_h),
                "valid_steps": int(np.sum(full_mask)),
                "fully_reachable_all_steps": fully_all,
                "strict_static_mean_hops": float(np.mean(strict_finite)) if fully_all and strict_finite.size else math.nan,
                "finite_mean_hops": float(np.mean(finite)) if finite.size else math.nan,
                "finite_min_hops": float(np.min(finite)) if finite.size else math.nan,
                "finite_max_hops": float(np.max(finite)) if finite.size else math.nan,
                "support": spec.support,
                "edges": spec.edges,
            }
        )

    def strict_sort_key(row: dict[str, Any]) -> tuple[int, float]:
        value = row["strict_static_mean_hops"]
        if isinstance(value, float) and math.isfinite(value):
            return (0, float(value))
        return (1, float(row["finite_mean_hops"]) if math.isfinite(float(row["finite_mean_hops"])) else math.inf)

    summary_rows.sort(key=strict_sort_key)
    summary_path = pair_dir / f"combined_motif_summary_sorted_102_706_hops_{pair_key}_strict_reachable.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    strict_compare_path = pair_dir / f"combined_compare_102_706_hops_{pair_key}_strict_reachable.csv"
    finite_compare_path = pair_dir / f"combined_compare_102_706_hops_{pair_key}_finite_only.csv"
    for path, values_by_name in ((strict_compare_path, strict_values_by_name), (finite_compare_path, finite_values_by_name)):
        with path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["step"] + topology_names + baseline_names
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, step in enumerate(steps):
                row: dict[str, Any] = {"step": int(step)}
                for name in topology_names:
                    value = float(values_by_name[name][idx])
                    row[name] = "" if not math.isfinite(value) else value
                for name in baseline_names:
                    value = float(pair_results[name]["mean_hops"][idx])
                    row[name] = "" if not math.isfinite(value) else value
                writer.writerow(row)

    matrix = np.column_stack([strict_values_by_name[name] for name in topology_names]).astype(np.float64)
    dynamic_values = np.full(len(steps), np.nan, dtype=np.float64)
    dynamic_names = np.full(len(steps), "", dtype=object)
    for row_idx in range(matrix.shape[0]):
        row = matrix[row_idx]
        if np.all(~np.isfinite(row)):
            continue
        best_idx = int(np.nanargmin(row))
        dynamic_values[row_idx] = float(row[best_idx])
        dynamic_names[row_idx] = topology_names[best_idx]
    dynamic_path = pair_dir / f"combined_dynamic_best_102_706_hops_by_step_{pair_key}_strict_reachable.csv"
    with dynamic_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["step", "best_topology", "best_library", "best_hops"] + [f"{name}_hops" for name in baseline_names]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(steps):
            best_name = str(dynamic_names[idx])
            best_library = best_name.split("_motif_", 1)[0] if best_name else ""
            row = {
                "step": int(step),
                "best_topology": best_name,
                "best_library": best_library,
                "best_hops": "" if not math.isfinite(float(dynamic_values[idx])) else float(dynamic_values[idx]),
            }
            for name in baseline_names:
                value = float(pair_results[name]["mean_hops"][idx])
                row[f"{name}_hops"] = "" if not math.isfinite(value) else value
            writer.writerow(row)

    best_static = summary_rows[0]
    best_small102 = next((row for row in summary_rows if row["library"] == "small102"), None)
    best_706 = next((row for row in summary_rows if row["library"] == "motifs706"), None)
    x_hours = np.asarray(steps, dtype=np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(16, 7.3), dpi=180)
    for name in topology_names:
        color = "#2563eb" if name.startswith("small102_") else "#6b7280"
        alpha = 0.07 if name.startswith("small102_") else 0.045
        ax.plot(x_hours, strict_values_by_name[name], color=color, alpha=alpha, linewidth=0.55)
    if best_small102:
        name = str(best_small102["topology"])
        ax.plot(
            x_hours,
            strict_values_by_name[name],
            color="#1d4ed8",
            linewidth=1.65,
            label=f"best static small102: {name.replace('small102_', '')}",
        )
    if best_706:
        name = str(best_706["topology"])
        ax.plot(
            x_hours,
            strict_values_by_name[name],
            color="#7c3aed",
            linewidth=1.65,
            label=f"best static 706: {name.replace('motifs706_', '')}",
        )
    ax.plot(x_hours, dynamic_values, color="#111827", linestyle="--", linewidth=1.65, label="per-time best among 102+706")
    if "gridplus" in baseline_names:
        ax.plot(x_hours, pair_results["gridplus"]["mean_hops"], color="#f97316", linewidth=1.35, label="gridplus")
    if "full_link" in baseline_names:
        ax.plot(x_hours, pair_results["full_link"]["mean_hops"], color="#16a34a", linewidth=1.35, label="full_link")
    ax.set_title(f"G60 {pair_label} mean shortest path length: small102 + primitive706 (strict reachable)")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(f"{pair_label} mean shortest path (hops)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    plot_path = pair_dir / f"mean_shortest_hops_combined_102_706_{pair_key}_strict_reachable.png"
    fig.savefig(plot_path)
    plt.close(fig)

    baseline_summary = {
        name: {
            "mean_hops": finite_mean(np.asarray(pair_results[name]["mean_hops"], dtype=np.float32)),
            "min_hops": float(np.nanmin(pair_results[name]["mean_hops"])),
            "max_hops": float(np.nanmax(pair_results[name]["mean_hops"])),
        }
        for name in baseline_names
    }
    result = {
        "pair_key": pair_key,
        "pair_label": pair_label,
        "num_steps": int(len(steps)),
        "motif_count": int(len(topology_names)),
        "best_static_topology": str(best_static["topology"]),
        "best_static_library": str(best_static["library"]),
        "best_static_mean_hops": float(best_static["strict_static_mean_hops"]),
        "dynamic_mean_hops": finite_mean(dynamic_values),
        "dynamic_unique_topologies": int(len(set(str(x) for x in dynamic_names if str(x)))),
        "full_time_reachable_motifs": int(sum(bool(row["fully_reachable_all_steps"]) for row in summary_rows)),
        "baseline_summary": baseline_summary,
        "strict_compare_csv": str(strict_compare_path),
        "finite_compare_csv": str(finite_compare_path),
        "dynamic_csv": str(dynamic_path),
        "summary_csv": str(summary_path),
        "plot": str(plot_path),
    }
    (pair_dir / f"combined_102_706_hops_summary_{pair_key}_strict_reachable.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    args = parse_args()
    if int(args.end) < int(args.start):
        raise ValueError("--end must be >= --start")
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")

    started_at = time.time()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty step range")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    motif_specs = read_topology_specs(Path(args.small102_csv), Path(args.motifs706_csv), int(args.limit_motifs))
    baseline_specs = [
        TopologySpec("gridplus", "baseline", None, "gridplus", None, None, None, "", "", baseline=True),
        TopologySpec("full_link", "baseline", None, "full_link", None, None, None, "", "", baseline=True),
    ]
    all_specs = motif_specs + baseline_specs
    print(
        f"[shortest-hops-808] motifs={len(motif_specs)} baselines={len(baseline_specs)} "
        f"steps={len(steps)} range={steps[0]}..{steps[-1]} stride={args.stride}",
        flush=True,
    )

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
    pair_states = build_pair_states(group_data=group_data, steps=steps, pairs=[str(x) for x in args.pairs])
    for pair_key, payload in pair_states.items():
        print(
            f"[shortest-hops-808] pair={pair_key} label={payload['label']} "
            f"unique_states={len(payload['states'])}",
            flush=True,
        )

    workers = auto_workers(int(args.max_workers))
    print(f"[shortest-hops-808] workers={workers}", flush=True)
    tasks = [
        (spec, pair_states, str(Path(args.gridplus_config)), bool(args.write_edges), str(out_dir / "topology_edges"))
        for spec in all_specs
    ]

    pair_results: dict[str, dict[str, dict[str, np.ndarray]]] = {pair_key: {} for pair_key in pair_states}
    edge_counts: dict[str, int] = {}
    topology_elapsed: dict[str, float] = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(compute_topology, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            name = str(result["name"])
            edge_counts[name] = int(result["num_edges"])
            topology_elapsed[name] = float(result["elapsed_s"])
            for pair_key, payload in result["pairs"].items():
                pair_results[pair_key][name] = payload
            completed += 1
            if int(args.progress_every) > 0 and (completed == len(futures) or completed % int(args.progress_every) == 0):
                print(
                    f"[shortest-hops-808] completed {completed}/{len(futures)} "
                    f"last={name} edges={edge_counts[name]} elapsed={time.time() - started_at:.1f}s",
                    flush=True,
                )

    write_topology_library(all_specs, out_dir / "topology_library_808_plus_baselines.csv", edge_counts)
    pair_summaries = []
    for pair_key in args.pairs:
        source_group, target_group, pair_label = PAIR_CONFIGS[str(pair_key)]
        summary = write_pair_outputs(
            pair_key=str(pair_key),
            pair_label=pair_label,
            out_dir=out_dir,
            steps=steps,
            motif_specs=motif_specs,
            baseline_specs=baseline_specs,
            pair_results=pair_results[str(pair_key)],
        )
        pair_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    meta = {
        "constellation": G60_CONFIG.name,
        "P": int(G60_CONFIG.P),
        "N": int(G60_CONFIG.N),
        "total_sats": int(G60_CONFIG.total_sats),
        "metric": "unweighted shortest path hop count",
        "motif_count": int(len(motif_specs)),
        "baseline_count": int(len(baseline_specs)),
        "small102_csv": str(Path(args.small102_csv)),
        "motifs706_csv": str(Path(args.motifs706_csv)),
        "start": int(steps[0]),
        "end": int(steps[-1]),
        "stride": int(args.stride),
        "num_steps": int(len(steps)),
        "pairs": list(args.pairs),
        "pair_summaries": pair_summaries,
        "topology_elapsed_s": topology_elapsed,
        "elapsed_s": float(time.time() - started_at),
    }
    (out_dir / "combined_102_706_shortest_hops_all_pairs_summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[shortest-hops-808] done elapsed={time.time() - started_at:.1f}s out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
