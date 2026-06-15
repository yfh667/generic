from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PyQt5 import QtWidgets
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_m56_local_patch_hybrid_topology import (  # noqa: E402
    DEFAULT_CONFIG,
    RegionPair,
    build_hybrid_edge_table,
    cyclic_band,
    degree_stats,
    normalize_motif_name,
    path_from,
    region_pairs_from_workflow,
)
from weighted_base_viewer import EdgeUsageTopology2DViewer  # noqa: E402

from src.link_delay.module.edge_options import EdgeTable, write_edges_csv  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.edge_betweenness import (  # noqa: E402
    build_undirected_adjacency,
    edge_betweenness_between_node_sets,
)
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.batch_shortest_hops import (  # noqa: E402
    TopologySpec,
    topology_specs_from_motif_csv,
)
from src.topology_workflow.module.config import (  # noqa: E402
    load_workflow_yaml,
    time_axis_from_config,
    viewer_config_from_workflow,
)
from src.topology_workflow.module.edge_tables import make_edge_table_from_records  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    idx: int
    name: str
    mode: str
    band_start: int | None
    band_end: int | None
    band_len: int
    edge_table: EdgeTable
    added_edges: int
    removed_edges: int
    degree: dict[str, Any]

    @property
    def changed_edges(self) -> int:
        return int(self.added_edges + self.removed_edges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a time-varying G60 topology sequence by dynamically splicing "
            "local 000040 y-bands into base motif 000056."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-motif", default="56")
    parser.add_argument("--patch-motif", default="40")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86100)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--target-pair", default="china_america")
    parser.add_argument("--preserve-pairs", nargs="+", default=("china_europe", "china_africa"))
    parser.add_argument("--metric-pairs", nargs="+", default=("china_europe", "china_america", "china_africa"))
    parser.add_argument("--usage-pairs", nargs="+", default=("china_america",))
    parser.add_argument("--patch-modes", nargs="+", default=("c", "cb", "all"), choices=("c", "cb", "all"))
    parser.add_argument(
        "--band-lengths",
        nargs="+",
        type=int,
        default=(6, 12, 18),
        help="Cyclic y-band lengths to sweep. Starts are all 0..N-1.",
    )
    parser.add_argument(
        "--preserve-eps",
        type=float,
        default=1e-9,
        help="Allowed mean-hop increase over the base motif for preserve pairs.",
    )
    parser.add_argument(
        "--prefer-fewer-changes-eps",
        type=float,
        default=1e-9,
        help="When target scores tie within this epsilon, prefer fewer edge changes.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("instant", "transition-dp"),
        default="instant",
        help="instant selects the best candidate at each step independently; transition-dp penalizes new setup edges between consecutive steps.",
    )
    parser.add_argument(
        "--transition-setup-penalty-per-edge",
        type=float,
        default=0.0,
        help="Hop-equivalent penalty per newly built edge when --selection-mode transition-dp.",
    )
    parser.add_argument(
        "--transition-max-new-edges",
        type=int,
        default=0,
        help="Optional hard cap on newly built edges per transition for transition-dp; 0 disables the cap.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--reuse", action="store_true", help="Load existing outputs instead of rebuilding them.")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument(
        "--skip-usage",
        action="store_true",
        help="Do not compute per-step edge-usage values; write a zero matrix for topology-only viewer runs.",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def sanitize_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def short_motif_token(name: str) -> str:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    if digits:
        return f"m{int(digits):03d}"
    token = sanitize_token(str(name))
    return token[:18] if len(token) > 18 else token


def short_pair_token(pair_key: str) -> str:
    mapping = {
        "china_europe": "ce",
        "china_america": "ca",
        "china_africa": "caf",
    }
    return mapping.get(str(pair_key), sanitize_token(str(pair_key))[:12])


def selection_suffix(
    *,
    selection_mode: str = "instant",
    transition_setup_penalty_per_edge: float = 0.0,
    transition_max_new_edges: int = 0,
) -> str:
    if str(selection_mode) == "instant":
        return ""
    penalty = f"{float(transition_setup_penalty_per_edge):g}".replace(".", "p").replace("-", "m")
    cap = int(transition_max_new_edges)
    suffix = f"_sel-{sanitize_token(str(selection_mode))}_pen{penalty}"
    if cap > 0:
        suffix += f"_cap{cap}"
    return suffix


def default_out_dir(
    *,
    run_out_dir: Path,
    base_name: str,
    patch_name: str,
    start: int,
    end: int,
    stride: int,
    target_pair: str,
    band_lengths: Iterable[int],
    patch_modes: Iterable[str],
    selection_mode: str = "instant",
    transition_setup_penalty_per_edge: float = 0.0,
    transition_max_new_edges: int = 0,
) -> Path:
    lengths = "-".join(str(int(x)) for x in band_lengths)
    modes = "-".join(str(x) for x in patch_modes)
    suffix = selection_suffix(
        selection_mode=str(selection_mode),
        transition_setup_penalty_per_edge=float(transition_setup_penalty_per_edge),
        transition_max_new_edges=int(transition_max_new_edges),
    )
    return (
        run_out_dir
        / "dynamic_splice_topologies"
        / (
            f"dyn_{short_motif_token(base_name)}_{short_motif_token(patch_name)}_"
            f"{short_pair_token(target_pair)}_b{lengths}_{modes}_t{start}_{end}_s{stride}{suffix}"
        )
    )


def edge_key_from_arrays(edge_table: EdgeTable, idx: int) -> tuple[int, int]:
    src = int(edge_table.src[idx])
    dst = int(edge_table.dst[idx])
    return (src, dst) if src <= dst else (dst, src)


def edge_record_from_idx(edge_table: EdgeTable, idx: int) -> tuple[int, int, int, int, int]:
    return (
        int(edge_table.src_plane[idx]),
        int(edge_table.src_y[idx]),
        int(edge_table.dst_plane[idx]),
        int(edge_table.dst_y[idx]),
        int(edge_table.option[idx]),
    )


def edge_records_from_table(edge_table: EdgeTable) -> list[tuple[int, int, int, int, int]]:
    return [edge_record_from_idx(edge_table, idx) for idx in range(edge_table.num_edges)]


def all_pairs_shortest_hops(edge_table: EdgeTable, total_nodes: int) -> np.ndarray:
    rows = np.concatenate([edge_table.src, edge_table.dst]).astype(np.int32)
    cols = np.concatenate([edge_table.dst, edge_table.src]).astype(np.int32)
    data = np.ones(rows.shape[0], dtype=np.int8)
    graph = csr_matrix((data, (rows, cols)), shape=(int(total_nodes), int(total_nodes)))
    dist = shortest_path(graph, directed=False, unweighted=True, return_predecessors=False)
    return np.asarray(dist, dtype=np.float32)


def mean_hops_from_dist(dist: np.ndarray, sources: tuple[int, ...], targets: tuple[int, ...]) -> float:
    if not sources or not targets:
        return math.nan
    block = dist[np.ix_(np.asarray(sources, dtype=np.int32), np.asarray(targets, dtype=np.int32))]
    finite = np.isfinite(block) & (block > 0)
    if not np.any(finite):
        return math.nan
    return float(np.mean(block[finite]))


def build_candidates(
    *,
    base_spec: TopologySpec,
    patch_spec: TopologySpec,
    p: int,
    n: int,
    band_lengths: Iterable[int],
    patch_modes: Iterable[str],
) -> list[Candidate]:
    candidates: list[Candidate] = [
        Candidate(
            idx=0,
            name=f"{base_spec.name}__base",
            mode="base",
            band_start=None,
            band_end=None,
            band_len=0,
            edge_table=base_spec.edge_table,
            added_edges=0,
            removed_edges=0,
            degree=degree_stats(base_spec.edge_table, total_sats=int(p) * int(n)),
        )
    ]

    seen: set[tuple[str, int, int, int]] = set()
    for mode in patch_modes:
        for band_len_raw in band_lengths:
            band_len = int(band_len_raw)
            if not (1 <= band_len <= int(n)):
                raise ValueError(f"band length must be in 1..N={n}, got {band_len}")
            for band_start in range(int(n)):
                band_end = (band_start + band_len - 1) % int(n)
                key = (str(mode), int(band_start), int(band_end), int(band_len))
                if key in seen:
                    continue
                seen.add(key)
                edge_table, added, removed, deg = build_hybrid_edge_table(
                    base_spec=base_spec,
                    patch_spec=patch_spec,
                    p=int(p),
                    n=int(n),
                    band=cyclic_band(band_start, band_end, n=int(n)),
                    patch_mode=str(mode),
                )
                if int(deg["max_out_degree"]) > 1 or int(deg["max_in_degree"]) > 1:
                    raise RuntimeError(f"candidate violates degree constraints: mode={mode} band={band_start}..{band_end}")
                candidates.append(
                    Candidate(
                        idx=len(candidates),
                        name=f"{base_spec.name}+{patch_spec.name}:{mode}:y{band_start}_{band_end}:len{band_len}",
                        mode=str(mode),
                        band_start=int(band_start),
                        band_end=int(band_end),
                        band_len=int(band_len),
                        edge_table=edge_table,
                        added_edges=len(added),
                        removed_edges=len(removed),
                        degree=deg,
                    )
                )
    return candidates


def pair_nodes_by_step(
    *,
    group_data: dict,
    steps: list[int],
    pairs: dict[str, RegionPair],
) -> dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    out: dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
    for key, pair in pairs.items():
        seq: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for step in steps:
            seq.append(
                (
                    group_nodes_for_step(group_data, int(step), int(pair.source_group_id)),
                    group_nodes_for_step(group_data, int(step), int(pair.target_group_id)),
                )
            )
        out[key] = seq
    return out


def candidate_metric_arrays(
    *,
    dist: np.ndarray,
    pair_nodes: dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]],
    pair_keys: list[str],
    num_steps: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in pair_keys:
        values = np.empty(int(num_steps), dtype=np.float32)
        for row_idx, (sources, targets) in enumerate(pair_nodes[key]):
            values[row_idx] = mean_hops_from_dist(dist, sources, targets)
        out[key] = values
    return out


def candidate_edge_key_sets(candidates: list[Candidate]) -> list[set[tuple[int, int]]]:
    out: list[set[tuple[int, int]]] = []
    for candidate in candidates:
        out.append(
            {
                edge_key_from_arrays(candidate.edge_table, edge_idx)
                for edge_idx in range(candidate.edge_table.num_edges)
            }
        )
    return out


def transition_new_edge_matrix(candidates: list[Candidate]) -> np.ndarray:
    edge_sets = candidate_edge_key_sets(candidates)
    n_candidates = len(edge_sets)
    matrix = np.zeros((n_candidates, n_candidates), dtype=np.int32)
    for prev_idx, prev_edges in enumerate(edge_sets):
        for curr_idx, curr_edges in enumerate(edge_sets):
            if prev_idx == curr_idx:
                continue
            matrix[prev_idx, curr_idx] = len(curr_edges - prev_edges)
    return matrix


def select_candidates_transition_dp(
    *,
    target_values: np.ndarray,
    valid_mask: np.ndarray,
    transition_new_edges: np.ndarray,
    penalty_per_new_edge: float,
    max_new_edges: int,
) -> np.ndarray:
    values = np.asarray(target_values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("target_values must have shape (steps, candidates)")
    n_steps, n_candidates = values.shape
    if valid.shape != values.shape:
        raise ValueError("valid_mask must match target_values shape")
    if int(n_steps) == 0 or int(n_candidates) == 0:
        raise ValueError("empty transition-dp problem")

    trans = np.asarray(transition_new_edges, dtype=np.float64) * float(penalty_per_new_edge)
    if int(max_new_edges) > 0:
        trans = trans.copy()
        trans[np.asarray(transition_new_edges) > int(max_new_edges)] = np.inf

    inf = np.inf
    cost = np.where(valid[0], values[0], inf)
    prev_choice = np.full((n_steps, n_candidates), -1, dtype=np.int32)
    if not np.any(np.isfinite(cost)):
        raise ValueError("transition-dp has no valid candidate at first step")

    for row_idx in range(1, n_steps):
        transition_cost = cost[:, None] + trans
        best_prev = np.argmin(transition_cost, axis=0).astype(np.int32)
        best_cost = transition_cost[best_prev, np.arange(n_candidates)]
        row_cost = best_cost + values[row_idx]
        row_cost = np.where(valid[row_idx], row_cost, inf)

        if not np.any(np.isfinite(row_cost)) and int(max_new_edges) > 0:
            relaxed = cost[:, None] + np.asarray(transition_new_edges, dtype=np.float64) * float(penalty_per_new_edge)
            best_prev = np.argmin(relaxed, axis=0).astype(np.int32)
            best_cost = relaxed[best_prev, np.arange(n_candidates)]
            row_cost = np.where(valid[row_idx], best_cost + values[row_idx], inf)

        if not np.any(np.isfinite(row_cost)):
            raise ValueError(f"transition-dp has no valid candidate at row {row_idx}")
        prev_choice[row_idx] = best_prev
        cost = row_cost

    selected = np.zeros(n_steps, dtype=np.int32)
    selected[-1] = int(np.argmin(cost))
    for row_idx in range(n_steps - 1, 0, -1):
        selected[row_idx - 1] = int(prev_choice[row_idx, selected[row_idx]])
    return selected


def should_replace_best(
    *,
    candidate_target: float,
    candidate_changes: int,
    best_target: float,
    best_changes: int,
    tie_eps: float,
) -> bool:
    if not math.isfinite(candidate_target):
        return False
    if not math.isfinite(best_target):
        return True
    if candidate_target < best_target - float(tie_eps):
        return True
    if abs(candidate_target - best_target) <= float(tie_eps) and int(candidate_changes) < int(best_changes):
        return True
    return False


def write_candidate_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_idx",
        "candidate",
        "mode",
        "band_start",
        "band_end",
        "band_len",
        "added_edges",
        "removed_edges",
        "changed_edges",
        "valid_steps",
        "selected_steps",
        "mean_target_hops",
        "mean_target_improvement",
        "max_target_improvement",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_dynamic_schedule(
    path: Path,
    *,
    steps: list[int],
    candidates: list[Candidate],
    selected_idx: np.ndarray,
    selected_metrics: dict[str, np.ndarray],
    base_metrics: dict[str, np.ndarray],
    target_pair: str,
    metric_pairs: list[str],
    transition_new_edges: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "candidate_idx",
        "candidate",
        "mode",
        "band_start",
        "band_end",
        "band_len",
        "added_edges",
        "removed_edges",
        "changed_edges",
        "new_edges_from_prev",
    ]
    for key in metric_pairs:
        fieldnames.append(f"{key}_mean_hops")
        fieldnames.append(f"{key}_base_mean_hops")
        fieldnames.append(f"{key}_delta_vs_base")
    fieldnames.append(f"{target_pair}_improvement_vs_base")

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, step in enumerate(steps):
            cand = candidates[int(selected_idx[row_idx])]
            row: dict[str, Any] = {
                "step": int(step),
                "candidate_idx": int(cand.idx),
                "candidate": cand.name,
                "mode": cand.mode,
                "band_start": "" if cand.band_start is None else int(cand.band_start),
                "band_end": "" if cand.band_end is None else int(cand.band_end),
                "band_len": int(cand.band_len),
                "added_edges": int(cand.added_edges),
                "removed_edges": int(cand.removed_edges),
                "changed_edges": int(cand.changed_edges),
                "new_edges_from_prev": (
                    "" if transition_new_edges is None else int(transition_new_edges[row_idx])
                ),
            }
            for key in metric_pairs:
                value = float(selected_metrics[key][row_idx])
                base = float(base_metrics[key][row_idx])
                row[f"{key}_mean_hops"] = value
                row[f"{key}_base_mean_hops"] = base
                row[f"{key}_delta_vs_base"] = value - base
            row[f"{target_pair}_improvement_vs_base"] = (
                float(base_metrics[target_pair][row_idx]) - float(selected_metrics[target_pair][row_idx])
            )
            writer.writerow(row)


def build_union_edge_table(candidates: list[Candidate], selected_idx: np.ndarray, *, p: int, n: int) -> tuple[EdgeTable, dict[tuple[int, int], int]]:
    records_by_key: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
    for cand_idx in sorted(set(int(x) for x in selected_idx.tolist())):
        table = candidates[int(cand_idx)].edge_table
        for edge_idx in range(table.num_edges):
            key = edge_key_from_arrays(table, edge_idx)
            records_by_key.setdefault(key, edge_record_from_idx(table, edge_idx))
    union = make_edge_table_from_records(p=int(p), n=int(n), records=list(records_by_key.values()))
    union_key_to_idx = {edge_key_from_arrays(union, idx): int(idx) for idx in range(union.num_edges)}
    return union, union_key_to_idx


def build_active_mask(
    *,
    candidates: list[Candidate],
    selected_idx: np.ndarray,
    union_key_to_idx: dict[tuple[int, int], int],
    num_edges: int,
) -> np.ndarray:
    mask = np.zeros((int(selected_idx.size), int(num_edges)), dtype=bool)
    edge_sets_by_candidate: dict[int, list[int]] = {}
    for cand_idx in sorted(set(int(x) for x in selected_idx.tolist())):
        table = candidates[int(cand_idx)].edge_table
        edge_sets_by_candidate[int(cand_idx)] = [
            union_key_to_idx[edge_key_from_arrays(table, edge_idx)]
            for edge_idx in range(table.num_edges)
        ]
    for row_idx, cand_idx in enumerate(selected_idx):
        mask[row_idx, edge_sets_by_candidate[int(cand_idx)]] = True
    return mask


def build_usage_values(
    *,
    candidates: list[Candidate],
    selected_idx: np.ndarray,
    steps: list[int],
    union_key_to_idx: dict[tuple[int, int], int],
    num_edges: int,
    config,
    group_data: dict,
    usage_pairs: list[RegionPair],
    progress_every: int = 100,
) -> np.ndarray:
    values = np.zeros((len(steps), int(num_edges)), dtype=np.float32)
    adjacency_by_candidate: dict[int, list[list[tuple[int, int]]]] = {}
    local_to_union_by_candidate: dict[int, np.ndarray] = {}
    for cand_idx in sorted(set(int(x) for x in selected_idx.tolist())):
        table = candidates[int(cand_idx)].edge_table
        adjacency_by_candidate[int(cand_idx)] = build_undirected_adjacency(table, int(config.total_sats))
        local_to_union_by_candidate[int(cand_idx)] = np.asarray(
            [union_key_to_idx[edge_key_from_arrays(table, edge_idx)] for edge_idx in range(table.num_edges)],
            dtype=np.int32,
        )

    for row_idx, step in enumerate(steps):
        cand_idx = int(selected_idx[row_idx])
        table = candidates[cand_idx].edge_table
        adjacency = adjacency_by_candidate[cand_idx]
        local_to_union = local_to_union_by_candidate[cand_idx]
        for pair in usage_pairs:
            local_values, _summary, _samples = edge_betweenness_between_node_sets(
                table,
                total_nodes=int(config.total_sats),
                source_nodes=group_nodes_for_step(group_data, int(step), int(pair.source_group_id)),
                target_nodes=group_nodes_for_step(group_data, int(step), int(pair.target_group_id)),
                adjacency=adjacency,
                sample_path_limit=0,
            )
            values[row_idx, local_to_union] += local_values
        if (row_idx + 1) % int(progress_every) == 0 or row_idx + 1 == len(steps):
            print(f"[dynamic-splice] edge usage {row_idx + 1}/{len(steps)} step={step}", flush=True)
    return values


def plot_dynamic_summary(
    path: Path,
    *,
    steps: list[int],
    target_pair: str,
    preserve_pairs: list[str],
    selected_metrics: dict[str, np.ndarray],
    base_metrics: dict[str, np.ndarray],
    patch_metrics: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_hours = np.asarray(steps, dtype=np.float64) / 3600.0
    fig, axes = plt.subplots(1 + len(preserve_pairs), 1, figsize=(15.5, 3.2 * (1 + len(preserve_pairs))), dpi=170, sharex=True)
    axes = np.atleast_1d(axes)
    all_pairs = [target_pair] + list(preserve_pairs)
    for ax, key in zip(axes, all_pairs):
        ax.plot(x_hours, base_metrics[key], color="#2563eb", linewidth=1.8, label="base 000056")
        ax.plot(x_hours, patch_metrics[key], color="#64748b", linewidth=1.2, alpha=0.75, label="static 000040")
        ax.plot(x_hours, selected_metrics[key], color="#111827", linewidth=1.8, linestyle="--", label="dynamic splice")
        ax.set_ylabel(f"{key}\nmean hops")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.55)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (hour)")
    fig.suptitle(f"Dynamic splice target={target_pair}: base 000056 + local 000040 y-band patches", y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def read_edge_table_csv(path: Path, *, total_sats: int) -> EdgeTable:
    src: list[int] = []
    dst: list[int] = []
    option: list[int] = []
    src_plane: list[int] = []
    src_y: list[int] = []
    dst_plane: list[int] = []
    dst_y: list[int] = []
    sat_ids = [str(i + 1) for i in range(int(total_sats))]
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            src.append(int(row["src_node"]))
            dst.append(int(row["dst_node"]))
            option.append(int(row["option"]))
            src_plane.append(int(row["src_plane"]))
            src_y.append(int(row["src_y"]))
            dst_plane.append(int(row["dst_plane"]))
            dst_y.append(int(row["dst_y"]))
    return EdgeTable(
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        option=np.asarray(option, dtype=np.int16),
        src_plane=np.asarray(src_plane, dtype=np.int16),
        src_y=np.asarray(src_y, dtype=np.int16),
        dst_plane=np.asarray(dst_plane, dtype=np.int16),
        dst_y=np.asarray(dst_y, dtype=np.int16),
        sat_ids=sat_ids,
    )


def load_schedule_steps(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [int(row["step"]) for row in csv.DictReader(f)]


def build_or_load_outputs(args: argparse.Namespace) -> dict[str, Any]:
    workflow = load_workflow_yaml(args.config)
    config = viewer_config_from_workflow(workflow)
    _cfg_start, _cfg_end, cfg_stride = time_axis_from_config(workflow)
    start = int(args.start)
    end = int(args.end)
    stride = int(args.stride if args.stride is not None else cfg_stride)
    steps = list(range(start, end + 1, stride))
    if not steps:
        raise ValueError("empty steps")

    paths_raw = workflow.get("paths", {})
    motif_raw = workflow.get("motif_library", {})
    name_prefix = str(motif_raw.get("name_prefix", "combined"))
    base_name = normalize_motif_name(args.base_motif, name_prefix=name_prefix)
    patch_name = normalize_motif_name(args.patch_motif, name_prefix=name_prefix)
    out_dir = args.out_dir or default_out_dir(
        run_out_dir=path_from(paths_raw, "out_dir"),
        base_name=base_name,
        patch_name=patch_name,
        start=start,
        end=end,
        stride=stride,
        target_pair=str(args.target_pair),
        band_lengths=args.band_lengths,
        patch_modes=args.patch_modes,
        selection_mode=str(args.selection_mode),
        transition_setup_penalty_per_edge=float(args.transition_setup_penalty_per_edge),
        transition_max_new_edges=int(args.transition_max_new_edges),
    )
    out_dir = Path(out_dir)

    if args.reuse:
        union_edge_table = read_edge_table_csv(out_dir / "union_edges.csv", total_sats=int(config.total_sats))
        return {
            "workflow": workflow,
            "config": config,
            "steps": load_schedule_steps(out_dir / "dynamic_schedule.csv"),
            "out_dir": out_dir,
            "union_edge_table": union_edge_table,
            "active_mask": np.load(out_dir / "edge_active_mask.npy"),
            "usage_values": np.load(out_dir / "edge_usage_values.npy"),
            "group_data": load_or_build_group_data(
                xml_file=path_from(paths_raw, "group_xml"),
                group_cache_dir=path_from(paths_raw, "group_cache_dir"),
                steps=load_schedule_steps(out_dir / "dynamic_schedule.csv"),
                station_groups=config.station_groups,
                total_sats=config.total_sats,
                constellation_name=config.name,
                stride=stride,
                enabled=True,
                force=False,
            ),
        }

    started_at = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    motif_csv = path_from(paths_raw, "motif_library_dir") / str(motif_raw.get("csv_name"))
    specs = topology_specs_from_motif_csv(
        motif_csv,
        config=config,
        library=name_prefix,
        name_prefix=name_prefix,
        add_intra_ring=True,
        wrap_planes=False,
    )
    specs_by_name = {spec.name: spec for spec in specs}
    if base_name not in specs_by_name or patch_name not in specs_by_name:
        raise ValueError(f"missing base or patch motif: {base_name}, {patch_name}")
    base_spec = specs_by_name[base_name]
    patch_spec = specs_by_name[patch_name]

    candidates = build_candidates(
        base_spec=base_spec,
        patch_spec=patch_spec,
        p=int(config.P),
        n=int(config.N),
        band_lengths=args.band_lengths,
        patch_modes=args.patch_modes,
    )
    print(f"[dynamic-splice] candidates={len(candidates)} steps={len(steps)} out_dir={out_dir}", flush=True)

    group_data = load_or_build_group_data(
        xml_file=path_from(paths_raw, "group_xml"),
        group_cache_dir=path_from(paths_raw, "group_cache_dir"),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=stride,
        enabled=True,
        force=bool(args.force_group_cache),
    )
    pair_by_key = region_pairs_from_workflow(workflow)
    metric_pairs = [str(x) for x in args.metric_pairs]
    preserve_pairs = [str(x) for x in args.preserve_pairs]
    target_pair = str(args.target_pair)
    required_pairs = list(dict.fromkeys(metric_pairs + preserve_pairs + [target_pair]))
    missing = [key for key in required_pairs if key not in pair_by_key]
    if missing:
        raise ValueError(f"unknown region pair(s): {missing}; available={sorted(pair_by_key)}")
    pair_nodes = pair_nodes_by_step(
        group_data=group_data,
        steps=steps,
        pairs={key: pair_by_key[key] for key in required_pairs},
    )

    base_dist = all_pairs_shortest_hops(base_spec.edge_table, int(config.total_sats))
    base_metrics = candidate_metric_arrays(
        dist=base_dist,
        pair_nodes=pair_nodes,
        pair_keys=required_pairs,
        num_steps=len(steps),
    )
    patch_dist = all_pairs_shortest_hops(patch_spec.edge_table, int(config.total_sats))
    patch_metrics = candidate_metric_arrays(
        dist=patch_dist,
        pair_nodes=pair_nodes,
        pair_keys=required_pairs,
        num_steps=len(steps),
    )

    best_idx = np.zeros(len(steps), dtype=np.int32)
    best_target = np.array(base_metrics[target_pair], dtype=np.float32)
    best_changes = np.zeros(len(steps), dtype=np.int32)
    selected_metrics = {key: np.array(base_metrics[key], dtype=np.float32) for key in required_pairs}
    candidate_summary: list[dict[str, Any]] = []
    candidate_metrics_store: list[dict[str, np.ndarray]] = []
    candidate_valid_store: list[np.ndarray] = []
    candidate_target_store: list[np.ndarray] = []

    for cand_pos, candidate in enumerate(candidates):
        if cand_pos == 0:
            metrics = base_metrics
            valid_mask = np.ones(len(steps), dtype=bool)
            target_values = base_metrics[target_pair]
        else:
            dist = all_pairs_shortest_hops(candidate.edge_table, int(config.total_sats))
            metrics = candidate_metric_arrays(
                dist=dist,
                pair_nodes=pair_nodes,
                pair_keys=required_pairs,
                num_steps=len(steps),
            )
            valid_mask = np.ones(len(steps), dtype=bool)
            for key in preserve_pairs:
                valid_mask &= np.isfinite(metrics[key])
                valid_mask &= metrics[key] <= base_metrics[key] + float(args.preserve_eps)

            target_values = metrics[target_pair]

        candidate_metrics_store.append({key: np.asarray(metrics[key], dtype=np.float32) for key in required_pairs})
        candidate_valid_store.append(np.asarray(valid_mask & np.isfinite(target_values), dtype=bool))
        candidate_target_store.append(np.asarray(target_values, dtype=np.float32))

        if str(args.selection_mode) == "instant":
            for row_idx in np.where(valid_mask)[0]:
                if should_replace_best(
                    candidate_target=float(target_values[row_idx]),
                    candidate_changes=int(candidate.changed_edges),
                    best_target=float(best_target[row_idx]),
                    best_changes=int(best_changes[row_idx]),
                    tie_eps=float(args.prefer_fewer_changes_eps),
                ):
                    best_idx[row_idx] = int(candidate.idx)
                    best_target[row_idx] = float(target_values[row_idx])
                    best_changes[row_idx] = int(candidate.changed_edges)
                    for key in required_pairs:
                        selected_metrics[key][row_idx] = metrics[key][row_idx]

        improvement = base_metrics[target_pair] - target_values
        candidate_summary.append(
            {
                "candidate_idx": int(candidate.idx),
                "candidate": candidate.name,
                "mode": candidate.mode,
                "band_start": "" if candidate.band_start is None else int(candidate.band_start),
                "band_end": "" if candidate.band_end is None else int(candidate.band_end),
                "band_len": int(candidate.band_len),
                "added_edges": int(candidate.added_edges),
                "removed_edges": int(candidate.removed_edges),
                "changed_edges": int(candidate.changed_edges),
                "valid_steps": int(np.count_nonzero(valid_mask)),
                "selected_steps": 0,
                "mean_target_hops": float(np.nanmean(target_values)),
                "mean_target_improvement": float(np.nanmean(improvement)),
                "max_target_improvement": float(np.nanmax(improvement)),
            }
        )
        if (cand_pos + 1) % 25 == 0 or cand_pos + 1 == len(candidates):
            print(f"[dynamic-splice] scored candidate {cand_pos + 1}/{len(candidates)}", flush=True)

    if str(args.selection_mode) == "transition-dp":
        target_matrix = np.stack(candidate_target_store, axis=1)
        valid_matrix = np.stack(candidate_valid_store, axis=1)
        transition_matrix = transition_new_edge_matrix(candidates)
        best_idx = select_candidates_transition_dp(
            target_values=target_matrix,
            valid_mask=valid_matrix,
            transition_new_edges=transition_matrix,
            penalty_per_new_edge=float(args.transition_setup_penalty_per_edge),
            max_new_edges=int(args.transition_max_new_edges),
        )
        rows = np.arange(len(steps), dtype=np.int32)
        for key in required_pairs:
            metric_matrix = np.stack([candidate_metrics_store[cand_idx][key] for cand_idx in range(len(candidates))], axis=1)
            selected_metrics[key] = metric_matrix[rows, best_idx].astype(np.float32)
        best_target = np.asarray(selected_metrics[target_pair], dtype=np.float32)
        best_changes = np.asarray([candidates[int(idx)].changed_edges for idx in best_idx], dtype=np.int32)

    selected_counts = Counter(int(idx) for idx in best_idx.tolist())
    for row in candidate_summary:
        row["selected_steps"] = int(selected_counts.get(int(row["candidate_idx"]), 0))

    transition_new = transition_new_edge_matrix(candidates)
    selected_transition_new_edges = np.zeros(len(steps), dtype=np.int32)
    if len(steps) > 1:
        selected_transition_new_edges[1:] = transition_new[best_idx[:-1], best_idx[1:]]

    union_edge_table, union_key_to_idx = build_union_edge_table(
        candidates,
        best_idx,
        p=int(config.P),
        n=int(config.N),
    )
    active_mask = build_active_mask(
        candidates=candidates,
        selected_idx=best_idx,
        union_key_to_idx=union_key_to_idx,
        num_edges=int(union_edge_table.num_edges),
    )
    if bool(args.skip_usage):
        usage_values = np.zeros((1, int(union_edge_table.num_edges)), dtype=np.float32)
        print("[dynamic-splice] skip_usage=True, wrote zero edge-usage matrix", flush=True)
    else:
        usage_pair_objs = [pair_by_key[key] for key in args.usage_pairs]
        usage_values = build_usage_values(
            candidates=candidates,
            selected_idx=best_idx,
            steps=steps,
            union_key_to_idx=union_key_to_idx,
            num_edges=int(union_edge_table.num_edges),
            config=config,
            group_data=group_data,
            usage_pairs=usage_pair_objs,
        )

    write_edges_csv(union_edge_table, out_dir / "union_edges.csv")
    np.save(out_dir / "edge_active_mask.npy", active_mask)
    np.save(out_dir / "edge_usage_values.npy", usage_values)
    write_dynamic_schedule(
        out_dir / "dynamic_schedule.csv",
        steps=steps,
        candidates=candidates,
        selected_idx=best_idx,
        selected_metrics=selected_metrics,
        base_metrics=base_metrics,
        target_pair=target_pair,
        metric_pairs=metric_pairs,
        transition_new_edges=selected_transition_new_edges,
    )
    write_candidate_summary(out_dir / "candidate_summary.csv", candidate_summary)
    plot_dynamic_summary(
        out_dir / "dynamic_splice_mean_hops.png",
        steps=steps,
        target_pair=target_pair,
        preserve_pairs=preserve_pairs,
        selected_metrics=selected_metrics,
        base_metrics=base_metrics,
        patch_metrics=patch_metrics,
    )

    selected_counts = {
        candidates[int(idx)].name: int(count)
        for idx, count in zip(*np.unique(best_idx, return_counts=True))
    }
    meta = {
        "base_topology": base_name,
        "patch_topology": patch_name,
        "target_pair": target_pair,
        "preserve_pairs": preserve_pairs,
        "metric_pairs": metric_pairs,
        "usage_pairs": list(args.usage_pairs),
        "skip_usage": bool(args.skip_usage),
        "patch_modes": list(args.patch_modes),
        "band_lengths": [int(x) for x in args.band_lengths],
        "selection_mode": str(args.selection_mode),
        "transition_setup_penalty_per_edge": float(args.transition_setup_penalty_per_edge),
        "transition_max_new_edges": int(args.transition_max_new_edges),
        "start": start,
        "end": end,
        "stride": stride,
        "num_steps": len(steps),
        "num_candidates": len(candidates),
        "num_selected_candidates": len(selected_counts),
        "selected_counts": selected_counts,
        "mean_new_edges_from_prev": float(np.mean(selected_transition_new_edges)),
        "max_new_edges_from_prev": int(np.max(selected_transition_new_edges)),
        "target_base_mean_hops": float(np.nanmean(base_metrics[target_pair])),
        "target_dynamic_mean_hops": float(np.nanmean(selected_metrics[target_pair])),
        "target_mean_improvement": float(np.nanmean(base_metrics[target_pair] - selected_metrics[target_pair])),
        "active_union_edges": int(union_edge_table.num_edges),
        "outputs": {
            "dynamic_schedule": str(out_dir / "dynamic_schedule.csv"),
            "candidate_summary": str(out_dir / "candidate_summary.csv"),
            "union_edges": str(out_dir / "union_edges.csv"),
            "edge_active_mask": str(out_dir / "edge_active_mask.npy"),
            "edge_usage_values": str(out_dir / "edge_usage_values.npy"),
            "plot": str(out_dir / "dynamic_splice_mean_hops.png"),
        },
        "elapsed_sec": float(time.time() - started_at),
    }
    (out_dir / "dynamic_splice_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[dynamic-splice] done elapsed={meta['elapsed_sec']:.1f}s "
        f"dynamic_mean={meta['target_dynamic_mean_hops']:.6f} "
        f"base_mean={meta['target_base_mean_hops']:.6f} out_dir={out_dir}",
        flush=True,
    )
    print(f"[dynamic-splice] selected topologies={len(selected_counts)}", flush=True)

    return {
        "workflow": workflow,
        "config": config,
        "steps": steps,
        "out_dir": out_dir,
        "union_edge_table": union_edge_table,
        "active_mask": active_mask,
        "usage_values": usage_values,
        "group_data": group_data,
    }


def main() -> int:
    args = parse_args()
    outputs = build_or_load_outputs(args)
    if args.check_only:
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
        edge_table=outputs["union_edge_table"],
        edge_active_mask=outputs["active_mask"],
        edge_usage_values=usage_values,
        value_max=float(np.nanmax(usage_values)) if usage_values.size else 1.0,
        window_title=f"G60 dynamic splice 000056 + local 000040 | {outputs['out_dir'].name}",
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
