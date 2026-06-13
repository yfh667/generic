from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.topology_metrics.module.group_states import GroupStateIndex, build_group_state_index

from .edge_tables import build_full_option_plus_intra_edge_table, build_motif_text_edge_table


@dataclass
class TopologySpec:
    name: str
    edge_table: EdgeTable
    library: str = ""
    motif_id: int | None = None
    motif: str = ""
    source_w: int | None = None
    source_h: int | None = None
    edge_count_local: int | None = None
    support: str = ""
    edges: str = ""
    baseline: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionPairSpec:
    key: str
    label: str
    source_group_id: int
    target_group_id: int


@dataclass(frozen=True)
class PairStatePayload:
    spec: RegionPairSpec
    state_index: GroupStateIndex


def auto_worker_count(requested: int = 0, *, max_cap: int = 32) -> int:
    if int(requested) > 0:
        return int(requested)
    cpu = os.cpu_count() or 2
    return max(1, min(int(max_cap), int(cpu) - 1))


def _optional_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def topology_specs_from_motif_csv(
    path: str | Path,
    *,
    config: ViewerConfig,
    library: str,
    name_prefix: str | None = None,
    limit: int = 0,
    default_source_w: int | None = None,
    default_source_h: int | None = None,
    add_intra_ring: bool = True,
) -> list[TopologySpec]:
    """Load motif rows and build one full-grid topology for each row.

    The CSV is expected to contain at least a ``motif`` column and preferably
    ``motif_id`` plus optional metadata columns such as ``source_w``,
    ``source_h``, ``support``, and ``edges``.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if int(limit) > 0:
        rows = rows[: int(limit)]

    specs: list[TopologySpec] = []
    prefix = str(name_prefix or library)
    for row_idx, row in enumerate(rows, start=1):
        motif_text = str(row["motif"])
        motif_id = _optional_int(row.get("motif_id"), row_idx)
        edge_table = build_motif_text_edge_table(
            motif_text=motif_text,
            config=config,
            add_intra_ring=bool(add_intra_ring),
        )
        specs.append(
            TopologySpec(
                name=f"{prefix}_motif_{int(motif_id):06d}",
                edge_table=edge_table,
                library=str(library),
                motif_id=int(motif_id),
                motif=motif_text,
                source_w=_optional_int(row.get("source_w") or row.get("w"), default_source_w),
                source_h=_optional_int(row.get("source_h") or row.get("h"), default_source_h),
                edge_count_local=_optional_int(row.get("edge_count")),
                support=str(row.get("support", "")),
                edges=str(row.get("edges", "")),
                baseline=False,
                meta={k: v for k, v in row.items() if k not in {"motif"}},
            )
        )
    return specs


def full_link_topology_spec(
    *,
    config: ViewerConfig,
    name: str = "full_link",
    options: tuple[int, ...] = (0, 1, 2, 4),
    add_intra_ring: bool = True,
) -> TopologySpec:
    return TopologySpec(
        name=str(name),
        edge_table=build_full_option_plus_intra_edge_table(
            config=config,
            options=tuple(int(x) for x in options),
            add_intra_ring=bool(add_intra_ring),
        ),
        library="baseline",
        motif="full_option_plus_intra",
        baseline=True,
        meta={"options": list(int(x) for x in options), "add_intra_ring": bool(add_intra_ring)},
    )


def build_pair_state_payloads(
    *,
    group_data: Mapping,
    steps: Sequence[int],
    pair_specs: Sequence[RegionPairSpec],
) -> dict[str, PairStatePayload]:
    out: dict[str, PairStatePayload] = {}
    for pair in pair_specs:
        state_index = build_group_state_index(
            group_data=group_data,
            steps=list(int(step) for step in steps),
            source_group_id=int(pair.source_group_id),
            target_group_id=int(pair.target_group_id),
        )
        out[str(pair.key)] = PairStatePayload(spec=pair, state_index=state_index)
    return out


def _build_adjacency_arrays(edge_table: EdgeTable, total_nodes: int) -> tuple[np.ndarray, np.ndarray]:
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
    indptr, neighbors = _build_adjacency_arrays(edge_table, int(total_nodes))
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


def summarize_group_state(
    dist: np.ndarray,
    source_nodes: tuple[int, ...],
    target_nodes: tuple[int, ...],
) -> tuple[float, int, int, bool, float, float]:
    required = int(len(source_nodes) * len(target_nodes))
    if required == 0:
        return math.nan, 0, required, False, math.nan, math.nan

    sources = np.asarray(source_nodes, dtype=np.int32)
    targets = np.asarray(target_nodes, dtype=np.int32)
    values = dist[sources[:, None], targets].astype(np.int16, copy=False)
    reachable = values >= 0
    reachable_count = int(np.sum(reachable))
    if reachable_count == 0:
        return math.nan, 0, required, False, math.nan, math.nan

    reachable_values = values[reachable].astype(np.float64, copy=False)
    return (
        float(np.mean(reachable_values)),
        reachable_count,
        required,
        bool(reachable_count == required),
        float(np.min(reachable_values)),
        float(np.max(reachable_values)),
    )


def _compute_topology_task(task: tuple[TopologySpec, dict[str, PairStatePayload], int, bool, str]) -> dict[str, Any]:
    spec, pair_payloads, total_nodes, write_edges, edges_root = task
    started_at = time.time()
    if write_edges:
        topology_dir = Path(edges_root) / spec.name
        topology_dir.mkdir(parents=True, exist_ok=True)
        write_edges_csv(spec.edge_table, topology_dir / "edges.csv")

    dist = all_pairs_hop_dist(spec.edge_table, int(total_nodes))
    by_pair: dict[str, dict[str, np.ndarray]] = {}
    for pair_key, payload in pair_payloads.items():
        states = payload.state_index.unique_states
        state_ids = np.asarray(payload.state_index.state_ids, dtype=np.int32)
        state_mean = np.full(len(states), np.nan, dtype=np.float32)
        state_reachable = np.zeros(len(states), dtype=np.int32)
        state_required = np.zeros(len(states), dtype=np.int32)
        state_full = np.zeros(len(states), dtype=np.bool_)
        state_min = np.full(len(states), np.nan, dtype=np.float32)
        state_max = np.full(len(states), np.nan, dtype=np.float32)

        for state_id, (source_nodes, target_nodes) in enumerate(states):
            mean, reachable, required, full, min_value, max_value = summarize_group_state(
                dist,
                source_nodes,
                target_nodes,
            )
            state_mean[state_id] = mean
            state_reachable[state_id] = reachable
            state_required[state_id] = required
            state_full[state_id] = full
            state_min[state_id] = min_value
            state_max[state_id] = max_value

        by_pair[str(pair_key)] = {
            "mean_hops": state_mean[state_ids].astype(np.float32, copy=False),
            "reachable_pairs": state_reachable[state_ids].astype(np.int32, copy=False),
            "required_pairs": state_required[state_ids].astype(np.int32, copy=False),
            "full_reachable": state_full[state_ids].astype(np.bool_, copy=False),
            "min_hops": state_min[state_ids].astype(np.float32, copy=False),
            "max_hops": state_max[state_ids].astype(np.float32, copy=False),
        }

    return {
        "name": spec.name,
        "library": spec.library,
        "baseline": bool(spec.baseline),
        "num_edges": int(spec.edge_table.num_edges),
        "elapsed_s": float(time.time() - started_at),
        "pairs": by_pair,
    }


def finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _write_topology_library(specs: Sequence[TopologySpec], path: Path, edge_counts: Mapping[str, int]) -> None:
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
                    "edge_count_local": "" if spec.edge_count_local is None else int(spec.edge_count_local),
                    "edge_count_full_topology": edge_counts.get(spec.name, ""),
                    "support": spec.support,
                    "edges": spec.edges,
                    "baseline": bool(spec.baseline),
                }
            )


def _sanitize_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def _write_pair_outputs(
    *,
    pair_payload: PairStatePayload,
    out_dir: Path,
    steps: Sequence[int],
    motif_specs: Sequence[TopologySpec],
    baseline_specs: Sequence[TopologySpec],
    pair_results: Mapping[str, Mapping[str, np.ndarray]],
    run_label: str,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair = pair_payload.spec
    pair_dir = out_dir / str(pair.key)
    pair_dir.mkdir(parents=True, exist_ok=True)
    topology_names = [spec.name for spec in motif_specs]
    baseline_names = [spec.name for spec in baseline_specs]
    safe_label = _sanitize_label(run_label)

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

    def strict_sort_key(row: Mapping[str, Any]) -> tuple[int, float]:
        value = row["strict_static_mean_hops"]
        if isinstance(value, float) and math.isfinite(value):
            return 0, float(value)
        finite_value = float(row["finite_mean_hops"]) if math.isfinite(float(row["finite_mean_hops"])) else math.inf
        return 1, finite_value

    summary_rows.sort(key=strict_sort_key)
    if not summary_rows:
        raise ValueError("motif_specs is empty; no pair output can be written")

    summary_path = pair_dir / f"motif_summary_sorted_{safe_label}_{pair.key}_strict_reachable.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    strict_compare_path = pair_dir / f"compare_{safe_label}_{pair.key}_strict_reachable.csv"
    finite_compare_path = pair_dir / f"compare_{safe_label}_{pair.key}_finite_only.csv"
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

    dynamic_path = pair_dir / f"dynamic_best_{safe_label}_{pair.key}_strict_reachable.csv"
    with dynamic_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["step", "best_topology", "best_library", "best_hops"] + [f"{name}_hops" for name in baseline_names]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        library_by_name = {spec.name: spec.library for spec in motif_specs}
        for idx, step in enumerate(steps):
            best_name = str(dynamic_names[idx])
            row = {
                "step": int(step),
                "best_topology": best_name,
                "best_library": library_by_name.get(best_name, ""),
                "best_hops": "" if not math.isfinite(float(dynamic_values[idx])) else float(dynamic_values[idx]),
            }
            for name in baseline_names:
                value = float(pair_results[name]["mean_hops"][idx])
                row[f"{name}_hops"] = "" if not math.isfinite(value) else value
            writer.writerow(row)

    best_static = summary_rows[0]
    best_by_library: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        best_by_library.setdefault(str(row["library"]), row)

    x_hours = np.asarray(steps, dtype=np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(16, 7.3), dpi=180)
    palette = ["#2563eb", "#6b7280", "#7c3aed", "#0f766e", "#be123c"]
    libraries = sorted({spec.library for spec in motif_specs})
    color_by_library = {lib: palette[idx % len(palette)] for idx, lib in enumerate(libraries)}
    for spec in motif_specs:
        ax.plot(
            x_hours,
            strict_values_by_name[spec.name],
            color=color_by_library.get(spec.library, "#6b7280"),
            alpha=0.06,
            linewidth=0.55,
        )
    for library, row in best_by_library.items():
        name = str(row["topology"])
        ax.plot(
            x_hours,
            strict_values_by_name[name],
            color=color_by_library.get(str(library), "#111827"),
            linewidth=1.65,
            label=f"best static {library}: {name}",
        )
    ax.plot(x_hours, dynamic_values, color="#111827", linestyle="--", linewidth=1.65, label="per-time best")
    baseline_colors = ["#f97316", "#16a34a", "#dc2626", "#0891b2"]
    for idx, name in enumerate(baseline_names):
        ax.plot(
            x_hours,
            pair_results[name]["mean_hops"],
            color=baseline_colors[idx % len(baseline_colors)],
            linewidth=1.35,
            label=name,
        )
    ax.set_title(f"{pair.label} mean shortest path length ({run_label}, strict reachable)")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(f"{pair.label} mean shortest path (hops)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    plot_path = pair_dir / f"mean_shortest_hops_{safe_label}_{pair.key}_strict_reachable.png"
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
        "pair_key": pair.key,
        "pair_label": pair.label,
        "source_group_id": int(pair.source_group_id),
        "target_group_id": int(pair.target_group_id),
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
    (pair_dir / f"summary_{safe_label}_{pair.key}_strict_reachable.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def compute_shortest_hops_batch(
    *,
    topology_specs: Sequence[TopologySpec],
    pair_specs: Sequence[RegionPairSpec],
    group_data: Mapping,
    steps: Sequence[int],
    total_nodes: int,
    out_dir: str | Path,
    run_label: str = "batch",
    max_workers: int = 0,
    progress_every: int = 25,
    write_edges: bool = False,
) -> dict[str, Any]:
    started_at = time.time()
    steps = [int(step) for step in steps]
    if not steps:
        raise ValueError("steps is empty")
    if not topology_specs:
        raise ValueError("topology_specs is empty")
    if not pair_specs:
        raise ValueError("pair_specs is empty")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))

    motif_specs = [spec for spec in topology_specs if not bool(spec.baseline)]
    baseline_specs = [spec for spec in topology_specs if bool(spec.baseline)]
    pair_payloads = build_pair_state_payloads(group_data=group_data, steps=steps, pair_specs=pair_specs)

    workers = auto_worker_count(int(max_workers))
    print(
        f"[topology-workflow] shortest_hops_batch topologies={len(topology_specs)} "
        f"motifs={len(motif_specs)} baselines={len(baseline_specs)} steps={len(steps)} workers={workers}",
        flush=True,
    )
    for pair_key, payload in pair_payloads.items():
        print(
            f"[topology-workflow] pair={pair_key} label={payload.spec.label} "
            f"unique_states={payload.state_index.num_states}",
            flush=True,
        )

    pair_results: dict[str, dict[str, dict[str, np.ndarray]]] = {pair.key: {} for pair in pair_specs}
    edge_counts: dict[str, int] = {}
    topology_elapsed: dict[str, float] = {}
    tasks = [
        (spec, pair_payloads, int(total_nodes), bool(write_edges), str(out_dir / "topology_edges"))
        for spec in topology_specs
    ]

    completed = 0
    if workers <= 1:
        for task in tasks:
            result = _compute_topology_task(task)
            name = str(result["name"])
            edge_counts[name] = int(result["num_edges"])
            topology_elapsed[name] = float(result["elapsed_s"])
            for pair_key, payload in result["pairs"].items():
                pair_results[str(pair_key)][name] = payload
            completed += 1
            if int(progress_every) > 0 and (completed == len(tasks) or completed % int(progress_every) == 0):
                print(
                    f"[topology-workflow] completed {completed}/{len(tasks)} last={name} "
                    f"elapsed={time.time() - started_at:.1f}s",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_compute_topology_task, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                name = str(result["name"])
                edge_counts[name] = int(result["num_edges"])
                topology_elapsed[name] = float(result["elapsed_s"])
                for pair_key, payload in result["pairs"].items():
                    pair_results[str(pair_key)][name] = payload
                completed += 1
                if int(progress_every) > 0 and (completed == len(futures) or completed % int(progress_every) == 0):
                    print(
                        f"[topology-workflow] completed {completed}/{len(futures)} last={name} "
                        f"elapsed={time.time() - started_at:.1f}s",
                        flush=True,
                    )

    _write_topology_library(topology_specs, out_dir / f"topology_library_{_sanitize_label(run_label)}.csv", edge_counts)
    pair_summaries = []
    for pair in pair_specs:
        summary = _write_pair_outputs(
            pair_payload=pair_payloads[pair.key],
            out_dir=out_dir,
            steps=steps,
            motif_specs=motif_specs,
            baseline_specs=baseline_specs,
            pair_results=pair_results[pair.key],
            run_label=run_label,
        )
        pair_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    meta = {
        "run_label": str(run_label),
        "metric": "unweighted shortest path hop count",
        "topology_count": int(len(topology_specs)),
        "motif_count": int(len(motif_specs)),
        "baseline_count": int(len(baseline_specs)),
        "start": int(steps[0]),
        "end": int(steps[-1]),
        "stride": int(steps[1] - steps[0]) if len(steps) > 1 else None,
        "num_steps": int(len(steps)),
        "pair_summaries": pair_summaries,
        "topology_elapsed_s": topology_elapsed,
        "elapsed_s": float(time.time() - started_at),
    }
    (out_dir / f"shortest_hops_batch_summary_{_sanitize_label(run_label)}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[topology-workflow] shortest_hops_batch done elapsed={time.time() - started_at:.1f}s out_dir={out_dir}", flush=True)
    return meta
