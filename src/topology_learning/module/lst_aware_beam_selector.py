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
from src.link_delay.module.position_cache import open_position_cache_for_interval
from src.link_delay.module.query import open_delay_store_for_interval
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_metrics.module.group_states import group_nodes_for_step
from src.topology_workflow.module.edge_tables import make_edge_table_from_records
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step

from .config_io import load_yaml_dict, optional_path
from .full_link_gap_selector import _metric_specs_from_raw, _viewer_config_from_raw, read_edge_weights_by_key
from .lst_schedule_eval import _edge_key, _edge_table_for_topology_name, _records_from_edge_table


@dataclass(frozen=True)
class LstAwarePairSpec:
    source_group_id: int
    target_group_id: int
    metric_prefix: str
    label: str = ""


@dataclass(frozen=True)
class LstAwareBeamResult:
    selector_dir: Path
    schedule_name: str
    selected_action: np.ndarray
    total_setup_commands: int
    max_setup_commands_per_step: int
    mean_stage_cost: float


def parse_pair_spec(text: str) -> LstAwarePairSpec:
    """Parse `source_group_id:target_group_id:metric_prefix[:label]`."""

    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) < 3:
        raise ValueError("pair must look like source_group_id:target_group_id:metric_prefix[:label]")
    return LstAwarePairSpec(
        source_group_id=int(parts[0]),
        target_group_id=int(parts[1]),
        metric_prefix=str(parts[2]),
        label=str(parts[3]) if len(parts) > 3 else str(parts[2]),
    )


def _metric_weights(raw_config: Mapping[str, Any], pair_specs: Sequence[LstAwarePairSpec]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for spec in _metric_specs_from_raw(raw_config.get("metrics")):
        weights[str(spec.name)] = float(spec.weight)
    names: list[str] = []
    for pair in pair_specs:
        names.append(f"{pair.metric_prefix}_delay_ms")
        names.append(f"{pair.metric_prefix}_hops")
    missing = [name for name in names if name not in weights]
    if missing:
        default = 1.0 / float(len(names))
        for name in missing:
            weights[name] = default
    return weights


def _load_config_with_g60_fallback(raw_config: Mapping[str, Any]) -> ViewerConfig:
    config = _viewer_config_from_raw(raw_config)
    if not config.station_groups and config.name == "G60" and int(config.P) == 18 and int(config.N) == 36:
        from src.config.viewer_config import G60_CONFIG

        return G60_CONFIG
    return config


def _build_union_table_and_action_masks(
    *,
    config: ViewerConfig,
    topology_names: Sequence[str],
    topology_library_csv: str | Path,
) -> tuple[Any, np.ndarray]:
    topology_library: dict[str, dict[str, Any]] = {}
    with Path(topology_library_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("name"):
                topology_library[str(row["name"])] = dict(row)

    tables = []
    records = []
    for name in topology_names:
        table = _edge_table_for_topology_name(
            name=str(name),
            config=config,
            topology_library=topology_library,
        )
        tables.append(table)
        records.extend(_records_from_edge_table(table))

    union_table = make_edge_table_from_records(p=int(config.P), n=int(config.N), records=records)
    key_to_col = {
        _edge_key(int(union_table.src[idx]), int(union_table.dst[idx])): int(idx)
        for idx in range(int(union_table.num_edges))
    }
    masks = np.zeros((len(tables), int(union_table.num_edges)), dtype=bool)
    for action, table in enumerate(tables):
        for idx in range(int(table.num_edges)):
            masks[action, key_to_col[_edge_key(int(table.src[idx]), int(table.dst[idx]))]] = True
    return union_table, masks


def _candidate_actions_for_row(
    *,
    row: int,
    stage_cost: np.ndarray,
    top_k: int,
    include_schedules: Sequence[np.ndarray],
) -> np.ndarray:
    values = np.asarray(stage_cost[int(row)], dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        raise RuntimeError(f"no finite actions at row={row}")
    k = min(int(top_k), int(finite.size))
    top = finite[np.argpartition(values[finite], k - 1)[:k]]
    extras = [int(schedule[int(row)]) for schedule in include_schedules]
    return np.asarray(sorted(set(int(x) for x in top.tolist() + extras)), dtype=np.int32)


def _mean_shortest_hops(
    *,
    n_nodes: int,
    src_edges: np.ndarray,
    dst_edges: np.ndarray,
    sources: Sequence[int],
    targets: Sequence[int],
    require_all_pairs: bool = False,
) -> float:
    if not sources or not targets or src_edges.size == 0:
        return math.inf
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    rows = np.concatenate([src_edges, dst_edges]).astype(np.int32, copy=False)
    cols = np.concatenate([dst_edges, src_edges]).astype(np.int32, copy=False)
    data = np.ones(rows.shape[0], dtype=np.float32)
    graph = csr_matrix((data, (rows, cols)), shape=(int(n_nodes), int(n_nodes)))
    dist = shortest_path(graph, directed=False, unweighted=True, indices=np.asarray(sources, dtype=np.int32))
    values = np.atleast_2d(np.asarray(dist, dtype=np.float64))[:, np.asarray(targets, dtype=np.int32)]
    finite = np.isfinite(values)
    if bool(require_all_pairs) and not bool(np.all(finite)):
        return math.inf
    return float(np.mean(values[finite])) if bool(np.any(finite)) else math.inf


def _mean_shortest_delay(
    *,
    n_nodes: int,
    src_edges: np.ndarray,
    dst_edges: np.ndarray,
    weights: np.ndarray,
    sources: Sequence[int],
    targets: Sequence[int],
    require_all_pairs: bool = False,
) -> float:
    if not sources or not targets or src_edges.size == 0:
        return math.inf
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    rows = np.concatenate([src_edges, dst_edges]).astype(np.int32, copy=False)
    cols = np.concatenate([dst_edges, src_edges]).astype(np.int32, copy=False)
    data = np.concatenate([weights, weights]).astype(np.float32, copy=False)
    graph = csr_matrix((data, (rows, cols)), shape=(int(n_nodes), int(n_nodes)))
    dist = dijkstra(graph, directed=False, indices=np.asarray(sources, dtype=np.int32))
    values = np.atleast_2d(np.asarray(dist, dtype=np.float64))[:, np.asarray(targets, dtype=np.int32)]
    finite = np.isfinite(values)
    if bool(require_all_pairs) and not bool(np.all(finite)):
        return math.inf
    return float(np.mean(values[finite])) if bool(np.any(finite)) else math.inf


def _relative_gap(value: float, reference: float, epsilon: float = 1e-9) -> float:
    if not math.isfinite(float(value)) or not math.isfinite(float(reference)):
        return math.inf
    denom = max(abs(float(reference)), float(epsilon))
    return abs(float(value) - float(reference)) / denom


def _active_mask_from_history(action_masks: np.ndarray, history: Sequence[int]) -> np.ndarray:
    current = np.ones(action_masks.shape[1], dtype=bool)
    for action in history:
        current &= action_masks[int(action)]
    return current


def _edge_weights_for_union_table(
    *,
    union_table: Any,
    edge_weight_by_key: Mapping[tuple[int, int], float],
    missing_edge_weight: float = 0.0,
) -> np.ndarray:
    return np.asarray(
        [
            float(edge_weight_by_key.get(_edge_key(int(src), int(dst)), float(missing_edge_weight)))
            for src, dst in zip(union_table.src, union_table.dst, strict=True)
        ],
        dtype=np.float32,
    )


def _weighted_added_edge_transition_counts_from_masks(
    *,
    action_masks: np.ndarray,
    edge_weights: np.ndarray,
) -> np.ndarray:
    masks = np.asarray(action_masks, dtype=bool)
    weights = np.asarray(edge_weights, dtype=np.float32).reshape(-1)
    if masks.ndim != 2:
        raise ValueError("action_masks must be 2D")
    if weights.shape[0] != masks.shape[1]:
        raise ValueError("edge_weights length must match action_masks columns")
    transition = np.empty((masks.shape[0], masks.shape[0]), dtype=np.float32)
    for next_idx in range(masks.shape[0]):
        added = masks[next_idx][None, :] & ~masks
        transition[:, next_idx] = added.astype(np.float32) @ weights
    return transition


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_selector_summary(path: Path, arrays_payload: Mapping[str, np.ndarray]) -> None:
    steps = np.asarray(arrays_payload["steps"])
    stage_cost = np.asarray(arrays_payload["stage_cost"], dtype=np.float64)
    transition_cost = np.asarray(arrays_payload.get("transition_cost", np.zeros((stage_cost.shape[1], stage_cost.shape[1]))), dtype=np.float64)
    metric_names = sorted(
        key[len("metric_") :]
        for key, value in arrays_payload.items()
        if key.startswith("metric_")
        and not key.endswith("_full_link")
        and not key.endswith("_gap")
        and np.asarray(value).ndim == 2
    )
    rows: list[dict[str, Any]] = []
    for key, value in arrays_payload.items():
        if not key.startswith("schedule_"):
            continue
        selected = np.asarray(value, dtype=np.int32)
        if selected.shape[0] != steps.shape[0]:
            continue
        row_idx = np.arange(selected.shape[0])
        transitions = (
            transition_cost[selected[:-1], selected[1:]]
            if selected.shape[0] > 1
            else np.asarray([], dtype=np.float64)
        )
        item: dict[str, Any] = {
            "schedule": key[len("schedule_") :],
            "mean_stage_cost": float(np.mean(stage_cost[row_idx, selected])),
            "total_stage_cost": float(np.sum(stage_cost[row_idx, selected])),
            "num_switches": int(np.count_nonzero(selected[1:] != selected[:-1])) if selected.shape[0] > 1 else 0,
            "sum_new_edge_transition_cost": float(np.sum(transitions)) if transitions.size else 0.0,
            "mean_new_edge_transition_cost": float(np.mean(transitions)) if transitions.size else 0.0,
            "num_unique_topologies": int(len(np.unique(selected))),
        }
        for metric_name in metric_names:
            values = np.asarray(arrays_payload[f"metric_{metric_name}"], dtype=np.float64)
            refs = np.asarray(arrays_payload[f"metric_{metric_name}_full_link"], dtype=np.float64)
            gaps = np.asarray(arrays_payload[f"metric_{metric_name}_gap"], dtype=np.float64)
            chosen_values = values[row_idx, selected]
            chosen_gaps = gaps[row_idx, selected]
            item[f"mean_{metric_name}_gap"] = float(np.mean(chosen_gaps))
            item[f"max_{metric_name}_gap"] = float(np.max(chosen_gaps))
            item[f"mean_{metric_name}_value"] = float(np.mean(chosen_values))
            item[f"mean_{metric_name}_full_link"] = float(np.mean(refs))
        rows.append(item)
    _write_rows(path, rows)


def run_lst_aware_beam_selector(
    *,
    selector_dir: str | Path,
    out_dir: str | Path,
    pair_specs: Sequence[LstAwarePairSpec],
    setup_time_seconds: int | float,
    delay_store_dir: str | Path,
    position_cache_dir: str | Path | None,
    group_xml: str | Path,
    group_cache_dir: str | Path,
    top_k: int = 12,
    beam_width: int = 8,
    setup_penalty: float = 0.0,
    burst_penalty: float = 0.0,
    critical_setup_penalty: float = 0.0,
    critical_edges_csv: str | Path | None = None,
    critical_weights_npy: str | Path | None = None,
    critical_normalize_weights: str = "max",
    critical_weight_power: float = 1.0,
    critical_weight_scale: float = 1.0,
    critical_base_new_edge_cost: float = 0.0,
    critical_missing_edge_weight: float = 0.0,
    start_row: int = 0,
    max_rows: int | None = None,
    include_schedule_names: Sequence[str] = (),
    force_group_cache: bool = False,
) -> LstAwareBeamResult:
    """Beam-search a topology sequence using active-after-LST graph metrics.

    This is intentionally a controlled approximation: each row considers only
    the `top_k` candidates from the precomputed full-link stage cost, plus any
    supplied existing schedules. Candidate scoring uses the active graph after
    LST, not the desired target graph.
    """

    selector_dir = Path(selector_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_config = load_yaml_dict(selector_dir / "selector_config.yaml")
    config = _load_config_with_g60_fallback(raw_config)
    topology_library_csv = optional_path(raw_config.get("inputs", {}).get("topology_library_csv"))
    if topology_library_csv is None:
        raise ValueError("selector_config.yaml inputs.topology_library_csv is required")

    arrays = np.load(selector_dir / "selector_arrays.npz", allow_pickle=False)
    original_num_steps = int(np.asarray(arrays["steps"]).shape[0])
    start_row = max(0, int(start_row))
    if start_row >= original_num_steps:
        raise ValueError(f"start_row={start_row} is outside steps length={original_num_steps}")
    end_row = original_num_steps if max_rows is None else min(original_num_steps, start_row + int(max_rows))
    row_slice = slice(start_row, end_row)
    steps = np.asarray(arrays["steps"], dtype=np.int64)[row_slice]
    topology_names = tuple(str(x) for x in np.asarray(arrays["topology_names"]))
    stage_cost = np.asarray(arrays["stage_cost"], dtype=np.float32)[row_slice]
    transition_counts = np.asarray(arrays["transition_counts"], dtype=np.float32)
    if steps.size < 1:
        raise ValueError("no steps available")

    include_schedules: list[np.ndarray] = []
    for name in include_schedule_names:
        key = f"schedule_{name}"
        if key not in arrays.files:
            raise KeyError(f"schedule {name!r} not found in selector arrays")
        include_schedules.append(np.asarray(arrays[key], dtype=np.int32)[row_slice])

    union_table, action_masks = _build_union_table_and_action_masks(
        config=config,
        topology_names=topology_names,
        topology_library_csv=topology_library_csv,
    )
    src_all = np.asarray(union_table.src, dtype=np.int32)
    dst_all = np.asarray(union_table.dst, dtype=np.int32)

    stride = int(steps[1] - steps[0]) if steps.size > 1 else 1
    lag_steps = int(math.ceil(float(setup_time_seconds) / float(stride))) if float(setup_time_seconds) > 0 else 0
    history_len = max(1, lag_steps + 1)
    max_transition = float(np.max(transition_counts)) or 1.0
    critical_transition_counts = None
    critical_max_transition = 1.0
    if critical_edges_csv is not None or critical_weights_npy is not None or float(critical_setup_penalty) != 0.0:
        if critical_edges_csv is None or critical_weights_npy is None:
            raise ValueError(
                "critical_setup_penalty requires both critical_edges_csv and critical_weights_npy"
            )
        critical_weights_by_key = read_edge_weights_by_key(
            edges_csv=critical_edges_csv,
            weights_npy=critical_weights_npy,
            normalize_weights=str(critical_normalize_weights),
            weight_power=float(critical_weight_power),
            weight_scale=float(critical_weight_scale),
            base_new_edge_cost=float(critical_base_new_edge_cost),
        )
        critical_edge_weights = _edge_weights_for_union_table(
            union_table=union_table,
            edge_weight_by_key=critical_weights_by_key,
            missing_edge_weight=float(critical_missing_edge_weight),
        )
        critical_transition_counts = _weighted_added_edge_transition_counts_from_masks(
            action_masks=action_masks,
            edge_weights=critical_edge_weights,
        )
        critical_max_transition = float(np.max(critical_transition_counts)) or 1.0

    group_data = load_or_build_group_data(
        xml_file=Path(group_xml),
        group_cache_dir=Path(group_cache_dir),
        steps=[int(x) for x in steps],
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(stride),
        enabled=True,
        force=bool(force_group_cache),
    )

    pair_nodes_by_row: list[list[tuple[LstAwarePairSpec, tuple[int, ...], tuple[int, ...]]]] = []
    for step in steps:
        row_pairs = []
        for pair in pair_specs:
            row_pairs.append(
                (
                    pair,
                    tuple(group_nodes_for_step(group_data, int(step), int(pair.source_group_id))),
                    tuple(group_nodes_for_step(group_data, int(step), int(pair.target_group_id))),
                )
            )
        pair_nodes_by_row.append(row_pairs)

    metric_weights = _metric_weights(raw_config, pair_specs)
    metric_refs: dict[str, np.ndarray] = {}
    for pair in pair_specs:
        for suffix in ("delay_ms", "hops"):
            metric_name = f"{pair.metric_prefix}_{suffix}"
            metric_refs[metric_name] = np.asarray(arrays[f"metric_{metric_name}_full_link"], dtype=np.float32)[row_slice]
    delay_store = open_delay_store_for_interval(
        int(steps[0]),
        int(steps[-1]),
        stride=int(stride),
        store_dir=Path(delay_store_dir),
        constellation_name=config.name,
    )
    delay_rows = delay_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(stride))
    position_store = None
    position_rows = None
    if position_cache_dir not in (None, "", False):
        position_store = open_position_cache_for_interval(
            int(steps[0]),
            int(steps[-1]),
            stride=int(stride),
            cache_dir=Path(position_cache_dir),
        )
        position_rows = position_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(stride))
    lookup = build_weight_lookup(
        union_table,
        delay_store,
        config=config,
        allow_intra_fallback=position_store is not None,
    )

    weights_by_row = []
    for row in range(int(steps.size)):
        weights_by_row.append(
            edge_weights_for_step(
                edge_table=union_table,
                lookup=lookup,
                delay_store=delay_store,
                position_store=position_store,
                delay_row=int(delay_rows[row]),
                position_row=int(position_rows[row]) if position_rows is not None else None,
            )
        )

    eval_cache: dict[tuple[int, tuple[int, ...]], float] = {}

    def score_active_graph(row: int, history: tuple[int, ...]) -> float:
        key = (int(row), tuple(int(x) for x in history[-history_len:]))
        cached = eval_cache.get(key)
        if cached is not None:
            return float(cached)
        active = _active_mask_from_history(action_masks, key[1])
        cols = np.flatnonzero(active)
        src_edges = src_all[cols]
        dst_edges = dst_all[cols]
        weights = np.asarray(weights_by_row[int(row)], dtype=np.float32)[cols]
        total = 0.0
        for pair, sources, targets in pair_nodes_by_row[int(row)]:
            delay_name = f"{pair.metric_prefix}_delay_ms"
            hops_name = f"{pair.metric_prefix}_hops"
            delay_value = _mean_shortest_delay(
                n_nodes=int(config.total_sats),
                src_edges=src_edges,
                dst_edges=dst_edges,
                weights=weights,
                sources=sources,
                targets=targets,
                require_all_pairs=True,
            )
            hops_value = _mean_shortest_hops(
                n_nodes=int(config.total_sats),
                src_edges=src_edges,
                dst_edges=dst_edges,
                sources=sources,
                targets=targets,
                require_all_pairs=True,
            )
            delay_ref = float(metric_refs[delay_name][int(row)])
            hops_ref = float(metric_refs[hops_name][int(row)])
            total += float(metric_weights.get(delay_name, 0.0)) * _relative_gap(delay_value, delay_ref)
            total += float(metric_weights.get(hops_name, 0.0)) * _relative_gap(hops_value, hops_ref)
        if not math.isfinite(total):
            total = 1.0e6
        eval_cache[key] = float(total)
        return float(total)

    candidates0 = _candidate_actions_for_row(
        row=0,
        stage_cost=stage_cost,
        top_k=int(top_k),
        include_schedules=include_schedules,
    )
    beam: list[tuple[float, tuple[int, ...], list[int], list[float]]] = []
    for action in candidates0:
        history = tuple([int(action)] * history_len)
        cost = score_active_graph(0, history)
        beam.append((cost, history, [int(action)], [cost]))
    beam.sort(key=lambda item: item[0])
    beam = beam[: int(beam_width)]

    for row in range(1, int(steps.size)):
        row_candidates = _candidate_actions_for_row(
            row=row,
            stage_cost=stage_cost,
            top_k=int(top_k),
            include_schedules=include_schedules,
        )
        next_beam: list[tuple[float, tuple[int, ...], list[int], list[float]]] = []
        for total_cost, history, path, row_costs in beam:
            prev_action = int(path[-1])
            candidates = np.asarray(
                sorted(set(int(x) for x in row_candidates.tolist() + [prev_action])),
                dtype=np.int32,
            )
            for action in candidates:
                new_edges = float(transition_counts[prev_action, int(action)])
                next_history = tuple((list(history) + [int(action)])[-history_len:])
                graph_cost = score_active_graph(row, next_history)
                transition_cost = float(setup_penalty) * (new_edges / max_transition)
                if critical_transition_counts is not None and float(critical_setup_penalty) != 0.0:
                    critical_new_edges = float(critical_transition_counts[prev_action, int(action)])
                    transition_cost += float(critical_setup_penalty) * (
                        critical_new_edges / critical_max_transition
                    )
                if float(burst_penalty) != 0.0:
                    transition_cost += float(burst_penalty) * (new_edges / max_transition) ** 2
                next_cost = float(total_cost) + float(graph_cost) + transition_cost
                next_beam.append((next_cost, next_history, path + [int(action)], row_costs + [float(graph_cost)]))
        next_beam.sort(key=lambda item: item[0])
        beam = next_beam[: int(beam_width)]
        if (row + 1) % 100 == 0 or row + 1 == int(steps.size):
            print(
                f"[topology-learning] lst-aware beam row={row + 1}/{steps.size} "
                f"best_mean_cost={beam[0][0] / float(row + 1):.6f} cache={len(eval_cache)}",
                flush=True,
            )

    best_total, _history, best_path, best_row_costs = beam[0]
    selected = np.asarray(best_path, dtype=np.int32)
    setup_counts = np.zeros(selected.size, dtype=np.int32)
    if selected.size > 1:
        setup_counts[1:] = transition_counts[selected[:-1], selected[1:]].astype(np.int32, copy=False)

    schedule_name = (
        f"lst_aware_beam_k{int(top_k)}_b{int(beam_width)}_"
        f"sp{float(setup_penalty):g}_cp{float(critical_setup_penalty):g}_"
        f"bp{float(burst_penalty):g}_rows{int(steps.size)}"
    )
    rows = []
    for row, action in enumerate(selected):
        rows.append(
            {
                "step": int(steps[row]),
                "action_idx": int(action),
                "topology": str(topology_names[int(action)]),
                "lst_aware_stage_cost": float(best_row_costs[row]),
                "setup_commands": int(setup_counts[row]),
            }
        )
    with (out_dir / f"{schedule_name}_by_step.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    arrays_payload = {}
    for key in arrays.files:
        value = arrays[key]
        if value.ndim >= 1 and int(value.shape[0]) == original_num_steps:
            arrays_payload[key] = value[row_slice]
        else:
            arrays_payload[key] = value
    arrays_payload[f"schedule_{schedule_name}"] = selected
    np.savez_compressed(out_dir / "selector_arrays.npz", **arrays_payload)
    shutil.copy2(selector_dir / "selector_config.yaml", out_dir / "selector_config.yaml")
    _write_selector_summary(out_dir / "summary.csv", arrays_payload)
    if (selector_dir / "summary.csv").exists():
        shutil.copy2(selector_dir / "summary.csv", out_dir / "source_selector_summary.csv")

    meta = {
        "source_selector_dir": str(selector_dir),
        "schedule_name": schedule_name,
        "setup_time_seconds": float(setup_time_seconds),
        "start_row": int(start_row),
        "start_step": int(steps[0]),
        "end_step": int(steps[-1]),
        "stride_seconds": int(stride),
        "lag_steps": int(lag_steps),
        "top_k": int(top_k),
        "beam_width": int(beam_width),
        "setup_penalty": float(setup_penalty),
        "critical_setup_penalty": float(critical_setup_penalty),
        "critical_edges_csv": None if critical_edges_csv is None else str(critical_edges_csv),
        "critical_weights_npy": None if critical_weights_npy is None else str(critical_weights_npy),
        "critical_normalize_weights": str(critical_normalize_weights),
        "critical_weight_power": float(critical_weight_power),
        "critical_weight_scale": float(critical_weight_scale),
        "critical_base_new_edge_cost": float(critical_base_new_edge_cost),
        "critical_missing_edge_weight": float(critical_missing_edge_weight),
        "burst_penalty": float(burst_penalty),
        "num_steps": int(steps.size),
        "num_actions": int(len(topology_names)),
        "num_union_edges": int(union_table.num_edges),
        "eval_cache_entries": int(len(eval_cache)),
        "mean_lst_aware_stage_cost": float(np.mean(best_row_costs)),
        "total_lst_aware_stage_cost": float(best_total),
        "total_setup_commands": int(np.sum(setup_counts, dtype=np.int64)),
        "max_setup_commands_per_step": int(np.max(setup_counts)) if setup_counts.size else 0,
        "num_switches": int(np.count_nonzero(selected[1:] != selected[:-1])) if selected.size > 1 else 0,
        "pairs": [
            {
                "source_group_id": int(pair.source_group_id),
                "target_group_id": int(pair.target_group_id),
                "metric_prefix": str(pair.metric_prefix),
                "label": str(pair.label),
            }
            for pair in pair_specs
        ],
        "outputs": {
            "selector_arrays": str(out_dir / "selector_arrays.npz"),
            "by_step_csv": str(out_dir / f"{schedule_name}_by_step.csv"),
            "meta_json": str(out_dir / "lst_aware_beam_meta.json"),
        },
    }
    (out_dir / "lst_aware_beam_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return LstAwareBeamResult(
        selector_dir=out_dir,
        schedule_name=schedule_name,
        selected_action=selected,
        total_setup_commands=int(meta["total_setup_commands"]),
        max_setup_commands_per_step=int(meta["max_setup_commands_per_step"]),
        mean_stage_cost=float(meta["mean_lst_aware_stage_cost"]),
    )
