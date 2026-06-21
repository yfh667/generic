from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.topology_workflow.module.edge_tables import build_full_option_plus_intra_edge_table, build_motif_text_edge_table

from .config_io import load_yaml_dict, optional_path
from .datasets import WideMetricTable, align_metric_tables, common_topology_names, read_wide_metric_csv


@dataclass(frozen=True)
class FullLinkGapMetric:
    name: str
    path: Path
    reference_topology: str = "full_link"
    weight: float = 1.0
    gap: str = "relative_abs"
    epsilon: float = 1e-9


@dataclass(frozen=True)
class FullLinkGapSelectionResult:
    steps: np.ndarray
    topology_names: tuple[str, ...]
    stage_cost: np.ndarray
    transition_cost: np.ndarray
    schedules: dict[str, np.ndarray]
    schedule_costs: dict[str, np.ndarray]
    metric_values: dict[str, np.ndarray]
    metric_references: dict[str, np.ndarray]
    metric_gaps: dict[str, np.ndarray]
    meta: dict[str, Any]


def _viewer_config_from_raw(raw: Mapping[str, Any]) -> ViewerConfig:
    if "constellation" in raw:
        raw = raw["constellation"]
    if not isinstance(raw, Mapping):
        raise ValueError("constellation must be a mapping")
    station_groups_raw = raw.get("station_groups", {}) or {}
    station_groups = {
        int(group_id): {
            "name": str(info.get("name", f"Group {int(group_id)}")),
            "stations": [int(x) for x in (info.get("stations", []) or [])],
        }
        for group_id, info in station_groups_raw.items()
    }
    return ViewerConfig(
        name=str(raw["name"]),
        P=int(raw["p"]),
        N=int(raw["n"]),
        station_groups=station_groups,
        group_colors=[str(x) for x in (raw.get("group_colors", []) or [])],
    )


def _metric_specs_from_raw(raw_metrics: Any) -> list[FullLinkGapMetric]:
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ValueError("metrics must be a non-empty list")
    specs: list[FullLinkGapMetric] = []
    for item in raw_metrics:
        if not isinstance(item, Mapping):
            raise ValueError("each metric item must be a mapping")
        enabled = bool(item.get("enabled", True))
        if not enabled:
            continue
        path = optional_path(item.get("path"))
        if path is None:
            continue
        specs.append(
            FullLinkGapMetric(
                name=str(item["name"]),
                path=path,
                reference_topology=str(item.get("reference_topology", "full_link")),
                weight=float(item.get("weight", 1.0)),
                gap=str(item.get("gap", "relative_abs")),
                epsilon=float(item.get("epsilon", 1e-9)),
            )
        )
    if not specs:
        raise ValueError("no enabled metrics with paths were provided")
    return specs


def _gap_to_reference(values: np.ndarray, reference: np.ndarray, *, mode: str, epsilon: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    mode = str(mode or "relative_abs").lower()
    denom = np.maximum(np.abs(reference), float(epsilon))
    if mode == "relative_abs":
        return np.abs(values - reference[:, None]) / denom[:, None]
    if mode == "relative_positive":
        return np.maximum(values - reference[:, None], 0.0) / denom[:, None]
    if mode == "absolute":
        return np.abs(values - reference[:, None])
    if mode == "signed":
        return values - reference[:, None]
    raise ValueError(f"unsupported full-link gap mode: {mode!r}")


def _read_reference_metric(
    spec: FullLinkGapMetric,
    *,
    step_column: str,
    max_rows: int | None,
) -> tuple[WideMetricTable, np.ndarray]:
    table = read_wide_metric_csv(
        spec.path,
        name=spec.name,
        step_column=step_column,
        topology_columns=None,
        max_rows=max_rows,
    )
    try:
        ref_idx = table.topology_names.index(str(spec.reference_topology))
    except ValueError as exc:
        raise ValueError(f"reference topology {spec.reference_topology!r} not found in {spec.path}") from exc
    reference = np.asarray(table.values[:, ref_idx], dtype=np.float64)
    keep_names = tuple(name for name in table.topology_names if name != str(spec.reference_topology))
    keep_idx = [idx for idx, name in enumerate(table.topology_names) if name != str(spec.reference_topology)]
    return (
        WideMetricTable(
            name=table.name,
            path=table.path,
            steps=table.steps,
            topology_names=keep_names,
            values=np.asarray(table.values[:, keep_idx], dtype=np.float64),
        ),
        reference,
    )


def _edge_key(src: int, dst: int) -> tuple[int, int]:
    return (int(src), int(dst)) if int(src) <= int(dst) else (int(dst), int(src))


def _edge_keys_from_table(edge_table) -> set[tuple[int, int]]:
    return {_edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])) for idx in range(int(edge_table.num_edges))}


def _read_topology_library(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path in (None, "", False):
        return {}
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {str(row.get("name", "")): dict(row) for row in rows if row.get("name")}


def _gridplus_edge_keys(config: ViewerConfig) -> set[tuple[int, int]]:
    return _edge_keys_from_table(build_motif_text_edge_table(motif_text="A", config=config, add_intra_ring=True))


def edge_key_sets_for_topologies(
    *,
    topology_names: Sequence[str],
    config: ViewerConfig,
    topology_library_csv: str | Path | None,
) -> dict[str, set[tuple[int, int]]]:
    library = _read_topology_library(topology_library_csv)
    out: dict[str, set[tuple[int, int]]] = {}
    for name in topology_names:
        name = str(name)
        if name == "gridplus":
            out[name] = _gridplus_edge_keys(config)
            continue
        row = library.get(name)
        if row is None:
            raise KeyError(
                f"topology {name!r} not found in topology library and is not a known baseline. "
                "Provide topology_library_csv or exclude this topology."
            )
        motif = str(row.get("motif", ""))
        if not motif:
            raise ValueError(f"topology {name!r} has no motif text in {topology_library_csv}")
        out[name] = _edge_keys_from_table(build_motif_text_edge_table(motif_text=motif, config=config, add_intra_ring=True))
    return out


def compute_added_edge_transition_cost(
    *,
    topology_names: Sequence[str],
    config: ViewerConfig,
    topology_library_csv: str | Path | None,
    normalize: str = "max",
) -> np.ndarray:
    """Transition cost = number of new edges in next topology not present before."""

    transition = compute_added_edge_transition_counts(
        topology_names=topology_names,
        config=config,
        topology_library_csv=topology_library_csv,
    )
    return normalize_transition_cost(transition, normalize=normalize)


def compute_added_edge_transition_counts(
    *,
    topology_names: Sequence[str],
    config: ViewerConfig,
    topology_library_csv: str | Path | None,
) -> np.ndarray:
    """Transition counts = number of new edges in next topology not present before."""

    names = tuple(str(x) for x in topology_names)
    key_sets = edge_key_sets_for_topologies(
        topology_names=names,
        config=config,
        topology_library_csv=topology_library_csv,
    )
    all_keys = sorted(set().union(*(key_sets[name] for name in names)))
    key_to_col = {key: idx for idx, key in enumerate(all_keys)}
    masks = np.zeros((len(names), len(all_keys)), dtype=bool)
    for row, name in enumerate(names):
        for key in key_sets[name]:
            masks[row, key_to_col[key]] = True

    transition = np.empty((len(names), len(names)), dtype=np.float32)
    for next_idx in range(len(names)):
        next_mask = masks[next_idx]
        transition[:, next_idx] = np.count_nonzero(next_mask[None, :] & ~masks, axis=1)
    return transition.astype(np.float32)


def compute_overlap_loss_transition_counts(
    *,
    topology_names: Sequence[str],
    config: ViewerConfig,
    topology_library_csv: str | Path | None,
) -> np.ndarray:
    """Transition counts = edges not shared by the previous and next topology.

    Under an edge-level LST model, the active graph just after a switch is close
    to the intersection of the previous and next topology. The symmetric
    difference is therefore a simple proxy for how much graph structure is lost
    while new links are still building.
    """

    names = tuple(str(x) for x in topology_names)
    key_sets = edge_key_sets_for_topologies(
        topology_names=names,
        config=config,
        topology_library_csv=topology_library_csv,
    )
    all_keys = sorted(set().union(*(key_sets[name] for name in names)))
    key_to_col = {key: idx for idx, key in enumerate(all_keys)}
    masks = np.zeros((len(names), len(all_keys)), dtype=bool)
    for row, name in enumerate(names):
        for key in key_sets[name]:
            masks[row, key_to_col[key]] = True

    transition = np.empty((len(names), len(names)), dtype=np.float32)
    for next_idx in range(len(names)):
        next_mask = masks[next_idx]
        transition[:, next_idx] = np.count_nonzero(np.logical_xor(masks, next_mask[None, :]), axis=1)
    return transition.astype(np.float32)


def read_edge_weights_by_key(
    *,
    edges_csv: str | Path,
    weights_npy: str | Path,
    normalize_weights: str = "max",
    weight_power: float = 1.0,
    weight_scale: float = 1.0,
    base_new_edge_cost: float = 0.0,
) -> dict[tuple[int, int], float]:
    """Read per-edge criticality weights aligned to an ``edges.csv`` file.

    The intended source is a topology-metrics usage-share vector such as
    ``combined_usage_share_max_over_time.npy``. The return value maps an
    undirected edge key to the cost contribution used when that edge is newly
    requested by a transition.
    """

    weights = np.asarray(np.load(Path(weights_npy), allow_pickle=False), dtype=np.float64).reshape(-1)
    mode = str(normalize_weights or "max").lower()
    if mode == "max":
        denom = float(np.nanmax(weights)) if weights.size else 0.0
        if denom > 0.0 and math.isfinite(denom):
            weights = weights / denom
    elif mode == "sum":
        denom = float(np.nansum(weights)) if weights.size else 0.0
        if denom > 0.0 and math.isfinite(denom):
            weights = weights / denom
    elif mode == "none":
        pass
    else:
        raise ValueError(f"unsupported critical edge weight normalization: {normalize_weights!r}")
    if float(weight_power) != 1.0:
        weights = np.power(np.maximum(weights, 0.0), float(weight_power))
    weights = float(base_new_edge_cost) + float(weight_scale) * weights

    out: dict[tuple[int, int], float] = {}
    with Path(edges_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            edge_idx = int(row["edge_idx"])
            if edge_idx < 0 or edge_idx >= weights.shape[0]:
                raise ValueError(f"edge_idx={edge_idx} outside weights length {weights.shape[0]}")
            key = _edge_key(int(row["src_node"]), int(row["dst_node"]))
            out[key] = float(weights[edge_idx])
    return out


def compute_weighted_added_edge_transition_counts(
    *,
    topology_names: Sequence[str],
    config: ViewerConfig,
    topology_library_csv: str | Path | None,
    edge_weight_by_key: Mapping[tuple[int, int], float],
    missing_edge_weight: float = 0.0,
) -> np.ndarray:
    """Transition cost = sum of weights of new edges in the next topology."""

    names = tuple(str(x) for x in topology_names)
    key_sets = edge_key_sets_for_topologies(
        topology_names=names,
        config=config,
        topology_library_csv=topology_library_csv,
    )
    all_keys = sorted(set().union(*(key_sets[name] for name in names)))
    key_to_col = {key: idx for idx, key in enumerate(all_keys)}
    weights = np.asarray(
        [float(edge_weight_by_key.get(key, float(missing_edge_weight))) for key in all_keys],
        dtype=np.float32,
    )
    masks = np.zeros((len(names), len(all_keys)), dtype=bool)
    for row, name in enumerate(names):
        for key in key_sets[name]:
            masks[row, key_to_col[key]] = True

    transition = np.empty((len(names), len(names)), dtype=np.float32)
    for next_idx in range(len(names)):
        next_mask = masks[next_idx]
        added = next_mask[None, :] & ~masks
        transition[:, next_idx] = added.astype(np.float32) @ weights
    return transition.astype(np.float32)


def normalize_transition_cost(transition: np.ndarray, *, normalize: str = "max") -> np.ndarray:
    transition = np.asarray(transition, dtype=np.float32)
    mode = str(normalize or "max").lower()
    if mode == "none":
        return transition.astype(np.float32, copy=False)
    if mode == "max":
        denom = float(np.max(transition))
    else:
        raise ValueError(f"unsupported transition normalization: {normalize!r}")
    if denom > 0:
        transition = transition / denom
    return transition.astype(np.float32)


def burst_transition_cost_from_counts(
    transition_counts: np.ndarray,
    *,
    power: float = 2.0,
    threshold: float = 0.0,
    normalize: str = "max",
) -> np.ndarray:
    counts = np.asarray(transition_counts, dtype=np.float32)
    values = np.maximum(counts - float(threshold), 0.0)
    if float(power) != 1.0:
        values = np.power(values, float(power), dtype=np.float32)
    return normalize_transition_cost(values, normalize=normalize)


def hard_cap_transition_cost_from_counts(
    transition_counts: np.ndarray,
    *,
    max_new_edges: int | float,
) -> np.ndarray:
    """Return a 0/inf transition matrix enforcing a new-edge cap."""

    counts = np.asarray(transition_counts, dtype=np.float32)
    cap = float(max_new_edges)
    if cap < 0:
        raise ValueError("max_new_edges must be >= 0")
    return np.where(counts <= cap, 0.0, math.inf).astype(np.float32)


def _safe_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(text)).strip("_") or "profile"


def _critical_edge_profiles_from_raw(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    profiles = raw.get("critical_edge_transition_profiles", []) or []
    singular = raw.get("critical_edge_transition")
    out: list[Mapping[str, Any]] = []
    if isinstance(singular, Mapping) and bool(singular.get("enabled", True)):
        out.append(singular)
    if isinstance(profiles, list):
        out.extend(item for item in profiles if isinstance(item, Mapping) and bool(item.get("enabled", True)))
    elif isinstance(profiles, Mapping) and bool(profiles.get("enabled", True)):
        out.append(profiles)
    return out


def build_full_link_gap_stage_cost(
    *,
    metric_specs: Sequence[FullLinkGapMetric],
    candidate_topologies: Sequence[str] | None = None,
    exclude_topologies: Sequence[str] = ("full_link",),
    step_column: str = "step",
    max_rows: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw_tables: list[WideMetricTable] = []
    references_by_name: dict[str, np.ndarray] = {}
    for spec in metric_specs:
        table, reference = _read_reference_metric(spec, step_column=step_column, max_rows=max_rows)
        raw_tables.append(table)
        references_by_name[spec.name] = reference

    if candidate_topologies is None:
        names = common_topology_names(raw_tables)
    else:
        names = tuple(str(x) for x in candidate_topologies)
    excluded = set(str(x) for x in exclude_topologies)
    names = tuple(name for name in names if name not in excluded)
    if not names:
        raise ValueError("candidate topology list is empty after exclusions")

    tables = align_metric_tables(raw_tables, topology_names=names)
    steps = np.asarray(tables[0].steps, dtype=np.int64)
    stage_cost = np.zeros_like(tables[0].values, dtype=np.float64)
    metric_values: dict[str, np.ndarray] = {}
    metric_references: dict[str, np.ndarray] = {}
    metric_gaps: dict[str, np.ndarray] = {}
    for spec, table in zip(metric_specs, tables):
        reference = np.asarray(references_by_name[spec.name], dtype=np.float64)
        if reference.shape[0] != steps.shape[0]:
            raise ValueError(f"reference metric length mismatch for {spec.name}")
        gap = _gap_to_reference(table.values, reference, mode=spec.gap, epsilon=float(spec.epsilon))
        stage_cost += float(spec.weight) * gap
        metric_values[spec.name] = np.asarray(table.values, dtype=np.float64)
        metric_references[spec.name] = reference
        metric_gaps[spec.name] = gap

    invalid = ~np.isfinite(stage_cost)
    stage_cost[invalid] = math.inf
    return stage_cost, names, steps, metric_values, metric_references, metric_gaps


def run_switch_penalty_dp(
    *,
    stage_cost: np.ndarray,
    transition_cost: np.ndarray,
    switch_penalty: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    stage = np.asarray(stage_cost, dtype=np.float32)
    transition = np.asarray(transition_cost, dtype=np.float32)
    if stage.ndim != 2:
        raise ValueError("stage_cost must be 2D")
    n_steps, n_actions = stage.shape
    if transition.shape != (n_actions, n_actions):
        raise ValueError("transition_cost must have shape (num_actions, num_actions)")
    if float(switch_penalty) == 0.0:
        selected = np.argmin(stage, axis=1).astype(np.int32)
        chosen_cost = stage[np.arange(n_steps), selected].astype(np.float64)
        return selected, chosen_cost, float(np.sum(chosen_cost))

    prev = np.full((n_steps, n_actions), -1, dtype=np.int32)
    prev_cost = stage[0, :].astype(np.float32, copy=True)
    trans = (float(switch_penalty) * transition).astype(np.float32, copy=False)
    actions = np.arange(n_actions)
    for row in range(1, n_steps):
        candidates = prev_cost[:, None] + trans
        best_prev = np.argmin(candidates, axis=0).astype(np.int32, copy=False)
        best_cost = candidates[best_prev, actions]
        finite_stage = np.isfinite(stage[row, :])
        current_cost = np.full(n_actions, math.inf, dtype=np.float32)
        current_cost[finite_stage] = stage[row, finite_stage] + best_cost[finite_stage]
        prev[row, finite_stage] = best_prev[finite_stage]
        prev_cost = current_cost

    last = int(np.argmin(prev_cost))
    if not math.isfinite(float(prev_cost[last])):
        raise RuntimeError("no finite dynamic schedule found")
    selected = np.empty(n_steps, dtype=np.int32)
    selected[-1] = last
    for row in range(n_steps - 1, 0, -1):
        selected[row - 1] = int(prev[row, int(selected[row])])
    chosen_cost = stage[np.arange(n_steps), selected].astype(np.float64)
    return selected, chosen_cost, float(prev_cost[last])


def run_min_dwell_switch_penalty_dp(
    *,
    stage_cost: np.ndarray,
    transition_cost: np.ndarray,
    switch_penalty: float,
    min_dwell_steps: int,
    require_final_dwell: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Dynamic programming with a minimum dwell time per selected topology.

    A transition is allowed only after the previous action has been held for at
    least ``min_dwell_steps`` sampled rows. The transition cost still uses the
    supplied matrix, typically the newly requested edge count/cost.
    """

    dwell = max(1, int(min_dwell_steps))
    if dwell <= 1:
        return run_switch_penalty_dp(
            stage_cost=stage_cost,
            transition_cost=transition_cost,
            switch_penalty=float(switch_penalty),
        )

    stage = np.asarray(stage_cost, dtype=np.float32)
    transition = np.asarray(transition_cost, dtype=np.float32)
    if stage.ndim != 2:
        raise ValueError("stage_cost must be 2D")
    n_steps, n_actions = stage.shape
    if transition.shape != (n_actions, n_actions):
        raise ValueError("transition_cost must have shape (num_actions, num_actions)")
    if dwell > n_steps:
        raise ValueError(f"min_dwell_steps={dwell} exceeds number of steps={n_steps}")

    trans = (float(switch_penalty) * transition).astype(np.float32, copy=True)
    np.fill_diagonal(trans, math.inf)
    actions = np.arange(n_actions)
    costs = np.full((dwell, n_actions), math.inf, dtype=np.float32)
    costs[0, :] = stage[0, :]

    prev_action = np.full((n_steps, dwell, n_actions), -1, dtype=np.int32)
    prev_age = np.full((n_steps, dwell, n_actions), -1, dtype=np.int16)

    for row in range(1, n_steps):
        next_costs = np.full((dwell, n_actions), math.inf, dtype=np.float32)

        for age in range(dwell):
            next_age = min(dwell - 1, age + 1)
            candidate = costs[age, :] + stage[row, :]
            mask = candidate < next_costs[next_age, :]
            if np.any(mask):
                next_costs[next_age, mask] = candidate[mask]
                prev_action[row, next_age, mask] = actions[mask]
                prev_age[row, next_age, mask] = age

        mature_cost = costs[dwell - 1, :]
        if np.any(np.isfinite(mature_cost)):
            candidates = mature_cost[:, None] + trans
            best_prev = np.argmin(candidates, axis=0).astype(np.int32, copy=False)
            best_transition = candidates[best_prev, actions]
            candidate = best_transition + stage[row, :]
            mask = candidate < next_costs[0, :]
            if np.any(mask):
                next_costs[0, mask] = candidate[mask]
                prev_action[row, 0, mask] = best_prev[mask]
                prev_age[row, 0, mask] = dwell - 1

        costs = next_costs

    final_costs = costs[dwell - 1, :] if bool(require_final_dwell) else costs.reshape(-1)
    final_idx = int(np.argmin(final_costs))
    if not math.isfinite(float(final_costs[final_idx])):
        raise RuntimeError("no finite min-dwell schedule found")
    if bool(require_final_dwell):
        age = dwell - 1
        action = final_idx
    else:
        age = int(final_idx // n_actions)
        action = int(final_idx % n_actions)

    selected = np.empty(n_steps, dtype=np.int32)
    selected[-1] = int(action)
    for row in range(n_steps - 1, 0, -1):
        pa = int(prev_action[row, int(age), int(action)])
        pg = int(prev_age[row, int(age), int(action)])
        if pa < 0 or pg < 0:
            raise RuntimeError("min-dwell backtrack failed")
        selected[row - 1] = pa
        action = pa
        age = pg

    chosen_cost = stage[np.arange(n_steps), selected].astype(np.float64)
    return selected, chosen_cost, float(final_costs[final_idx])


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _schedule_rows(
    *,
    steps: np.ndarray,
    topology_names: Sequence[str],
    selected: np.ndarray,
    stage_cost: np.ndarray,
    transition_cost: np.ndarray,
    metric_values: Mapping[str, np.ndarray],
    metric_references: Mapping[str, np.ndarray],
    metric_gaps: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_idx, step in enumerate(steps):
        action = int(selected[row_idx])
        prev_action = int(selected[row_idx - 1]) if row_idx > 0 else action
        item: dict[str, Any] = {
            "step": int(step),
            "action_idx": action,
            "topology": str(topology_names[action]),
            "stage_cost": float(stage_cost[row_idx, action]),
            "new_edge_transition_cost": float(transition_cost[prev_action, action]) if row_idx > 0 else 0.0,
        }
        for metric_name, values in metric_values.items():
            item[f"{metric_name}_value"] = float(values[row_idx, action])
            item[f"{metric_name}_full_link"] = float(metric_references[metric_name][row_idx])
            item[f"{metric_name}_gap"] = float(metric_gaps[metric_name][row_idx, action])
        rows.append(item)
    return rows


def _summary_row(
    *,
    name: str,
    selected: np.ndarray,
    stage_cost: np.ndarray,
    transition_cost: np.ndarray,
    metric_gaps: Mapping[str, np.ndarray],
    metric_values: Mapping[str, np.ndarray],
    metric_references: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    idx = np.asarray(selected, dtype=np.int32)
    rows = np.arange(idx.size)
    transitions = np.asarray([float(transition_cost[int(idx[t - 1]), int(idx[t])]) for t in range(1, idx.size)], dtype=np.float64)
    item: dict[str, Any] = {
        "schedule": str(name),
        "mean_stage_cost": float(np.mean(stage_cost[rows, idx])),
        "total_stage_cost": float(np.sum(stage_cost[rows, idx])),
        "num_switches": int(np.count_nonzero(idx[1:] != idx[:-1])) if idx.size > 1 else 0,
        "sum_new_edge_transition_cost": float(np.sum(transitions)) if transitions.size else 0.0,
        "mean_new_edge_transition_cost": float(np.mean(transitions)) if transitions.size else 0.0,
        "num_unique_topologies": int(len(np.unique(idx))),
    }
    for metric_name, gaps in metric_gaps.items():
        chosen_gaps = gaps[rows, idx]
        chosen_values = metric_values[metric_name][rows, idx]
        ref = metric_references[metric_name]
        item[f"mean_{metric_name}_gap"] = float(np.mean(chosen_gaps))
        item[f"max_{metric_name}_gap"] = float(np.max(chosen_gaps))
        item[f"mean_{metric_name}_value"] = float(np.mean(chosen_values))
        item[f"mean_{metric_name}_full_link"] = float(np.mean(ref))
        ref_mean = float(np.mean(ref))
        item[f"mean_{metric_name}_global_gap"] = (
            float((np.mean(chosen_values) - ref_mean) / ref_mean)
            if abs(ref_mean) > 1e-12
            else math.nan
        )
    global_delay_gaps = [
        float(value)
        for key, value in item.items()
        if key.startswith("mean_") and key.endswith("_global_gap") and "_delay" in key and math.isfinite(float(value))
    ]
    global_hop_gaps = [
        float(value)
        for key, value in item.items()
        if key.startswith("mean_") and key.endswith("_global_gap") and "_hops" in key and math.isfinite(float(value))
    ]
    if global_delay_gaps:
        item["mean_delay_global_gap"] = float(np.mean(global_delay_gaps))
    if global_hop_gaps:
        item["mean_hop_global_gap"] = float(np.mean(global_hop_gaps))
    if global_delay_gaps and global_hop_gaps:
        item["mean_all_global_gap"] = float((item["mean_delay_global_gap"] + item["mean_hop_global_gap"]) / 2.0)
    return item


def _plot_schedules(
    path: Path,
    *,
    steps: np.ndarray,
    schedules: Mapping[str, np.ndarray],
    topology_names: Sequence[str],
    metric_values: Mapping[str, np.ndarray],
    metric_references: Mapping[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = list(metric_values.keys())
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(15, 4.5 * len(metric_names)), dpi=170, sharex=True)
    if len(metric_names) == 1:
        axes = [axes]
    x = np.asarray(steps, dtype=np.float64) / 3600.0
    for ax, metric_name in zip(axes, metric_names):
        ax.plot(x, metric_references[metric_name], color="black", linewidth=1.3, label=f"full_link {metric_name}")
        values = metric_values[metric_name]
        for schedule_name, selected in schedules.items():
            y = values[np.arange(len(selected)), selected]
            ax.plot(x, y, linewidth=0.9, label=schedule_name)
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time (hour)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_full_link_gap_selector_from_yaml(
    config_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    max_rows: int | None = None,
) -> Path:
    config_path = Path(config_path)
    raw = load_yaml_dict(config_path)
    config = _viewer_config_from_raw(raw)
    metric_specs = _metric_specs_from_raw(raw.get("metrics"))

    selector_raw = raw.get("selector", {})
    if not isinstance(selector_raw, Mapping):
        raise ValueError("selector must be a mapping when present")
    outputs_raw = raw.get("outputs", {})
    if not isinstance(outputs_raw, Mapping):
        raise ValueError("outputs must be a mapping when present")
    inputs_raw = raw.get("inputs", {})
    if not isinstance(inputs_raw, Mapping):
        raise ValueError("inputs must be a mapping when present")

    exclude = tuple(str(x) for x in (selector_raw.get("exclude_topologies", ["full_link"]) or []))
    stage_cost, topology_names, steps, metric_values, metric_refs, metric_gaps = build_full_link_gap_stage_cost(
        metric_specs=metric_specs,
        exclude_topologies=exclude,
        step_column=str(selector_raw.get("step_column", "step")),
        max_rows=max_rows,
    )

    topology_library_csv = optional_path(inputs_raw.get("topology_library_csv"))
    transition_counts = compute_added_edge_transition_counts(
        topology_names=topology_names,
        config=config,
        topology_library_csv=topology_library_csv,
    )
    transition_cost = normalize_transition_cost(
        transition_counts,
        normalize=str(selector_raw.get("transition_normalize", "max")),
    )
    burst_penalties = [float(x) for x in (selector_raw.get("burst_penalties", []) or [])]
    burst_power = float(selector_raw.get("burst_power", 2.0))
    burst_threshold = float(selector_raw.get("burst_threshold", 0.0))
    overlap_penalties = [float(x) for x in (selector_raw.get("overlap_penalties", []) or [])]
    transition_caps = [float(x) for x in (selector_raw.get("transition_caps", []) or [])]
    min_dwell_steps_values = [int(x) for x in (selector_raw.get("min_dwell_steps", []) or [])]
    min_dwell_penalties = [float(x) for x in (selector_raw.get("min_dwell_switch_penalties", []) or [])]
    min_dwell_burst_penalties = [float(x) for x in (selector_raw.get("min_dwell_burst_penalties", []) or [])]
    min_dwell_overlap_penalties = [float(x) for x in (selector_raw.get("min_dwell_overlap_penalties", []) or [])]
    critical_transition_profiles: list[dict[str, Any]] = []
    burst_transition_cost = None
    if burst_penalties:
        burst_transition_cost = burst_transition_cost_from_counts(
            transition_counts,
            power=burst_power,
            threshold=burst_threshold,
            normalize=str(selector_raw.get("burst_transition_normalize", selector_raw.get("transition_normalize", "max"))),
        )
    overlap_transition_counts = None
    overlap_transition_cost = None
    if overlap_penalties or min_dwell_overlap_penalties:
        overlap_transition_counts = compute_overlap_loss_transition_counts(
            topology_names=topology_names,
            config=config,
            topology_library_csv=topology_library_csv,
        )
        overlap_transition_cost = normalize_transition_cost(
            overlap_transition_counts,
            normalize=str(selector_raw.get("overlap_transition_normalize", selector_raw.get("transition_normalize", "max"))),
        )
    for profile_raw in _critical_edge_profiles_from_raw(selector_raw):
        name = _safe_token(str(profile_raw.get("name", f"critical{len(critical_transition_profiles) + 1}")))
        edges_csv = optional_path(profile_raw.get("edges_csv"))
        weights_npy = optional_path(profile_raw.get("weights_npy"))
        if edges_csv is None or weights_npy is None:
            raise ValueError(f"critical edge transition profile {name!r} requires edges_csv and weights_npy")
        penalties_for_profile = [float(x) for x in (profile_raw.get("penalties", []) or [])]
        if not penalties_for_profile:
            continue
        weights_by_key = read_edge_weights_by_key(
            edges_csv=edges_csv,
            weights_npy=weights_npy,
            normalize_weights=str(profile_raw.get("normalize_weights", "max")),
            weight_power=float(profile_raw.get("weight_power", 1.0)),
            weight_scale=float(profile_raw.get("weight_scale", 1.0)),
            base_new_edge_cost=float(profile_raw.get("base_new_edge_cost", 0.0)),
        )
        critical_counts = compute_weighted_added_edge_transition_counts(
            topology_names=topology_names,
            config=config,
            topology_library_csv=topology_library_csv,
            edge_weight_by_key=weights_by_key,
            missing_edge_weight=float(profile_raw.get("missing_edge_weight", 0.0)),
        )
        critical_cost = normalize_transition_cost(
            critical_counts,
            normalize=str(profile_raw.get("transition_normalize", selector_raw.get("transition_normalize", "max"))),
        )
        critical_transition_profiles.append(
            {
                "name": name,
                "penalties": penalties_for_profile,
                "transition_counts": critical_counts,
                "transition_cost": critical_cost,
                "edges_csv": str(edges_csv),
                "weights_npy": str(weights_npy),
                "normalize_weights": str(profile_raw.get("normalize_weights", "max")),
                "weight_power": float(profile_raw.get("weight_power", 1.0)),
                "weight_scale": float(profile_raw.get("weight_scale", 1.0)),
                "base_new_edge_cost": float(profile_raw.get("base_new_edge_cost", 0.0)),
                "missing_edge_weight": float(profile_raw.get("missing_edge_weight", 0.0)),
            }
        )

    penalties = [float(x) for x in (selector_raw.get("switch_penalties", [0.0]) or [0.0])]
    schedules: dict[str, np.ndarray] = {}
    schedule_costs: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []

    greedy = np.argmin(stage_cost, axis=1).astype(np.int32)
    schedules["greedy_no_switch_penalty"] = greedy
    schedule_costs["greedy_no_switch_penalty"] = stage_cost[np.arange(stage_cost.shape[0]), greedy]
    summary_rows.append(
        _summary_row(
            name="greedy_no_switch_penalty",
            selected=greedy,
            stage_cost=stage_cost,
            transition_cost=transition_cost,
            metric_gaps=metric_gaps,
            metric_values=metric_values,
            metric_references=metric_refs,
        )
    )

    static_mean = np.mean(stage_cost, axis=0)
    static_idx = int(np.argmin(static_mean))
    static = np.full(stage_cost.shape[0], static_idx, dtype=np.int32)
    schedules["static_best"] = static
    schedule_costs["static_best"] = stage_cost[np.arange(stage_cost.shape[0]), static]
    summary_rows.append(
        _summary_row(
            name="static_best",
            selected=static,
            stage_cost=stage_cost,
            transition_cost=transition_cost,
            metric_gaps=metric_gaps,
            metric_values=metric_values,
            metric_references=metric_refs,
        )
    )

    for penalty in penalties:
        selected, chosen_cost, _total = run_switch_penalty_dp(
            stage_cost=stage_cost,
            transition_cost=transition_cost,
            switch_penalty=float(penalty),
        )
        label = f"dp_new_edge_penalty_{float(penalty):g}"
        schedules[label] = selected
        schedule_costs[label] = chosen_cost
        row = _summary_row(
            name=label,
            selected=selected,
            stage_cost=stage_cost,
            transition_cost=transition_cost,
            metric_gaps=metric_gaps,
            metric_values=metric_values,
            metric_references=metric_refs,
        )
        row["switch_penalty"] = float(penalty)
        row["transition_profile"] = "linear_new_edges"
        summary_rows.append(row)

    if burst_transition_cost is not None:
        for penalty in burst_penalties:
            selected, chosen_cost, _total = run_switch_penalty_dp(
                stage_cost=stage_cost,
                transition_cost=burst_transition_cost,
                switch_penalty=float(penalty),
            )
            label = f"dp_burst{burst_power:g}_penalty_{float(penalty):g}"
            schedules[label] = selected
            schedule_costs[label] = chosen_cost
            row = _summary_row(
                name=label,
                selected=selected,
                stage_cost=stage_cost,
                transition_cost=transition_cost,
                metric_gaps=metric_gaps,
                metric_values=metric_values,
                metric_references=metric_refs,
            )
            row["switch_penalty"] = float(penalty)
            row["transition_profile"] = "burst_new_edges"
            row["burst_power"] = float(burst_power)
            row["burst_threshold"] = float(burst_threshold)
            summary_rows.append(row)

    if overlap_transition_cost is not None:
        for penalty in overlap_penalties:
            selected, chosen_cost, _total = run_switch_penalty_dp(
                stage_cost=stage_cost,
                transition_cost=overlap_transition_cost,
                switch_penalty=float(penalty),
            )
            label = f"dp_overlap_penalty_{float(penalty):g}"
            schedules[label] = selected
            schedule_costs[label] = chosen_cost
            row = _summary_row(
                name=label,
                selected=selected,
                stage_cost=stage_cost,
                transition_cost=transition_cost,
                metric_gaps=metric_gaps,
                metric_values=metric_values,
                metric_references=metric_refs,
            )
            row["switch_penalty"] = float(penalty)
            row["transition_profile"] = "overlap_loss"
            summary_rows.append(row)

    for profile in critical_transition_profiles:
        profile_name = str(profile["name"])
        profile_cost = np.asarray(profile["transition_cost"], dtype=np.float32)
        for penalty in profile["penalties"]:
            selected, chosen_cost, _total = run_switch_penalty_dp(
                stage_cost=stage_cost,
                transition_cost=profile_cost,
                switch_penalty=float(penalty),
            )
            label = f"dp_critical_{profile_name}_penalty_{float(penalty):g}"
            schedules[label] = selected
            schedule_costs[label] = chosen_cost
            row = _summary_row(
                name=label,
                selected=selected,
                stage_cost=stage_cost,
                transition_cost=transition_cost,
                metric_gaps=metric_gaps,
                metric_values=metric_values,
                metric_references=metric_refs,
            )
            row["switch_penalty"] = float(penalty)
            row["transition_profile"] = f"critical_edge_{profile_name}"
            row["critical_edges_csv"] = str(profile["edges_csv"])
            row["critical_weights_npy"] = str(profile["weights_npy"])
            row["critical_base_new_edge_cost"] = float(profile["base_new_edge_cost"])
            row["critical_weight_scale"] = float(profile["weight_scale"])
            row["critical_weight_power"] = float(profile["weight_power"])
            summary_rows.append(row)

    for cap in transition_caps:
        cap_transition_cost = hard_cap_transition_cost_from_counts(transition_counts, max_new_edges=float(cap))
        selected, chosen_cost, _total = run_switch_penalty_dp(
            stage_cost=stage_cost,
            transition_cost=cap_transition_cost,
            switch_penalty=1.0,
        )
        label = f"dp_new_edge_cap_{float(cap):g}"
        schedules[label] = selected
        schedule_costs[label] = chosen_cost
        row = _summary_row(
            name=label,
            selected=selected,
            stage_cost=stage_cost,
            transition_cost=transition_cost,
            metric_gaps=metric_gaps,
            metric_values=metric_values,
            metric_references=metric_refs,
        )
        row["transition_profile"] = "hard_new_edge_cap"
        row["transition_cap_new_edges"] = float(cap)
        summary_rows.append(row)

    for dwell_steps in min_dwell_steps_values:
        for penalty in min_dwell_penalties:
            selected, chosen_cost, _total = run_min_dwell_switch_penalty_dp(
                stage_cost=stage_cost,
                transition_cost=transition_cost,
                switch_penalty=float(penalty),
                min_dwell_steps=int(dwell_steps),
                require_final_dwell=True,
            )
            label = f"dp_dwell{int(dwell_steps)}_new_edge_penalty_{float(penalty):g}"
            schedules[label] = selected
            schedule_costs[label] = chosen_cost
            row = _summary_row(
                name=label,
                selected=selected,
                stage_cost=stage_cost,
                transition_cost=transition_cost,
                metric_gaps=metric_gaps,
                metric_values=metric_values,
                metric_references=metric_refs,
            )
            row["transition_profile"] = "min_dwell_linear_new_edges"
            row["min_dwell_steps"] = int(dwell_steps)
            row["switch_penalty"] = float(penalty)
            summary_rows.append(row)

    if burst_transition_cost is not None:
        for dwell_steps in min_dwell_steps_values:
            for penalty in min_dwell_burst_penalties:
                selected, chosen_cost, _total = run_min_dwell_switch_penalty_dp(
                    stage_cost=stage_cost,
                    transition_cost=burst_transition_cost,
                    switch_penalty=float(penalty),
                    min_dwell_steps=int(dwell_steps),
                    require_final_dwell=True,
                )
                label = f"dp_dwell{int(dwell_steps)}_burst{burst_power:g}_penalty_{float(penalty):g}"
                schedules[label] = selected
                schedule_costs[label] = chosen_cost
                row = _summary_row(
                    name=label,
                    selected=selected,
                    stage_cost=stage_cost,
                    transition_cost=transition_cost,
                    metric_gaps=metric_gaps,
                    metric_values=metric_values,
                    metric_references=metric_refs,
                )
                row["transition_profile"] = "min_dwell_burst_new_edges"
                row["min_dwell_steps"] = int(dwell_steps)
                row["switch_penalty"] = float(penalty)
                row["burst_power"] = float(burst_power)
                row["burst_threshold"] = float(burst_threshold)
                summary_rows.append(row)

    if overlap_transition_cost is not None:
        for dwell_steps in min_dwell_steps_values:
            for penalty in min_dwell_overlap_penalties:
                selected, chosen_cost, _total = run_min_dwell_switch_penalty_dp(
                    stage_cost=stage_cost,
                    transition_cost=overlap_transition_cost,
                    switch_penalty=float(penalty),
                    min_dwell_steps=int(dwell_steps),
                    require_final_dwell=True,
                )
                label = f"dp_dwell{int(dwell_steps)}_overlap_penalty_{float(penalty):g}"
                schedules[label] = selected
                schedule_costs[label] = chosen_cost
                row = _summary_row(
                    name=label,
                    selected=selected,
                    stage_cost=stage_cost,
                    transition_cost=transition_cost,
                    metric_gaps=metric_gaps,
                    metric_values=metric_values,
                    metric_references=metric_refs,
                )
                row["transition_profile"] = "min_dwell_overlap_loss"
                row["min_dwell_steps"] = int(dwell_steps)
                row["switch_penalty"] = float(penalty)
                summary_rows.append(row)

    output_dir = Path(out_dir if out_dir is not None else outputs_raw["out_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "selector_config.yaml")

    np.savez_compressed(
        output_dir / "selector_arrays.npz",
        steps=steps.astype(np.int64),
        topology_names=np.asarray(topology_names, dtype="U128"),
        stage_cost=stage_cost.astype(np.float32),
        transition_counts=transition_counts.astype(np.float32),
        transition_cost=transition_cost.astype(np.float32),
        **({"burst_transition_cost": burst_transition_cost.astype(np.float32)} if burst_transition_cost is not None else {}),
        **({"overlap_transition_counts": overlap_transition_counts.astype(np.float32)} if overlap_transition_counts is not None else {}),
        **({"overlap_transition_cost": overlap_transition_cost.astype(np.float32)} if overlap_transition_cost is not None else {}),
        **{
            f"critical_{str(profile['name'])}_transition_counts": np.asarray(profile["transition_counts"], dtype=np.float32)
            for profile in critical_transition_profiles
        },
        **{
            f"critical_{str(profile['name'])}_transition_cost": np.asarray(profile["transition_cost"], dtype=np.float32)
            for profile in critical_transition_profiles
        },
        **{f"schedule_{name}": selected.astype(np.int32) for name, selected in schedules.items()},
        **{f"metric_{name}": values.astype(np.float32) for name, values in metric_values.items()},
        **{f"metric_{name}_full_link": values.astype(np.float32) for name, values in metric_refs.items()},
        **{f"metric_{name}_gap": values.astype(np.float32) for name, values in metric_gaps.items()},
    )
    _write_rows(output_dir / "summary.csv", summary_rows)
    for schedule_name, selected in schedules.items():
        _write_rows(
            output_dir / f"{schedule_name}_by_step.csv",
            _schedule_rows(
                steps=steps,
                topology_names=topology_names,
                selected=selected,
                stage_cost=stage_cost,
                transition_cost=transition_cost,
                metric_values=metric_values,
                metric_references=metric_refs,
                metric_gaps=metric_gaps,
            ),
        )
    _plot_schedules(
        output_dir / "schedule_metric_comparison.png",
        steps=steps,
        schedules=schedules,
        topology_names=topology_names,
        metric_values=metric_values,
        metric_references=metric_refs,
    )

    meta = {
        "name": str(raw.get("name", config_path.stem)),
        "config_path": str(config_path),
        "num_steps": int(steps.size),
        "num_candidate_topologies": int(len(topology_names)),
        "metrics": [asdict(spec) | {"path": str(spec.path)} for spec in metric_specs],
        "exclude_topologies": list(exclude),
        "topology_library_csv": str(topology_library_csv) if topology_library_csv is not None else None,
        "transition_normalize": str(selector_raw.get("transition_normalize", "max")),
        "switch_penalties": penalties,
        "burst_penalties": burst_penalties,
        "burst_power": burst_power if burst_penalties else None,
        "burst_threshold": burst_threshold if burst_penalties else None,
        "overlap_penalties": overlap_penalties,
        "critical_edge_transition_profiles": [
            {
                key: value
                for key, value in profile.items()
                if key not in {"transition_counts", "transition_cost"}
            }
            for profile in critical_transition_profiles
        ],
        "transition_caps": transition_caps,
        "min_dwell_steps": min_dwell_steps_values,
        "min_dwell_switch_penalties": min_dwell_penalties,
        "min_dwell_burst_penalties": min_dwell_burst_penalties,
        "min_dwell_overlap_penalties": min_dwell_overlap_penalties,
        "outputs": {
            "selector_arrays_npz": str(output_dir / "selector_arrays.npz"),
            "summary_csv": str(output_dir / "summary.csv"),
            "plot": str(output_dir / "schedule_metric_comparison.png"),
        },
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(output_dir), "num_steps": int(steps.size), "num_actions": int(len(topology_names))}, ensure_ascii=False, indent=2), flush=True)
    return output_dir
