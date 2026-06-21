from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def metric_maps_from_selector_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    metric_gaps = {
        key[len("metric_") : -len("_gap")]: np.asarray(value, dtype=np.float64)
        for key, value in arrays.items()
        if str(key).startswith("metric_") and str(key).endswith("_gap")
    }
    metric_values = {
        key[len("metric_") :]: np.asarray(value, dtype=np.float64)
        for key, value in arrays.items()
        if str(key).startswith("metric_") and not str(key).endswith("_gap") and not str(key).endswith("_full_link")
    }
    metric_references = {
        key[len("metric_") : -len("_full_link")]: np.asarray(value, dtype=np.float64)
        for key, value in arrays.items()
        if str(key).startswith("metric_") and str(key).endswith("_full_link")
    }
    return metric_gaps, metric_values, metric_references


def read_replacement_metric_series(
    path: str | Path,
    *,
    topology: str,
    pair_to_metric_prefix: Mapping[str, str],
    step_column: str = "step",
    pair_column: str = "pair",
    topology_column: str = "topology",
    delay_column: str = "mean_delay_ms",
    hops_column: str = "mean_hops",
) -> dict[str, dict[int, float]]:
    """Read by-step replacement metric values.

    The returned mapping uses selector metric names such as ``ce_delay_ms`` and
    ``ce_hops``. It is intentionally independent of G60 naming; callers provide
    the pair-to-prefix mapping.
    """

    out: dict[str, dict[int, float]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get(topology_column, "")) != str(topology):
                continue
            pair = str(row[pair_column])
            if pair not in pair_to_metric_prefix:
                continue
            prefix = str(pair_to_metric_prefix[pair])
            step = int(float(row[step_column]))
            out.setdefault(f"{prefix}_delay_ms", {})[step] = float(row[delay_column])
            out.setdefault(f"{prefix}_hops", {})[step] = float(row[hops_column])
    if not out:
        raise ValueError(f"no replacement rows found for topology={topology!r} in {path}")
    return out


def read_candidate_meta_from_summary(
    path: str | Path,
    *,
    topology: str,
    topology_column: str = "topology",
    meta_column: str = "candidate_meta",
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get(topology_column, "")) != str(topology):
                continue
            raw = row.get(meta_column, "")
            if not raw:
                return {}
            return dict(json.loads(raw))
    raise ValueError(f"topology={topology!r} not found in {path}")


def rows_for_replacement_steps(steps: Sequence[int], replacement: Mapping[str, Mapping[int, float]]) -> tuple[int, int]:
    step_to_row = {int(step): row for row, step in enumerate(steps)}
    replacement_steps = sorted({int(step) for by_step in replacement.values() for step in by_step})
    if not replacement_steps:
        raise ValueError("replacement contains no steps")
    missing = [step for step in replacement_steps if step not in step_to_row]
    if missing:
        raise ValueError(f"replacement steps are not present in selector steps: {missing[:10]}")
    rows = [step_to_row[step] for step in replacement_steps]
    left = min(rows)
    right = max(rows)
    expected = set(range(left, right + 1))
    if set(rows) != expected:
        raise ValueError("replacement steps must form one contiguous row window")
    return int(left), int(right)


def chosen_metric_series(
    *,
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
) -> dict[str, np.ndarray]:
    _metric_gaps, metric_values, _metric_refs = metric_maps_from_selector_arrays(arrays)
    idx = np.asarray(selected, dtype=np.int32)
    rows = np.arange(idx.size)
    return {
        name: np.asarray(values, dtype=np.float64)[rows, idx].copy()
        for name, values in metric_values.items()
    }


def apply_metric_replacement(
    *,
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
    replacement: Mapping[str, Mapping[int, float]],
    left_row: int | None = None,
    right_row: int | None = None,
) -> tuple[dict[str, np.ndarray], int, int]:
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    if left_row is None or right_row is None:
        inferred_left, inferred_right = rows_for_replacement_steps(steps, replacement)
        left_row = inferred_left if left_row is None else int(left_row)
        right_row = inferred_right if right_row is None else int(right_row)
    left = int(left_row)
    right = int(right_row)
    if not (0 <= left <= right < steps.size):
        raise ValueError("replacement rows must satisfy 0 <= left <= right < num_steps")

    series = chosen_metric_series(arrays=arrays, selected=selected)
    step_to_row = {int(step): row for row, step in enumerate(steps)}
    for metric_name, by_step in replacement.items():
        if metric_name not in series:
            raise KeyError(f"metric {metric_name!r} is not present in selector arrays")
        for step, value in by_step.items():
            row = step_to_row[int(step)]
            if row < left or row > right:
                raise ValueError(f"replacement step {step} is outside rows {left}..{right}")
            series[metric_name][row] = float(value)
    return series, left, right


def setup_counts_with_local_replacement(
    *,
    selected: np.ndarray,
    transition_counts: np.ndarray,
    left_row: int,
    right_row: int,
    entry_setup: int,
    exit_setup: int,
) -> np.ndarray:
    selected = np.asarray(selected, dtype=np.int32)
    transition = np.asarray(transition_counts, dtype=np.float64)
    counts = np.zeros(selected.size, dtype=np.float64)
    if selected.size > 1:
        counts[1:] = transition[selected[:-1], selected[1:]]

    left = int(left_row)
    right = int(right_row)
    if left > 0:
        counts[left] = float(entry_setup)
    for row in range(max(left + 1, 1), right + 1):
        counts[row] = 0.0
    if right + 1 < selected.size:
        counts[right + 1] = float(exit_setup)
    return counts


def summarize_local_metric_replacement(
    *,
    name: str,
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
    replacement: Mapping[str, Mapping[int, float]],
    entry_setup: int,
    exit_setup: int,
    left_row: int | None = None,
    right_row: int | None = None,
    quality_threshold: float | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    selected = np.asarray(selected, dtype=np.int32)
    metric_values, left, right = apply_metric_replacement(
        arrays=arrays,
        selected=selected,
        replacement=replacement,
        left_row=left_row,
        right_row=right_row,
    )
    _metric_gaps, _base_values, metric_refs = metric_maps_from_selector_arrays(arrays)
    setup_counts = setup_counts_with_local_replacement(
        selected=selected,
        transition_counts=np.asarray(arrays["transition_counts"], dtype=np.float64),
        left_row=left,
        right_row=right,
        entry_setup=int(entry_setup),
        exit_setup=int(exit_setup),
    )
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    summary: dict[str, Any] = {
        "schedule": str(name),
        "left_row": int(left),
        "right_row": int(right),
        "left_step": int(steps[left]),
        "right_step": int(steps[right]),
        "entry_setup": int(entry_setup),
        "exit_setup": int(exit_setup),
        "total_setup_commands": int(np.sum(setup_counts)),
        "max_setup_commands_per_step": int(np.max(setup_counts)) if setup_counts.size else 0,
        "mean_setup_commands_per_step": float(np.mean(setup_counts)) if setup_counts.size else 0.0,
        "num_setup_events": int(np.count_nonzero(setup_counts)),
    }
    delay_global: list[float] = []
    hop_global: list[float] = []
    for metric_name, values in sorted(metric_values.items()):
        ref = np.asarray(metric_refs[metric_name], dtype=np.float64)
        ref_mean = float(np.mean(ref))
        value_mean = float(np.mean(values))
        global_gap = float((value_mean - ref_mean) / ref_mean) if abs(ref_mean) > 1e-12 else math.nan
        abs_step_gap = np.abs(values - ref) / np.maximum(np.abs(ref), 1e-9)
        summary[f"mean_{metric_name}_value"] = value_mean
        summary[f"mean_{metric_name}_full_link"] = ref_mean
        summary[f"mean_{metric_name}_gap"] = float(np.mean(abs_step_gap))
        summary[f"max_{metric_name}_gap"] = float(np.max(abs_step_gap))
        summary[f"mean_{metric_name}_global_gap"] = global_gap
        if math.isfinite(global_gap) and "delay" in metric_name:
            delay_global.append(global_gap)
        if math.isfinite(global_gap) and "hops" in metric_name:
            hop_global.append(global_gap)
    if delay_global:
        summary["mean_delay_global_gap"] = float(np.mean(delay_global))
    if hop_global:
        summary["mean_hop_global_gap"] = float(np.mean(hop_global))
    if delay_global and hop_global:
        summary["mean_all_global_gap"] = float((summary["mean_delay_global_gap"] + summary["mean_hop_global_gap"]) / 2.0)
    if quality_threshold is not None:
        summary["quality_threshold"] = float(quality_threshold)
        summary["within_quality_threshold"] = bool(float(summary.get("mean_all_global_gap", math.inf)) <= float(quality_threshold))
    return summary, metric_values


def replacement_by_step_rows(
    *,
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
    replacement_name: str,
    left_row: int,
    right_row: int,
    metric_values: Mapping[str, np.ndarray],
    setup_counts: np.ndarray,
) -> list[dict[str, Any]]:
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    topology_names = np.asarray(arrays["topology_names"])
    selected = np.asarray(selected, dtype=np.int32)
    rows: list[dict[str, Any]] = []
    for row, step in enumerate(steps):
        in_replacement = int(left_row) <= row <= int(right_row)
        item: dict[str, Any] = {
            "row": int(row),
            "step": int(step),
            "action_idx": -1 if in_replacement else int(selected[row]),
            "topology": str(replacement_name) if in_replacement else str(topology_names[int(selected[row])]),
            "setup_commands": float(setup_counts[row]),
            "is_replacement": bool(in_replacement),
        }
        for metric_name, values in sorted(metric_values.items()):
            item[f"{metric_name}_value"] = float(values[row])
        rows.append(item)
    return rows
