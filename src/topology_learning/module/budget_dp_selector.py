from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .full_link_gap_selector import _summary_row, _write_rows


@dataclass(frozen=True)
class BudgetDpRun:
    name: str
    selected: np.ndarray
    score_total: float
    used_budget: int


def _schedule_keys(npz: Mapping[str, Any]) -> list[str]:
    return sorted(str(key) for key in npz.keys() if str(key).startswith("schedule_"))


def _load_selector_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {str(key): np.asarray(data[key]) for key in data.files}


def _metric_gap_names(arrays: Mapping[str, np.ndarray]) -> list[str]:
    names: list[str] = []
    for key in arrays:
        if str(key).startswith("metric_") and str(key).endswith("_gap"):
            names.append(str(key)[len("metric_") : -len("_gap")])
    return sorted(names)


def score_matrix_for_kind(
    arrays: Mapping[str, np.ndarray],
    *,
    score_kind: str,
) -> np.ndarray:
    """Build the DP objective matrix.

    The output is only used to choose a schedule. Summary rows are always
    computed from the original metric gaps in ``arrays``.
    """

    kind = str(score_kind)
    if kind == "default_stage":
        return np.asarray(arrays["stage_cost"], dtype=np.float32)

    use_mean_reference_objective = False
    if kind == "meanref" or kind.startswith("meanref_"):
        use_mean_reference_objective = True
        kind = kind[len("meanref") :].lstrip("_") or "delay1_hop1"

    gap_names = _metric_gap_names(arrays)
    if not gap_names:
        raise ValueError("selector arrays do not contain metric_*_gap fields")
    delay_names = [name for name in gap_names if "delay" in name]
    hop_names = [name for name in gap_names if "hop" in name]
    if not delay_names or not hop_names:
        raise ValueError("weighted score kinds require both delay and hop metric gaps")

    delay_weight = 1.0
    hop_weight = 1.0
    parts = kind.split("_")
    for part in parts:
        if part.startswith("delay"):
            delay_weight = float(part[len("delay") :] or 1.0)
        elif part.startswith("hop"):
            hop_weight = float(part[len("hop") :] or 1.0)

    sample_key = f"metric_{gap_names[0]}_gap"
    weighted = np.zeros_like(np.asarray(arrays[sample_key], dtype=np.float32), dtype=np.float32)
    denom = 0.0
    if use_mean_reference_objective:
        for name in delay_names:
            ref = np.asarray(arrays[f"metric_{name}_full_link"], dtype=np.float64)
            ref_mean = float(np.nanmean(ref))
            if not math.isfinite(ref_mean) or abs(ref_mean) < 1e-12:
                raise ValueError(f"invalid full_link reference mean for metric {name!r}")
            weighted += float(delay_weight) * (
                np.asarray(arrays[f"metric_{name}"], dtype=np.float32) / float(ref_mean)
            )
            denom += float(delay_weight)
        for name in hop_names:
            ref = np.asarray(arrays[f"metric_{name}_full_link"], dtype=np.float64)
            ref_mean = float(np.nanmean(ref))
            if not math.isfinite(ref_mean) or abs(ref_mean) < 1e-12:
                raise ValueError(f"invalid full_link reference mean for metric {name!r}")
            weighted += float(hop_weight) * (
                np.asarray(arrays[f"metric_{name}"], dtype=np.float32) / float(ref_mean)
            )
            denom += float(hop_weight)
    else:
        for name in delay_names:
            weighted += float(delay_weight) * np.asarray(arrays[f"metric_{name}_gap"], dtype=np.float32)
            denom += float(delay_weight)
        for name in hop_names:
            weighted += float(hop_weight) * np.asarray(arrays[f"metric_{name}_gap"], dtype=np.float32)
            denom += float(hop_weight)
    if denom <= 0.0:
        raise ValueError(f"invalid score weights for {score_kind!r}")
    return (weighted / float(denom)).astype(np.float32, copy=False)


def collect_candidate_actions_by_row(
    *,
    source_dirs: Sequence[str | Path],
    top_k_stage: int = 0,
) -> tuple[dict[str, np.ndarray], list[np.ndarray], list[str]]:
    """Collect per-row action candidates from schedule arrays in source dirs."""

    if not source_dirs:
        raise ValueError("source_dirs must be non-empty")

    first = Path(source_dirs[0])
    base = _load_selector_npz(first / "selector_arrays.npz")
    steps = np.asarray(base["steps"])
    topology_names = np.asarray(base["topology_names"])
    n_steps, n_actions = np.asarray(base["stage_cost"]).shape

    candidate_sets = [set() for _ in range(n_steps)]
    source_schedule_names: list[str] = []

    for source_dir in source_dirs:
        source = Path(source_dir)
        arrays = _load_selector_npz(source / "selector_arrays.npz")
        if not np.array_equal(np.asarray(arrays["steps"]), steps):
            raise ValueError(f"steps differ in {source}")
        if not np.array_equal(np.asarray(arrays["topology_names"]), topology_names):
            raise ValueError(f"topology_names differ in {source}")
        for key in _schedule_keys(arrays):
            selected = np.asarray(arrays[key], dtype=np.int32)
            if selected.shape != (n_steps,):
                raise ValueError(f"{source}/{key} has shape {selected.shape}, expected {(n_steps,)}")
            source_schedule_names.append(f"{source.name}:{key[len('schedule_'):]}")
            for row, action in enumerate(selected):
                candidate_sets[row].add(int(action))

    k = int(top_k_stage)
    if k > 0:
        stage = np.asarray(base["stage_cost"], dtype=np.float32)
        kk = min(k, n_actions)
        for row in range(n_steps):
            finite = np.isfinite(stage[row])
            finite_idx = np.flatnonzero(finite)
            if finite_idx.size == 0:
                continue
            local_k = min(kk, finite_idx.size)
            best_local = np.argpartition(stage[row, finite_idx], local_k - 1)[:local_k]
            for action in finite_idx[best_local]:
                candidate_sets[row].add(int(action))

    candidates = [np.asarray(sorted(row_set), dtype=np.int32) for row_set in candidate_sets]
    for row, actions in enumerate(candidates):
        if actions.size == 0:
            raise ValueError(f"no candidate action at row {row}")
    return base, candidates, source_schedule_names


def add_top_k_score_candidates(
    candidate_actions_by_row: Sequence[np.ndarray],
    score: np.ndarray,
    *,
    top_k_score: int = 0,
) -> list[np.ndarray]:
    """Return row candidates plus the top-k actions under the current score.

    ``top_k_stage`` in :func:`collect_candidate_actions_by_row` always uses the
    original selector ``stage_cost``. This helper is useful when the DP objective
    is a transformed score, for example ``meanref_delay1.25_hop1``.
    """

    k = int(top_k_score)
    if k <= 0:
        return [np.asarray(actions, dtype=np.int32) for actions in candidate_actions_by_row]

    score_arr = np.asarray(score, dtype=np.float32)
    if score_arr.ndim != 2:
        raise ValueError("score must be 2D")
    if len(candidate_actions_by_row) != score_arr.shape[0]:
        raise ValueError("candidate_actions_by_row length must match score rows")

    out: list[np.ndarray] = []
    n_actions = int(score_arr.shape[1])
    kk = min(k, n_actions)
    for row, actions in enumerate(candidate_actions_by_row):
        row_set = set(int(x) for x in np.asarray(actions, dtype=np.int32))
        finite_idx = np.flatnonzero(np.isfinite(score_arr[row]))
        if finite_idx.size:
            local_k = min(kk, finite_idx.size)
            best_local = np.argpartition(score_arr[row, finite_idx], local_k - 1)[:local_k]
            for action in finite_idx[best_local]:
                row_set.add(int(action))
        out.append(np.asarray(sorted(row_set), dtype=np.int32))
    return out


def run_budget_constrained_dp(
    *,
    score: np.ndarray,
    transition_counts: np.ndarray,
    candidate_actions_by_row: Sequence[np.ndarray],
    budget: int,
    name: str,
    max_transition_count: int | None = None,
) -> BudgetDpRun:
    """Minimise score subject to a total new-edge setup budget."""

    stage = np.asarray(score, dtype=np.float32)
    trans = np.asarray(transition_counts, dtype=np.int32)
    if stage.ndim != 2:
        raise ValueError("score must be 2D")
    n_steps, n_actions = stage.shape
    if trans.shape != (n_actions, n_actions):
        raise ValueError("transition_counts shape must match action count")
    if len(candidate_actions_by_row) != n_steps:
        raise ValueError("candidate_actions_by_row length must match number of steps")
    max_budget = int(budget)
    if max_budget < 0:
        raise ValueError("budget must be >= 0")
    cap = None if max_transition_count is None else int(max_transition_count)
    if cap is not None and cap < 0:
        raise ValueError("max_transition_count must be >= 0")

    budget_dtype = np.int16 if max_budget <= np.iinfo(np.int16).max else np.int32
    local_dtype = np.int16
    actions0 = np.asarray(candidate_actions_by_row[0], dtype=np.int32)
    cost_prev = np.full((max_budget + 1, actions0.size), math.inf, dtype=np.float32)
    cost_prev[0, :] = stage[0, actions0]

    prev_budget_rows: list[np.ndarray | None] = [None] * n_steps
    prev_local_rows: list[np.ndarray | None] = [None] * n_steps

    prev_actions = actions0
    for row in range(1, n_steps):
        curr_actions = np.asarray(candidate_actions_by_row[row], dtype=np.int32)
        cost_curr = np.full((max_budget + 1, curr_actions.size), math.inf, dtype=np.float32)
        prev_budget = np.full((max_budget + 1, curr_actions.size), -1, dtype=budget_dtype)
        prev_local = np.full((max_budget + 1, curr_actions.size), -1, dtype=local_dtype)

        for prev_local_idx, prev_action in enumerate(prev_actions):
            prev_col = cost_prev[:, prev_local_idx]
            if not np.any(np.isfinite(prev_col)):
                continue
            for curr_local_idx, curr_action in enumerate(curr_actions):
                setup = int(trans[int(prev_action), int(curr_action)])
                if cap is not None and setup > cap:
                    continue
                if setup > max_budget:
                    continue
                upto = max_budget - setup + 1
                candidate = prev_col[:upto] + stage[row, int(curr_action)]
                target = cost_curr[setup:, curr_local_idx]
                mask = candidate < target
                if not np.any(mask):
                    continue
                target[mask] = candidate[mask]
                idx = np.flatnonzero(mask)
                prev_budget[setup:, curr_local_idx][idx] = idx.astype(budget_dtype, copy=False)
                prev_local[setup:, curr_local_idx][idx] = int(prev_local_idx)

        if not np.any(np.isfinite(cost_curr)):
            raise RuntimeError(f"no feasible state at row {row} within budget {max_budget}")
        prev_budget_rows[row] = prev_budget
        prev_local_rows[row] = prev_local
        cost_prev = cost_curr
        prev_actions = curr_actions

    flat = int(np.argmin(cost_prev))
    best_budget, best_local = np.unravel_index(flat, cost_prev.shape)
    best_score = float(cost_prev[best_budget, best_local])
    if not math.isfinite(best_score):
        raise RuntimeError(f"no finite budget-DP schedule for budget {max_budget}")

    selected = np.empty(n_steps, dtype=np.int32)
    budget_cursor = int(best_budget)
    local_cursor = int(best_local)
    selected[-1] = int(candidate_actions_by_row[-1][local_cursor])
    for row in range(n_steps - 1, 0, -1):
        pb = prev_budget_rows[row]
        pl = prev_local_rows[row]
        if pb is None or pl is None:
            raise RuntimeError("budget-DP backtrack state is missing")
        next_budget = int(pb[budget_cursor, local_cursor])
        next_local = int(pl[budget_cursor, local_cursor])
        if next_budget < 0 or next_local < 0:
            raise RuntimeError(f"budget-DP backtrack failed at row {row}")
        budget_cursor = next_budget
        local_cursor = next_local
        selected[row - 1] = int(candidate_actions_by_row[row - 1][local_cursor])

    return BudgetDpRun(name=str(name), selected=selected, score_total=best_score, used_budget=int(best_budget))


def _prune_frontier_labels(
    labels: list[tuple[int, float, int]],
    *,
    max_labels: int | None = None,
) -> list[tuple[int, float, int]]:
    """Keep non-dominated labels `(budget, score, prev_label_id)`.

    A label is dominated when another label for the same action uses no more
    setup budget and has no larger score. The optional ``max_labels`` is an
    approximate safety valve for very large candidate sets; leaving it unset is
    exact for the row-action candidate graph.
    """

    if not labels:
        return []
    best_by_budget: dict[int, tuple[float, int]] = {}
    for budget, score, prev_label in labels:
        budget = int(budget)
        score = float(score)
        old = best_by_budget.get(budget)
        if old is None or score < old[0]:
            best_by_budget[budget] = (score, int(prev_label))

    out: list[tuple[int, float, int]] = []
    best_score = math.inf
    for budget in sorted(best_by_budget):
        score, prev_label = best_by_budget[budget]
        if score < best_score:
            out.append((int(budget), float(score), int(prev_label)))
            best_score = float(score)

    if max_labels is not None and int(max_labels) > 0 and len(out) > int(max_labels):
        # Keep a deterministic mix of low-score and low-budget labels. This is
        # approximate, so callers should use it only for exploratory scans.
        keep: set[int] = set()
        by_score = sorted(range(len(out)), key=lambda idx: (out[idx][1], out[idx][0]))
        by_budget = sorted(range(len(out)), key=lambda idx: (out[idx][0], out[idx][1]))
        half = max(1, int(max_labels) // 2)
        keep.update(by_score[:half])
        keep.update(by_budget[: int(max_labels) - len(keep)])
        cursor = 0
        while len(keep) < int(max_labels) and cursor < len(by_score):
            keep.add(by_score[cursor])
            cursor += 1
        out = [out[idx] for idx in sorted(keep)]
    return out


def _backtrack_frontier_selection(row_records: Sequence[Mapping[str, np.ndarray]], label_id: int) -> np.ndarray:
    selected = np.empty(len(row_records), dtype=np.int32)
    label_cursor = int(label_id)
    for row in range(len(row_records) - 1, -1, -1):
        records = row_records[row]
        selected[row] = int(records["action"][label_cursor])
        label_cursor = int(records["prev"][label_cursor])
        if row > 0 and label_cursor < 0:
            raise RuntimeError(f"frontier-DP backtrack failed at row {row}")
    return selected


def run_budget_constrained_frontier_dp(
    *,
    score: np.ndarray,
    transition_counts: np.ndarray,
    candidate_actions_by_row: Sequence[np.ndarray],
    budget: int,
    name: str,
    max_transition_count: int | None = None,
    max_labels_per_action: int | None = None,
    progress_every: int = 0,
) -> BudgetDpRun:
    """Minimise score with a label-setting resource-constrained DP.

    Compared with :func:`run_budget_constrained_dp`, this solver stores only the
    non-dominated `(budget, score)` frontier for each row/action. It is useful
    when the setup budget is moderately large but each action has only a small
    number of meaningful budget/score trade-offs.
    """

    stage = np.asarray(score, dtype=np.float32)
    trans = np.asarray(transition_counts, dtype=np.int32)
    if stage.ndim != 2:
        raise ValueError("score must be 2D")
    n_steps, n_actions = stage.shape
    if trans.shape != (n_actions, n_actions):
        raise ValueError("transition_counts shape must match action count")
    if len(candidate_actions_by_row) != n_steps:
        raise ValueError("candidate_actions_by_row length must match number of steps")
    max_budget = int(budget)
    if max_budget < 0:
        raise ValueError("budget must be >= 0")
    cap = None if max_transition_count is None else int(max_transition_count)
    if cap is not None and cap < 0:
        raise ValueError("max_transition_count must be >= 0")

    row_records: list[dict[str, np.ndarray]] = []
    prev_by_action: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    actions0 = np.asarray(candidate_actions_by_row[0], dtype=np.int32)
    action_values: list[int] = []
    budget_values: list[int] = []
    score_values: list[float] = []
    prev_values: list[int] = []
    for action in actions0:
        value = float(stage[0, int(action)])
        if not math.isfinite(value):
            continue
        label_id = len(action_values)
        action_values.append(int(action))
        budget_values.append(0)
        score_values.append(value)
        prev_values.append(-1)
        prev_by_action[int(action)] = (
            np.asarray([0], dtype=np.int32),
            np.asarray([value], dtype=np.float32),
            np.asarray([label_id], dtype=np.int32),
        )
    if not prev_by_action:
        raise RuntimeError("no feasible initial labels")
    row_records.append(
        {
            "action": np.asarray(action_values, dtype=np.int32),
            "budget": np.asarray(budget_values, dtype=np.int32),
            "score": np.asarray(score_values, dtype=np.float32),
            "prev": np.asarray(prev_values, dtype=np.int32),
        }
    )

    for row in range(1, n_steps):
        curr_actions = np.asarray(candidate_actions_by_row[row], dtype=np.int32)
        pending: dict[int, list[tuple[int, float, int]]] = {int(action): [] for action in curr_actions}
        for prev_action, (prev_budgets, prev_scores, prev_label_ids) in prev_by_action.items():
            for curr_action in curr_actions:
                setup = int(trans[int(prev_action), int(curr_action)])
                if cap is not None and setup > cap:
                    continue
                if setup > max_budget:
                    continue
                allowed = prev_budgets <= (max_budget - setup)
                if not np.any(allowed):
                    continue
                next_budgets = prev_budgets[allowed] + setup
                next_scores = prev_scores[allowed] + float(stage[row, int(curr_action)])
                next_prev_ids = prev_label_ids[allowed]
                finite = np.isfinite(next_scores)
                if not np.any(finite):
                    continue
                pending[int(curr_action)].extend(
                    (
                        int(next_budgets[idx]),
                        float(next_scores[idx]),
                        int(next_prev_ids[idx]),
                    )
                    for idx in np.flatnonzero(finite)
                )

        action_values = []
        budget_values = []
        score_values = []
        prev_values = []
        curr_by_action: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for action in curr_actions:
            labels = _prune_frontier_labels(
                pending.get(int(action), []),
                max_labels=max_labels_per_action,
            )
            if not labels:
                continue
            start = len(action_values)
            for budget_value, score_value, prev_label in labels:
                action_values.append(int(action))
                budget_values.append(int(budget_value))
                score_values.append(float(score_value))
                prev_values.append(int(prev_label))
            stop = len(action_values)
            curr_by_action[int(action)] = (
                np.asarray(budget_values[start:stop], dtype=np.int32),
                np.asarray(score_values[start:stop], dtype=np.float32),
                np.arange(start, stop, dtype=np.int32),
            )
        if not curr_by_action:
            raise RuntimeError(f"no feasible frontier labels at row {row} within budget {max_budget}")
        row_records.append(
            {
                "action": np.asarray(action_values, dtype=np.int32),
                "budget": np.asarray(budget_values, dtype=np.int32),
                "score": np.asarray(score_values, dtype=np.float32),
                "prev": np.asarray(prev_values, dtype=np.int32),
            }
        )
        prev_by_action = curr_by_action
        if int(progress_every) > 0 and (row + 1 == n_steps or (row + 1) % int(progress_every) == 0):
            label_count = sum(len(values[0]) for values in prev_by_action.values())
            print(
                f"[budget-dp-frontier] row={row + 1}/{n_steps} actions={len(prev_by_action)} labels={label_count}",
                flush=True,
            )

    last = row_records[-1]
    best_label = int(np.argmin(last["score"]))
    best_score = float(last["score"][best_label])
    best_budget = int(last["budget"][best_label])
    if not math.isfinite(best_score):
        raise RuntimeError(f"no finite frontier-DP schedule for budget {max_budget}")

    selected = _backtrack_frontier_selection(row_records, best_label)

    return BudgetDpRun(name=str(name), selected=selected, score_total=best_score, used_budget=best_budget)


def run_budget_constrained_frontier_curve_dp(
    *,
    score: np.ndarray,
    transition_counts: np.ndarray,
    candidate_actions_by_row: Sequence[np.ndarray],
    budgets: Sequence[int],
    name_prefix: str,
    max_transition_count: int | None = None,
    max_labels_per_action: int | None = None,
    progress_every: int = 0,
) -> list[BudgetDpRun]:
    """Run frontier DP once and return the best schedule for each budget."""

    requested = sorted({int(budget) for budget in budgets})
    if not requested:
        raise ValueError("budgets must be non-empty")
    max_budget = max(requested)

    stage = np.asarray(score, dtype=np.float32)
    trans = np.asarray(transition_counts, dtype=np.int32)
    if stage.ndim != 2:
        raise ValueError("score must be 2D")
    n_steps, n_actions = stage.shape
    if trans.shape != (n_actions, n_actions):
        raise ValueError("transition_counts shape must match action count")
    if len(candidate_actions_by_row) != n_steps:
        raise ValueError("candidate_actions_by_row length must match number of steps")
    if max_budget < 0:
        raise ValueError("budget must be >= 0")
    cap = None if max_transition_count is None else int(max_transition_count)
    if cap is not None and cap < 0:
        raise ValueError("max_transition_count must be >= 0")

    row_records: list[dict[str, np.ndarray]] = []
    prev_by_action: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    actions0 = np.asarray(candidate_actions_by_row[0], dtype=np.int32)
    action_values: list[int] = []
    budget_values: list[int] = []
    score_values: list[float] = []
    prev_values: list[int] = []
    for action in actions0:
        value = float(stage[0, int(action)])
        if not math.isfinite(value):
            continue
        label_id = len(action_values)
        action_values.append(int(action))
        budget_values.append(0)
        score_values.append(value)
        prev_values.append(-1)
        prev_by_action[int(action)] = (
            np.asarray([0], dtype=np.int32),
            np.asarray([value], dtype=np.float32),
            np.asarray([label_id], dtype=np.int32),
        )
    if not prev_by_action:
        raise RuntimeError("no feasible initial labels")
    row_records.append(
        {
            "action": np.asarray(action_values, dtype=np.int32),
            "budget": np.asarray(budget_values, dtype=np.int32),
            "score": np.asarray(score_values, dtype=np.float32),
            "prev": np.asarray(prev_values, dtype=np.int32),
        }
    )

    for row in range(1, n_steps):
        curr_actions = np.asarray(candidate_actions_by_row[row], dtype=np.int32)
        pending: dict[int, list[tuple[int, float, int]]] = {int(action): [] for action in curr_actions}
        for prev_action, (prev_budgets, prev_scores, prev_label_ids) in prev_by_action.items():
            for curr_action in curr_actions:
                setup = int(trans[int(prev_action), int(curr_action)])
                if cap is not None and setup > cap:
                    continue
                if setup > max_budget:
                    continue
                allowed = prev_budgets <= (max_budget - setup)
                if not np.any(allowed):
                    continue
                next_budgets = prev_budgets[allowed] + setup
                next_scores = prev_scores[allowed] + float(stage[row, int(curr_action)])
                next_prev_ids = prev_label_ids[allowed]
                finite = np.isfinite(next_scores)
                if not np.any(finite):
                    continue
                pending[int(curr_action)].extend(
                    (
                        int(next_budgets[idx]),
                        float(next_scores[idx]),
                        int(next_prev_ids[idx]),
                    )
                    for idx in np.flatnonzero(finite)
                )

        action_values = []
        budget_values = []
        score_values = []
        prev_values = []
        curr_by_action: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for action in curr_actions:
            labels = _prune_frontier_labels(
                pending.get(int(action), []),
                max_labels=max_labels_per_action,
            )
            if not labels:
                continue
            start = len(action_values)
            for budget_value, score_value, prev_label in labels:
                action_values.append(int(action))
                budget_values.append(int(budget_value))
                score_values.append(float(score_value))
                prev_values.append(int(prev_label))
            stop = len(action_values)
            curr_by_action[int(action)] = (
                np.asarray(budget_values[start:stop], dtype=np.int32),
                np.asarray(score_values[start:stop], dtype=np.float32),
                np.arange(start, stop, dtype=np.int32),
            )
        if not curr_by_action:
            raise RuntimeError(f"no feasible frontier labels at row {row} within budget {max_budget}")
        row_records.append(
            {
                "action": np.asarray(action_values, dtype=np.int32),
                "budget": np.asarray(budget_values, dtype=np.int32),
                "score": np.asarray(score_values, dtype=np.float32),
                "prev": np.asarray(prev_values, dtype=np.int32),
            }
        )
        prev_by_action = curr_by_action
        if int(progress_every) > 0 and (row + 1 == n_steps or (row + 1) % int(progress_every) == 0):
            label_count = sum(len(values[0]) for values in prev_by_action.values())
            print(
                f"[budget-dp-frontier] row={row + 1}/{n_steps} actions={len(prev_by_action)} labels={label_count}",
                flush=True,
            )

    last = row_records[-1]
    runs: list[BudgetDpRun] = []
    for budget in requested:
        valid = np.flatnonzero((last["budget"] <= int(budget)) & np.isfinite(last["score"]))
        if valid.size == 0:
            raise RuntimeError(f"no finite frontier-DP schedule for budget {budget}")
        local = int(valid[np.argmin(last["score"][valid])])
        selected = _backtrack_frontier_selection(row_records, local)
        runs.append(
            BudgetDpRun(
                name=f"{name_prefix}_b{int(budget)}",
                selected=selected,
                score_total=float(last["score"][local]),
                used_budget=int(last["budget"][local]),
            )
        )
    return runs


def _mean_gap(arrays: Mapping[str, np.ndarray], selected: np.ndarray, contains: str | None = None) -> float:
    rows = np.arange(selected.size)
    values: list[np.ndarray] = []
    for name in _metric_gap_names(arrays):
        if contains is not None and contains not in name:
            continue
        values.append(np.asarray(arrays[f"metric_{name}_gap"], dtype=np.float64)[rows, selected])
    if not values:
        return float("nan")
    return float(np.mean(np.vstack(values)))


def _write_schedule_step_csv(
    path: Path,
    *,
    steps: np.ndarray,
    topology_names: np.ndarray,
    selected: np.ndarray,
    stage_cost: np.ndarray,
    transition_counts: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "action_idx", "topology", "stage_cost", "new_edge_transition_count"],
        )
        writer.writeheader()
        for row, action in enumerate(selected):
            prev = int(selected[row - 1]) if row > 0 else int(action)
            writer.writerow(
                {
                    "step": int(steps[row]),
                    "action_idx": int(action),
                    "topology": str(topology_names[int(action)]),
                    "stage_cost": float(stage_cost[row, int(action)]),
                    "new_edge_transition_count": int(transition_counts[prev, int(action)]) if row > 0 else 0,
                }
            )


def run_budget_dp_selector(
    *,
    source_dirs: Sequence[str | Path],
    out_dir: str | Path,
    budgets: Sequence[int],
    score_kinds: Sequence[str] = ("default_stage",),
    max_transition_counts: Sequence[int | None] = (None,),
    top_k_stage: int = 0,
    top_k_score: int = 0,
    quality_threshold: float | None = None,
    copy_config_from: str | Path | None = None,
    solver: str = "dense",
    max_labels_per_action: int | None = None,
    progress_every: int = 0,
) -> Path:
    base, candidate_actions, source_schedule_names = collect_candidate_actions_by_row(
        source_dirs=source_dirs,
        top_k_stage=int(top_k_stage),
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps = np.asarray(base["steps"], dtype=np.int64)
    topology_names = np.asarray(base["topology_names"])
    stage_cost = np.asarray(base["stage_cost"], dtype=np.float32)
    transition_counts = np.asarray(base["transition_counts"], dtype=np.float32)

    metric_values = {
        key[len("metric_") :]: np.asarray(value, dtype=np.float64)
        for key, value in base.items()
        if key.startswith("metric_") and not key.endswith("_gap") and not key.endswith("_full_link")
    }
    metric_refs = {
        key[len("metric_") : -len("_full_link")]: np.asarray(value, dtype=np.float64)
        for key, value in base.items()
        if key.startswith("metric_") and key.endswith("_full_link")
    }
    metric_gaps = {
        key[len("metric_") : -len("_gap")]: np.asarray(value, dtype=np.float64)
        for key, value in base.items()
        if key.startswith("metric_") and key.endswith("_gap")
    }

    schedules: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    caps = tuple(max_transition_counts) if max_transition_counts else (None,)
    solver_name = str(solver or "dense").lower()
    if solver_name not in {"dense", "frontier"}:
        raise ValueError("solver must be 'dense' or 'frontier'")

    def record_result(
        *,
        label: str,
        result: BudgetDpRun,
        budget: int,
        cap: int | None,
        score_kind: str,
    ) -> None:
        selected = result.selected.astype(np.int32, copy=False)
        schedules[label] = selected
        row = _summary_row(
            name=label,
            selected=selected,
            stage_cost=stage_cost,
            transition_cost=transition_counts,
            metric_gaps=metric_gaps,
            metric_values=metric_values,
            metric_references=metric_refs,
        )
        row["mean_all_gap"] = _mean_gap(base, selected)
        row["mean_delay_gap"] = _mean_gap(base, selected, "delay")
        row["mean_hop_gap"] = _mean_gap(base, selected, "hop")
        row["mean_all_step_gap"] = float(row["mean_all_gap"])
        row["mean_delay_step_gap"] = float(row["mean_delay_gap"])
        row["mean_hop_step_gap"] = float(row["mean_hop_gap"])
        row["total_setup_commands"] = int(row["sum_new_edge_transition_cost"])
        row["max_setup_commands_per_step"] = int(
            max(
                [0]
                + [
                    int(transition_counts[int(selected[row - 1]), int(selected[row])])
                    for row in range(1, selected.size)
                ]
            )
        )
        row["transition_profile"] = "budget_dp_row_action_union"
        row["budget_dp_solver"] = solver_name
        row["max_labels_per_action"] = "" if max_labels_per_action is None else int(max_labels_per_action)
        row["budget_limit"] = int(budget)
        row["transition_cap_new_edges"] = "" if cap is None else int(cap)
        row["score_kind"] = str(score_kind)
        row["top_k_score"] = int(top_k_score)
        row["score_total"] = float(result.score_total)
        row["dp_used_budget"] = int(result.used_budget)
        if quality_threshold is not None:
            quality_gap_field = "mean_all_global_gap"
            quality_gap = float(row.get(quality_gap_field, math.nan))
            if not math.isfinite(quality_gap):
                quality_gap_field = "mean_all_gap"
                quality_gap = float(row["mean_all_gap"])
            row["quality_threshold_gap_field"] = quality_gap_field
            row["within_quality_threshold_estimate"] = bool(quality_gap <= float(quality_threshold))
        summary_rows.append(row)
        _write_schedule_step_csv(
            out / f"{label}_by_step.csv",
            steps=steps,
            topology_names=topology_names,
            selected=selected,
            stage_cost=stage_cost,
            transition_counts=transition_counts,
        )

    budget_values = sorted({int(budget) for budget in budgets})
    for score_kind in score_kinds:
        score = score_matrix_for_kind(base, score_kind=str(score_kind))
        score_candidate_actions = add_top_k_score_candidates(
            candidate_actions,
            score,
            top_k_score=int(top_k_score),
        )
        for cap in caps:
            cap_token = "" if cap is None else f"_cap{int(cap)}"
            if solver_name == "frontier" and len(budget_values) > 1:
                results = run_budget_constrained_frontier_curve_dp(
                    score=score,
                    transition_counts=transition_counts,
                    candidate_actions_by_row=score_candidate_actions,
                    budgets=budget_values,
                    name_prefix=f"budget_dp{cap_token}_{str(score_kind)}",
                    max_transition_count=cap,
                    max_labels_per_action=max_labels_per_action,
                    progress_every=int(progress_every),
                )
                for budget, result in zip(budget_values, results):
                    label = f"budget_dp_b{int(budget)}{cap_token}_{str(score_kind)}"
                    record_result(label=label, result=result, budget=int(budget), cap=cap, score_kind=str(score_kind))
            else:
                for budget in budget_values:
                    label = f"budget_dp_b{int(budget)}{cap_token}_{str(score_kind)}"
                    if solver_name == "frontier":
                        result = run_budget_constrained_frontier_dp(
                            score=score,
                            transition_counts=transition_counts,
                            candidate_actions_by_row=score_candidate_actions,
                            budget=int(budget),
                            name=label,
                            max_transition_count=cap,
                            max_labels_per_action=max_labels_per_action,
                            progress_every=int(progress_every),
                        )
                    else:
                        result = run_budget_constrained_dp(
                            score=score,
                            transition_counts=transition_counts,
                            candidate_actions_by_row=score_candidate_actions,
                            budget=int(budget),
                            name=label,
                            max_transition_count=cap,
                        )
                    record_result(label=label, result=result, budget=int(budget), cap=cap, score_kind=str(score_kind))

    arrays_to_save = {
        key: value
        for key, value in base.items()
        if key == "steps"
        or key == "topology_names"
        or key == "stage_cost"
        or key == "transition_counts"
        or key == "transition_cost"
        or key.startswith("metric_")
    }
    arrays_to_save.update({f"schedule_{name}": selected.astype(np.int32) for name, selected in schedules.items()})
    np.savez_compressed(out / "selector_arrays.npz", **arrays_to_save)
    _write_rows(out / "summary.csv", summary_rows)

    config_source = Path(copy_config_from) if copy_config_from is not None else Path(source_dirs[0]) / "selector_config.yaml"
    if config_source.is_dir():
        config_source = config_source / "selector_config.yaml"
    if config_source.exists():
        shutil.copy2(config_source, out / "selector_config.yaml")

    meta = {
        "source_dirs": [str(Path(path)) for path in source_dirs],
        "num_source_schedules": int(len(source_schedule_names)),
        "source_schedules": source_schedule_names,
        "num_steps": int(steps.size),
        "num_topologies": int(topology_names.size),
        "row_action_count_min": int(min(actions.size for actions in candidate_actions)),
        "row_action_count_max": int(max(actions.size for actions in candidate_actions)),
        "row_action_count_mean": float(np.mean([actions.size for actions in candidate_actions])),
        "budgets": [int(x) for x in budgets],
        "score_kinds": [str(x) for x in score_kinds],
        "max_transition_counts": [None if x is None else int(x) for x in caps],
        "top_k_stage": int(top_k_stage),
        "top_k_score": int(top_k_score),
        "solver": solver_name,
        "max_labels_per_action": None if max_labels_per_action is None else int(max_labels_per_action),
        "quality_threshold": None if quality_threshold is None else float(quality_threshold),
        "outputs": {
            "selector_arrays_npz": str(out / "selector_arrays.npz"),
            "summary_csv": str(out / "summary.csv"),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "num_schedules": len(schedules)}, ensure_ascii=False, indent=2), flush=True)
    return out
