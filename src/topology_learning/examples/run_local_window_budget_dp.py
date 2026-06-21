from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.examples.score_selector_schedule import score_selected_schedule
from src.topology_learning.module.budget_dp_selector import (
    run_budget_constrained_frontier_dp,
    score_matrix_for_kind,
)
from src.topology_learning.module.full_link_gap_selector import _write_rows


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {str(key): np.asarray(data[key]) for key in data.files}


def _schedule_key(name: str) -> str:
    return f"schedule_{name}"


def _read_by_step_actions(path: Path) -> np.ndarray:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty by-step CSV: {path}")
    if "action_idx" not in rows[0]:
        raise ValueError(f"by-step CSV must contain action_idx column: {path}")
    return np.asarray([int(row["action_idx"]) for row in rows], dtype=np.int32)


def _setup_total(selected: np.ndarray, transition: np.ndarray) -> int:
    if selected.size <= 1:
        return 0
    return int(np.sum(transition[selected[:-1], selected[1:]], dtype=np.int64))


def _window_budget_from_target(
    *,
    selected: np.ndarray,
    transition: np.ndarray,
    left_row: int,
    right_row: int,
    target_total_setup: int,
) -> tuple[int, int, int]:
    setup_counts = np.zeros(selected.size, dtype=np.int32)
    setup_counts[1:] = transition[selected[:-1], selected[1:]]
    total_setup = int(np.sum(setup_counts, dtype=np.int64))
    old_window_setup = int(np.sum(setup_counts[int(left_row) : int(right_row) + 1], dtype=np.int64))
    if int(right_row) + 1 < selected.size:
        old_window_setup += int(setup_counts[int(right_row) + 1])
    outside_setup = int(total_setup - old_window_setup)
    local_budget = int(target_total_setup - outside_setup)
    return local_budget, outside_setup, old_window_setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep a base schedule outside one time window, then run all-action "
            "budget DP inside the window with entry/exit setup costs included."
        )
    )
    parser.add_argument("--selector-dir", type=Path, required=True)
    base_group = parser.add_mutually_exclusive_group(required=True)
    base_group.add_argument("--base-schedule", type=str, default=None)
    base_group.add_argument("--base-by-step-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--left-row", type=int, required=True)
    parser.add_argument("--right-row", type=int, required=True)
    parser.add_argument("--target-total-setup", type=int, required=True)
    parser.add_argument("--local-budget", type=int, default=None)
    parser.add_argument("--max-transition-count", type=int, default=None)
    parser.add_argument("--score-kind", type=str, default="meanref")
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument(
        "--top-k-actions",
        type=int,
        default=None,
        help="Optional per-row action cap inside the window. Keeps top-k by score plus the base action.",
    )
    parser.add_argument("--max-labels-per-action", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selector_dir = Path(args.selector_dir)
    arrays = _load_npz(selector_dir / "selector_arrays.npz")
    if args.base_by_step_csv is not None:
        base = _read_by_step_actions(Path(args.base_by_step_csv))
        base_label = Path(args.base_by_step_csv).stem.removesuffix("_by_step")
    else:
        key = _schedule_key(str(args.base_schedule))
        if key not in arrays:
            raise KeyError(f"{key!r} not found in {selector_dir / 'selector_arrays.npz'}")
        base = np.asarray(arrays[key], dtype=np.int32)
        base_label = str(args.base_schedule)
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    topology_names = np.asarray(arrays["topology_names"])
    transition = np.asarray(arrays["transition_counts"], dtype=np.int32)
    score = score_matrix_for_kind(arrays, score_kind=str(args.score_kind)).astype(np.float32)
    if base.shape != steps.shape:
        raise ValueError(f"base schedule length {base.shape[0]} does not match selector steps {steps.shape[0]}")

    left = int(args.left_row)
    right = int(args.right_row)
    if not (0 <= left <= right < base.size):
        raise ValueError("window rows must satisfy 0 <= left <= right < num_steps")
    if left == 0 or right + 1 >= base.size:
        raise ValueError("this helper currently expects one fixed action before and after the window")

    computed_budget, outside_setup, old_window_setup = _window_budget_from_target(
        selected=base,
        transition=transition,
        left_row=left,
        right_row=right,
        target_total_setup=int(args.target_total_setup),
    )
    local_budget = int(computed_budget if args.local_budget is None else args.local_budget)
    if local_budget < 0:
        raise ValueError(f"local budget is negative: {local_budget}")

    prev_action = int(base[left - 1])
    next_action = int(base[right + 1])
    all_actions = np.arange(score.shape[1], dtype=np.int32)
    local_score = np.zeros((right - left + 3, score.shape[1]), dtype=np.float32)
    local_score[1:-1, :] = score[left : right + 1, :]
    if args.top_k_actions is None or int(args.top_k_actions) <= 0:
        window_candidates = [all_actions for _ in range(right - left + 1)]
    else:
        k = min(int(args.top_k_actions), int(score.shape[1]))
        window_candidates = []
        for row in range(left, right + 1):
            finite_idx = np.flatnonzero(np.isfinite(score[row]))
            if finite_idx.size == 0:
                raise RuntimeError(f"no finite action scores at row {row}")
            local_k = min(k, int(finite_idx.size))
            top_local = finite_idx[np.argpartition(score[row, finite_idx], local_k - 1)[:local_k]]
            keep = {int(action) for action in top_local.tolist()}
            keep.add(int(base[row]))
            window_candidates.append(np.asarray(sorted(keep), dtype=np.int32))
    candidates = [np.asarray([prev_action], dtype=np.int32)] + window_candidates + [
        np.asarray([next_action], dtype=np.int32)
    ]

    result = run_budget_constrained_frontier_dp(
        score=local_score,
        transition_counts=transition,
        candidate_actions_by_row=candidates,
        budget=local_budget,
        name=f"local_window_{left}_{right}_b{local_budget}",
        max_transition_count=args.max_transition_count,
        max_labels_per_action=args.max_labels_per_action,
        progress_every=int(args.progress_every),
    )
    local_selected = result.selected[1:-1].astype(np.int32, copy=False)
    selected = base.copy()
    selected[left : right + 1] = local_selected

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "selected_action.npy", selected.astype(np.int32, copy=False))
    np.save(out_dir / "local_selected_action.npy", local_selected.astype(np.int32, copy=False))
    setup_counts = np.zeros(selected.shape[0], dtype=np.int32)
    if selected.size > 1:
        setup_counts[1:] = transition[selected[:-1], selected[1:]].astype(np.int32, copy=False)
    by_step_rows = []
    for row, action in enumerate(selected):
        by_step_rows.append(
            {
                "step": int(steps[row]),
                "action_idx": int(action),
                "topology": str(topology_names[int(action)]),
                "stage_cost": float(score[row, int(action)]),
                "new_edge_transition_count": int(setup_counts[row]),
            }
        )
    _write_rows(out_dir / f"local_window_{left}_{right}_by_step.csv", by_step_rows)

    score_row = score_selected_schedule(
        arrays=arrays,
        selected=selected,
        name=f"local_window_{left}_{right}_b{local_budget}",
        quality_threshold=args.quality_threshold,
    )
    score_row.update(
        {
            "left_row": int(left),
            "right_row": int(right),
            "left_step": int(steps[left]),
            "right_step": int(steps[right]),
            "local_budget": int(local_budget),
            "local_used_budget": int(result.used_budget),
            "outside_setup": int(outside_setup),
            "old_window_setup_with_exit": int(old_window_setup),
            "target_total_setup": int(args.target_total_setup),
            "same_as_base": bool(np.array_equal(selected, base)),
        }
    )
    _write_rows(out_dir / "score.csv", [score_row])

    segments = []
    start = 0
    for idx in range(1, local_selected.size + 1):
        if idx == local_selected.size or int(local_selected[idx]) != int(local_selected[start]):
            action = int(local_selected[start])
            segments.append(
                {
                    "local_start": int(start),
                    "local_end": int(idx - 1),
                    "left_row": int(left + start),
                    "right_row": int(left + idx - 1),
                    "left_step": int(steps[left + start]),
                    "right_step": int(steps[left + idx - 1]),
                    "action_idx": action,
                    "topology": str(topology_names[action]),
                }
            )
            start = idx
    _write_rows(out_dir / "local_segments.csv", segments)

    meta = {
        "selector_dir": str(selector_dir),
        "base_schedule": str(base_label),
        "base_by_step_csv": None if args.base_by_step_csv is None else str(Path(args.base_by_step_csv)),
        "left_row": int(left),
        "right_row": int(right),
        "prev_action": int(prev_action),
        "next_action": int(next_action),
        "target_total_setup": int(args.target_total_setup),
        "local_budget": int(local_budget),
        "local_used_budget": int(result.used_budget),
        "outside_setup": int(outside_setup),
        "old_window_setup_with_exit": int(old_window_setup),
        "max_transition_count": None if args.max_transition_count is None else int(args.max_transition_count),
            "score_kind": str(args.score_kind),
            "quality_threshold": None if args.quality_threshold is None else float(args.quality_threshold),
            "top_k_actions": None if args.top_k_actions is None else int(args.top_k_actions),
            "max_labels_per_action": None if args.max_labels_per_action is None else int(args.max_labels_per_action),
            "candidate_count_min": int(min(len(actions) for actions in candidates)),
            "candidate_count_max": int(max(len(actions) for actions in candidates)),
            "candidate_count_mean": float(np.mean([len(actions) for actions in candidates])),
            "num_local_segments": int(len(segments)),
        "score": score_row,
        "outputs": {
            "selected_action_npy": str(out_dir / "selected_action.npy"),
            "local_selected_action_npy": str(out_dir / "local_selected_action.npy"),
            "by_step_csv": str(out_dir / f"local_window_{left}_{right}_by_step.csv"),
            "score_csv": str(out_dir / "score.csv"),
            "local_segments_csv": str(out_dir / "local_segments.csv"),
            "meta_json": str(out_dir / "meta.json"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
