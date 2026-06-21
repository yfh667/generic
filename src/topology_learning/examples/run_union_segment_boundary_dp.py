from __future__ import annotations

import argparse
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


def _segment_boundaries_from_schedules(schedules: list[np.ndarray]) -> list[int]:
    if not schedules:
        raise ValueError("at least one boundary schedule is required")
    n = int(schedules[0].size)
    boundaries = {0, n}
    for selected in schedules:
        selected = np.asarray(selected, dtype=np.int32)
        if selected.shape != (n,):
            raise ValueError("all boundary schedules must have the same length")
        boundaries.update((np.flatnonzero(selected[1:] != selected[:-1]) + 1).astype(int).tolist())
    return sorted(boundaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all-action budget DP on the union of segment boundaries from one or more "
            "existing schedules."
        )
    )
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--boundary-schedule", type=str, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--max-transition-count", type=int, default=None)
    parser.add_argument("--score-kind", type=str, default="meanref")
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--max-labels-per-action",
        type=int,
        default=None,
        help="Optional approximate cap per segment/action frontier. Omit for exact pruning.",
    )
    parser.add_argument(
        "--top-k-actions",
        type=int,
        default=None,
        help=(
            "Optional per-segment candidate cap. Keeps the top-k actions by segment score "
            "plus the actions used by the boundary schedules. Omit to consider all actions."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selector_dir = Path(args.selector_dir)
    arrays = _load_npz(selector_dir / "selector_arrays.npz")

    boundary_schedules = []
    for schedule in args.boundary_schedule:
        key = _schedule_key(str(schedule))
        if key not in arrays:
            raise KeyError(f"{key!r} not found in {selector_dir / 'selector_arrays.npz'}")
        boundary_schedules.append(np.asarray(arrays[key], dtype=np.int32))

    steps = np.asarray(arrays["steps"], dtype=np.int64)
    topology_names = np.asarray(arrays["topology_names"])
    transition = np.asarray(arrays["transition_counts"], dtype=np.int32)
    score = score_matrix_for_kind(arrays, score_kind=str(args.score_kind)).astype(np.float32)
    boundaries = _segment_boundaries_from_schedules(boundary_schedules)
    segments = [(int(boundaries[i]), int(boundaries[i + 1] - 1)) for i in range(len(boundaries) - 1)]

    segment_score = np.zeros((len(segments), score.shape[1]), dtype=np.float32)
    for idx, (left, right) in enumerate(segments):
        segment_score[idx, :] = np.sum(score[left : right + 1, :], axis=0)

    candidate_actions_by_segment: list[np.ndarray] = []
    if args.top_k_actions is None or int(args.top_k_actions) <= 0:
        candidate_actions_by_segment = [np.arange(score.shape[1], dtype=np.int32) for _ in segments]
    else:
        k = min(int(args.top_k_actions), int(score.shape[1]))
        for segment_id, (left, _right) in enumerate(segments):
            finite_idx = np.flatnonzero(np.isfinite(segment_score[segment_id]))
            if finite_idx.size == 0:
                raise RuntimeError(f"no finite actions for segment {segment_id}")
            local_k = min(k, int(finite_idx.size))
            top_local = finite_idx[np.argpartition(segment_score[segment_id, finite_idx], local_k - 1)[:local_k]]
            keep = {int(action) for action in top_local.tolist()}
            for boundary in boundary_schedules:
                keep.add(int(boundary[int(left)]))
            candidate_actions_by_segment.append(np.asarray(sorted(keep), dtype=np.int32))

    result = run_budget_constrained_frontier_dp(
        score=segment_score,
        transition_counts=transition,
        candidate_actions_by_row=candidate_actions_by_segment,
        budget=int(args.budget),
        name=f"union_segment_boundary_b{int(args.budget)}",
        max_transition_count=args.max_transition_count,
        max_labels_per_action=args.max_labels_per_action,
        progress_every=int(args.progress_every),
    )

    selected = np.empty(steps.size, dtype=np.int32)
    for action, (left, right) in zip(result.selected, segments):
        selected[left : right + 1] = int(action)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "selected_action.npy", selected.astype(np.int32, copy=False))
    np.save(out_dir / "segment_actions.npy", result.selected.astype(np.int32, copy=False))

    setup_counts = np.zeros(selected.shape[0], dtype=np.int32)
    if selected.size > 1:
        setup_counts[1:] = transition[selected[:-1], selected[1:]].astype(np.int32, copy=False)
    segment_rows = []
    for segment_id, (action, (left, right)) in enumerate(zip(result.selected, segments)):
        segment_rows.append(
            {
                "segment_id": int(segment_id),
                "left_row": int(left),
                "right_row": int(right),
                "left_step": int(steps[left]),
                "right_step": int(steps[right]),
                "action_idx": int(action),
                "topology": str(topology_names[int(action)]),
                "setup_at_start": int(setup_counts[left]),
                "segment_score_sum": float(segment_score[segment_id, int(action)]),
            }
        )
    _write_rows(out_dir / "segments.csv", segment_rows)
    by_step_rows = []
    for row, action in enumerate(selected):
        prev = int(selected[row - 1]) if row > 0 else int(action)
        by_step_rows.append(
            {
                "step": int(steps[row]),
                "action_idx": int(action),
                "topology": str(topology_names[int(action)]),
                "stage_cost": float(score[row, int(action)]),
                "new_edge_transition_count": int(transition[prev, int(action)]) if row > 0 else 0,
            }
        )
    _write_rows(out_dir / f"union_segment_boundary_b{int(args.budget)}_by_step.csv", by_step_rows)

    score_row = score_selected_schedule(
        arrays=arrays,
        selected=selected,
        name=f"union_segment_boundary_b{int(args.budget)}",
        quality_threshold=args.quality_threshold,
    )
    score_row["budget"] = int(args.budget)
    score_row["result_used_budget"] = int(result.used_budget)
    score_row["boundary_schedules"] = ";".join(str(x) for x in args.boundary_schedule)
    _write_rows(out_dir / "score.csv", [score_row])

    same_as = {}
    for schedule, boundary in zip(args.boundary_schedule, boundary_schedules):
        same_as[str(schedule)] = bool(np.array_equal(selected, boundary))
    meta = {
        "selector_dir": str(selector_dir),
        "boundary_schedules": [str(x) for x in args.boundary_schedule],
        "budget": int(args.budget),
        "max_transition_count": None if args.max_transition_count is None else int(args.max_transition_count),
        "score_kind": str(args.score_kind),
        "quality_threshold": None if args.quality_threshold is None else float(args.quality_threshold),
        "max_labels_per_action": None if args.max_labels_per_action is None else int(args.max_labels_per_action),
        "top_k_actions": None if args.top_k_actions is None else int(args.top_k_actions),
        "num_segments": int(len(segments)),
        "num_actions": int(score.shape[1]),
        "segment_candidate_count_min": int(min(actions.size for actions in candidate_actions_by_segment)),
        "segment_candidate_count_max": int(max(actions.size for actions in candidate_actions_by_segment)),
        "segment_candidate_count_mean": float(np.mean([actions.size for actions in candidate_actions_by_segment])),
        "result_used_budget": int(result.used_budget),
        "same_as_boundary_schedule": same_as,
        "score": score_row,
        "outputs": {
            "selected_action_npy": str(out_dir / "selected_action.npy"),
            "segment_actions_npy": str(out_dir / "segment_actions.npy"),
            "segments_csv": str(out_dir / "segments.csv"),
            "by_step_csv": str(out_dir / f"union_segment_boundary_b{int(args.budget)}_by_step.csv"),
            "score_csv": str(out_dir / "score.csv"),
            "meta_json": str(out_dir / "meta.json"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
