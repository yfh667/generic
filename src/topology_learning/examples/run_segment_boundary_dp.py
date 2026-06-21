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


def _segments_from_selected(selected: np.ndarray) -> list[tuple[int, int]]:
    selected = np.asarray(selected, dtype=np.int32)
    if selected.size == 0:
        return []
    starts = [0] + (np.flatnonzero(selected[1:] != selected[:-1]) + 1).astype(int).tolist()
    ends = [start - 1 for start in starts[1:]] + [int(selected.size - 1)]
    return [(int(left), int(right)) for left, right in zip(starts, ends)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all-action budget DP on the fixed segment boundaries of a base selector schedule."
    )
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--base-schedule", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--max-transition-count", type=int, default=None)
    parser.add_argument("--score-kind", type=str, default="meanref")
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selector_dir = Path(args.selector_dir)
    arrays = _load_npz(selector_dir / "selector_arrays.npz")
    key = f"schedule_{args.base_schedule}"
    if key not in arrays:
        raise KeyError(f"{key!r} not found in selector_arrays.npz")

    base = np.asarray(arrays[key], dtype=np.int32)
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    topology_names = np.asarray(arrays["topology_names"])
    transition = np.asarray(arrays["transition_counts"], dtype=np.int32)
    score = score_matrix_for_kind(arrays, score_kind=str(args.score_kind)).astype(np.float32)
    segments = _segments_from_selected(base)
    if not segments:
        raise ValueError("base schedule has no segments")

    segment_score = np.zeros((len(segments), score.shape[1]), dtype=np.float32)
    for idx, (left, right) in enumerate(segments):
        segment_score[idx, :] = np.sum(score[left : right + 1, :], axis=0)

    result = run_budget_constrained_frontier_dp(
        score=segment_score,
        transition_counts=transition,
        candidate_actions_by_row=[np.arange(score.shape[1], dtype=np.int32) for _ in segments],
        budget=int(args.budget),
        name=f"segment_boundary_b{int(args.budget)}",
        max_transition_count=args.max_transition_count,
        progress_every=int(args.progress_every),
    )
    selected = np.empty(base.size, dtype=np.int32)
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

    score_row = score_selected_schedule(
        arrays=arrays,
        selected=selected,
        name=f"segment_boundary_b{int(args.budget)}",
        quality_threshold=args.quality_threshold,
    )
    _write_rows(out_dir / "score.csv", [score_row])
    meta = {
        "selector_dir": str(selector_dir),
        "base_schedule": str(args.base_schedule),
        "budget": int(args.budget),
        "max_transition_count": None if args.max_transition_count is None else int(args.max_transition_count),
        "score_kind": str(args.score_kind),
        "quality_threshold": None if args.quality_threshold is None else float(args.quality_threshold),
        "num_segments": int(len(segments)),
        "num_actions": int(score.shape[1]),
        "used_budget": int(result.used_budget),
        "same_as_base": bool(np.array_equal(selected, base)),
        "score": score_row,
        "outputs": {
            "selected_action_npy": str(out_dir / "selected_action.npy"),
            "segment_actions_npy": str(out_dir / "segment_actions.npy"),
            "segments_csv": str(out_dir / "segments.csv"),
            "score_csv": str(out_dir / "score.csv"),
            "meta_json": str(out_dir / "meta.json"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
