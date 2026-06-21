from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_learning.module.budget_dp_selector import score_matrix_for_kind
from src.topology_learning.module.full_link_gap_selector import _write_rows


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {str(key): np.asarray(data[key]) for key in data.files}


def _schedule_keys(arrays: Mapping[str, np.ndarray]) -> list[str]:
    return sorted(key for key in arrays if key.startswith("schedule_"))


def _load_donor_schedules(donor_dirs: Sequence[Path], shape: tuple[int, ...]) -> list[tuple[str, str, np.ndarray]]:
    out: list[tuple[str, str, np.ndarray]] = []
    for donor_dir in donor_dirs:
        path = Path(donor_dir) / "selector_arrays.npz"
        if not path.exists():
            continue
        arrays = _load_npz(path)
        for key in _schedule_keys(arrays):
            selected = np.asarray(arrays[key], dtype=np.int32)
            if selected.shape == shape:
                out.append((Path(donor_dir).name, key[len("schedule_") :], selected))
    return out


def _setup_total(selected: np.ndarray, transition: np.ndarray) -> int:
    selected = np.asarray(selected, dtype=np.int32)
    if selected.size <= 1:
        return 0
    return int(np.sum(transition[selected[:-1], selected[1:]], dtype=np.int64))


def _interval_setup(selected: np.ndarray, transition: np.ndarray, left: int, right: int) -> int:
    cost = 0
    for row in range(int(left), int(right) + 1):
        if row > 0:
            cost += int(transition[int(selected[row - 1]), int(selected[row])])
    if int(right) + 1 < selected.size:
        cost += int(transition[int(selected[right]), int(selected[right + 1])])
    return int(cost)


def _mixed_interval_setup(base: np.ndarray, donor: np.ndarray, transition: np.ndarray, left: int, right: int) -> int:
    cost = 0
    prev = int(base[left - 1]) if int(left) > 0 else int(donor[left])
    for row in range(int(left), int(right) + 1):
        if row > 0:
            cost += int(transition[prev, int(donor[row])])
        prev = int(donor[row])
    if int(right) + 1 < base.size:
        cost += int(transition[prev, int(base[right + 1])])
    return int(cost)


def _apply_splice_intervals(base: np.ndarray, intervals: Sequence[tuple[np.ndarray, int, int]]) -> np.ndarray:
    """Return a schedule with donor slices copied into ``base``.

    Pairwise setup deltas are not additive when two intervals touch or when the
    replacement changes the boundary action of another interval. Always rebuild
    the full schedule before declaring a candidate feasible.
    """

    selected = np.asarray(base, dtype=np.int32).copy()
    for donor, left, right in intervals:
        selected[int(left) : int(right) + 1] = np.asarray(donor, dtype=np.int32)[int(left) : int(right) + 1]
    return selected


def _score_total(score: np.ndarray, selected: np.ndarray) -> float:
    rows = np.arange(np.asarray(selected).size)
    return float(np.sum(np.asarray(score, dtype=np.float64)[rows, np.asarray(selected, dtype=np.int32)]))


def _score_gap(score: np.ndarray, selected: np.ndarray) -> float:
    rows = np.arange(np.asarray(selected).size)
    return float(np.mean(np.asarray(score, dtype=np.float64)[rows, np.asarray(selected, dtype=np.int32)]) - 1.0)


def _elementary_intervals(base: np.ndarray, donor: np.ndarray) -> list[tuple[int, int]]:
    boundaries = {0, int(base.size)}
    for selected in (base, donor):
        boundaries.update((np.flatnonzero(selected[1:] != selected[:-1]) + 1).astype(int).tolist())
    ordered = sorted(boundaries)
    out: list[tuple[int, int]] = []
    for idx in range(len(ordered) - 1):
        left = int(ordered[idx])
        right = int(ordered[idx + 1] - 1)
        if np.any(donor[left : right + 1] != base[left : right + 1]):
            out.append((left, right))
    return out


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search one- and two-interval schedule splices that reduce setup while preserving quality."
    )
    parser.add_argument("--base-selector-dir", type=Path, required=True)
    parser.add_argument("--base-schedule", type=str, required=True)
    parser.add_argument("--donor-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--score-kind", type=str, default="meanref")
    parser.add_argument("--quality-threshold", type=float, required=True)
    parser.add_argument("--target-setup", type=int, required=True)
    parser.add_argument("--max-merged-intervals", type=int, default=8)
    parser.add_argument("--max-quality-candidates", type=int, default=5000)
    parser.add_argument("--max-saving-candidates", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = _load_npz(Path(args.base_selector_dir) / "selector_arrays.npz")
    key = f"schedule_{args.base_schedule}"
    if key not in arrays:
        raise KeyError(f"{key!r} not found in base selector arrays")
    base = np.asarray(arrays[key], dtype=np.int32)
    steps = np.asarray(arrays["steps"], dtype=np.int64)
    transition = np.asarray(arrays["transition_counts"], dtype=np.int32)
    score = score_matrix_for_kind(arrays, score_kind=str(args.score_kind)).astype(np.float64)
    row_idx = np.arange(base.size)
    base_score = score[row_idx, base]
    base_gap = float(np.mean(base_score) - 1.0)
    base_setup = _setup_total(base, transition)
    slack_score_sum = float((1.0 + float(args.quality_threshold) - np.mean(base_score)) * base.size)

    donors = _load_donor_schedules(args.donor_dir, base.shape)
    donor_lookup = {(source_run, schedule): donor for source_run, schedule, donor in donors}
    intervals: list[dict[str, Any]] = []
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    for source_run, schedule, donor in donors:
        if np.array_equal(donor, base):
            continue
        donor_score = score[row_idx, donor]
        elementary = _elementary_intervals(base, donor)
        candidates: list[tuple[int, int]] = []
        max_merge = max(1, int(args.max_merged_intervals))
        for idx in range(len(elementary)):
            left = int(elementary[idx][0])
            for end_idx in range(idx, min(len(elementary), idx + max_merge)):
                candidates.append((left, int(elementary[end_idx][1])))
        for left, right in sorted(set(candidates)):
            old_setup = _interval_setup(base, transition, left, right)
            new_setup = _mixed_interval_setup(base, donor, transition, left, right)
            setup_delta = int(new_setup - old_setup)
            score_delta = float(np.sum(donor_score[left : right + 1] - base_score[left : right + 1]))
            if setup_delta >= 0 and score_delta >= 0 and score_delta > slack_score_sum:
                continue
            signature = (int(left), int(right), tuple(int(x) for x in donor[left : right + 1].tolist()))
            if signature in seen:
                continue
            seen.add(signature)
            intervals.append(
                {
                    "source_run": source_run,
                    "schedule": schedule,
                    "left_row": int(left),
                    "right_row": int(right),
                    "left_step": int(steps[left]),
                    "right_step": int(steps[right]),
                    "setup_delta": setup_delta,
                    "score_delta_sum": score_delta,
                    "old_interval_setup": int(old_setup),
                    "new_interval_setup": int(new_setup),
                    "new_total_setup_if_single": int(base_setup + setup_delta),
                    "new_gap_if_single": float(base_gap + score_delta / base.size),
                }
            )

    feasible: list[dict[str, Any]] = []
    for interval in intervals:
        donor = donor_lookup[(str(interval["source_run"]), str(interval["schedule"]))]
        selected = _apply_splice_intervals(
            base,
            [(donor, int(interval["left_row"]), int(interval["right_row"]))],
        )
        actual_setup = _setup_total(selected, transition)
        actual_gap = _score_gap(score, selected)
        if actual_setup <= int(args.target_setup) and actual_gap <= float(args.quality_threshold):
            feasible.append(
                {
                    "splice_type": "single",
                    **interval,
                    "new_total_setup": int(actual_setup),
                    "new_gap": float(actual_gap),
                    "score_delta_sum_actual": float(_score_total(score, selected) - np.sum(base_score)),
                }
            )

    saving = [item for item in intervals if int(item["setup_delta"]) < 0]
    quality = [item for item in intervals if float(item["score_delta_sum"]) <= slack_score_sum]
    saving = sorted(saving, key=lambda item: (int(item["setup_delta"]), float(item["score_delta_sum"])))[
        : max(0, int(args.max_saving_candidates))
    ]
    quality = sorted(quality, key=lambda item: (float(item["score_delta_sum"]), int(item["setup_delta"])))[
        : max(0, int(args.max_quality_candidates))
    ]
    checked_pairs = 0
    for first in saving:
        for second in quality:
            if first is second:
                continue
            if not (int(first["right_row"]) < int(second["left_row"]) or int(second["right_row"]) < int(first["left_row"])):
                continue
            first_donor = donor_lookup[(str(first["source_run"]), str(first["schedule"]))]
            second_donor = donor_lookup[(str(second["source_run"]), str(second["schedule"]))]
            selected = _apply_splice_intervals(
                base,
                [
                    (first_donor, int(first["left_row"]), int(first["right_row"])),
                    (second_donor, int(second["left_row"]), int(second["right_row"])),
                ],
            )
            new_setup = _setup_total(selected, transition)
            if new_setup > int(args.target_setup):
                continue
            score_delta = float(_score_total(score, selected) - np.sum(base_score))
            new_gap = _score_gap(score, selected)
            checked_pairs += 1
            if new_gap <= float(args.quality_threshold):
                feasible.append(
                    {
                        "splice_type": "pair",
                        "new_total_setup": int(new_setup),
                        "new_gap": new_gap,
                        "setup_delta": int(new_setup - base_setup),
                        "score_delta_sum": score_delta,
                        "first_source_run": first["source_run"],
                        "first_schedule": first["schedule"],
                        "first_left_row": first["left_row"],
                        "first_right_row": first["right_row"],
                        "first_left_step": first["left_step"],
                        "first_right_step": first["right_step"],
                        "first_setup_delta": first["setup_delta"],
                        "first_score_delta_sum": first["score_delta_sum"],
                        "second_source_run": second["source_run"],
                        "second_schedule": second["schedule"],
                        "second_left_row": second["left_row"],
                        "second_right_row": second["right_row"],
                        "second_left_step": second["left_step"],
                        "second_right_step": second["right_step"],
                        "second_setup_delta": second["setup_delta"],
                        "second_score_delta_sum": second["score_delta_sum"],
                    }
                )

    interval_rows = sorted(intervals, key=lambda item: (int(item["setup_delta"]), float(item["score_delta_sum"])))
    feasible_rows = sorted(
        feasible,
        key=lambda item: (
            int(item.get("new_total_setup", item.get("new_total_setup_if_single", 10**9))),
            float(item.get("new_gap", item.get("new_gap_if_single", 10**9))),
        ),
    )
    _write_rows(out_dir / "interval_candidates.csv", interval_rows)
    _write_rows(out_dir / "feasible_splices.csv", feasible_rows)
    payload = {
        "base_selector_dir": str(Path(args.base_selector_dir)),
        "base_schedule": str(args.base_schedule),
        "score_kind": str(args.score_kind),
        "quality_threshold": float(args.quality_threshold),
        "target_setup": int(args.target_setup),
        "base_gap": base_gap,
        "base_setup": int(base_setup),
        "slack_score_sum": slack_score_sum,
        "num_donor_schedules": int(len(donors)),
        "num_interval_candidates": int(len(interval_rows)),
        "num_saving_candidates_used": int(len(saving)),
        "num_quality_candidates_used": int(len(quality)),
        "checked_pairs": int(checked_pairs),
        "num_feasible_splices": int(len(feasible_rows)),
        "outputs": {
            "interval_candidates_csv": str(out_dir / "interval_candidates.csv"),
            "feasible_splices_csv": str(out_dir / "feasible_splices.csv"),
            "summary_json": str(out_dir / "splice_search_summary.json"),
        },
    }
    _write_json(out_dir / "splice_search_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
