from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_SWITCH_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_switch_056_061_056_china_europe\t0_86160_stride60\switch36000_54000"
)
DEFAULT_SWEEP_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_lst_sweep_056_061_056_china_europe\t0_86160_stride60"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute interval-density lower bounds for LST peak concurrency."
    )
    parser.add_argument("--switch-root", type=Path, default=DEFAULT_SWITCH_ROOT)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--lst-start", type=int, default=10)
    parser.add_argument("--lst-end", type=int, default=140)
    parser.add_argument("--lst-step", type=int, default=10)
    return parser.parse_args()


def parse_int_maybe(value: object) -> int | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return int(float(text))


def step_floor(value: int, *, step: int) -> int:
    return int(math.floor(int(value) / int(step)) * int(step))


def step_ceil(value: int, *, step: int) -> int:
    return int(math.ceil(int(value) / int(step)) * int(step))


def load_required_plan(root: Path, mode: str) -> pd.DataFrame:
    path = Path(root) / f"setup600_{mode}" / "usage_driven_right_link_switch_plan_required.csv"
    return pd.read_csv(path)


def load_peak_summary(root: Path, mode: str) -> pd.DataFrame:
    path = Path(root) / f"{mode}_after-old-last-work_exact-cap" / "lst_sweep_summary.csv"
    return pd.read_csv(path).set_index("lst")


def feasible_windows(df: pd.DataFrame, *, lst: int, start: int, stride: int) -> list[tuple[int, int]]:
    duration_slots = max(1, int(math.ceil(float(lst) / float(stride))))
    windows: list[tuple[int, int]] = []
    for row in df.itertuples(index=False):
        target = int(row.target_first_work_step)
        old_last = parse_int_maybe(row.old_last_work_step_before_deadline)
        earliest = int(start) if old_last is None else step_ceil(old_last + int(stride), step=int(stride))
        latest = step_floor(target - int(lst), step=int(stride))
        if earliest > latest:
            continue
        earliest_idx = int((earliest - int(start)) // int(stride))
        latest_idx = int((latest - int(start)) // int(stride))
        windows.append((earliest_idx, latest_idx + duration_slots))
    return windows


def density_lower_bound(windows: list[tuple[int, int]], *, duration_slots: int) -> tuple[int, int, int, int]:
    if not windows:
        return 0, 0, 0, 0
    starts = sorted({w[0] for w in windows})
    completions = sorted({w[1] for w in windows})
    best_lb = 1
    best_start = starts[0]
    best_end = completions[-1]
    best_jobs = 0
    for interval_start in starts:
        candidates = sorted(completion for start, completion in windows if start >= interval_start)
        if not candidates:
            continue
        cursor = 0
        for interval_end in completions:
            if interval_end <= interval_start:
                continue
            while cursor < len(candidates) and candidates[cursor] <= interval_end:
                cursor += 1
            if cursor == 0:
                continue
            interval_len = int(interval_end) - int(interval_start)
            lb = int(math.ceil(cursor * int(duration_slots) / interval_len))
            if lb > best_lb or (lb == best_lb and best_jobs == 0):
                best_lb = lb
                best_start = int(interval_start)
                best_end = int(interval_end)
                best_jobs = int(cursor)
    return int(best_lb), int(best_start), int(best_end), int(best_jobs)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.sweep_root) / "peak_lower_bounds"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for mode in ("delay", "any"):
        required = load_required_plan(Path(args.switch_root), mode)
        peak_summary = load_peak_summary(Path(args.sweep_root), mode)
        for lst in range(int(args.lst_start), int(args.lst_end) + 1, int(args.lst_step)):
            duration_slots = max(1, int(math.ceil(float(lst) / float(args.stride))))
            windows = feasible_windows(required, lst=int(lst), start=int(args.start), stride=int(args.stride))
            lb, lb_start_idx, lb_end_idx, lb_jobs = density_lower_bound(windows, duration_slots=duration_slots)
            peak = int(peak_summary.loc[int(lst), "peak_concurrency"])
            rows.append(
                {
                    "mode": mode,
                    "lst": int(lst),
                    "duration_slots": int(duration_slots),
                    "schedulable_windows": int(len(windows)),
                    "density_lower_bound": int(lb),
                    "scheduled_peak": int(peak),
                    "proven_optimal_by_density": bool(int(lb) == int(peak)),
                    "critical_interval_start": int(args.start) + int(lb_start_idx) * int(args.stride),
                    "critical_interval_end": int(args.start) + int(lb_end_idx) * int(args.stride),
                    "critical_interval_jobs": int(lb_jobs),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "lst_peak_lower_bounds.csv", index=False, encoding="utf-8-sig")
    print(f"out_dir={out_dir}")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
