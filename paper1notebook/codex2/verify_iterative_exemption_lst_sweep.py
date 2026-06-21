from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SWEEP_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_iterative_exemption_lst_sweep_056_061_china_europe"
    r"\t0_86160_stride60\switch36000_54000\releaseguard1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify generated LST sweep setup schedules against release/deadline and concurrency files."
    )
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--lst-start", type=int, default=10)
    parser.add_argument("--lst-end", type=int, default=140)
    parser.add_argument("--lst-step", type=int, default=10)
    return parser.parse_args()


def read_csv_maybe_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def fail(errors: list[str], msg: str) -> None:
    errors.append(msg)


def verify_one_lst(
    *,
    sweep_dir: Path,
    summary_row: pd.Series,
    lst: int,
    start: int,
    end: int,
    errors: list[str],
) -> dict[str, object]:
    lst_dir = sweep_dir / f"lst{int(lst):03d}"
    if not lst_dir.exists():
        fail(errors, f"LST {lst}: missing directory {lst_dir}")
        return {"lst": int(lst), "ok": False}

    jobs = read_csv_maybe_empty(lst_dir / "schedule_jobs.csv")
    schedule = read_csv_maybe_empty(lst_dir / "schedule_1s.csv")
    infeasible = read_csv_maybe_empty(lst_dir / "schedule_window_infeasible.csv")
    conflicts = read_csv_maybe_empty(lst_dir / "remaining_conflicts.csv")
    dynamic_conflicts = read_csv_maybe_empty(lst_dir / "remaining_dynamic_conflicts.csv")
    window_conflicts = read_csv_maybe_empty(lst_dir / "remaining_schedule_window_conflicts.csv")
    counts_path = lst_dir / "building_concurrency_1s.npy"

    if len(conflicts) != 0:
        fail(errors, f"LST {lst}: remaining_conflicts has {len(conflicts)} rows")
    if len(dynamic_conflicts) != 0:
        fail(errors, f"LST {lst}: remaining_dynamic_conflicts has {len(dynamic_conflicts)} rows")
    if len(window_conflicts) != 0:
        fail(errors, f"LST {lst}: remaining_schedule_window_conflicts has {len(window_conflicts)} rows")
    if len(infeasible) != int(summary_row["schedule_window_infeasible"]):
        fail(
            errors,
            f"LST {lst}: schedule_window_infeasible count mismatch "
            f"{len(infeasible)} != {summary_row['schedule_window_infeasible']}",
        )
    if len(jobs) != int(summary_row["schedule_jobs"]):
        fail(errors, f"LST {lst}: schedule_jobs count mismatch {len(jobs)} != {summary_row['schedule_jobs']}")
    if len(schedule) != int(summary_row["schedule_jobs"]):
        fail(errors, f"LST {lst}: schedule_1s count mismatch {len(schedule)} != {summary_row['schedule_jobs']}")

    if not counts_path.exists():
        fail(errors, f"LST {lst}: missing {counts_path}")
        counts = np.zeros(int(end) - int(start) + 1, dtype=np.int32)
    else:
        counts = np.load(counts_path)
    expected_len = int(end) - int(start) + 1
    if len(counts) != expected_len:
        fail(errors, f"LST {lst}: count length mismatch {len(counts)} != {expected_len}")

    recomputed = np.zeros(expected_len, dtype=np.int32)
    if not schedule.empty:
        required_cols = {"job_id", "release", "latest_start", "target_first_work_step", "plan_start", "plan_end"}
        missing = required_cols - set(schedule.columns)
        if missing:
            fail(errors, f"LST {lst}: schedule_1s missing columns {sorted(missing)}")
        else:
            for row in schedule.itertuples(index=False):
                plan_start = int(row.plan_start)
                plan_end = int(row.plan_end)
                release = int(row.release)
                latest_start = int(row.latest_start)
                deadline = int(row.target_first_work_step)
                if plan_start < release:
                    fail(errors, f"LST {lst}: job {row.job_id} starts before release")
                if plan_start > latest_start:
                    fail(errors, f"LST {lst}: job {row.job_id} starts after latest_start")
                if plan_end != plan_start + int(lst):
                    fail(errors, f"LST {lst}: job {row.job_id} duration is not {lst}s")
                if plan_end > deadline:
                    fail(errors, f"LST {lst}: job {row.job_id} ends after deadline")
                left = plan_start - int(start)
                right = plan_end - int(start)
                if left < 0 or right > len(recomputed):
                    fail(errors, f"LST {lst}: job {row.job_id} interval outside axis")
                else:
                    recomputed[left:right] += 1

    if counts.shape == recomputed.shape and not np.array_equal(counts, recomputed):
        fail(errors, f"LST {lst}: building_concurrency_1s.npy does not match schedule_1s.csv")

    peak = int(np.max(recomputed)) if recomputed.size else 0
    if peak != int(summary_row["smoothed_peak_concurrency"]):
        fail(errors, f"LST {lst}: peak mismatch {peak} != {summary_row['smoothed_peak_concurrency']}")
    if peak != int(summary_row["peak_concurrency"]):
        fail(errors, f"LST {lst}: peak_concurrency mismatch {peak} != {summary_row['peak_concurrency']}")

    total_work = int(len(schedule)) * int(lst)
    busy_seconds = int(np.count_nonzero(recomputed > 0))
    mean_full = float(total_work / max(1, expected_len))
    if abs(mean_full - float(summary_row["mean_concurrency_full_horizon"])) > 1e-9:
        fail(errors, f"LST {lst}: full-horizon mean mismatch")

    return {
        "lst": int(lst),
        "jobs": int(len(schedule)),
        "unscheduled_by_exemption": int(summary_row["iterative_unscheduled_links"]),
        "remaining_conflicts": int(len(conflicts)),
        "window_infeasible": int(len(infeasible)),
        "peak_concurrency": int(peak),
        "busy_seconds": int(busy_seconds),
        "mean_concurrency_full_horizon": float(mean_full),
        "lower_bound_matched": bool(summary_row["lower_bound_matched"]),
        "ok": True,
    }


def main() -> int:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    summary_path = sweep_dir / "lst_sweep_concurrency_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    expected_lsts = list(range(int(args.lst_start), int(args.lst_end) + 1, int(args.lst_step)))
    got_lsts = [int(x) for x in summary["lst"].tolist()]

    errors: list[str] = []
    if got_lsts != expected_lsts:
        fail(errors, f"LST list mismatch {got_lsts} != {expected_lsts}")

    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        rows.append(
            verify_one_lst(
                sweep_dir=sweep_dir,
                summary_row=row,
                lst=int(row["lst"]),
                start=int(args.start),
                end=int(args.end),
                errors=errors,
            )
        )

    report = {
        "sweep_dir": str(sweep_dir),
        "summary_csv": str(summary_path),
        "expected_lsts": expected_lsts,
        "verified_lsts": got_lsts,
        "errors": errors,
        "ok": not errors,
        "checks": [
            "LST values 10..140",
            "all remaining conflict CSVs are empty",
            "schedule_window_infeasible matches summary",
            "all scheduled jobs satisfy release <= start <= latest_start and end <= deadline",
            "all scheduled jobs have duration equal to LST",
            "building_concurrency_1s.npy matches schedule_1s.csv",
            "summary peak and mean metrics match recomputed values",
        ],
        "per_lst": rows,
    }
    out_path = sweep_dir / "verification_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
