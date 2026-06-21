from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


DEFAULT_SWITCH_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_switch_056_061_056_china_europe\t0_86160_stride60\switch36000_54000"
)
DEFAULT_SWEEP_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_lst_sweep_056_061_056_china_europe\t0_86160_stride60"
)


@dataclass(frozen=True)
class Job:
    job_id: int
    transition: str
    owner: int
    old_edge: str
    new_edge: str
    target_first_work_step: int
    old_last_work_step_before_deadline: int | None


@dataclass(frozen=True)
class WindowJob:
    job: Job
    earliest_idx: int
    latest_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MILP verification of minimum peak concurrency for usage-driven LST scheduling."
    )
    parser.add_argument("--switch-root", type=Path, default=DEFAULT_SWITCH_ROOT)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--modes", nargs="+", default=["delay", "any"], choices=["delay", "any", "hop"])
    parser.add_argument("--lst-start", type=int, default=10)
    parser.add_argument("--lst-end", type=int, default=140)
    parser.add_argument("--lst-step", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--time-limit", type=float, default=60.0)
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


def load_jobs(path: Path) -> list[Job]:
    df = pd.read_csv(path)
    jobs: list[Job] = []
    for idx, row in enumerate(df.itertuples(index=False), 1):
        jobs.append(
            Job(
                job_id=int(getattr(row, "job_id", idx)),
                transition=str(row.transition),
                owner=int(row.owner),
                old_edge="" if pd.isna(row.old_edge) else str(row.old_edge),
                new_edge="" if pd.isna(row.new_edge) else str(row.new_edge),
                target_first_work_step=int(row.target_first_work_step),
                old_last_work_step_before_deadline=parse_int_maybe(row.old_last_work_step_before_deadline),
            )
        )
    return jobs


def window_jobs(
    jobs: list[Job],
    *,
    lst: int,
    start: int,
    stride: int,
) -> tuple[list[WindowJob], list[Job]]:
    feasible: list[WindowJob] = []
    infeasible: list[Job] = []
    for job in jobs:
        earliest = int(start)
        if job.old_last_work_step_before_deadline is not None:
            earliest = step_ceil(job.old_last_work_step_before_deadline + int(stride), step=int(stride))
        latest = step_floor(job.target_first_work_step - int(lst), step=int(stride))
        if earliest > latest:
            infeasible.append(job)
            continue
        feasible.append(
            WindowJob(
                job=job,
                earliest_idx=int((earliest - int(start)) // int(stride)),
                latest_idx=int((latest - int(start)) // int(stride)),
            )
        )
    return feasible, infeasible


def solve_cap(
    feasible_jobs: list[WindowJob],
    *,
    cap: int,
    duration_slots: int,
    num_time_slots: int,
    time_limit: float,
) -> tuple[bool, object, list[tuple[int, int, int]]]:
    var_job: list[int] = []
    var_start: list[int] = []
    for local_job, item in enumerate(feasible_jobs):
        for start_idx in range(int(item.earliest_idx), int(item.latest_idx) + 1):
            var_job.append(int(local_job))
            var_start.append(int(start_idx))
    num_vars = len(var_job)
    if num_vars == 0:
        return True, None, []

    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []

    for var_idx, local_job in enumerate(var_job):
        row_idx.append(int(local_job))
        col_idx.append(int(var_idx))
        data.append(1.0)

    for var_idx, start_idx in enumerate(var_start):
        for offset in range(int(duration_slots)):
            t = int(start_idx) + int(offset)
            if 0 <= t < int(num_time_slots):
                row_idx.append(len(feasible_jobs) + t)
                col_idx.append(int(var_idx))
                data.append(1.0)

    num_rows = len(feasible_jobs) + int(num_time_slots)
    matrix = coo_matrix((data, (row_idx, col_idx)), shape=(num_rows, num_vars)).tocsr()
    lower = np.full(num_rows, -np.inf, dtype=np.float64)
    upper = np.full(num_rows, float(cap), dtype=np.float64)
    lower[: len(feasible_jobs)] = 1.0
    upper[: len(feasible_jobs)] = 1.0

    result = milp(
        c=np.zeros(num_vars, dtype=np.float64),
        integrality=np.ones(num_vars, dtype=np.int8),
        bounds=Bounds(np.zeros(num_vars), np.ones(num_vars)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": float(time_limit), "mip_rel_gap": 0.0},
    )
    if not result.success:
        return False, result, []

    selected = np.flatnonzero(np.asarray(result.x) > 0.5)
    starts = [(int(var_job[var_idx]), int(var_start[var_idx]), int(var_idx)) for var_idx in selected]
    return True, result, starts


def write_schedule(path: Path, feasible_jobs: list[WindowJob], selected: list[tuple[int, int, int]], *, start: int, stride: int, lst: int) -> None:
    selected_by_job = {job_idx: start_idx for job_idx, start_idx, _ in selected}
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "job_id",
                "transition",
                "owner",
                "old_edge",
                "new_edge",
                "target_first_work_step",
                "old_last_work_step_before_deadline",
                "earliest_start",
                "latest_start",
                "plan_start",
                "plan_end",
            ]
        )
        for local_idx, item in enumerate(feasible_jobs):
            job = item.job
            start_idx = selected_by_job.get(local_idx)
            writer.writerow(
                [
                    int(job.job_id),
                    job.transition,
                    int(job.owner),
                    job.old_edge,
                    job.new_edge,
                    int(job.target_first_work_step),
                    "" if job.old_last_work_step_before_deadline is None else int(job.old_last_work_step_before_deadline),
                    int(start) + int(item.earliest_idx) * int(stride),
                    int(start) + int(item.latest_idx) * int(stride),
                    "" if start_idx is None else int(start) + int(start_idx) * int(stride),
                    "" if start_idx is None else int(start) + int(start_idx) * int(stride) + int(lst),
                ]
            )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.sweep_root) / "milp_min_peak_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    num_time_slots = int((int(args.end) - int(args.start)) // int(args.stride)) + 1
    rows: list[dict[str, object]] = []

    for mode in args.modes:
        plan_csv = Path(args.switch_root) / f"setup600_{mode}" / "usage_driven_right_link_switch_plan_required.csv"
        old_summary_csv = Path(args.sweep_root) / f"{mode}_after-old-last-work_exact-cap" / "lst_sweep_summary.csv"
        old_summary = pd.read_csv(old_summary_csv).set_index("lst")
        jobs = load_jobs(plan_csv)
        mode_dir = out_dir / str(mode)
        mode_dir.mkdir(parents=True, exist_ok=True)

        for lst in range(int(args.lst_start), int(args.lst_end) + 1, int(args.lst_step)):
            duration_slots = max(1, int(math.ceil(float(lst) / float(args.stride))))
            feasible, infeasible = window_jobs(jobs, lst=int(lst), start=int(args.start), stride=int(args.stride))
            old_peak = int(old_summary.loc[int(lst), "peak_concurrency"])
            best_cap = None
            best_selected: list[tuple[int, int, int]] = []
            status_parts: list[str] = []
            for cap in range(1, max(1, old_peak) + 1):
                ok, result, selected = solve_cap(
                    feasible,
                    cap=int(cap),
                    duration_slots=int(duration_slots),
                    num_time_slots=int(num_time_slots),
                    time_limit=float(args.time_limit),
                )
                message = "empty" if result is None else str(result.message)
                status_parts.append(f"cap{cap}:{'ok' if ok else 'no'}")
                if ok:
                    best_cap = int(cap)
                    best_selected = selected
                    break
                if result is not None and "Time limit" in message:
                    status_parts[-1] += ":time_limit"
                    break

            proof_status = "optimal_proved" if best_cap is not None else "not_proved"
            if best_cap is not None:
                schedule_dir = mode_dir / f"lst{int(lst):03d}"
                schedule_dir.mkdir(parents=True, exist_ok=True)
                write_schedule(
                    schedule_dir / "milp_schedule.csv",
                    feasible,
                    best_selected,
                    start=int(args.start),
                    stride=int(args.stride),
                    lst=int(lst),
                )

            rows.append(
                {
                    "mode": str(mode),
                    "lst": int(lst),
                    "duration_slots": int(duration_slots),
                    "jobs": int(len(jobs)),
                    "feasible_jobs": int(len(feasible)),
                    "infeasible_window_jobs": int(len(infeasible)),
                    "old_exact_cap_peak": int(old_peak),
                    "milp_min_peak": "" if best_cap is None else int(best_cap),
                    "proof_status": proof_status,
                    "cap_checks": ";".join(status_parts),
                }
            )
            print(rows[-1])

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "milp_min_peak_summary.csv", index=False, encoding="utf-8-sig")
    print(f"out_dir={out_dir}")
    print(result_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
