from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DEFAULT_PLAN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_switch_056_061_056_china_europe\t0_86160_stride60"
    r"\switch36000_54000\setup600_any"
)
DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_lst_sweep_056_061_056_china_europe"
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
    first_work_value: float
    old_last_work_value: float


@dataclass(frozen=True)
class ScheduledJob:
    job: Job
    start: int
    end: int
    earliest_start: int
    latest_start: int
    feasible_window: bool
    conflict_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep link setup time (LST) for usage-driven right-link switches and "
            "compute a concurrency-smoothed schedule."
        )
    )
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--lst-start", type=int, default=10)
    parser.add_argument("--lst-end", type=int, default=140)
    parser.add_argument("--lst-step", type=int, default=10)
    parser.add_argument("--working-mode", type=str, default="any")
    parser.add_argument("--scheduler", choices=("greedy", "exact-cap"), default="greedy")
    parser.add_argument(
        "--release-policy",
        choices=("after-old-last-work", "allow-overlap"),
        default="after-old-last-work",
        help=(
            "after-old-last-work forbids tearing down the old right link while it is working. "
            "allow-overlap uses only target first-use deadlines and records overlap conflicts separately."
        ),
    )
    return parser.parse_args()


def parse_int_maybe(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return int(float(text))


def load_jobs(plan_csv: Path) -> list[Job]:
    df = pd.read_csv(plan_csv)
    df = df[df["required_by_working"].astype(bool)].copy()
    jobs: list[Job] = []
    for job_id, row in enumerate(df.itertuples(index=False), 1):
        jobs.append(
            Job(
                job_id=int(job_id),
                transition=str(row.transition),
                owner=int(row.owner),
                old_edge=str(row.old_edge) if not pd.isna(row.old_edge) else "",
                new_edge=str(row.new_edge) if not pd.isna(row.new_edge) else "",
                target_first_work_step=int(row.target_first_work_step),
                old_last_work_step_before_deadline=parse_int_maybe(row.old_last_work_step_before_deadline),
                first_work_value=float(row.target_first_work_value),
                old_last_work_value=float(row.old_last_work_value),
            )
        )
    return jobs


def step_floor(value: int, *, step: int) -> int:
    return int(math.floor(int(value) / int(step)) * int(step))


def step_ceil(value: int, *, step: int) -> int:
    return int(math.ceil(int(value) / int(step)) * int(step))


def job_window(
    job: Job,
    *,
    lst: int,
    stride: int,
    global_start: int,
    release_policy: str,
) -> tuple[int, int, bool, str]:
    latest_start = step_floor(int(job.target_first_work_step) - int(lst), step=int(stride))
    if str(release_policy) == "allow-overlap":
        earliest_start = int(global_start)
    else:
        if job.old_last_work_step_before_deadline is None:
            earliest_start = int(global_start)
        else:
            earliest_start = step_ceil(int(job.old_last_work_step_before_deadline) + int(stride), step=int(stride))
    feasible = earliest_start <= latest_start
    reason = "" if feasible else "old_link_working_until_after_latest_start"
    return int(earliest_start), int(latest_start), bool(feasible), reason


def peak_concurrency_for_starts(
    *,
    starts: dict[int, int],
    duration: int,
    time_axis: np.ndarray,
) -> tuple[np.ndarray, int]:
    counts = np.zeros(len(time_axis), dtype=np.int32)
    for start in starts.values():
        end = int(start) + int(duration)
        rows = np.flatnonzero((time_axis >= int(start)) & (time_axis < int(end)))
        counts[rows] += 1
    return counts, int(np.max(counts)) if counts.size else 0


def can_schedule_with_cap(
    jobs: list[Job],
    *,
    lst: int,
    stride: int,
    global_start: int,
    global_end: int,
    cap: int,
    release_policy: str,
) -> tuple[bool, list[ScheduledJob], np.ndarray, list[dict[str, object]]]:
    time_axis = np.arange(int(global_start), int(global_end) + 1, int(stride), dtype=np.int64)
    counts = np.zeros(len(time_axis), dtype=np.int32)
    scheduled: list[ScheduledJob] = []
    unscheduled: list[dict[str, object]] = []

    job_records: list[tuple[int, int, Job, bool, str]] = []
    for job in jobs:
        earliest, latest, feasible, reason = job_window(
            job,
            lst=int(lst),
            stride=int(stride),
            global_start=int(global_start),
            release_policy=str(release_policy),
        )
        if not feasible:
            unscheduled.append(
                {
                    "job_id": int(job.job_id),
                    "transition": job.transition,
                    "owner": int(job.owner),
                    "old_edge": job.old_edge,
                    "new_edge": job.new_edge,
                    "target_first_work_step": int(job.target_first_work_step),
                    "old_last_work_step_before_deadline": "" if job.old_last_work_step_before_deadline is None else int(job.old_last_work_step_before_deadline),
                    "earliest_start": int(earliest),
                    "latest_start": int(latest),
                    "reason": reason,
                }
            )
            continue
        job_records.append((int(latest), int(earliest), job, feasible, reason))

    # Earliest-deadline first; within a job window choose the lowest-load feasible slot.
    for latest, earliest, job, feasible, reason in sorted(job_records, key=lambda item: (item[0], item[1], item[2].owner)):
        candidates = list(range(int(earliest), int(latest) + 1, int(stride)))
        best: tuple[int, int, int] | None = None
        best_rows: np.ndarray | None = None
        for start in candidates:
            end = int(start) + int(lst)
            rows = np.flatnonzero((time_axis >= int(start)) & (time_axis < int(end)))
            if rows.size == 0:
                continue
            if int(np.max(counts[rows])) + 1 > int(cap):
                continue
            score_peak = int(np.max(counts[rows]))
            score_sum = int(np.sum(counts[rows]))
            candidate = (score_peak, score_sum, int(start))
            if best is None or candidate < best:
                best = candidate
                best_rows = rows
        if best is None or best_rows is None:
            unscheduled.append(
                {
                    "job_id": int(job.job_id),
                    "transition": job.transition,
                    "owner": int(job.owner),
                    "old_edge": job.old_edge,
                    "new_edge": job.new_edge,
                    "target_first_work_step": int(job.target_first_work_step),
                    "old_last_work_step_before_deadline": "" if job.old_last_work_step_before_deadline is None else int(job.old_last_work_step_before_deadline),
                    "earliest_start": int(earliest),
                    "latest_start": int(latest),
                    "reason": f"no_slot_under_cap_{int(cap)}",
                }
            )
            continue
        start = int(best[2])
        counts[best_rows] += 1
        scheduled.append(
            ScheduledJob(
                job=job,
                start=start,
                end=start + int(lst),
                earliest_start=int(earliest),
                latest_start=int(latest),
                feasible_window=bool(feasible),
                conflict_reason=str(reason),
            )
        )
    return len(unscheduled) == 0, scheduled, counts, unscheduled


def find_min_cap(
    jobs: list[Job],
    *,
    lst: int,
    stride: int,
    global_start: int,
    global_end: int,
    release_policy: str,
) -> tuple[int, list[ScheduledJob], np.ndarray, list[dict[str, object]]]:
    feasible_jobs: list[Job] = []
    infeasible_unscheduled: list[dict[str, object]] = []
    for job in jobs:
        earliest, latest, feasible, reason = job_window(
            job,
            lst=int(lst),
            stride=int(stride),
            global_start=int(global_start),
            release_policy=str(release_policy),
        )
        if feasible:
            feasible_jobs.append(job)
        else:
            infeasible_unscheduled.append(
                {
                    "job_id": int(job.job_id),
                    "transition": job.transition,
                    "owner": int(job.owner),
                    "old_edge": job.old_edge,
                    "new_edge": job.new_edge,
                    "target_first_work_step": int(job.target_first_work_step),
                    "old_last_work_step_before_deadline": "" if job.old_last_work_step_before_deadline is None else int(job.old_last_work_step_before_deadline),
                    "earliest_start": int(earliest),
                    "latest_start": int(latest),
                    "reason": reason,
                }
            )
    time_axis = np.arange(int(global_start), int(global_end) + 1, int(stride), dtype=np.int64)
    if not feasible_jobs:
        return 0, [], np.zeros(len(time_axis), dtype=np.int32), infeasible_unscheduled

    greedy_peak, greedy_scheduled, greedy_counts, greedy_unscheduled = schedule_lowest_load(
        feasible_jobs,
        lst=int(lst),
        stride=int(stride),
        global_start=int(global_start),
        global_end=int(global_end),
        release_policy=str(release_policy),
    )
    low = 1
    high = max(1, int(greedy_peak) - 1)
    best: tuple[int, list[ScheduledJob], np.ndarray, list[dict[str, object]]] | None = (
        int(greedy_peak),
        greedy_scheduled,
        greedy_counts,
        infeasible_unscheduled + greedy_unscheduled,
    )
    while low <= high:
        mid = (low + high) // 2
        ok, scheduled, counts, unscheduled = can_schedule_with_cap(
            feasible_jobs,
            lst=int(lst),
            stride=int(stride),
            global_start=int(global_start),
            global_end=int(global_end),
            cap=int(mid),
            release_policy=str(release_policy),
        )
        if ok:
            best = (int(mid), scheduled, counts, infeasible_unscheduled + unscheduled)
            high = mid - 1
        else:
            low = mid + 1
    if best is None:
        ok, scheduled, counts, unscheduled = can_schedule_with_cap(
            feasible_jobs,
            lst=int(lst),
            stride=int(stride),
            global_start=int(global_start),
            global_end=int(global_end),
            cap=len(feasible_jobs),
            release_policy=str(release_policy),
        )
        return len(feasible_jobs), scheduled, counts, infeasible_unscheduled + unscheduled
    return best


def schedule_lowest_load(
    jobs: list[Job],
    *,
    lst: int,
    stride: int,
    global_start: int,
    global_end: int,
    release_policy: str,
) -> tuple[int, list[ScheduledJob], np.ndarray, list[dict[str, object]]]:
    time_axis = np.arange(int(global_start), int(global_end) + 1, int(stride), dtype=np.int64)
    counts = np.zeros(len(time_axis), dtype=np.int32)
    scheduled: list[ScheduledJob] = []
    unscheduled: list[dict[str, object]] = []
    duration_slots = max(1, int(math.ceil(float(lst) / float(stride))))

    job_records: list[tuple[int, int, Job, bool, str]] = []
    for job in jobs:
        earliest, latest, feasible, reason = job_window(
            job,
            lst=int(lst),
            stride=int(stride),
            global_start=int(global_start),
            release_policy=str(release_policy),
        )
        if not feasible:
            unscheduled.append(
                {
                    "job_id": int(job.job_id),
                    "transition": job.transition,
                    "owner": int(job.owner),
                    "old_edge": job.old_edge,
                    "new_edge": job.new_edge,
                    "target_first_work_step": int(job.target_first_work_step),
                    "old_last_work_step_before_deadline": "" if job.old_last_work_step_before_deadline is None else int(job.old_last_work_step_before_deadline),
                    "earliest_start": int(earliest),
                    "latest_start": int(latest),
                    "reason": reason,
                }
            )
            continue
        job_records.append((int(latest), int(earliest), job, feasible, reason))

    for latest, earliest, job, feasible, reason in sorted(job_records, key=lambda item: (item[0], item[1], item[2].owner)):
        first_idx = max(0, int((int(earliest) - int(global_start)) // int(stride)))
        last_idx = min(
            len(time_axis) - duration_slots,
            int((int(latest) - int(global_start)) // int(stride)),
        )
        if first_idx > last_idx:
            unscheduled.append(
                {
                    "job_id": int(job.job_id),
                    "transition": job.transition,
                    "owner": int(job.owner),
                    "old_edge": job.old_edge,
                    "new_edge": job.new_edge,
                    "target_first_work_step": int(job.target_first_work_step),
                    "old_last_work_step_before_deadline": "" if job.old_last_work_step_before_deadline is None else int(job.old_last_work_step_before_deadline),
                    "earliest_start": int(earliest),
                    "latest_start": int(latest),
                    "reason": "no_discrete_slot_in_window",
                }
            )
            continue

        candidate_indices = np.arange(first_idx, last_idx + 1, dtype=np.int32)
        window_counts = np.vstack([counts[candidate_indices + offset] for offset in range(duration_slots)])
        peak_scores = np.max(window_counts, axis=0)
        sum_scores = np.sum(window_counts, axis=0)
        # Lexicographic: minimize local peak, then total load, then use the earliest start.
        order = np.lexsort((candidate_indices, sum_scores, peak_scores))
        chosen_idx = int(candidate_indices[int(order[0])])
        counts[chosen_idx:chosen_idx + duration_slots] += 1
        start = int(time_axis[chosen_idx])
        scheduled.append(
            ScheduledJob(
                job=job,
                start=start,
                end=start + int(lst),
                earliest_start=int(earliest),
                latest_start=int(latest),
                feasible_window=bool(feasible),
                conflict_reason=str(reason),
            )
        )

    peak = int(np.max(counts)) if counts.size else 0
    return peak, scheduled, counts, unscheduled


def write_schedule(path: Path, scheduled: list[ScheduledJob], unscheduled: list[dict[str, object]]) -> None:
    fieldnames = [
        "status",
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
        "reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in scheduled:
            old_overlap = (
                item.job.old_last_work_step_before_deadline is not None
                and int(item.start) <= int(item.job.old_last_work_step_before_deadline)
            )
            writer.writerow(
                {
                    "status": "scheduled",
                    "job_id": int(item.job.job_id),
                    "transition": item.job.transition,
                    "owner": int(item.job.owner),
                    "old_edge": item.job.old_edge,
                    "new_edge": item.job.new_edge,
                    "target_first_work_step": int(item.job.target_first_work_step),
                    "old_last_work_step_before_deadline": "" if item.job.old_last_work_step_before_deadline is None else int(item.job.old_last_work_step_before_deadline),
                    "earliest_start": int(item.earliest_start),
                    "latest_start": int(item.latest_start),
                    "plan_start": int(item.start),
                    "plan_end": int(item.end),
                    "reason": "old_link_working_after_plan_start" if old_overlap else item.conflict_reason,
                }
            )
        for row in unscheduled:
            writer.writerow(
                {
                    "status": "unscheduled",
                    "job_id": int(row["job_id"]),
                    "transition": row["transition"],
                    "owner": int(row["owner"]),
                    "old_edge": row["old_edge"],
                    "new_edge": row["new_edge"],
                    "target_first_work_step": int(row["target_first_work_step"]),
                    "old_last_work_step_before_deadline": row["old_last_work_step_before_deadline"],
                    "earliest_start": int(row["earliest_start"]),
                    "latest_start": int(row["latest_start"]),
                    "plan_start": "",
                    "plan_end": "",
                    "reason": row["reason"],
                }
            )


def plot_sweep(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.4), dpi=170, sharex=True)
    x = summary["lst"].to_numpy(dtype=np.float64)
    axes[0].plot(x, summary["peak_concurrency"], marker="o", color="#b91c1c", linewidth=1.6)
    axes[0].set_ylabel("peak concurrency")
    axes[0].grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    axes[0].set_title("Usage-driven LST sweep")
    axes[1].plot(x, summary["infeasible_window_jobs"], marker="o", color="#7c2d12", linewidth=1.4, label="window conflicts")
    if "old_work_overlap_jobs" in summary.columns:
        axes[1].plot(x, summary["old_work_overlap_jobs"], marker="^", color="#be123c", linewidth=1.2, label="old-work overlap")
    axes[1].plot(x, summary["unscheduled_at_min_cap"], marker="s", color="#1d4ed8", linewidth=1.2, label="unscheduled at min cap")
    axes[1].set_ylabel("jobs")
    axes[1].set_xlabel("link setup time (s)")
    axes[1].grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_concurrency(counts: np.ndarray, *, start: int, stride: int, out_path: Path, title: str) -> None:
    steps = np.arange(int(start), int(start) + len(counts) * int(stride), int(stride), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(14.2, 4.7), dpi=170)
    ax.plot(steps / 3600.0, counts, color="#b91c1c", linewidth=1.35)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("building links")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    plan_csv = Path(args.plan_dir) / "usage_driven_right_link_switch_plan_required.csv"
    jobs = load_jobs(plan_csv)
    lst_values = list(range(int(args.lst_start), int(args.lst_end) + 1, int(args.lst_step)))
    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT
        / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
        / f"{args.working_mode}_{args.release_policy}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    representative_counts: dict[int, np.ndarray] = {}
    for lst in lst_values:
        if str(args.scheduler) == "exact-cap":
            peak_cap, scheduled, counts, unscheduled = find_min_cap(
                jobs,
                lst=int(lst),
                stride=int(args.stride),
                global_start=int(args.start),
                global_end=int(args.end),
                release_policy=str(args.release_policy),
            )
        else:
            peak_cap, scheduled, counts, unscheduled = schedule_lowest_load(
                jobs,
                lst=int(lst),
                stride=int(args.stride),
                global_start=int(args.start),
                global_end=int(args.end),
                release_policy=str(args.release_policy),
            )
        window_conflicts = [
            job
            for job in jobs
            if not job_window(
                job,
                lst=int(lst),
                stride=int(args.stride),
                global_start=int(args.start),
                release_policy=str(args.release_policy),
            )[2]
        ]
        lst_dir = out_dir / f"lst{int(lst):03d}"
        lst_dir.mkdir(parents=True, exist_ok=True)
        write_schedule(lst_dir / "schedule.csv", scheduled, unscheduled)
        old_work_overlap_jobs = sum(
            1
            for item in scheduled
            if item.job.old_last_work_step_before_deadline is not None
            and int(item.start) <= int(item.job.old_last_work_step_before_deadline)
        )
        np.save(lst_dir / "building_concurrency.npy", counts)
        plot_concurrency(
            counts,
            start=int(args.start),
            stride=int(args.stride),
            out_path=lst_dir / "building_concurrency.png",
            title=f"LST={int(lst)}s, {args.scheduler} peak concurrency={int(peak_cap)}",
        )
        representative_counts[int(lst)] = counts
        rows.append(
            {
                "lst": int(lst),
                "jobs": int(len(jobs)),
                "scheduled_at_min_cap": int(len(scheduled)),
                "unscheduled_at_min_cap": int(len(unscheduled)),
                "infeasible_window_jobs": int(len(window_conflicts)),
                "old_work_overlap_jobs": int(old_work_overlap_jobs),
                "peak_concurrency": int(peak_cap),
                "scheduler": str(args.scheduler),
                "observed_peak_concurrency": int(np.max(counts)) if counts.size else 0,
                "mean_concurrency": float(np.mean(counts)) if counts.size else 0.0,
                "nonzero_steps": int(np.count_nonzero(counts)) if counts.size else 0,
                "release_policy": str(args.release_policy),
                "working_mode": str(args.working_mode),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "lst_sweep_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_sweep(summary, out_dir / "lst_sweep_summary.png")
    meta = {
        "plan_csv": str(plan_csv),
        "out_dir": str(out_dir),
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "lst_values": lst_values,
        "working_mode": str(args.working_mode),
        "release_policy": str(args.release_policy),
        "scheduler": str(args.scheduler),
        "num_jobs": int(len(jobs)),
        "outputs": {
            "summary": "lst_sweep_summary.csv",
            "summary_plot": "lst_sweep_summary.png",
            "per_lst_schedule": "lstXXX/schedule.csv",
            "per_lst_concurrency": "lstXXX/building_concurrency.png",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
