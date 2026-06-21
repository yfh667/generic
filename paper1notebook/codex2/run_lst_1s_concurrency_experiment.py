from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DEFAULT_SWITCH_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_switch_056_061_056_china_europe\t0_86160_stride60\switch36000_54000"
)
DEFAULT_OUT_ROOT = Path(
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
    release: int
    latest_start: int
    deadline: int
    old_last_work_step_before_deadline: int | None


@dataclass(frozen=True)
class ScheduledJob:
    job: Job
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun usage-driven LST scheduling with one-second setup granularity. "
            "The edge-usage deadlines are still read from the existing sampled betweenness plan."
        )
    )
    parser.add_argument("--switch-root", type=Path, default=DEFAULT_SWITCH_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--modes", nargs="+", default=["delay", "any"], choices=["delay", "any", "hop"])
    parser.add_argument("--lst-start", type=int, default=10)
    parser.add_argument("--lst-end", type=int, default=140)
    parser.add_argument("--lst-step", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--schedule-resolution", type=int, default=1)
    parser.add_argument(
        "--release-guard-seconds",
        type=int,
        default=60,
        help=(
            "Because the current edge-usage deadlines are sampled every 60s, keep the old working link "
            "reserved until old_last_work_step + this guard. Use 1 only after rebuilding edge usage at 1s."
        ),
    )
    return parser.parse_args()


def parse_int_maybe(value: object) -> int | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return int(float(text))


def load_jobs(
    path: Path,
    *,
    lst: int,
    start: int,
    release_guard_seconds: int,
) -> tuple[list[Job], list[Job]]:
    df = pd.read_csv(path)
    jobs: list[Job] = []
    infeasible: list[Job] = []
    for idx, row in enumerate(df.itertuples(index=False), 1):
        old_last = parse_int_maybe(row.old_last_work_step_before_deadline)
        release = int(start) if old_last is None else int(old_last) + int(release_guard_seconds)
        deadline = int(row.target_first_work_step)
        latest_start = int(deadline) - int(lst)
        job = Job(
            job_id=int(getattr(row, "job_id", idx)),
            transition=str(row.transition),
            owner=int(row.owner),
            old_edge="" if pd.isna(row.old_edge) else str(row.old_edge),
            new_edge="" if pd.isna(row.new_edge) else str(row.new_edge),
            release=int(release),
            latest_start=int(latest_start),
            deadline=int(deadline),
            old_last_work_step_before_deadline=old_last,
        )
        if release > latest_start:
            infeasible.append(job)
        else:
            jobs.append(job)
    return jobs, infeasible


def add_interval(counts: np.ndarray, *, start: int, duration: int, axis_start: int) -> None:
    left = int(start) - int(axis_start)
    right = left + int(duration)
    counts[left:right] += 1


def naive_schedule_peak(jobs: list[Job], *, lst: int, axis_start: int, axis_end: int) -> int:
    counts = np.zeros(int(axis_end) - int(axis_start) + 1, dtype=np.int32)
    for job in jobs:
        add_interval(counts, start=int(job.latest_start), duration=int(lst), axis_start=int(axis_start))
    return int(np.max(counts)) if counts.size else 0


def density_lower_bound(jobs: list[Job], *, lst: int) -> tuple[int, int, int, int]:
    if not jobs:
        return 0, 0, 0, 0
    releases = sorted({int(job.release) for job in jobs})
    deadlines = sorted({int(job.deadline) for job in jobs})
    best_lb = 1
    best_start = releases[0]
    best_end = deadlines[-1]
    best_jobs = 0
    for interval_start in releases:
        completions = sorted(job.deadline for job in jobs if int(job.release) >= int(interval_start))
        cursor = 0
        for interval_end in deadlines:
            if interval_end <= interval_start:
                continue
            while cursor < len(completions) and int(completions[cursor]) <= int(interval_end):
                cursor += 1
            if cursor == 0:
                continue
            width = int(interval_end) - int(interval_start)
            lb = int(math.ceil(cursor * int(lst) / width))
            if lb > best_lb or (lb == best_lb and best_jobs == 0):
                best_lb = int(lb)
                best_start = int(interval_start)
                best_end = int(interval_end)
                best_jobs = int(cursor)
    return int(best_lb), int(best_start), int(best_end), int(best_jobs)


def try_smoothed_schedule(
    jobs: list[Job],
    *,
    lst: int,
    cap: int,
    axis_start: int,
    axis_end: int,
    placement: str,
) -> tuple[bool, list[ScheduledJob], np.ndarray]:
    counts = np.zeros(int(axis_end) - int(axis_start) + 1, dtype=np.int32)
    scheduled: list[ScheduledJob] = []
    # Tight, early-deadline jobs first; this keeps near-deadline conflicts visible.
    ordered = sorted(jobs, key=lambda job: (job.deadline, job.latest_start - job.release, job.release, job.owner))
    for job in ordered:
        first = int(job.release) - int(axis_start)
        last = int(job.latest_start) - int(axis_start)
        if first < 0 or last < first or last + int(lst) > len(counts):
            return False, scheduled, counts
        full = (counts >= int(cap)).astype(np.int16)
        full_prefix = np.concatenate(([0], np.cumsum(full, dtype=np.int32)))
        load_prefix = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        starts = np.arange(first, last + 1, dtype=np.int64)
        blocked = (full_prefix[starts + int(lst)] - full_prefix[starts]) > 0
        if np.all(blocked):
            return False, scheduled, counts
        feasible_offsets = np.flatnonzero(~blocked)
        if str(placement) == "earliest_feasible":
            chosen = int(starts[int(feasible_offsets[0])])
        elif str(placement) == "latest_feasible":
            chosen = int(starts[int(feasible_offsets[-1])])
        elif str(placement) in {"min_load_earliest", "min_load_latest"}:
            load = load_prefix[starts + int(lst)] - load_prefix[starts]
            load = np.where(blocked, np.iinfo(np.int64).max, load)
            best_load = int(np.min(load))
            best_offsets = np.flatnonzero(load == best_load)
            chosen_offset = int(best_offsets[0] if str(placement) == "min_load_earliest" else best_offsets[-1])
            chosen = int(starts[chosen_offset])
        else:
            raise ValueError(f"unknown placement strategy: {placement}")
        counts[chosen:chosen + int(lst)] += 1
        scheduled.append(ScheduledJob(job=job, start=int(axis_start) + chosen, end=int(axis_start) + chosen + int(lst)))
    return True, scheduled, counts


def find_smoothed_schedule(
    jobs: list[Job],
    *,
    lst: int,
    lower_bound: int,
    naive_peak: int,
    axis_start: int,
    axis_end: int,
) -> tuple[int, list[ScheduledJob], np.ndarray]:
    upper = max(int(lower_bound), int(naive_peak), 1)
    placements = ["earliest_feasible", "min_load_earliest", "min_load_latest", "latest_feasible"]
    for cap in range(max(1, int(lower_bound)), upper + 1):
        for placement in placements:
            ok, scheduled, counts = try_smoothed_schedule(
                jobs,
                lst=int(lst),
                cap=int(cap),
                axis_start=int(axis_start),
                axis_end=int(axis_end),
                placement=str(placement),
            )
            if ok:
                return int(cap), scheduled, counts
    for placement in placements:
        ok, scheduled, counts = try_smoothed_schedule(
            jobs,
            lst=int(lst),
            cap=int(upper),
            axis_start=int(axis_start),
            axis_end=int(axis_end),
            placement=str(placement),
        )
        if ok:
            return int(upper), scheduled, counts
    raise RuntimeError(f"failed to schedule even at naive peak cap={upper}")


def write_schedule(path: Path, scheduled: list[ScheduledJob]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "job_id",
                "transition",
                "owner",
                "old_edge",
                "new_edge",
                "old_last_work_step_before_deadline",
                "release",
                "latest_start",
                "target_first_work_step",
                "plan_start",
                "plan_end",
            ]
        )
        for item in scheduled:
            job = item.job
            writer.writerow(
                [
                    int(job.job_id),
                    job.transition,
                    int(job.owner),
                    job.old_edge,
                    job.new_edge,
                    "" if job.old_last_work_step_before_deadline is None else int(job.old_last_work_step_before_deadline),
                    int(job.release),
                    int(job.latest_start),
                    int(job.deadline),
                    int(item.start),
                    int(item.end),
                ]
            )


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.2), dpi=180, sharex=True)
    colors = {"delay": "#b91c1c", "any": "#1d4ed8", "hop": "#047857"}
    for mode, df in summary.groupby("mode", sort=False):
        color = colors.get(str(mode), "#111827")
        x = df["lst"].to_numpy()
        axes[0].plot(x, df["naive_peak_concurrency"], "--", marker="x", color=color, label=f"{mode} naive")
        axes[0].plot(x, df["density_lower_bound"], ":", marker="^", color=color, label=f"{mode} lower bound")
        axes[0].plot(x, df["smoothed_peak_concurrency"], "-", marker="o", color=color, label=f"{mode} smoothed")
        axes[1].plot(x, df["peak_reduction"], marker="s", color=color, label=mode)
        axes[2].plot(x, df["infeasible_window_jobs"], marker="^", color=color, label=mode)
    axes[0].set_ylabel("peak concurrency")
    axes[0].set_title("Usage-driven setup with 1-second scheduling granularity")
    axes[1].set_ylabel("naive - smoothed")
    axes[2].set_ylabel("window conflicts")
    axes[2].set_xlabel("link setup time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT / f"lst_1s_schedule_resolution_guard{int(args.release_guard_seconds)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for mode in args.modes:
        plan_csv = Path(args.switch_root) / f"setup600_{mode}" / "usage_driven_right_link_switch_plan_required.csv"
        mode_dir = out_dir / str(mode)
        mode_dir.mkdir(parents=True, exist_ok=True)
        for lst in range(int(args.lst_start), int(args.lst_end) + 1, int(args.lst_step)):
            jobs, infeasible = load_jobs(
                plan_csv,
                lst=int(lst),
                start=int(args.start),
                release_guard_seconds=int(args.release_guard_seconds),
            )
            lb, lb_start, lb_end, lb_jobs = density_lower_bound(jobs, lst=int(lst))
            naive_peak = naive_schedule_peak(
                jobs,
                lst=int(lst),
                axis_start=int(args.start),
                axis_end=int(args.end),
            )
            smoothed_peak, scheduled, counts = find_smoothed_schedule(
                jobs,
                lst=int(lst),
                lower_bound=int(lb),
                naive_peak=int(naive_peak),
                axis_start=int(args.start),
                axis_end=int(args.end),
            )
            lst_dir = mode_dir / f"lst{int(lst):03d}"
            lst_dir.mkdir(parents=True, exist_ok=True)
            write_schedule(lst_dir / "schedule_1s.csv", scheduled)
            np.save(lst_dir / "building_concurrency_1s.npy", counts)
            rows.append(
                {
                    "mode": str(mode),
                    "lst": int(lst),
                    "schedule_resolution": int(args.schedule_resolution),
                    "release_guard_seconds": int(args.release_guard_seconds),
                    "jobs": int(len(jobs) + len(infeasible)),
                    "schedulable_jobs": int(len(jobs)),
                    "infeasible_window_jobs": int(len(infeasible)),
                    "density_lower_bound": int(lb),
                    "smoothed_peak_concurrency": int(smoothed_peak),
                    "naive_peak_concurrency": int(naive_peak),
                    "peak_reduction": int(naive_peak) - int(smoothed_peak),
                    "lower_bound_matched": bool(int(lb) == int(smoothed_peak)),
                    "critical_interval_start": int(lb_start),
                    "critical_interval_end": int(lb_end),
                    "critical_interval_jobs": int(lb_jobs),
                }
            )
            print(rows[-1])

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "lst_1s_schedule_summary.csv", index=False, encoding="utf-8-sig")
    plot_summary(summary, out_dir / "lst_1s_schedule_summary.png")
    print(f"out_dir={out_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
