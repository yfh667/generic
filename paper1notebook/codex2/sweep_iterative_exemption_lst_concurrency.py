from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_usage_driven_switch_with_36000_exemption as base  # noqa: E402
import plan_usage_driven_switch_with_iterative_exemption as iterative  # noqa: E402
from build_ideal_splice_right_link_state_0056_0061 import DEFAULT_DELAY_USAGE_ROOT  # noqa: E402
from run_lst_1s_concurrency_experiment import (  # noqa: E402
    Job,
    ScheduledJob,
    density_lower_bound,
    naive_schedule_peak,
    try_smoothed_schedule,
    write_schedule,
)


DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_iterative_exemption_lst_sweep_056_061_china_europe"
)


@dataclass(frozen=True)
class SweepResult:
    lst: int
    plan_rows: list[dict[str, object]]
    seed_exemptions: list[dict[str, object]]
    remaining_conflicts: list[dict[str, object]]
    iteration_rows: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep LST=10..140 for motif000056 -> motif000061 usage-driven switching. "
            "Each LST first applies iterative working-port exemptions, then smooths setup starts "
            "to reduce simultaneous building pressure."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--return-switch-step", type=int, default=54000)
    parser.add_argument("--lst-start", type=int, default=10)
    parser.add_argument("--lst-end", type=int, default=140)
    parser.add_argument("--lst-step", type=int, default=10)
    parser.add_argument("--source-motif-id", type=int, default=56)
    parser.add_argument("--target-motif-id", type=int, default=61)
    parser.add_argument("--working-threshold", type=float, default=0.0)
    parser.add_argument(
        "--release-guard-seconds",
        type=int,
        default=1,
        help=(
            "Guard after the last sampled working point before a setup may start. "
            "Use 1 for strict sampled-time logic; use 60 for a conservative interpretation of 60s edge-usage samples."
        ),
    )
    parser.add_argument("--delay-usage-root", type=Path, default=DEFAULT_DELAY_USAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-iterations", type=int, default=100)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def last_work_step_before(
    *,
    steps: np.ndarray,
    usage: np.ndarray,
    edge_idx: int,
    start: int,
    deadline: int,
    threshold: float,
) -> int | None:
    rows = np.flatnonzero((steps >= int(start)) & (steps < int(deadline)))
    if rows.size == 0:
        return None
    values = np.asarray(usage[rows, int(edge_idx)], dtype=np.float32)
    hit = np.flatnonzero(values > float(threshold))
    if hit.size == 0:
        return None
    return int(steps[int(rows[int(hit[-1])])])


def release_for_plan_row(
    *,
    row: dict[str, object],
    source: base.UsageBundle,
    target: base.UsageBundle,
    steps: np.ndarray,
    axis_start: int,
    threshold: float,
    release_guard_seconds: int,
) -> tuple[int, list[str]]:
    owner = int(row["owner"])
    deadline = int(row["target_first_work_step"])
    old = source.topo.right_by_owner.get(owner)
    new = target.topo.right_by_owner.get(owner)
    if new is None:
        return int(axis_start), []

    release = int(axis_start)
    reasons: list[str] = []
    for label, link in (
        ("owner_right_old", old),
        ("target_left_old", source.topo.left_by_right.get(int(new.right))),
    ):
        if link is None:
            continue
        last_step = last_work_step_before(
            steps=steps,
            usage=source.any_usage,
            edge_idx=int(link.edge_idx),
            start=int(axis_start),
            deadline=int(deadline),
            threshold=float(threshold),
        )
        if last_step is None:
            continue
        candidate = int(last_step) + int(release_guard_seconds)
        release = max(release, candidate)
        reasons.append(f"{label}:{base.link_text(link)} last_work={last_step} release={candidate}")
    return int(release), reasons


def schedule_window_conflicts(
    *,
    plan_rows: list[dict[str, object]],
    source: base.UsageBundle,
    target: base.UsageBundle,
    steps: np.ndarray,
    lst: int,
    axis_start: int,
    threshold: float,
    release_guard_seconds: int,
) -> list[dict[str, object]]:
    conflicts: list[dict[str, object]] = []
    for row in plan_rows:
        if not bool(row.get("scheduled", False)):
            continue
        owner = int(row["owner"])
        deadline = int(row["target_first_work_step"])
        latest_start = int(deadline) - int(lst)
        new = target.topo.right_by_owner.get(owner)
        if new is None:
            continue

        candidates = [
            ("owner_right_old_release", source.topo.right_by_owner.get(owner)),
            ("target_left_old_release", source.topo.left_by_right.get(int(new.right))),
        ]
        for blocker_type, link in candidates:
            if link is None:
                continue
            last_step = last_work_step_before(
                steps=steps,
                usage=source.any_usage,
                edge_idx=int(link.edge_idx),
                start=int(axis_start),
                deadline=int(deadline),
                threshold=float(threshold),
            )
            if last_step is None:
                continue
            release = int(last_step) + int(release_guard_seconds)
            if release <= latest_start:
                continue
            payload = dict(row)
            payload.update(
                {
                    "blocker_type": str(blocker_type),
                    "blocker_state": "source",
                    "blocker_link": base.link_text(link),
                    "blocker_symbol": str(link.symbol),
                    "blocker_edge_key": base.edge_key_text(link),
                    "blocker_owner": int(link.owner),
                    "blocker_right": int(link.right),
                    "blocker_first_step_inside_setup": int(last_step),
                    "blocker_mode_at_first": "release_after_latest_start",
                    "release": int(release),
                    "latest_start": int(latest_start),
                    "deadline": int(deadline),
                    "release_guard_seconds": int(release_guard_seconds),
                }
            )
            conflicts.append(payload)
    return conflicts


def job_rows_from_plan(
    *,
    plan_rows: list[dict[str, object]],
    source: base.UsageBundle,
    target: base.UsageBundle,
    steps: np.ndarray,
    lst: int,
    axis_start: int,
    threshold: float,
    release_guard_seconds: int,
) -> tuple[list[Job], list[dict[str, object]], list[dict[str, object]]]:
    jobs: list[Job] = []
    job_meta_rows: list[dict[str, object]] = []
    infeasible_rows: list[dict[str, object]] = []
    job_id = 0
    for row in plan_rows:
        if not bool(row.get("scheduled", False)):
            continue
        deadline = int(row["target_first_work_step"])
        latest_start = int(deadline) - int(lst)
        release, release_reasons = release_for_plan_row(
            row=row,
            source=source,
            target=target,
            steps=steps,
            axis_start=int(axis_start),
            threshold=float(threshold),
            release_guard_seconds=int(release_guard_seconds),
        )
        payload = dict(row)
        payload["release"] = int(release)
        payload["latest_start"] = int(latest_start)
        payload["deadline"] = int(deadline)
        payload["release_reasons"] = "; ".join(release_reasons)
        payload["schedule_window_feasible"] = int(release) <= int(latest_start)
        if int(release) > int(latest_start):
            payload["window_reason"] = "release_after_latest_start"
            infeasible_rows.append(payload)
            continue
        job_id += 1
        jobs.append(
            Job(
                job_id=job_id,
                transition=str(row.get("transition", "switch_056_to_061")),
                owner=int(row["owner"]),
                old_edge=str(row.get("old_edge_key", "")),
                new_edge=str(row.get("new_edge_key", "")),
                release=int(release),
                latest_start=int(latest_start),
                deadline=int(deadline),
                old_last_work_step_before_deadline=None,
            )
        )
        payload["job_id"] = int(job_id)
        payload["window_reason"] = "ok"
        job_meta_rows.append(payload)
    return jobs, job_meta_rows, infeasible_rows


def count_stats(counts: np.ndarray, *, axis_start: int, axis_end: int, lst: int, jobs: int) -> dict[str, object]:
    counts = np.asarray(counts, dtype=np.int32)
    positive = counts[counts > 0]
    horizon_seconds = int(axis_end) - int(axis_start) + 1
    total_work = int(jobs) * int(lst)
    return {
        "total_build_work_seconds": int(total_work),
        "horizon_seconds": int(horizon_seconds),
        "mean_concurrency_full_horizon": float(total_work / max(1, horizon_seconds)),
        "busy_seconds": int(positive.size),
        "mean_concurrency_when_busy": float(np.mean(positive)) if positive.size else 0.0,
        "p50_concurrency_when_busy": float(np.percentile(positive, 50)) if positive.size else 0.0,
        "p95_concurrency_when_busy": float(np.percentile(positive, 95)) if positive.size else 0.0,
        "p99_concurrency_when_busy": float(np.percentile(positive, 99)) if positive.size else 0.0,
        "peak_concurrency": int(np.max(counts)) if counts.size else 0,
    }


def choose_schedule(
    jobs: list[Job],
    *,
    lst: int,
    lower_bound: int,
    naive_peak: int,
    axis_start: int,
    axis_end: int,
) -> tuple[int, list[ScheduledJob], np.ndarray, str]:
    placements = ["min_load_earliest", "min_load_latest", "earliest_feasible", "latest_feasible"]
    upper = max(int(lower_bound), int(naive_peak), 1)
    for cap in range(max(1, int(lower_bound)), upper + 1):
        candidates: list[tuple[tuple[float, float, int], str, list[ScheduledJob], np.ndarray]] = []
        for placement in placements:
            ok, scheduled, counts = try_smoothed_schedule(
                jobs,
                lst=int(lst),
                cap=int(cap),
                axis_start=int(axis_start),
                axis_end=int(axis_end),
                placement=str(placement),
            )
            if not ok:
                continue
            stats = count_stats(counts, axis_start=axis_start, axis_end=axis_end, lst=lst, jobs=len(jobs))
            key = (
                float(stats["mean_concurrency_when_busy"]),
                float(stats["p95_concurrency_when_busy"]),
                int(stats["busy_seconds"]) * -1,
            )
            candidates.append((key, placement, scheduled, counts))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            _key, placement, scheduled, counts = candidates[0]
            return int(cap), scheduled, counts, str(placement)
    raise RuntimeError(f"failed to find feasible schedule up to cap={upper}")


def run_one_lst(
    *,
    lst: int,
    source: base.UsageBundle,
    target: base.UsageBundle,
    steps: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, object]:
    contexts, seed_exemptions = iterative.build_contexts(
        source=source,
        target=target,
        steps=steps,
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
        lst=int(lst),
        threshold=float(args.working_threshold),
    )
    iteration_rows: list[dict[str, object]] = []
    for iteration in range(int(args.max_iterations) + 1):
        plan_rows = iterative.make_plan_rows(
            contexts=contexts,
            target=target,
            lst=int(lst),
            threshold=float(args.working_threshold),
        )
        dynamic_conflicts = base.dynamic_conflicts_after_schedule(
            plan_rows=plan_rows,
            source=source,
            target=target,
            steps=steps,
            threshold=float(args.working_threshold),
        )
        window_conflicts = schedule_window_conflicts(
            plan_rows=plan_rows,
            source=source,
            target=target,
            steps=steps,
            lst=int(lst),
            axis_start=int(args.start),
            threshold=float(args.working_threshold),
            release_guard_seconds=int(args.release_guard_seconds),
        )
        conflicts = dynamic_conflicts + window_conflicts
        iteration_rows.append(
            {
                "iteration": int(iteration),
                "scheduled": int(sum(1 for row in plan_rows if bool(row.get("scheduled", False)))),
                "unscheduled": int(sum(1 for row in plan_rows if not bool(row.get("scheduled", False)))),
                "conflict_rows": int(len(conflicts)),
                "dynamic_conflict_rows": int(len(dynamic_conflicts)),
                "schedule_window_conflict_rows": int(len(window_conflicts)),
                "conflict_owners": int(len({int(row["owner"]) for row in conflicts})) if conflicts else 0,
            }
        )
        if not conflicts:
            break
        changed = iterative.apply_one_conflict_round(contexts=contexts, conflicts=conflicts)
        if changed == 0:
            break

    plan_rows = iterative.make_plan_rows(
        contexts=contexts,
        target=target,
        lst=int(lst),
        threshold=float(args.working_threshold),
    )
    remaining_dynamic_conflicts = base.dynamic_conflicts_after_schedule(
        plan_rows=plan_rows,
        source=source,
        target=target,
        steps=steps,
        threshold=float(args.working_threshold),
    )
    remaining_window_conflicts = schedule_window_conflicts(
        plan_rows=plan_rows,
        source=source,
        target=target,
        steps=steps,
        lst=int(lst),
        axis_start=int(args.start),
        threshold=float(args.working_threshold),
        release_guard_seconds=int(args.release_guard_seconds),
    )
    remaining_conflicts = remaining_dynamic_conflicts + remaining_window_conflicts
    jobs, job_meta_rows, infeasible_rows = job_rows_from_plan(
        plan_rows=plan_rows,
        source=source,
        target=target,
        steps=steps,
        lst=int(lst),
        axis_start=int(args.start),
        threshold=float(args.working_threshold),
        release_guard_seconds=int(args.release_guard_seconds),
    )

    lst_dir = out_dir / f"lst{int(lst):03d}"
    lst_dir.mkdir(parents=True, exist_ok=True)
    write_rows(lst_dir / "iterative_exempted_switch_plan.csv", plan_rows)
    write_rows(lst_dir / "seed_exemptions_at_36000.csv", seed_exemptions)
    write_rows(lst_dir / "iteration_summary.csv", iteration_rows)
    write_rows(lst_dir / "remaining_conflicts.csv", remaining_conflicts)
    write_rows(lst_dir / "remaining_dynamic_conflicts.csv", remaining_dynamic_conflicts)
    write_rows(lst_dir / "remaining_schedule_window_conflicts.csv", remaining_window_conflicts)
    write_rows(lst_dir / "schedule_jobs.csv", job_meta_rows)
    write_rows(lst_dir / "schedule_window_infeasible.csv", infeasible_rows)

    if jobs:
        lb, lb_start, lb_end, lb_jobs = density_lower_bound(jobs, lst=int(lst))
        naive_peak = naive_schedule_peak(jobs, lst=int(lst), axis_start=int(args.start), axis_end=int(args.end))
        smoothed_peak, scheduled, counts, placement = choose_schedule(
            jobs,
            lst=int(lst),
            lower_bound=int(lb),
            naive_peak=int(naive_peak),
            axis_start=int(args.start),
            axis_end=int(args.end),
        )
        write_schedule(lst_dir / "schedule_1s.csv", scheduled)
        np.save(lst_dir / "building_concurrency_1s.npy", counts)
        stats = count_stats(counts, axis_start=int(args.start), axis_end=int(args.end), lst=int(lst), jobs=len(jobs))
    else:
        lb = lb_start = lb_end = lb_jobs = naive_peak = smoothed_peak = 0
        placement = ""
        counts = np.zeros(int(args.end) - int(args.start) + 1, dtype=np.int32)
        write_schedule(lst_dir / "schedule_1s.csv", [])
        np.save(lst_dir / "building_concurrency_1s.npy", counts)
        stats = count_stats(counts, axis_start=int(args.start), axis_end=int(args.end), lst=int(lst), jobs=0)

    propagated = [
        ctx for ctx in contexts.values()
        if int(len(ctx["skipped"])) > (1 if bool(ctx["seed_exempted"]) else 0)
    ]
    result = {
        "lst": int(lst),
        "changed_links_with_target": int(len(plan_rows)),
        "iterative_scheduled_links": int(sum(1 for row in plan_rows if bool(row.get("scheduled", False)))),
        "iterative_unscheduled_links": int(sum(1 for row in plan_rows if not bool(row.get("scheduled", False)))),
        "seed_exemptions_36000": int(len(seed_exemptions)),
        "propagated_exemption_owners": int(len(propagated)),
        "remaining_conflict_rows": int(len(remaining_conflicts)),
        "remaining_dynamic_conflict_rows": int(len(remaining_dynamic_conflicts)),
        "remaining_schedule_window_conflict_rows": int(len(remaining_window_conflicts)),
        "remaining_conflict_owners": int(len({int(row['owner']) for row in remaining_conflicts})) if remaining_conflicts else 0,
        "schedule_jobs": int(len(jobs)),
        "schedule_window_infeasible": int(len(infeasible_rows)),
        "density_lower_bound": int(lb),
        "critical_interval_start": int(lb_start),
        "critical_interval_end": int(lb_end),
        "critical_interval_jobs": int(lb_jobs),
        "naive_peak_concurrency": int(naive_peak),
        "smoothed_peak_concurrency": int(smoothed_peak),
        "peak_reduction": int(naive_peak) - int(smoothed_peak),
        "lower_bound_matched": bool(int(lb) == int(smoothed_peak)),
        "chosen_placement": str(placement),
        **stats,
        "lst_dir": str(lst_dir),
    }
    (lst_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13.6, 13.0), dpi=180, sharex=True)
    x = summary["lst"].to_numpy()
    axes[0].plot(x, summary["naive_peak_concurrency"], "--", marker="x", color="#7f1d1d", label="naive peak")
    axes[0].plot(x, summary["density_lower_bound"], ":", marker="^", color="#111827", label="density lower bound")
    axes[0].plot(x, summary["smoothed_peak_concurrency"], "-", marker="o", color="#2563eb", label="smoothed peak")
    axes[0].set_ylabel("peak concurrent links")
    axes[0].legend(loc="best")

    axes[1].plot(x, summary["mean_concurrency_full_horizon"], marker="o", color="#047857", label="full-horizon mean")
    axes[1].plot(x, summary["mean_concurrency_when_busy"], marker="s", color="#f97316", label="busy-time mean")
    axes[1].plot(x, summary["p95_concurrency_when_busy"], marker="^", color="#9333ea", label="busy p95")
    axes[1].set_ylabel("mean / p95 concurrency")
    axes[1].legend(loc="best")

    axes[2].plot(x, summary["schedule_jobs"], marker="o", color="#2563eb", label="schedulable setup jobs")
    axes[2].plot(x, summary["iterative_unscheduled_links"], marker="s", color="#dc2626", label="unscheduled by exemptions")
    axes[2].plot(x, summary["schedule_window_infeasible"], marker="^", color="#7c2d12", label="window infeasible")
    axes[2].set_ylabel("link count")
    axes[2].legend(loc="best")

    axes[3].plot(x, summary["seed_exemptions_36000"], marker="o", color="#475569", label="seed exemptions")
    axes[3].plot(x, summary["propagated_exemption_owners"], marker="s", color="#0f766e", label="propagated owners")
    axes[3].set_ylabel("exemptions")
    axes[3].set_xlabel("link setup time LST (s)")
    axes[3].legend(loc="best")

    for ax in axes:
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    fig.suptitle("Iterative-exemption usage-driven setup sweep: motif000056 -> motif000061", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "lst_sweep_concurrency_summary.png")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    steps = np.asarray(list(range(int(args.start), int(args.end) + 1, int(args.stride))), dtype=np.int64)
    source = base.usage_bundle(
        int(args.source_motif_id),
        delay_usage_root=Path(args.delay_usage_root),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    target = base.usage_bundle(
        int(args.target_motif_id),
        delay_usage_root=Path(args.delay_usage_root),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT
        / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
        / f"switch{int(args.switch_step)}_{int(args.return_switch_step)}"
        / f"releaseguard{int(args.release_guard_seconds)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for lst in range(int(args.lst_start), int(args.lst_end) + 1, int(args.lst_step)):
        result = run_one_lst(
            lst=int(lst),
            source=source,
            target=target,
            steps=steps,
            args=args,
            out_dir=out_dir,
        )
        rows.append(result)
        print(result, flush=True)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "lst_sweep_concurrency_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_summary(summary, out_dir)
    meta = {
        "description": (
            "LST sweep using iterative working-port exemptions, then 1-second setup scheduling. "
            "Full-horizon mean concurrency equals total setup work divided by horizon and is not a scheduling objective; "
            "smoothed_peak_concurrency and busy-time metrics quantify simultaneous setup pressure."
        ),
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "switch_step": int(args.switch_step),
        "return_switch_step": int(args.return_switch_step),
        "lst_values": [int(x) for x in summary["lst"].tolist()],
        "release_guard_seconds": int(args.release_guard_seconds),
        "outputs": {
            "summary_csv": str(summary_path),
            "summary_plot": str(out_dir / "lst_sweep_concurrency_summary.png"),
            "per_lst_dirs": "lstXXX/",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
