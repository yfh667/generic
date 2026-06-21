from __future__ import annotations

import argparse
import math
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
DEFAULT_SWEEP_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_lst_sweep_056_061_056_china_europe\t0_86160_stride60"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot naive just-in-time setup against MILP-proved minimum peak.")
    parser.add_argument("--switch-root", type=Path, default=DEFAULT_SWITCH_ROOT)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
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


def naive_peak(df: pd.DataFrame, *, lst: int, start: int, end: int, stride: int) -> tuple[int, int]:
    axis = np.arange(int(start), int(end) + int(stride), int(stride), dtype=np.int64)
    counts = np.zeros(len(axis), dtype=np.int32)
    infeasible = 0
    duration_slots = max(1, int(math.ceil(float(lst) / float(stride))))
    for row in df.itertuples(index=False):
        target = int(row.target_first_work_step)
        old_last = parse_int_maybe(row.old_last_work_step_before_deadline)
        earliest = int(start) if old_last is None else step_ceil(old_last + int(stride), step=int(stride))
        latest = step_floor(target - int(lst), step=int(stride))
        if earliest > latest:
            infeasible += 1
            continue
        start_idx = int((latest - int(start)) // int(stride))
        counts[start_idx:start_idx + duration_slots] += 1
    return int(np.max(counts)) if counts.size else 0, int(infeasible)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.sweep_root) / "naive_vs_milp"
    out_dir.mkdir(parents=True, exist_ok=True)

    milp = pd.read_csv(Path(args.sweep_root) / "milp_min_peak_verification" / "milp_min_peak_summary.csv")
    rows: list[dict[str, object]] = []
    for mode in ("delay", "any"):
        plan = pd.read_csv(Path(args.switch_root) / f"setup600_{mode}" / "usage_driven_right_link_switch_plan_required.csv")
        for row in milp[milp["mode"] == mode].itertuples(index=False):
            naive, naive_infeasible = naive_peak(
                plan,
                lst=int(row.lst),
                start=int(args.start),
                end=int(args.end),
                stride=int(args.stride),
            )
            rows.append(
                {
                    "mode": mode,
                    "lst": int(row.lst),
                    "required_jobs": int(row.jobs),
                    "feasible_jobs": int(row.feasible_jobs),
                    "infeasible_window_jobs": int(row.infeasible_window_jobs),
                    "naive_peak_concurrency": int(naive),
                    "milp_min_peak": int(row.milp_min_peak),
                    "peak_reduction": int(naive) - int(row.milp_min_peak),
                    "proof_status": str(row.proof_status),
                }
            )

    combined = pd.DataFrame(rows)
    combined.to_csv(out_dir / "naive_vs_milp_lst.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.2), dpi=180, sharex=True)
    colors = {"delay": "#b91c1c", "any": "#1d4ed8"}
    for mode, df in combined.groupby("mode", sort=False):
        x = df["lst"].to_numpy()
        axes[0].plot(x, df["naive_peak_concurrency"], "--", marker="x", color=colors[mode], label=f"{mode} naive")
        axes[0].plot(x, df["milp_min_peak"], "-", marker="o", color=colors[mode], label=f"{mode} MILP min")
        axes[1].plot(x, df["peak_reduction"], marker="s", color=colors[mode], label=mode)
        axes[2].plot(x, df["infeasible_window_jobs"], marker="^", color=colors[mode], label=mode)

    axes[0].set_ylabel("peak concurrency")
    axes[0].set_title("Usage-driven setup: naive deadline scheduling vs MILP-proved minimum concurrency")
    axes[1].set_ylabel("peak reduction")
    axes[2].set_ylabel("window conflicts")
    axes[2].set_xlabel("link setup time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "naive_vs_milp_lst.png")
    plt.close(fig)

    print(f"out_dir={out_dir}")
    print(combined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
