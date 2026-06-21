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
    parser = argparse.ArgumentParser(
        description="Compare naive just-in-time setup with exact-cap smoothed setup."
    )
    parser.add_argument("--switch-root", type=Path, default=DEFAULT_SWITCH_ROOT)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--lst-start", type=int, default=10)
    parser.add_argument("--lst-end", type=int, default=140)
    parser.add_argument("--lst-step", type=int, default=10)
    return parser.parse_args()


def step_floor(value: int, *, step: int) -> int:
    return int(math.floor(int(value) / int(step)) * int(step))


def step_ceil(value: int, *, step: int) -> int:
    return int(math.ceil(int(value) / int(step)) * int(step))


def parse_int_maybe(value: object) -> int | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return int(float(text))


def naive_peak(df: pd.DataFrame, *, lst: int, start: int, end: int, stride: int) -> tuple[int, int, int]:
    axis = np.arange(int(start), int(end) + int(stride), int(stride), dtype=np.int64)
    counts = np.zeros(len(axis), dtype=np.int32)
    feasible_jobs = 0
    infeasible_jobs = 0
    duration_slots = max(1, int(math.ceil(float(lst) / float(stride))))
    for row in df.itertuples(index=False):
        target = int(row.target_first_work_step)
        old_last = parse_int_maybe(row.old_last_work_step_before_deadline)
        earliest = int(start) if old_last is None else step_ceil(old_last + int(stride), step=int(stride))
        latest = step_floor(target - int(lst), step=int(stride))
        if earliest > latest:
            infeasible_jobs += 1
            continue
        feasible_jobs += 1
        idx = int((latest - int(start)) // int(stride))
        if idx < 0:
            infeasible_jobs += 1
            feasible_jobs -= 1
            continue
        counts[idx:idx + duration_slots] += 1
    return int(np.max(counts)) if counts.size else 0, int(feasible_jobs), int(infeasible_jobs)


def load_required_plan(root: Path, mode: str) -> pd.DataFrame:
    path = Path(root) / f"setup600_{mode}" / "usage_driven_right_link_switch_plan_required.csv"
    return pd.read_csv(path)


def load_exact_summary(root: Path, mode: str) -> pd.DataFrame:
    path = Path(root) / f"{mode}_after-old-last-work_exact-cap" / "lst_sweep_summary.csv"
    return pd.read_csv(path)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.sweep_root) / "naive_vs_smoothed"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for mode in ("delay", "any"):
        required = load_required_plan(Path(args.switch_root), mode)
        exact = load_exact_summary(Path(args.sweep_root), mode).set_index("lst")
        for lst in range(int(args.lst_start), int(args.lst_end) + 1, int(args.lst_step)):
            naive, feasible, infeasible = naive_peak(
                required,
                lst=int(lst),
                start=int(args.start),
                end=int(args.end),
                stride=int(args.stride),
            )
            exact_row = exact.loc[int(lst)]
            rows.append(
                {
                    "mode": mode,
                    "lst": int(lst),
                    "required_jobs": int(len(required)),
                    "naive_peak_concurrency": int(naive),
                    "naive_feasible_jobs": int(feasible),
                    "naive_infeasible_jobs": int(infeasible),
                    "smoothed_peak_concurrency": int(exact_row["peak_concurrency"]),
                    "smoothed_schedulable_jobs": int(exact_row["scheduled_at_min_cap"]),
                    "smoothed_infeasible_jobs": int(exact_row["infeasible_window_jobs"]),
                    "peak_reduction": int(naive) - int(exact_row["peak_concurrency"]),
                }
            )

    combined = pd.DataFrame(rows)
    combined.to_csv(out_dir / "naive_vs_smoothed_lst.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.0), dpi=180, sharex=True)
    colors = {"delay": "#b91c1c", "any": "#1d4ed8"}
    for mode, df in combined.groupby("mode", sort=False):
        x = df["lst"].to_numpy()
        axes[0].plot(
            x,
            df["naive_peak_concurrency"].to_numpy(),
            color=colors[mode],
            linestyle="--",
            marker="x",
            linewidth=1.45,
            label=f"{mode} naive",
        )
        axes[0].plot(
            x,
            df["smoothed_peak_concurrency"].to_numpy(),
            color=colors[mode],
            linestyle="-",
            marker="o",
            linewidth=1.75,
            label=f"{mode} smoothed",
        )
        axes[1].plot(
            x,
            df["peak_reduction"].to_numpy(),
            color=colors[mode],
            marker="s",
            linewidth=1.55,
            label=mode,
        )
    axes[0].set_ylabel("peak concurrency")
    axes[0].set_title("Naive just-in-time setup vs smoothed exact-cap setup")
    axes[1].set_ylabel("peak reduction")
    axes[1].set_xlabel("link setup time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "naive_vs_smoothed_lst.png")
    plt.close(fig)

    print(f"out_dir={out_dir}")
    print(combined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
