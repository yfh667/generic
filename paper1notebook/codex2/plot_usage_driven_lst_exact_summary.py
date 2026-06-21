from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


DEFAULT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_lst_sweep_056_061_056_china_europe\t0_86160_stride60"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot exact-cap LST sweep summaries for usage-driven dynamic motif switching."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--delay-dir", type=str, default="delay_after-old-last-work_exact-cap")
    parser.add_argument("--any-dir", type=str, default="any_after-old-last-work_exact-cap")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_summary(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["case"] = label
    return df


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir is not None else root / "exact_cap_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    delay = load_summary(root / args.delay_dir / "lst_sweep_summary.csv", "delay")
    any_df = load_summary(root / args.any_dir / "lst_sweep_summary.csv", "any")
    combined = pd.concat([delay, any_df], ignore_index=True)
    combined.to_csv(out_dir / "lst_sweep_exact_cap_combined.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.2), dpi=180, sharex=True)
    colors = {"delay": "#b91c1c", "any": "#1d4ed8"}
    markers = {"delay": "o", "any": "s"}
    for case, df in combined.groupby("case", sort=False):
        x = df["lst"].to_numpy()
        axes[0].plot(
            x,
            df["peak_concurrency"].to_numpy(),
            color=colors[case],
            marker=markers[case],
            linewidth=1.7,
            label=case,
        )
        axes[1].plot(
            x,
            df["infeasible_window_jobs"].to_numpy(),
            color=colors[case],
            marker=markers[case],
            linewidth=1.5,
            label=case,
        )
        axes[2].plot(
            x,
            df["scheduled_at_min_cap"].to_numpy(),
            color=colors[case],
            marker=markers[case],
            linewidth=1.5,
            label=case,
        )

    axes[0].set_ylabel("min peak concurrency")
    axes[0].set_title("Usage-driven link setup time impact, strict no-loss scheduling")
    axes[1].set_ylabel("window conflicts")
    axes[2].set_ylabel("schedulable jobs")
    axes[2].set_xlabel("link setup time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "lst_sweep_exact_cap_comparison.png")
    plt.close(fig)

    print(f"out_dir={out_dir}")
    print(combined[[
        "case",
        "lst",
        "jobs",
        "scheduled_at_min_cap",
        "infeasible_window_jobs",
        "peak_concurrency",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
