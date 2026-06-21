from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_REVIEW_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_final_review"
)

PAIR_LABELS = {
    "china_europe": "China-Europe",
    "china_america": "China-America",
    "china_africa": "China-Africa",
}

POLICIES = {
    "dynamic_l078": {
        "label": "dynamic l078 LST120",
        "dir": "dynamic_l078_c005_cnt025",
        "color": "#111827",
        "linestyle": "-",
        "linewidth": 1.2,
    },
    "motif000056": {
        "label": "motif000056 DBD | --B",
        "dir": "motif000056_static_lst120",
        "color": "#2563EB",
        "linestyle": "--",
        "linewidth": 1.05,
    },
    "motif000040": {
        "label": "motif000040 CBD | CB-",
        "dir": "motif000040_static_lst120",
        "color": "#DC2626",
        "linestyle": ":",
        "linewidth": 1.15,
    },
    "plus_grid": {
        "label": "+grid option0",
        "dir": "plus_grid_option0_fixed",
        "color": "#16A34A",
        "linestyle": "-.",
        "linewidth": 1.1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot three-pair shortest-hop and shortest-delay comparison for l078, motif000056, motif000040, and +grid."
    )
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def read_metric_csv(path: Path, *, value_column: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            value = row.get(value_column, "")
            if value in ("", None):
                continue
            out[row["pair"]][int(row["step"])] = float(value)
    return out


def load_policy_series(review_dir: Path) -> dict[str, dict[str, dict[str, dict[int, float]]]]:
    base_dir = review_dir / "three_pair_lsts"
    out: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    missing: list[Path] = []
    for policy_key, info in POLICIES.items():
        policy_dir = base_dir / str(info["dir"])
        hops_csv = policy_dir / "lst_active_hops_timeseries.csv"
        delay_csv = policy_dir / "lst_active_delay_sample_timeseries.csv"
        if not hops_csv.exists():
            missing.append(hops_csv)
        if not delay_csv.exists():
            missing.append(delay_csv)
        if missing:
            continue
        out[policy_key] = {
            "hops": read_metric_csv(hops_csv, value_column="mean_hops"),
            "delay_ms": read_metric_csv(delay_csv, value_column="mean_delay_ms"),
        }
    if missing:
        raise FileNotFoundError("Missing input CSV files:\n" + "\n".join(str(p) for p in missing))
    return out


def sorted_steps(series: dict[str, dict[str, dict[str, dict[int, float]]]], *, metric: str, pair: str) -> list[int]:
    step_sets = [
        set(policy_data[metric].get(pair, {}).keys())
        for policy_data in series.values()
    ]
    if not step_sets:
        return []
    common = set.intersection(*step_sets)
    return sorted(common)


def values_for_steps(data: dict[int, float], steps: list[int]) -> np.ndarray:
    return np.asarray([data.get(step, np.nan) for step in steps], dtype=np.float64)


def write_long_timeseries(
    path: Path,
    *,
    series: dict[str, dict[str, dict[str, dict[int, float]]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "hour", "pair", "policy", "metric", "value"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in PAIR_LABELS:
            for metric in ("hops", "delay_ms"):
                steps = sorted_steps(series, metric=metric, pair=pair)
                for policy_key, policy_data in series.items():
                    values = policy_data[metric].get(pair, {})
                    for step in steps:
                        writer.writerow(
                            {
                                "step": int(step),
                                "hour": float(step) / 3600.0,
                                "pair": pair,
                                "policy": policy_key,
                                "metric": metric,
                                "value": values.get(step, ""),
                            }
                        )


def write_summary(path: Path, *, series: dict[str, dict[str, dict[str, dict[int, float]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pair in PAIR_LABELS:
        for metric in ("hops", "delay_ms"):
            steps = sorted_steps(series, metric=metric, pair=pair)
            for policy_key, policy_data in series.items():
                arr = values_for_steps(policy_data[metric].get(pair, {}), steps)
                finite = arr[np.isfinite(arr)]
                info = POLICIES[policy_key]
                rows.append(
                    {
                        "pair": pair,
                        "pair_label": PAIR_LABELS[pair],
                        "metric": metric,
                        "policy": policy_key,
                        "policy_label": info["label"],
                        "mean": float(np.mean(finite)) if finite.size else "",
                        "min": float(np.min(finite)) if finite.size else "",
                        "max": float(np.max(finite)) if finite.size else "",
                        "p05": float(np.percentile(finite, 5)) if finite.size else "",
                        "p50": float(np.percentile(finite, 50)) if finite.size else "",
                        "p95": float(np.percentile(finite, 95)) if finite.size else "",
                        "finite_steps": int(finite.size),
                    }
                )

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_all_panels(
    out_path: Path,
    *,
    series: dict[str, dict[str, dict[str, dict[int, float]]]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(16.8, 9.4), dpi=180, sharex="col")
    for row_idx, pair in enumerate(PAIR_LABELS):
        for col_idx, (metric, title, y_label) in enumerate(
            (
                ("hops", "mean shortest hops", "mean hops"),
                ("delay_ms", "mean shortest delay", "delay (ms)"),
            )
        ):
            ax = axes[row_idx, col_idx]
            steps = sorted_steps(series, metric=metric, pair=pair)
            x = np.asarray(steps, dtype=np.float64) / 3600.0
            for policy_key, policy_data in series.items():
                info = POLICIES[policy_key]
                y = values_for_steps(policy_data[metric].get(pair, {}), steps)
                mean = float(np.nanmean(y))
                ax.plot(
                    x,
                    y,
                    color=str(info["color"]),
                    linestyle=str(info["linestyle"]),
                    linewidth=float(info["linewidth"]),
                    label=f"{info['label']} | mean={mean:.3f}",
                )
            ax.grid(alpha=0.25, linestyle="--", linewidth=0.55)
            ax.legend(fontsize=7.2, loc="best")
            ax.set_ylabel(f"{PAIR_LABELS[pair]}\n{y_label}")
            if row_idx == 0:
                ax.set_title(title)
            if row_idx == len(PAIR_LABELS) - 1:
                ax.set_xlabel("time (hour)")
    fig.suptitle("G60 region-internal +grid: dynamic l078 vs motif000056 / motif000040 / +grid", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric_panels(
    out_path: Path,
    *,
    series: dict[str, dict[str, dict[str, dict[int, float]]]],
    metric: str,
    title: str,
    y_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(15.8, 8.9), dpi=180, sharex=True)
    for row_idx, pair in enumerate(PAIR_LABELS):
        ax = axes[row_idx]
        steps = sorted_steps(series, metric=metric, pair=pair)
        x = np.asarray(steps, dtype=np.float64) / 3600.0
        for policy_key, policy_data in series.items():
            info = POLICIES[policy_key]
            y = values_for_steps(policy_data[metric].get(pair, {}), steps)
            ax.plot(
                x,
                y,
                color=str(info["color"]),
                linestyle=str(info["linestyle"]),
                linewidth=float(info["linewidth"]),
                label=f"{info['label']} | mean={np.nanmean(y):.3f}",
            )
        ax.set_ylabel(PAIR_LABELS[pair])
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.55)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_title(title)
    axes[-1].set_xlabel("time (hour)")
    fig.supylabel(y_label, x=0.012)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pair_panels(
    out_dir: Path,
    *,
    series: dict[str, dict[str, dict[str, dict[int, float]]]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    for pair, pair_label in PAIR_LABELS.items():
        fig, axes = plt.subplots(2, 1, figsize=(15.6, 8.0), dpi=180, sharex=True)
        for ax, metric, y_label in (
            (axes[0], "hops", "mean shortest hops"),
            (axes[1], "delay_ms", "mean shortest delay (ms)"),
        ):
            steps = sorted_steps(series, metric=metric, pair=pair)
            x = np.asarray(steps, dtype=np.float64) / 3600.0
            for policy_key, policy_data in series.items():
                info = POLICIES[policy_key]
                y = values_for_steps(policy_data[metric].get(pair, {}), steps)
                ax.plot(
                    x,
                    y,
                    color=str(info["color"]),
                    linestyle=str(info["linestyle"]),
                    linewidth=float(info["linewidth"]),
                    label=f"{info['label']} | mean={np.nanmean(y):.3f}",
                )
            ax.set_ylabel(y_label)
            ax.grid(alpha=0.25, linestyle="--", linewidth=0.55)
            ax.legend(fontsize=8, loc="best")
        axes[0].set_title(f"{pair_label}: dynamic l078 vs motif000056 / motif000040 / +grid")
        axes[-1].set_xlabel("time (hour)")
        fig.tight_layout()
        out_path = out_dir / f"{pair}_lst120_hops_delay_compare.png"
        fig.savefig(out_path)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else review_dir / "plots" / "three_pair_topology_compare"
    table_dir = review_dir / "tables" / "three_pair_topology_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    series = load_policy_series(review_dir)
    timeseries_csv = table_dir / "lst120_l078_vs_motif0040_0056_plus_grid_timeseries_long.csv"
    summary_csv = table_dir / "lst120_l078_vs_motif0040_0056_plus_grid_summary.csv"
    write_long_timeseries(timeseries_csv, series=series)
    write_summary(summary_csv, series=series)

    panel_path = out_dir / "lst120_l078_vs_motif0040_0056_plus_grid_all_panels.png"
    hops_path = out_dir / "lst120_l078_vs_motif0040_0056_plus_grid_hops.png"
    delay_path = out_dir / "lst120_l078_vs_motif0040_0056_plus_grid_delay_ms.png"
    plot_all_panels(panel_path, series=series)
    plot_metric_panels(
        hops_path,
        series=series,
        metric="hops",
        title="G60 mean shortest hops: dynamic l078 vs motif000056 / motif000040 / +grid",
        y_label="mean shortest hops",
    )
    plot_metric_panels(
        delay_path,
        series=series,
        metric="delay_ms",
        title="G60 mean shortest delay: dynamic l078 vs motif000056 / motif000040 / +grid",
        y_label="mean shortest delay (ms)",
    )
    pair_paths = plot_pair_panels(out_dir, series=series)

    print(f"summary_csv={summary_csv}")
    print(f"timeseries_csv={timeseries_csv}")
    for path in (panel_path, hops_path, delay_path, *pair_paths):
        print(f"plot={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
