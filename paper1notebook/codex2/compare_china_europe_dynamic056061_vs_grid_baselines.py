from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRIC_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_w_le4_h_le3\paper_style_808_no_region_grid_t0_86160_stride60"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_w_le4_h_le3\two_motif_056_061_china_europe_t0_86160_stride60_switch36000_54000"
    r"\baseline_compare"
)
PAIR = "china_europe"
BASE = "combined_motif_000056"
PATCH = "combined_motif_000061"
XGRID = "combined_motif_000002"
PLUS_GRID = "combined_motif_000001"
FULL_LINK = "full_link"


SERIES = {
    "dynamic_056_061_10h_15h": {
        "label": "dynamic 056->061, 10h-15h",
        "color": "#f59e0b",
        "width": 1.7,
        "alpha": 0.98,
    },
    "motif000056": {
        "label": "motif000056 DBD | --B",
        "color": "#1B4F9C",
        "width": 1.2,
        "alpha": 0.95,
    },
    "motif000061": {
        "label": "motif000061 DCD | C--",
        "color": "#C1121F",
        "width": 1.0,
        "alpha": 0.68,
    },
    "plus_grid": {
        "label": "+grid / motif000001 A",
        "color": "#16a34a",
        "width": 1.05,
        "alpha": 0.75,
    },
    "xgrid": {
        "label": "xgrid / motif000002 CB",
        "color": "#7c3aed",
        "width": 1.05,
        "alpha": 0.75,
    },
    "full_link": {
        "label": "full-link upper bound",
        "color": "#111827",
        "width": 1.2,
        "alpha": 0.78,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dynamic motif000056/000061 window schedule with motif/grid baselines."
    )
    parser.add_argument("--metric-root", type=Path, default=DEFAULT_METRIC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-start", type=int, default=36000)
    parser.add_argument("--window-end", type=int, default=54000)
    return parser.parse_args()


def metric_csv(metric_root: Path, metric_kind: str) -> Path:
    if metric_kind == "hops":
        filename = "compare_mean_shortest_hops_808_strict_reachable.csv"
        subdir = "shortest_hops"
    elif metric_kind == "delay_ms":
        filename = "compare_mean_shortest_delay_ms_808_strict_reachable.csv"
        subdir = "shortest_delay"
    else:
        raise ValueError(metric_kind)
    return metric_root / subdir / PAIR / filename


def read_one(metric_root: Path, metric_kind: str) -> pd.DataFrame:
    path = metric_csv(metric_root, metric_kind)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=["step", BASE, PATCH, XGRID, PLUS_GRID, FULL_LINK])
    df = df.rename(
        columns={
            BASE: "motif000056",
            PATCH: "motif000061",
            XGRID: "xgrid",
            PLUS_GRID: "plus_grid",
            FULL_LINK: "full_link",
        }
    )
    return df.sort_values("step").reset_index(drop=True)


def build_compare_frame(metric_root: Path, *, window_start: int, window_end: int) -> pd.DataFrame:
    hops = read_one(metric_root, "hops").rename(
        columns={name: f"hops_{name}" for name in ("motif000056", "motif000061", "xgrid", "plus_grid", "full_link")}
    )
    delay = read_one(metric_root, "delay_ms").rename(
        columns={name: f"delay_ms_{name}" for name in ("motif000056", "motif000061", "xgrid", "plus_grid", "full_link")}
    )
    df = hops.merge(delay, on="step", validate="one_to_one")
    df["hour"] = df["step"] / 3600.0
    mask = (df["step"] >= int(window_start)) & (df["step"] <= int(window_end))
    df["use_motif000061"] = mask.astype(int)
    for metric in ("hops", "delay_ms"):
        df[f"{metric}_dynamic_056_061_10h_15h"] = np.where(
            mask.to_numpy(),
            df[f"{metric}_motif000061"].to_numpy(dtype=np.float64),
            df[f"{metric}_motif000056"].to_numpy(dtype=np.float64),
        )
    return df


def write_timeseries(df: pd.DataFrame, out_dir: Path) -> Path:
    columns = ["step", "hour", "use_motif000061"]
    for metric in ("hops", "delay_ms"):
        for name in SERIES:
            columns.append(f"{metric}_{name}")
    path = out_dir / "china_europe_dynamic056061_vs_056_061_plusgrid_xgrid_timeseries.csv"
    df[columns].to_csv(path, index=False, encoding="utf-8-sig")
    return path


def summarize(df: pd.DataFrame, out_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    baseline_dynamic = "dynamic_056_061_10h_15h"
    for metric in ("hops", "delay_ms"):
        dynamic_values = df[f"{metric}_{baseline_dynamic}"].to_numpy(dtype=np.float64)
        for name in SERIES:
            values = df[f"{metric}_{name}"].to_numpy(dtype=np.float64)
            diff_vs_dynamic = values - dynamic_values
            rows.append(
                {
                    "metric": metric,
                    "topology": name,
                    "label": SERIES[name]["label"],
                    "mean": float(np.nanmean(values)),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                    "p95": float(np.nanpercentile(values, 95)),
                    "mean_minus_dynamic": float(np.nanmean(diff_vs_dynamic)),
                    "dynamic_better_steps": int(np.count_nonzero(dynamic_values < values)),
                    "dynamic_worse_steps": int(np.count_nonzero(dynamic_values > values)),
                    "equal_steps": int(np.count_nonzero(np.isclose(dynamic_values, values, rtol=0.0, atol=1e-9))),
                    "steps": int(values.size),
                }
            )
    path = out_dir / "china_europe_dynamic056061_vs_056_061_plusgrid_xgrid_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_metric(df: pd.DataFrame, out_path: Path, *, metric: str, y_label: str, window_start: int, window_end: int) -> None:
    fig, ax = plt.subplots(figsize=(15.8, 6.7), dpi=180)
    x = df["hour"].to_numpy(dtype=np.float64)
    for name, spec in SERIES.items():
        values = df[f"{metric}_{name}"].to_numpy(dtype=np.float64)
        ax.plot(
            x,
            values,
            color=str(spec["color"]),
            linewidth=float(spec["width"]),
            alpha=float(spec["alpha"]),
            label=f"{spec['label']} | mean={np.nanmean(values):.3f}",
        )
    ax.axvspan(window_start / 3600.0, window_end / 3600.0, color="#f59e0b", alpha=0.08, linewidth=0)
    ax.set_title(f"G60 China-Europe: dynamic 056/061 vs motif and grid baselines, {y_label}")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    ax.legend(loc="best", fontsize=8.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_bar_summary(summary_path: Path, out_path: Path) -> None:
    summary = pd.read_csv(summary_path)
    order = list(SERIES.keys())
    labels = [SERIES[name]["label"] for name in order]
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.2), dpi=180)
    for ax, metric, title in zip(axes, ("hops", "delay_ms"), ("mean shortest hops", "mean shortest delay (ms)")):
        part = summary[summary["metric"] == metric].set_index("topology").loc[order]
        colors = [SERIES[name]["color"] for name in order]
        bars = ax.bar(range(len(order)), part["mean"].to_numpy(dtype=float), color=colors, alpha=0.82)
        for bar, value in zip(bars, part["mean"].to_numpy(dtype=float)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_title(title)
        ax.set_xticks(range(len(order)), labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("China-Europe mean metric comparison", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_compare_frame(Path(args.metric_root), window_start=int(args.window_start), window_end=int(args.window_end))
    timeseries_path = write_timeseries(df, out_dir)
    summary_path = summarize(df, out_dir)
    plot_metric(
        df,
        out_dir / "china_europe_dynamic056061_vs_056_061_plusgrid_xgrid_hops.png",
        metric="hops",
        y_label="mean shortest hops",
        window_start=int(args.window_start),
        window_end=int(args.window_end),
    )
    plot_metric(
        df,
        out_dir / "china_europe_dynamic056061_vs_056_061_plusgrid_xgrid_delay_ms.png",
        metric="delay_ms",
        y_label="mean shortest delay (ms)",
        window_start=int(args.window_start),
        window_end=int(args.window_end),
    )
    plot_bar_summary(summary_path, out_dir / "china_europe_dynamic056061_vs_056_061_plusgrid_xgrid_mean_bars.png")
    print(f"[dynamic056061-compare] out_dir={out_dir}", flush=True)
    print(f"[dynamic056061-compare] timeseries={timeseries_path}", flush=True)
    print(f"[dynamic056061-compare] summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
