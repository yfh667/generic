from __future__ import annotations

import argparse
import csv
import json
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
    r"\motif_w_le4_h_le3\two_motif_056_061_china_europe_t0_86160_stride60"
)
PAIR = "china_europe"
BASE_NAME = "combined_motif_000056"
PATCH_NAME = "combined_motif_000061"
BASE_LABEL = "motif 000056: DBD | --B"
PATCH_LABEL = "motif 000061: DCD | C--"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine motif 000056 and 000061 using existing China-Europe metric CSVs."
    )
    parser.add_argument("--metric-root", type=Path, default=DEFAULT_METRIC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window-start", type=int, default=41040)
    parser.add_argument("--window-end", type=int, default=53820)
    return parser.parse_args()


def contiguous_segments(mask: np.ndarray, steps: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    start_idx: int | None = None
    for idx, enabled in enumerate(mask.astype(bool)):
        if enabled and start_idx is None:
            start_idx = idx
        if (not enabled or idx == len(mask) - 1) and start_idx is not None:
            end_idx = idx if enabled and idx == len(mask) - 1 else idx - 1
            out.append(
                {
                    "start_step": int(steps[start_idx]),
                    "end_step": int(steps[end_idx]),
                    "rows": int(end_idx - start_idx + 1),
                    "duration_h": float((end_idx - start_idx + 1) * infer_stride(steps) / 3600.0),
                }
            )
            start_idx = None
    return out


def infer_stride(steps: np.ndarray) -> int:
    if len(steps) <= 1:
        return 0
    return int(np.median(np.diff(steps)))


def read_metrics(metric_root: Path) -> pd.DataFrame:
    hops_path = (
        metric_root
        / "shortest_hops"
        / PAIR
        / "compare_mean_shortest_hops_808_strict_reachable.csv"
    )
    delay_path = (
        metric_root
        / "shortest_delay"
        / PAIR
        / "compare_mean_shortest_delay_ms_808_strict_reachable.csv"
    )
    if not hops_path.exists():
        raise FileNotFoundError(hops_path)
    if not delay_path.exists():
        raise FileNotFoundError(delay_path)
    hops = pd.read_csv(hops_path, usecols=["step", BASE_NAME, PATCH_NAME]).rename(
        columns={BASE_NAME: "hops_000056", PATCH_NAME: "hops_000061"}
    )
    delay = pd.read_csv(delay_path, usecols=["step", BASE_NAME, PATCH_NAME]).rename(
        columns={BASE_NAME: "delay_ms_000056", PATCH_NAME: "delay_ms_000061"}
    )
    df = hops.merge(delay, on="step", validate="one_to_one").sort_values("step").reset_index(drop=True)
    df["hop_gain_061_vs_056"] = df["hops_000056"] - df["hops_000061"]
    df["delay_gain_ms_061_vs_056"] = df["delay_ms_000056"] - df["delay_ms_000061"]
    return df


def add_dynamic_columns(df: pd.DataFrame, *, window_start: int, window_end: int) -> pd.DataFrame:
    out = df.copy()
    window_mask = (out["step"] >= int(window_start)) & (out["step"] <= int(window_end))
    out["use_061_window"] = window_mask.astype(int)
    for metric in ("hops", "delay_ms"):
        base = out[f"{metric}_000056"].to_numpy(dtype=np.float64)
        patch = out[f"{metric}_000061"].to_numpy(dtype=np.float64)
        out[f"{metric}_dynamic_window"] = np.where(window_mask.to_numpy(), patch, base)
    return out


def write_summary(df: pd.DataFrame, out_dir: Path, *, window_start: int, window_end: int) -> Path:
    rows: list[dict[str, Any]] = []
    series = [
        ("static_000056", "000056 all day", "hops_000056", "delay_ms_000056"),
        ("static_000061", "000061 all day", "hops_000061", "delay_ms_000061"),
        (
            f"dynamic_window_{int(window_start)}_{int(window_end)}",
            f"use 000061 in the specified window {int(window_start)}..{int(window_end)}s",
            "hops_dynamic_window",
            "delay_ms_dynamic_window",
        ),
    ]
    for name, rule, hop_col, delay_col in series:
        rows.append(
            {
                "topology": name,
                "selection_rule": rule,
                "mean_hops": float(df[hop_col].mean()),
                "mean_delay_ms": float(df[delay_col].mean()),
                "min_hops": float(df[hop_col].min()),
                "max_hops": float(df[hop_col].max()),
                "min_delay_ms": float(df[delay_col].min()),
                "max_delay_ms": float(df[delay_col].max()),
            }
        )
    path = out_dir / "china_europe_motif0056_0061_splice_summary.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_metric(
    *,
    df: pd.DataFrame,
    out_path: Path,
    metric: str,
    y_label: str,
    title: str,
    window_start: int,
    window_end: int,
) -> None:
    x = df["step"].to_numpy(dtype=np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(15.4, 6.5), dpi=180)
    lines = [
        (f"{metric}_000056", BASE_LABEL, "#1B4F9C", 1.25, 0.95),
        (f"{metric}_000061", PATCH_LABEL, "#C1121F", 1.05, 0.72),
        (
            f"{metric}_dynamic_window",
            f"dynamic window: 061 from {int(window_start)}..{int(window_end)}s",
            "#f59e0b",
            1.20,
            0.92,
        ),
    ]
    for column, label, color, width, alpha in lines:
        values = df[column].to_numpy(dtype=np.float64)
        ax.plot(
            x,
            values,
            color=color,
            linewidth=width,
            alpha=alpha,
            label=f"{label} | mean={np.nanmean(values):.3f}",
        )
    ax.axvspan(window_start / 3600.0, window_end / 3600.0, color="#f59e0b", alpha=0.08, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.26, linestyle="--", linewidth=0.55)
    ax.legend(loc="best", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_schedule(df: pd.DataFrame, out_dir: Path, *, window_start: int, window_end: int) -> dict[str, Any]:
    schedule_path = out_dir / "china_europe_motif0056_0061_splice_by_step.csv"
    cols = [
        "step",
        "hops_000056",
        "hops_000061",
        "delay_ms_000056",
        "delay_ms_000061",
        "hop_gain_061_vs_056",
        "delay_gain_ms_061_vs_056",
        "use_061_window",
        "hops_dynamic_window",
        "delay_ms_dynamic_window",
    ]
    df[cols].to_csv(schedule_path, index=False, encoding="utf-8-sig")
    steps = df["step"].to_numpy(dtype=np.int64)
    window_segments = contiguous_segments(df["use_061_window"].to_numpy(dtype=bool), steps)
    segment_path = out_dir / "china_europe_motif0056_0061_splice_segments.json"
    meta = {
        "pair": PAIR,
        "base": {"id": 56, "label": BASE_LABEL},
        "patch": {"id": 61, "label": PATCH_LABEL},
        "step_start": int(steps[0]),
        "step_end": int(steps[-1]),
        "stride": infer_stride(steps),
        "main_window_start": int(window_start),
        "main_window_end": int(window_end),
        "main_window_rows": int(df["use_061_window"].sum()),
        "main_window_hours": float(df["use_061_window"].sum() * infer_stride(steps) / 3600.0),
        "main_window_segments": window_segments,
        "schedule_csv": str(schedule_path),
    }
    segment_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = add_dynamic_columns(
        read_metrics(Path(args.metric_root)),
        window_start=int(args.window_start),
        window_end=int(args.window_end),
    )
    schedule_meta = write_schedule(df, out_dir, window_start=int(args.window_start), window_end=int(args.window_end))
    summary_path = write_summary(
        df,
        out_dir,
        window_start=int(args.window_start),
        window_end=int(args.window_end),
    )
    plot_metric(
        df=df,
        out_path=out_dir / "china_europe_motif0056_0061_splice_mean_shortest_hops.png",
        metric="hops",
        y_label="mean shortest hops",
        title="G60 China-Europe: motif 000056 / 000061 splice, mean shortest hops",
        window_start=int(args.window_start),
        window_end=int(args.window_end),
    )
    plot_metric(
        df=df,
        out_path=out_dir / "china_europe_motif0056_0061_splice_mean_shortest_delay_ms.png",
        metric="delay_ms",
        y_label="mean shortest delay (ms)",
        title="G60 China-Europe: motif 000056 / 000061 splice, mean shortest delay",
        window_start=int(args.window_start),
        window_end=int(args.window_end),
    )
    print(f"[motif0056-0061-splice] out_dir={out_dir}", flush=True)
    print(f"[motif0056-0061-splice] summary={summary_path}", flush=True)
    print(
        "[motif0056-0061-splice] "
        f"main_window={schedule_meta['main_window_start']}..{schedule_meta['main_window_end']} "
        f"main_window_hours={schedule_meta['main_window_hours']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
