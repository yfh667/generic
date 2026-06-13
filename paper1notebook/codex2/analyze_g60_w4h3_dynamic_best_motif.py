# -*- coding: utf-8 -*-
"""
Analyze the dynamic best motif among the selected 4x3 motif shortest-delay runs.

This script does not recompute shortest paths. It reads the existing time-series
CSV whose columns are motif/topology names and whose rows are time steps, then
builds an oracle schedule that picks the lowest-delay selected motif at each
time step.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "data"
    / "linshi"
    / "g60_w4h3_selected100_shortest_delay_t0_86164_stride60"
)
DEFAULT_COMPARE_CSV = DEFAULT_RESULT_DIR / "compare_100motifs_gridplus_full_link.csv"
DEFAULT_MOTIF_SUMMARY_CSV = DEFAULT_RESULT_DIR / "motif_summary_sorted.csv"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "linshi"
    / "g60_w4h3_dynamic_best_selected100_t0_86164_stride60"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the per-time-step dynamic best motif among selected motif delay curves."
    )
    parser.add_argument("--compare-csv", type=Path, default=DEFAULT_COMPARE_CSV)
    parser.add_argument("--motif-summary-csv", type=Path, default=DEFAULT_MOTIF_SUMMARY_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--static-best",
        type=str,
        default=None,
        help="Static baseline topology name. Defaults to the first row of motif_summary_sorted.csv.",
    )
    parser.add_argument(
        "--plot-all-motifs",
        action="store_true",
        help="Also draw all 100 motif curves as faint background lines.",
    )
    return parser.parse_args()


def motif_id_from_name(name: str) -> int | None:
    match = re.fullmatch(r"motif_(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def read_motif_metadata(path: Path) -> tuple[dict[str, dict[str, str]], str | None]:
    if not path.exists():
        return {}, None
    df = pd.read_csv(path)
    metadata: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        topology = str(row["topology"])
        metadata[topology] = {
            "motif_id": str(int(row["motif_id"])),
            "motif": str(row["motif"]),
            "edges": str(row.get("edges", "")),
            "mean_delay_ms": str(row.get("mean_delay_ms", "")),
        }
    static_best = str(df.iloc[0]["topology"]) if len(df) else None
    return metadata, static_best


def build_dynamic_table(
    compare_df: pd.DataFrame,
    motif_cols: list[str],
    static_best: str,
    metadata: dict[str, dict[str, str]],
) -> pd.DataFrame:
    values = compare_df[motif_cols].to_numpy(dtype=np.float64)
    best_idx = np.argmin(values, axis=1)
    best_values = values[np.arange(values.shape[0]), best_idx]

    if values.shape[1] > 1:
        two_idx = np.argpartition(values, kth=1, axis=1)[:, :2]
        two_vals = values[np.arange(values.shape[0])[:, None], two_idx]
        order = np.argsort(two_vals, axis=1)
        second_idx = two_idx[np.arange(values.shape[0]), order[:, 1]]
        second_values = values[np.arange(values.shape[0]), second_idx]
    else:
        second_idx = best_idx
        second_values = np.full(values.shape[0], np.nan)

    best_names = np.array(motif_cols, dtype=object)[best_idx]
    second_names = np.array(motif_cols, dtype=object)[second_idx]
    steps = compare_df["step"].to_numpy(dtype=np.int64)
    static_values = compare_df[static_best].to_numpy(dtype=np.float64)

    out = pd.DataFrame(
        {
            "step": steps,
            "dynamic_best_topology": best_names,
            "dynamic_best_motif_id": [motif_id_from_name(str(x)) for x in best_names],
            "dynamic_best_motif": [metadata.get(str(x), {}).get("motif", "") for x in best_names],
            "dynamic_best_delay_ms": best_values,
            "second_best_topology": second_names,
            "second_best_delay_ms": second_values,
            "gap_to_second_best_ms": second_values - best_values,
            "static_best_topology": static_best,
            "static_best_delay_ms": static_values,
            "improvement_vs_static_best_ms": static_values - best_values,
            "improvement_vs_static_best_pct": (static_values - best_values) / static_values * 100.0,
        }
    )
    for baseline in ("gridplus", "full_link"):
        if baseline in compare_df.columns:
            out[f"{baseline}_delay_ms"] = compare_df[baseline].to_numpy(dtype=np.float64)
            out[f"improvement_vs_{baseline}_ms"] = out[f"{baseline}_delay_ms"] - best_values
    return out


def build_winner_counts(
    dynamic_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    motif_cols: list[str],
    metadata: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows = []
    total_steps = len(dynamic_df)
    winners = dynamic_df["dynamic_best_topology"]
    for topology in motif_cols:
        mask = winners == topology
        all_values = compare_df[topology].to_numpy(dtype=np.float64)
        row = {
            "topology": topology,
            "motif_id": motif_id_from_name(topology),
            "motif": metadata.get(topology, {}).get("motif", ""),
            "win_count": int(mask.sum()),
            "win_fraction": float(mask.sum() / total_steps) if total_steps else 0.0,
            "mean_delay_all_ms": float(np.mean(all_values)),
            "min_delay_all_ms": float(np.min(all_values)),
            "max_delay_all_ms": float(np.max(all_values)),
        }
        if mask.any():
            row["mean_delay_when_win_ms"] = float(dynamic_df.loc[mask, "dynamic_best_delay_ms"].mean())
            row["first_win_step"] = int(dynamic_df.loc[mask, "step"].iloc[0])
            row["last_win_step"] = int(dynamic_df.loc[mask, "step"].iloc[-1])
        else:
            row["mean_delay_when_win_ms"] = np.nan
            row["first_win_step"] = np.nan
            row["last_win_step"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["win_count", "mean_delay_all_ms"], ascending=[False, True])


def build_segments(dynamic_df: pd.DataFrame) -> pd.DataFrame:
    if dynamic_df.empty:
        return pd.DataFrame()
    rows = []
    start_idx = 0
    names = dynamic_df["dynamic_best_topology"].to_numpy(dtype=object)
    steps = dynamic_df["step"].to_numpy(dtype=np.int64)
    for idx in range(1, len(dynamic_df) + 1):
        if idx == len(dynamic_df) or names[idx] != names[start_idx]:
            block = dynamic_df.iloc[start_idx:idx]
            rows.append(
                {
                    "segment_index": len(rows),
                    "topology": str(names[start_idx]),
                    "motif_id": motif_id_from_name(str(names[start_idx])),
                    "motif": str(block["dynamic_best_motif"].iloc[0]),
                    "start_step": int(steps[start_idx]),
                    "end_step": int(steps[idx - 1]),
                    "num_samples": int(idx - start_idx),
                    "mean_dynamic_delay_ms": float(block["dynamic_best_delay_ms"].mean()),
                    "min_dynamic_delay_ms": float(block["dynamic_best_delay_ms"].min()),
                    "max_dynamic_delay_ms": float(block["dynamic_best_delay_ms"].max()),
                }
            )
            start_idx = idx
    return pd.DataFrame(rows)


def write_examples(dynamic_df: pd.DataFrame, out_dir: Path) -> None:
    improved = dynamic_df[dynamic_df["improvement_vs_static_best_ms"] > 1e-9].copy()
    improved = improved.sort_values("improvement_vs_static_best_ms", ascending=False)
    improved.head(30).to_csv(out_dir / "largest_improvements_vs_static_best.csv", index=False)

    static_wins = dynamic_df[
        dynamic_df["dynamic_best_topology"] == dynamic_df["static_best_topology"]
    ].copy()
    static_wins.head(30).to_csv(out_dir / "examples_static_best_is_timewise_best.csv", index=False)


def plot_outputs(
    compare_df: pd.DataFrame,
    dynamic_df: pd.DataFrame,
    winner_counts: pd.DataFrame,
    out_dir: Path,
    motif_cols: list[str],
    plot_all_motifs: bool,
) -> None:
    import matplotlib.pyplot as plt

    x_hours = dynamic_df["step"].to_numpy(dtype=np.float64) / 3600.0

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    if plot_all_motifs:
        for col in motif_cols:
            ax.plot(x_hours, compare_df[col], color="#b0b0b0", linewidth=0.45, alpha=0.22)
    ax.plot(x_hours, dynamic_df["dynamic_best_delay_ms"], color="#d62728", linewidth=2.0, label="dynamic best among 100 motifs")
    ax.plot(x_hours, dynamic_df["static_best_delay_ms"], color="#1f77b4", linewidth=1.4, label=f"static best {dynamic_df['static_best_topology'].iloc[0]}")
    if "gridplus_delay_ms" in dynamic_df:
        ax.plot(x_hours, dynamic_df["gridplus_delay_ms"], color="#2ca02c", linewidth=1.2, label="gridplus")
    if "full_link_delay_ms" in dynamic_df:
        ax.plot(x_hours, dynamic_df["full_link_delay_ms"], color="#111111", linewidth=1.2, label="full link")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("mean shortest delay (ms)")
    ax.set_title("Dynamic best selected motif vs static baselines")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "dynamic_best_vs_static_gridplus_full_link.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=160)
    y = dynamic_df["dynamic_best_motif_id"].to_numpy(dtype=np.float64)
    ax.scatter(x_hours, y, s=9, c=dynamic_df["dynamic_best_delay_ms"], cmap="turbo_r", linewidths=0)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("winning motif id")
    ax.set_title("Timewise winning motif identity")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "dynamic_winner_timeline.png")
    plt.close(fig)

    top = winner_counts[winner_counts["win_count"] > 0].head(25).copy()
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=160)
    labels = [f"{int(row.motif_id):04d}\n{row.motif}" for row in top.itertuples()]
    ax.bar(range(len(top)), top["win_count"], color="#b2182b")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("winning samples")
    ax.set_title("Top timewise winners among selected motifs")
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "dynamic_winner_counts_top25.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=160)
    ax.plot(x_hours, dynamic_df["improvement_vs_static_best_ms"], color="#9467bd", linewidth=1.1)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("static best - dynamic best (ms)")
    ax.set_title("Dynamic motif switching gain over the best static selected motif")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "dynamic_gain_vs_static_best.png")
    plt.close(fig)


def write_summary(
    args: argparse.Namespace,
    dynamic_df: pd.DataFrame,
    winner_counts: pd.DataFrame,
    segments: pd.DataFrame,
    motif_cols: list[str],
    out_dir: Path,
) -> None:
    static_best = str(dynamic_df["static_best_topology"].iloc[0])
    static_wins = int((dynamic_df["dynamic_best_topology"] == static_best).sum())
    positive_gain = dynamic_df["improvement_vs_static_best_ms"] > 1e-9
    summary = {
        "compare_csv": str(Path(args.compare_csv)),
        "motif_summary_csv": str(Path(args.motif_summary_csv)),
        "out_dir": str(out_dir),
        "num_selected_motifs": int(len(motif_cols)),
        "num_steps": int(len(dynamic_df)),
        "start_step": int(dynamic_df["step"].iloc[0]),
        "end_step": int(dynamic_df["step"].iloc[-1]),
        "static_best_topology": static_best,
        "static_best_mean_delay_ms": float(dynamic_df["static_best_delay_ms"].mean()),
        "dynamic_best_mean_delay_ms": float(dynamic_df["dynamic_best_delay_ms"].mean()),
        "dynamic_gain_mean_ms": float(dynamic_df["improvement_vs_static_best_ms"].mean()),
        "dynamic_gain_max_ms": float(dynamic_df["improvement_vs_static_best_ms"].max()),
        "dynamic_gain_positive_steps": int(positive_gain.sum()),
        "dynamic_gain_positive_fraction": float(positive_gain.mean()),
        "static_best_timewise_win_count": static_wins,
        "static_best_timewise_win_fraction": float(static_wins / len(dynamic_df)),
        "unique_timewise_winner_count": int((winner_counts["win_count"] > 0).sum()),
        "switch_count": int(max(len(segments) - 1, 0)),
        "segment_count": int(len(segments)),
        "top_winners": winner_counts.head(10).to_dict(orient="records"),
    }
    for baseline in ("gridplus", "full_link"):
        col = f"{baseline}_delay_ms"
        if col in dynamic_df:
            summary[f"{baseline}_mean_delay_ms"] = float(dynamic_df[col].mean())
            summary[f"dynamic_gain_vs_{baseline}_mean_ms"] = float(
                (dynamic_df[col] - dynamic_df["dynamic_best_delay_ms"]).mean()
            )
    with (out_dir / "dynamic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    compare_path = Path(args.compare_csv)
    if not compare_path.exists():
        raise FileNotFoundError(compare_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compare_df = pd.read_csv(compare_path)
    if "step" not in compare_df.columns:
        raise ValueError(f"{compare_path} must contain a 'step' column")
    motif_cols = [c for c in compare_df.columns if c.startswith("motif_")]
    if not motif_cols:
        raise ValueError(f"No motif columns found in {compare_path}")

    metadata, summary_static_best = read_motif_metadata(Path(args.motif_summary_csv))
    static_best = args.static_best or summary_static_best or motif_cols[0]
    if static_best not in motif_cols:
        raise ValueError(f"static best topology {static_best!r} is not a motif column in {compare_path}")

    dynamic_df = build_dynamic_table(compare_df, motif_cols, static_best, metadata)
    winner_counts = build_winner_counts(dynamic_df, compare_df, motif_cols, metadata)
    segments = build_segments(dynamic_df)

    dynamic_df.to_csv(out_dir / "dynamic_best_by_step.csv", index=False)
    winner_counts.to_csv(out_dir / "dynamic_winner_counts.csv", index=False)
    segments.to_csv(out_dir / "dynamic_winner_segments.csv", index=False)
    write_examples(dynamic_df, out_dir)
    write_summary(args, dynamic_df, winner_counts, segments, motif_cols, out_dir)
    plot_outputs(compare_df, dynamic_df, winner_counts, out_dir, motif_cols, args.plot_all_motifs)

    print(f"[dynamic-best] selected motifs: {len(motif_cols)}")
    print(f"[dynamic-best] steps: {len(dynamic_df)} range={dynamic_df['step'].iloc[0]}..{dynamic_df['step'].iloc[-1]}")
    print(f"[dynamic-best] static best: {static_best}")
    print(f"[dynamic-best] dynamic mean delay: {dynamic_df['dynamic_best_delay_ms'].mean():.6f} ms")
    print(f"[dynamic-best] static best mean delay: {dynamic_df['static_best_delay_ms'].mean():.6f} ms")
    print(f"[dynamic-best] unique winners: {(winner_counts['win_count'] > 0).sum()}")
    print(f"[dynamic-best] switches: {max(len(segments) - 1, 0)}")
    print(f"[dynamic-best] out_dir: {out_dir}")


if __name__ == "__main__":
    main()
