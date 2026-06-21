from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DELAY_CSV = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3"
    r"\paper_style_808_no_region_grid_t0_86160_stride60\shortest_delay\china_europe"
    r"\compare_mean_shortest_delay_ms_808_strict_reachable.csv"
)
HOPS_CSV = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3"
    r"\paper_style_808_no_region_grid_t0_86160_stride60\shortest_hops\china_europe"
    r"\compare_mean_shortest_hops_808_strict_reachable.csv"
)
MOTIF_LIBRARY = Path(
    r"E:\paper11\data\satnet_experiments\libraries\motif\exact_box"
    r"\w_le_4_h_le_3\combined_w_le4_h_le3_808.csv"
)
COMMON_SPIKE_SUMMARY = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_0056_0061_common_spike_analysis\china_europe_t0_86160_stride60"
    r"\top_common_delay_spikes_summary.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_0056_0061_common_spike_analysis\spike_resistant_motif_search_808"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find motifs that suppress the local endpoint-handover spikes seen in motif000056/000061."
    )
    parser.add_argument("--delay-csv", type=Path, default=DELAY_CSV)
    parser.add_argument("--hops-csv", type=Path, default=HOPS_CSV)
    parser.add_argument("--motif-library", type=Path, default=MOTIF_LIBRARY)
    parser.add_argument("--spike-summary", type=Path, default=COMMON_SPIKE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--plot-top", type=int, default=8)
    return parser.parse_args()


def motif_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("combined_motif_")]


def local_spike_table(
    df: pd.DataFrame,
    *,
    spike_steps: list[int],
    metric_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = motif_columns(df)
    if "full_link" in df.columns:
        cols_with_full = cols + ["full_link"]
    else:
        cols_with_full = cols
    by_step = df.set_index("step")
    rows: list[dict[str, object]] = []
    for step in spike_steps:
        prev = int(step) - int(df["step"].iloc[1] - df["step"].iloc[0])
        if prev not in by_step.index or int(step) not in by_step.index:
            continue
        diff = by_step.loc[int(step), cols_with_full] - by_step.loc[int(prev), cols_with_full]
        for col in cols_with_full:
            value = float(diff[col])
            if not np.isfinite(value):
                positive_jump = float("nan")
            else:
                positive_jump = max(0.0, value)
            rows.append(
                {
                    "metric": metric_label,
                    "step": int(step),
                    "hour": float(step) / 3600.0,
                    "topology": col,
                    "positive_jump": positive_jump,
                    "signed_jump": value,
                    "value_before": float(by_step.loc[int(prev), col]),
                    "value_after": float(by_step.loc[int(step), col]),
                }
            )
    step_detail = pd.DataFrame(rows)
    grouped = step_detail.groupby("topology", as_index=False)
    summary = grouped.agg(
        local_positive_jump_mean=("positive_jump", "mean"),
        local_positive_jump_max=("positive_jump", "max"),
        local_positive_jump_sum=("positive_jump", "sum"),
        finite_spike_points=("positive_jump", lambda x: int(np.count_nonzero(np.isfinite(np.asarray(x, dtype=float))))),
        local_signed_jump_mean=("signed_jump", "mean"),
        local_abs_signed_jump_mean=("signed_jump", lambda x: float(np.mean(np.abs(np.asarray(x, dtype=float))))),
        local_value_after_mean=("value_after", "mean"),
        local_value_after_max=("value_after", "max"),
    )
    full = summary[summary["topology"] == "full_link"]
    if not full.empty:
        full_mean = float(full["local_positive_jump_mean"].iloc[0])
        full_max = float(full["local_positive_jump_max"].iloc[0])
        summary["excess_positive_jump_mean_vs_full_link"] = summary["local_positive_jump_mean"] - full_mean
        summary["excess_positive_jump_max_vs_full_link"] = summary["local_positive_jump_max"] - full_max
    summary.insert(0, "metric", metric_label)
    return summary, step_detail


def add_motif_meta(summary: pd.DataFrame, motif_lib: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["motif_id"] = out["topology"].str.extract(r"combined_motif_(\d+)").astype(float)
    meta = motif_lib.copy()
    meta["motif_id"] = meta["motif_id"].astype(float)
    out = out.merge(
        meta[["motif_id", "source_w", "source_h", "motif", "edge_count", "support", "edges"]],
        on="motif_id",
        how="left",
    )
    return out


def robust_rank(summary: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    finite = summary[summary["topology"] != "full_link"].copy()
    finite = finite.replace([np.inf, -np.inf], np.nan)
    required_points = int(summary["finite_spike_points"].max())
    finite = finite[finite["finite_spike_points"] >= required_points]
    finite = finite.dropna(
        subset=[
            "local_positive_jump_mean",
            "local_positive_jump_max",
            "local_value_after_mean",
            "local_value_after_max",
        ]
    )
    # Local problem only: rank spike suppression first. Average value is only a tie-breaker.
    finite = finite.sort_values(
        ["local_positive_jump_mean", "local_positive_jump_max", "local_value_after_mean"],
        ascending=[True, True, True],
    )
    return finite.head(int(top_n)).reset_index(drop=True)


def plot_candidates(
    *,
    df: pd.DataFrame,
    candidates: list[str],
    spike_steps: list[int],
    out_path: Path,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14.8, 6.4), dpi=150)
    hours = df["step"].to_numpy(dtype=float) / 3600.0
    styles = {
        "full_link": ("full_link", "#111827", 2.0, 0.9),
        "combined_motif_000056": ("motif000056", "#1B4F9C", 1.4, 0.7),
        "combined_motif_000061": ("motif000061", "#C1121F", 1.4, 0.7),
    }
    for col, (label, color, width, alpha) in styles.items():
        if col in df.columns:
            ax.plot(hours, df[col], label=label, color=color, linewidth=width, alpha=alpha)
    palette = ["#d97706", "#0f766e", "#7c3aed", "#be123c", "#0369a1", "#4d7c0f", "#9333ea", "#ca8a04"]
    for idx, col in enumerate(candidates):
        if col not in df.columns:
            continue
        ax.plot(
            hours,
            df[col],
            label=col.replace("combined_motif_", "motif"),
            color=palette[idx % len(palette)],
            linewidth=1.1,
            alpha=0.82,
        )
    for step in spike_steps:
        ax.axvline(float(step) / 3600.0, color="#64748b", linewidth=0.55, alpha=0.18)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.set_title("Candidate motifs that suppress local endpoint-handover spikes")
    ax.grid(True, alpha=0.24, linewidth=0.6)
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spike_summary = pd.read_csv(args.spike_summary)
    spike_steps = [int(x) for x in spike_summary["step"].tolist()]
    motif_lib = pd.read_csv(args.motif_library)

    delay = pd.read_csv(args.delay_csv)
    hops = pd.read_csv(args.hops_csv)

    delay_summary, delay_detail = local_spike_table(delay, spike_steps=spike_steps, metric_label="delay_ms")
    hops_summary, hops_detail = local_spike_table(hops, spike_steps=spike_steps, metric_label="hops")
    delay_summary = add_motif_meta(delay_summary, motif_lib)
    hops_summary = add_motif_meta(hops_summary, motif_lib)

    delay_rank = robust_rank(delay_summary, top_n=int(args.top_n))
    hops_rank = robust_rank(hops_summary, top_n=int(args.top_n))

    # Combined local score keeps delay primary, hops secondary.
    combined = delay_summary[delay_summary["topology"] != "full_link"].copy()
    combined = combined.merge(
        hops_summary[
            [
                "topology",
                "finite_spike_points",
                "local_positive_jump_mean",
                "local_positive_jump_max",
                "local_value_after_mean",
            ]
        ].rename(
            columns={
                "finite_spike_points": "hops_finite_spike_points",
                "local_positive_jump_mean": "hops_local_positive_jump_mean",
                "local_positive_jump_max": "hops_local_positive_jump_max",
                "local_value_after_mean": "hops_local_value_after_mean",
            }
        ),
        on="topology",
        how="left",
    )
    required_delay_points = int(delay_summary["finite_spike_points"].max())
    required_hops_points = int(hops_summary["finite_spike_points"].max())
    combined = combined[
        (combined["finite_spike_points"] >= required_delay_points)
        & (combined["hops_finite_spike_points"] >= required_hops_points)
    ]
    for col in [
        "local_positive_jump_mean",
        "local_positive_jump_max",
        "hops_local_positive_jump_mean",
        "hops_local_positive_jump_max",
    ]:
        values = combined[col].to_numpy(dtype=float)
        mn, mx = np.nanmin(values), np.nanmax(values)
        combined[col + "_norm"] = (values - mn) / max(1e-9, mx - mn)
    combined["local_spike_score"] = (
        0.55 * combined["local_positive_jump_mean_norm"]
        + 0.25 * combined["local_positive_jump_max_norm"]
        + 0.15 * combined["hops_local_positive_jump_mean_norm"]
        + 0.05 * combined["hops_local_positive_jump_max_norm"]
    )
    combined_rank = combined.sort_values(
        ["local_spike_score", "local_positive_jump_mean", "local_positive_jump_max"],
        ascending=[True, True, True],
    ).head(int(args.top_n))

    delay_summary.to_csv(out_dir / "all_motifs_local_delay_spike_summary.csv", index=False, encoding="utf-8-sig")
    hops_summary.to_csv(out_dir / "all_motifs_local_hops_spike_summary.csv", index=False, encoding="utf-8-sig")
    delay_detail.to_csv(out_dir / "all_motifs_local_delay_spike_by_step.csv", index=False, encoding="utf-8-sig")
    hops_detail.to_csv(out_dir / "all_motifs_local_hops_spike_by_step.csv", index=False, encoding="utf-8-sig")
    delay_rank.to_csv(out_dir / "top_delay_spike_resistant_motifs.csv", index=False, encoding="utf-8-sig")
    hops_rank.to_csv(out_dir / "top_hops_spike_resistant_motifs.csv", index=False, encoding="utf-8-sig")
    combined_rank.to_csv(out_dir / "top_combined_spike_resistant_motifs.csv", index=False, encoding="utf-8-sig")

    plot_candidates(
        df=delay,
        candidates=combined_rank["topology"].head(int(args.plot_top)).tolist(),
        spike_steps=spike_steps,
        out_path=out_dir / "top_spike_resistant_motifs_delay_timeseries.png",
        ylabel="mean shortest delay (ms)",
    )
    plot_candidates(
        df=hops,
        candidates=combined_rank["topology"].head(int(args.plot_top)).tolist(),
        spike_steps=spike_steps,
        out_path=out_dir / "top_spike_resistant_motifs_hops_timeseries.png",
        ylabel="mean shortest hops",
    )
    meta = {
        "spike_steps": spike_steps,
        "definition": (
            "Local spike resistance is evaluated only at the common positive-jump steps "
            "previously identified from motif000056/motif000061. Overall mean performance is not part of "
            "the primary ranking; it is only used as a tie-breaker."
        ),
        "outputs": [
            "top_delay_spike_resistant_motifs.csv",
            "top_hops_spike_resistant_motifs.csv",
            "top_combined_spike_resistant_motifs.csv",
            "top_spike_resistant_motifs_delay_timeseries.png",
            "top_spike_resistant_motifs_hops_timeseries.png",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [
        "topology",
        "motif",
        "source_w",
        "source_h",
        "edge_count",
        "local_spike_score",
        "local_positive_jump_mean",
        "local_positive_jump_max",
        "hops_local_positive_jump_mean",
        "hops_local_positive_jump_max",
        "local_value_after_mean",
    ]
    print(combined_rank[cols].head(int(args.top_n)).round(4).to_string(index=False))
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
