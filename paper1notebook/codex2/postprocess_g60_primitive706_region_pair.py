from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build summary CSV/plot for one G60 4x3 primitive-706 region-pair run."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pair-key", type=str, required=True, help="Example: china_europe")
    parser.add_argument("--pair-label", type=str, required=True, help="Example: China-Europe")
    parser.add_argument(
        "--compare-name",
        type=str,
        default="compare_100motifs_gridplus_full_link.csv",
        help="Kept for compatibility with the selected-motif batch script.",
    )
    parser.add_argument(
        "--strict-reachable",
        action="store_true",
        help=(
            "Mask a topology value to NaN for a step unless every source-target "
            "pair is reachable at that step. This avoids finite-only averages "
            "making disconnected motifs look better than full_link."
        ),
    )
    parser.add_argument(
        "--library-label",
        type=str,
        default="706 primitive 4x3 motifs",
        help="Human-readable motif library label used in plot titles and legends.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="706motifs",
        help="Filename stem used for dynamic_best and plot outputs.",
    )
    return parser.parse_args()


def full_reachability_mask(topology_dir: Path, steps: np.ndarray) -> np.ndarray:
    summary_path = topology_dir / "step_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    required = summary["source_nodes"].to_numpy(dtype=np.int64) * summary["target_nodes"].to_numpy(dtype=np.int64)
    ok = summary["reachable_pairs"].to_numpy(dtype=np.int64) == required
    by_step = {int(step): bool(flag) for step, flag in zip(summary["step"].to_numpy(dtype=np.int64), ok)}
    return np.asarray([by_step.get(int(step), False) for step in steps], dtype=bool)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    compare_path = out_dir / args.compare_name
    summary_path = out_dir / "motif_summary_sorted.csv"
    meta_path = out_dir / "meta.json"

    compare = pd.read_csv(compare_path)
    motif_cols = [col for col in compare.columns if col.startswith("motif_")]
    if not motif_cols:
        raise ValueError(f"No motif_* columns found in {compare_path}")

    steps = compare["step"].to_numpy(dtype=np.int64)
    values = compare[motif_cols].to_numpy(dtype=np.float64)
    reachability: dict[str, int | float] = {}
    suffix = ""
    if args.strict_reachable:
        suffix = "_strict_reachable"
        valid_counts = []
        for col_idx, col in enumerate(motif_cols):
            mask = full_reachability_mask(out_dir / col, steps)
            valid_counts.append(int(np.sum(mask)))
            values[~mask, col_idx] = np.nan
        valid_counts_arr = np.asarray(valid_counts, dtype=np.int64)
        reachability = {
            "strict_reachable": True,
            "full_time_reachable_motifs": int(np.sum(valid_counts_arr == len(steps))),
            "min_valid_steps_per_motif": int(np.min(valid_counts_arr)),
            "mean_valid_steps_per_motif": float(np.mean(valid_counts_arr)),
            "total_masked_motif_step_values": int(np.sum(np.isnan(values))),
        }
    else:
        valid_counts_arr = np.full(len(motif_cols), len(steps), dtype=np.int64)
        reachability = {"strict_reachable": False}

    static_means = np.full(len(motif_cols), np.inf, dtype=np.float64)
    fully_valid = valid_counts_arr == len(steps)
    if np.any(fully_valid):
        static_means[fully_valid] = np.nanmean(values[:, fully_valid], axis=0)
    else:
        static_means = np.nanmean(values, axis=0)
    best_static_idx = int(np.nanargmin(static_means))
    best_static_col = motif_cols[best_static_idx]

    dynamic_idx = np.full(values.shape[0], -1, dtype=np.int64)
    dynamic_values = np.full(values.shape[0], np.nan, dtype=np.float64)
    for row_idx in range(values.shape[0]):
        row = values[row_idx]
        if np.all(np.isnan(row)):
            continue
        dynamic_idx[row_idx] = int(np.nanargmin(row))
        dynamic_values[row_idx] = float(row[dynamic_idx[row_idx]])
    dynamic_topologies = np.asarray(
        [motif_cols[idx] if idx >= 0 else "" for idx in dynamic_idx],
        dtype=object,
    )

    dynamic_df = pd.DataFrame(
        {
            "step": steps,
            "best_topology": dynamic_topologies,
            "best_motif_id": [
                int(str(name).split("_")[1]) if str(name).startswith("motif_") else None
                for name in dynamic_topologies
            ],
            "best_delay_ms": dynamic_values,
        }
    )
    if "gridplus" in compare:
        dynamic_df["gridplus_delay_ms"] = compare["gridplus"].to_numpy(dtype=np.float64)
    if "full_link" in compare:
        dynamic_df["full_link_delay_ms"] = compare["full_link"].to_numpy(dtype=np.float64)

    selected_rows_path = out_dir / "selected_motifs_original_rows.csv"
    selected_lookup = {}
    if selected_rows_path.exists():
        selected_df = pd.read_csv(selected_rows_path)
        if "motif_id" in selected_df:
            for row in selected_df.to_dict(orient="records"):
                selected_lookup[f"motif_{int(row['motif_id']):06d}"] = row
    strict_summary_rows = []
    for col_idx, col in enumerate(motif_cols):
        finite = values[:, col_idx][np.isfinite(values[:, col_idx])]
        source = selected_lookup.get(col, {})
        strict_summary_rows.append(
            {
                "motif_id": int(str(col).split("_")[1]),
                "topology": col,
                "motif": source.get("motif", ""),
                "source_w": source.get("source_w", ""),
                "source_h": source.get("source_h", ""),
                "valid_steps": int(valid_counts_arr[col_idx]),
                "fully_reachable_all_steps": bool(valid_counts_arr[col_idx] == len(steps)),
                "strict_static_mean_delay_ms": float(static_means[col_idx]),
                "finite_mean_delay_ms": float(np.mean(finite)) if finite.size else np.nan,
                "finite_min_delay_ms": float(np.min(finite)) if finite.size else np.nan,
                "finite_max_delay_ms": float(np.max(finite)) if finite.size else np.nan,
                "edges": source.get("edges", ""),
                "output_dir": str(out_dir / col),
            }
        )
    strict_summary_rows.sort(key=lambda item: item["strict_static_mean_delay_ms"])
    strict_summary_path = out_dir / f"motif_summary_sorted_{args.output_stem}_{args.pair_key}{suffix}.csv"
    pd.DataFrame(strict_summary_rows).to_csv(strict_summary_path, index=False, encoding="utf-8-sig")

    dynamic_path = out_dir / f"dynamic_best_{args.output_stem}_by_step_{args.pair_key}{suffix}.csv"
    dynamic_df.to_csv(dynamic_path, index=False, encoding="utf-8-sig")
    if not args.strict_reachable:
        # Stable compatibility name for notebooks/scripts that do not care which pair was used.
        dynamic_df.to_csv(out_dir / "dynamic_best_706motifs_by_step.csv", index=False, encoding="utf-8-sig")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = out_dir / f"mean_shortest_delay_{args.output_stem}_gridplus_full_link_{args.pair_key}{suffix}.png"
    fig, ax = plt.subplots(figsize=(16, 7.2))
    for col in motif_cols:
        ax.plot(steps, compare[col].to_numpy(dtype=np.float64), color="#6b7280", alpha=0.075, linewidth=0.55)
    ax.plot(
        steps,
        values[:, best_static_idx],
        color="#0f62fe",
        linewidth=1.7,
        label=f"best static among {args.output_stem}: {best_static_col}",
    )
    ax.plot(
        steps,
        dynamic_values,
        color="#111827",
        linewidth=1.4,
        linestyle="--",
        label=f"per-time best envelope among {args.output_stem}",
    )
    if "gridplus" in compare:
        ax.plot(steps, compare["gridplus"].to_numpy(dtype=np.float64), color="#f97316", linewidth=1.45, label="gridplus")
    if "full_link" in compare:
        ax.plot(steps, compare["full_link"].to_numpy(dtype=np.float64), color="#16a34a", linewidth=1.45, label="full_link")
    title_suffix = " (strict reachable)" if args.strict_reachable else ""
    ax.set_title(f"G60 {args.pair_label} mean shortest delay: {args.library_label}{title_suffix}")
    ax.set_xlabel("time step (s)")
    ax.set_ylabel(f"{args.pair_label} mean shortest delay (ms)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=220)
    if not args.strict_reachable:
        # Stable compatibility name.
        fig.savefig(out_dir / "mean_shortest_delay_706motifs_gridplus_full_link.png", dpi=220)
    plt.close(fig)

    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if not summary.empty and "topology" in summary:
        summary["output_dir"] = summary["topology"].map(lambda name: str(out_dir / str(name)))
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    result = {
        "pair_key": args.pair_key,
        "pair_label": args.pair_label,
        "out_dir": str(out_dir),
        "compare_csv": str(compare_path),
        "motif_count": int(len(motif_cols)),
        "num_steps": int(len(steps)),
        "actual_start_step": int(steps[0]) if len(steps) else None,
        "actual_end_step": int(steps[-1]) if len(steps) else None,
        "best_static_topology": str(best_static_col),
        "best_static_mean_delay_ms": float(static_means[best_static_idx]),
        "dynamic_unique_topologies": int(dynamic_df["best_topology"].nunique()),
        "dynamic_best_csv": str(dynamic_path),
        "strict_motif_summary_csv": str(strict_summary_path),
        "plot": str(plot_path),
        "meta_selected_count": meta.get("selected_count"),
    }
    if not summary.empty:
        result["summary_top5"] = summary.head(5).to_dict(orient="records")

    result.update(reachability)

    summary_json = out_dir / f"region_pair_summary_{args.pair_key}{suffix}.json"
    summary_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.strict_reachable:
        (out_dir / "region_pair_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
