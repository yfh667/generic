from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
LINSHI_ROOT = PROJECT_ROOT / "data" / "linshi"
DEFAULT_OUT_DIR = LINSHI_ROOT / "g60_combined_102_706_region_pair_plots_t0_86160_stride60"


PAIR_CONFIGS = {
    "china_europe": {
        "label": "China-Europe",
        "small102_dir": LINSHI_ROOT / "g60_small102_up_to_w4h3_china_europe_t0_86160_stride60",
        "motifs706_dir": LINSHI_ROOT
        / "g60_w4h3_primitive706_region_pairs_t0_86160_stride60"
        / "china_europe"
        / "motifs706",
    },
    "china_america": {
        "label": "China-America",
        "small102_dir": LINSHI_ROOT / "g60_small102_up_to_w4h3_china_america_t0_86160_stride60",
        "motifs706_dir": LINSHI_ROOT
        / "g60_w4h3_primitive706_region_pairs_t0_86160_stride60"
        / "china_america"
        / "motifs706",
    },
    "china_africa": {
        "label": "China-Africa",
        "small102_dir": LINSHI_ROOT / "g60_small102_up_to_w4h3_china_africa_t0_86160_stride60",
        "motifs706_dir": LINSHI_ROOT
        / "g60_w4h3_primitive706_region_pairs_t0_86160_stride60"
        / "china_africa"
        / "motifs706",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot merged G60 small102 + primitive706 region-pair delay results.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=["china_europe", "china_america", "china_africa"],
        choices=sorted(PAIR_CONFIGS),
    )
    parser.add_argument("--non-strict", action="store_true", help="Do not mask disconnected motif-step values.")
    parser.add_argument("--write-wide-compare", action="store_true", help="Also write the 808-motif wide compare CSV.")
    return parser.parse_args()


def motif_columns(compare: pd.DataFrame) -> list[str]:
    return [col for col in compare.columns if col.startswith("motif_")]


def full_reachability_mask(topology_dir: Path, steps: np.ndarray) -> np.ndarray:
    summary_path = topology_dir / "step_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    required = summary["source_nodes"].to_numpy(dtype=np.int64) * summary["target_nodes"].to_numpy(dtype=np.int64)
    ok = summary["reachable_pairs"].to_numpy(dtype=np.int64) == required
    by_step = {int(step): bool(flag) for step, flag in zip(summary["step"].to_numpy(dtype=np.int64), ok)}
    return np.asarray([by_step.get(int(step), False) for step in steps], dtype=bool)


def load_library(
    library_dir: Path,
    *,
    prefix: str,
    steps: np.ndarray | None,
    strict: bool,
) -> tuple[np.ndarray, pd.DataFrame, list[str], dict[str, int]]:
    compare_path = Path(library_dir) / "compare_100motifs_gridplus_full_link.csv"
    compare = pd.read_csv(compare_path)
    current_steps = compare["step"].to_numpy(dtype=np.int64)
    if steps is None:
        steps = current_steps
    elif not np.array_equal(steps, current_steps):
        raise ValueError(f"step mismatch for {library_dir}")

    cols = motif_columns(compare)
    data = compare[cols].to_numpy(dtype=np.float64)
    valid_counts = np.full(len(cols), len(current_steps), dtype=np.int64)
    if strict:
        for col_idx, col in enumerate(cols):
            mask = full_reachability_mask(Path(library_dir) / col, current_steps)
            valid_counts[col_idx] = int(np.sum(mask))
            data[~mask, col_idx] = np.nan

    renamed_cols = [f"{prefix}_{col}" for col in cols]
    wide = pd.DataFrame(data, columns=renamed_cols)
    stats = {
        "motif_count": int(len(cols)),
        "full_time_reachable_motifs": int(np.sum(valid_counts == len(current_steps))),
        "min_valid_steps_per_motif": int(np.min(valid_counts)) if len(valid_counts) else 0,
        "total_masked_values": int(np.sum(~np.isfinite(data))),
    }
    return steps, wide, renamed_cols, stats


def best_static(values: np.ndarray, cols: list[str]) -> tuple[str, int, float]:
    finite_all = np.all(np.isfinite(values), axis=0)
    means = np.full(values.shape[1], np.inf, dtype=np.float64)
    if np.any(finite_all):
        means[finite_all] = np.nanmean(values[:, finite_all], axis=0)
    else:
        means = np.nanmean(values, axis=0)
    idx = int(np.nanargmin(means))
    return cols[idx], idx, float(means[idx])


def dynamic_best(values: np.ndarray, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    best_values = np.full(values.shape[0], np.nan, dtype=np.float64)
    best_names = np.full(values.shape[0], "", dtype=object)
    for row_idx in range(values.shape[0]):
        row = values[row_idx]
        if np.all(~np.isfinite(row)):
            continue
        idx = int(np.nanargmin(row))
        best_values[row_idx] = float(row[idx])
        best_names[row_idx] = cols[idx]
    return best_values, best_names


def load_baselines(library_dir: Path, steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    compare = pd.read_csv(Path(library_dir) / "compare_100motifs_gridplus_full_link.csv")
    if not np.array_equal(compare["step"].to_numpy(dtype=np.int64), steps):
        raise ValueError(f"baseline step mismatch for {library_dir}")
    return compare["gridplus"].to_numpy(dtype=np.float64), compare["full_link"].to_numpy(dtype=np.float64)


def plot_pair(pair_key: str, pair_cfg: dict[str, object], out_dir: Path, *, strict: bool, write_wide: bool) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair_out = out_dir / pair_key
    pair_out.mkdir(parents=True, exist_ok=True)

    steps: np.ndarray | None = None
    steps, small_wide, small_cols, small_stats = load_library(
        Path(pair_cfg["small102_dir"]),
        prefix="small102",
        steps=steps,
        strict=strict,
    )
    steps, motif706_wide, motif706_cols, motif706_stats = load_library(
        Path(pair_cfg["motifs706_dir"]),
        prefix="motifs706",
        steps=steps,
        strict=strict,
    )
    assert steps is not None

    gridplus, full_link = load_baselines(Path(pair_cfg["motifs706_dir"]), steps)
    all_wide = pd.concat([small_wide, motif706_wide], axis=1)
    all_cols = list(all_wide.columns)
    all_values = all_wide.to_numpy(dtype=np.float64)

    small_best_name, small_best_idx, small_best_mean = best_static(small_wide.to_numpy(dtype=np.float64), small_cols)
    motif706_best_name, motif706_best_idx, motif706_best_mean = best_static(
        motif706_wide.to_numpy(dtype=np.float64),
        motif706_cols,
    )
    combined_best_name, combined_best_idx, combined_best_mean = best_static(all_values, all_cols)
    dyn_values, dyn_names = dynamic_best(all_values, all_cols)

    dynamic_df = pd.DataFrame(
        {
            "step": steps,
            "best_topology": dyn_names,
            "best_delay_ms": dyn_values,
            "gridplus_delay_ms": gridplus,
            "full_link_delay_ms": full_link,
        }
    )
    suffix = "strict_reachable" if strict else "finite_only"
    dynamic_path = pair_out / f"combined_dynamic_best_102_706_by_step_{pair_key}_{suffix}.csv"
    dynamic_df.to_csv(dynamic_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    for col_idx, col in enumerate(all_cols):
        series = all_values[:, col_idx]
        finite = series[np.isfinite(series)]
        summary_rows.append(
            {
                "topology": col,
                "library": "small102" if col.startswith("small102_") else "motifs706",
                "valid_steps": int(np.sum(np.isfinite(series))),
                "fully_reachable_all_steps": bool(np.all(np.isfinite(series))),
                "static_mean_delay_ms": float(np.mean(finite)) if finite.size else np.nan,
                "min_delay_ms": float(np.min(finite)) if finite.size else np.nan,
                "max_delay_ms": float(np.max(finite)) if finite.size else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("static_mean_delay_ms", na_position="last")
    summary_path = pair_out / f"combined_motif_summary_sorted_102_706_{pair_key}_{suffix}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if write_wide:
        compare_out = pd.concat([pd.DataFrame({"step": steps}), all_wide], axis=1)
        compare_out["gridplus"] = gridplus
        compare_out["full_link"] = full_link
        compare_out.to_csv(pair_out / f"combined_compare_102_706_{pair_key}_{suffix}.csv", index=False)

    x_hours = steps / 3600.0
    label = str(pair_cfg["label"])
    fig, ax = plt.subplots(figsize=(16, 7.4), dpi=180)
    for col in small_cols:
        ax.plot(x_hours, small_wide[col].to_numpy(dtype=np.float64), color="#2563eb", alpha=0.07, linewidth=0.55)
    for col in motif706_cols:
        ax.plot(x_hours, motif706_wide[col].to_numpy(dtype=np.float64), color="#6b7280", alpha=0.045, linewidth=0.50)
    ax.plot(
        x_hours,
        small_wide[small_best_name].to_numpy(dtype=np.float64),
        color="#1d4ed8",
        linewidth=1.65,
        label=f"best static small102: {small_best_name.replace('small102_', '')}",
    )
    ax.plot(
        x_hours,
        motif706_wide[motif706_best_name].to_numpy(dtype=np.float64),
        color="#7c3aed",
        linewidth=1.65,
        label=f"best static 706: {motif706_best_name.replace('motifs706_', '')}",
    )
    ax.plot(
        x_hours,
        dyn_values,
        color="#111827",
        linewidth=1.65,
        linestyle="--",
        label="per-time best among 102+706",
    )
    ax.plot(x_hours, gridplus, color="#f97316", linewidth=1.35, label="gridplus")
    ax.plot(x_hours, full_link, color="#16a34a", linewidth=1.35, label="full_link")
    title_suffix = "strict reachable" if strict else "finite-only"
    ax.set_title(f"G60 {label} mean shortest delay: small102 + primitive706 ({title_suffix})")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(f"{label} mean shortest delay (ms)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    plot_path = pair_out / f"mean_shortest_delay_combined_102_706_{pair_key}_{suffix}.png"
    fig.savefig(plot_path)
    plt.close(fig)

    result = {
        "pair_key": pair_key,
        "pair_label": label,
        "strict": bool(strict),
        "steps": int(len(steps)),
        "start": int(steps[0]),
        "end": int(steps[-1]),
        "small102": {
            **small_stats,
            "best_static": small_best_name,
            "best_static_mean_delay_ms": small_best_mean,
        },
        "motifs706": {
            **motif706_stats,
            "best_static": motif706_best_name,
            "best_static_mean_delay_ms": motif706_best_mean,
        },
        "combined": {
            "motif_count": int(len(all_cols)),
            "best_static": combined_best_name,
            "best_static_mean_delay_ms": combined_best_mean,
            "dynamic_mean_delay_ms": float(np.nanmean(dyn_values)),
            "dynamic_unique_topologies": int(pd.Series(dyn_names).replace("", np.nan).nunique()),
            "gridplus_mean_delay_ms": float(np.nanmean(gridplus)),
            "full_link_mean_delay_ms": float(np.nanmean(full_link)),
        },
        "plot": str(plot_path),
        "dynamic_csv": str(dynamic_path),
        "summary_csv": str(summary_path),
    }
    (pair_out / f"combined_102_706_summary_{pair_key}_{suffix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strict = not bool(args.non_strict)
    results = []
    for pair_key in args.pairs:
        result = plot_pair(pair_key, PAIR_CONFIGS[pair_key], out_dir, strict=strict, write_wide=bool(args.write_wide_compare))
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    (out_dir / "combined_102_706_all_pairs_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
