from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.topology_workflow.module.config import load_workflow_yaml


DEFAULT_CONFIG = THIS_DIR / "configs" / "g60_w_le4_h_le3_shortest_hops.yaml"
DEFAULT_PAIRS = ("china_europe", "china_america", "china_africa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one fixed motif against each region pair's best-static/dynamic shortest-hop curves."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--motif", default="combined_motif_000056")
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--out-name", default=None)
    return parser.parse_args()


def pair_label(pair_key: str) -> str:
    return {
        "china_europe": "China-Europe",
        "china_america": "China-America",
        "china_africa": "China-Africa",
    }.get(str(pair_key), str(pair_key))


def paths_for_pair(out_dir: Path, run_label: str, pair: str) -> dict[str, Path]:
    pair_dir = out_dir / pair
    return {
        "compare": pair_dir / f"compare_{run_label}_{pair}_strict_reachable.csv",
        "summary": pair_dir / f"motif_summary_sorted_{run_label}_{pair}_strict_reachable.csv",
        "dynamic": pair_dir / "paper_style_dynamic_best_mean_shortest_hops.csv",
    }


def load_summary(path: Path, motif_name: str) -> tuple[dict[str, Any], dict[str, Any], int]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"summary CSV is empty: {path}")
    best_row = rows[0]
    fixed_rank = -1
    fixed_row: dict[str, Any] | None = None
    for idx, row in enumerate(rows, start=1):
        if row.get("topology") == motif_name:
            fixed_rank = idx
            fixed_row = row
            break
    if fixed_row is None:
        raise ValueError(f"{motif_name} not found in {path}")
    return best_row, fixed_row, fixed_rank


def read_fixed_motif_series(compare_csv: Path, motif_name: str) -> pd.DataFrame:
    df = pd.read_csv(compare_csv, usecols=["step", motif_name])
    df[motif_name] = pd.to_numeric(df[motif_name], errors="coerce")
    return df


def main() -> int:
    args = parse_args()
    raw = load_workflow_yaml(args.config)
    out_dir = Path(str(raw["paths"]["out_dir"]))
    run_label = str(raw["run"]["label"])
    motif_name = str(args.motif)

    fig, axes = plt.subplots(len(args.pairs), 1, figsize=(15.5, 10.5), sharex=True)
    if len(args.pairs) == 1:
        axes = [axes]

    summary_rows: list[dict[str, Any]] = []
    for ax, pair in zip(axes, args.pairs):
        pair_paths = paths_for_pair(out_dir, run_label, str(pair))
        best_row, fixed_row, fixed_rank = load_summary(pair_paths["summary"], motif_name)
        dyn = pd.read_csv(pair_paths["dynamic"])
        fixed = read_fixed_motif_series(pair_paths["compare"], motif_name)
        df = dyn.merge(fixed, on="step", how="inner")
        if df.empty:
            raise ValueError(f"empty merged time series for pair={pair}")

        hours = df["step"].to_numpy(dtype=float) / 3600.0
        fixed_values = df[motif_name].to_numpy(dtype=float)
        best_values = df["best_static_hops"].to_numpy(dtype=float)
        dynamic_values = df["dynamic_best_hops"].to_numpy(dtype=float)
        full_values = df["full_link_hops"].to_numpy(dtype=float)
        grid_values = df["gridplus_hops"].to_numpy(dtype=float)

        best_name = str(best_row["topology"])
        fixed_mean = float(np.nanmean(fixed_values))
        best_mean = float(np.nanmean(best_values))
        dynamic_mean = float(np.nanmean(dynamic_values))
        full_mean = float(np.nanmean(full_values))
        grid_mean = float(np.nanmean(grid_values))
        fixed_minus_best = fixed_mean - best_mean
        fixed_minus_dynamic = fixed_mean - dynamic_mean
        fixed_minus_full = fixed_mean - full_mean
        dynamic_uses_fixed = int((df["dynamic_best_topology"].astype(str) == motif_name).sum())

        ax.plot(hours, full_values, color="#111111", linewidth=1.8, label=f"full link mean={full_mean:.3f}")
        ax.plot(hours, grid_values, color="#8A8A8A", linewidth=1.1, alpha=0.75, label=f"gridplus mean={grid_mean:.3f}")
        ax.plot(
            hours,
            dynamic_values,
            color="#E76F00",
            linewidth=2.2,
            label=f"dynamic best mean={dynamic_mean:.3f}",
        )
        if best_name == motif_name:
            ax.plot(
                hours,
                fixed_values,
                color="#1F5AA6",
                linewidth=2.6,
                label=f"{motif_name} = best static mean={fixed_mean:.3f}",
            )
        else:
            ax.plot(
                hours,
                best_values,
                color="#2A9D55",
                linewidth=2.0,
                label=f"best static {best_name} mean={best_mean:.3f}",
            )
            ax.plot(
                hours,
                fixed_values,
                color="#1F5AA6",
                linewidth=2.4,
                linestyle="--",
                label=f"{motif_name} rank={fixed_rank}, mean={fixed_mean:.3f}",
            )

        ax.set_title(
            f"{pair_label(str(pair))}: fixed {motif_name} vs pair-specific best",
            loc="left",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_ylabel("mean shortest hops")
        ax.grid(True, color="#DADADA", linewidth=0.6, alpha=0.75)
        ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)

        summary_rows.append(
            {
                "pair": str(pair),
                "pair_label": pair_label(str(pair)),
                "fixed_topology": motif_name,
                "fixed_rank": fixed_rank,
                "fixed_mean_hops": fixed_mean,
                "best_static_topology": best_name,
                "best_static_mean_hops": best_mean,
                "dynamic_mean_hops": dynamic_mean,
                "full_link_mean_hops": full_mean,
                "gridplus_mean_hops": grid_mean,
                "fixed_minus_best_static": fixed_minus_best,
                "fixed_minus_dynamic": fixed_minus_dynamic,
                "fixed_minus_full_link": fixed_minus_full,
                "dynamic_uses_fixed_steps": dynamic_uses_fixed,
                "total_steps": int(len(df)),
                "dynamic_uses_fixed_ratio": float(dynamic_uses_fixed / max(1, len(df))),
                "fixed_motif_text": str(fixed_row.get("motif", "")),
                "fixed_support": str(fixed_row.get("support", "")),
                "fixed_edges": str(fixed_row.get("edges", "")),
                "best_motif_text": str(best_row.get("motif", "")),
                "best_support": str(best_row.get("support", "")),
                "best_edges": str(best_row.get("edges", "")),
            }
        )

    axes[-1].set_xlabel("time (hour)")
    fig.suptitle(
        f"Fixed topology {motif_name} compared with region-pair shortest-hop optima",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    out_name = args.out_name or f"{motif_name}_vs_pair_bests_shortest_hops"
    png_path = out_dir / f"{out_name}.png"
    csv_path = out_dir / f"{out_name}_summary.csv"
    json_path = out_dir / f"{out_name}_summary.json"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[fixed-motif-compare] figure={png_path}", flush=True)
    print(f"[fixed-motif-compare] summary_csv={csv_path}", flush=True)
    print(f"[fixed-motif-compare] summary_json={json_path}", flush=True)
    for row in summary_rows:
        print(
            f"[fixed-motif-compare] {row['pair']}: fixed_rank={row['fixed_rank']}, "
            f"fixed_mean={row['fixed_mean_hops']:.6f}, "
            f"best={row['best_static_topology']} ({row['best_static_mean_hops']:.6f}), "
            f"delta_best={row['fixed_minus_best_static']:.6f}, "
            f"dynamic_uses_fixed={row['dynamic_uses_fixed_steps']}/{row['total_steps']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
