# -*- coding: utf-8 -*-
"""Collect real LST120 metrics for 000040/000056 fusion policies.

This is a lightweight reporting helper: it reads existing
``lst_dynamic_metric_summary.json`` files and regenerates one CSV plus one
scatter plot. It does not recompute topology, LST, hop, or delay metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


G60_RUN_ROOT = Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60")
DEFAULT_OUT_DIR = G60_RUN_ROOT / "motif0040_0056_region_internal_plus_grid_mdp_learning_comparison"


POLICY_SUMMARIES: list[tuple[str, Path]] = [
    (
        "best_critical_count_k0",
        G60_RUN_ROOT
        / "motif0040_0056_region_internal_plus_grid_row_mask_beam_critical_drop_aware"
        / "beam_lam080_crit005_cnt003_build003_dynamic_topology"
        / "lst120s_backward_metrics"
        / "metric_summary"
        / "lst_dynamic_metric_summary.json",
    ),
    (
        "l078_c005_cnt025",
        G60_RUN_ROOT / "eval_short" / "l078_c005_cnt025_metrics" / "lst_dynamic_metric_summary.json",
    ),
    (
        "l078_c004_cnt025",
        G60_RUN_ROOT / "eval_short" / "l078_c004_cnt025_metrics" / "lst_dynamic_metric_summary.json",
    ),
    (
        "teacher_dp_sw05",
        G60_RUN_ROOT
        / "motif0040_0056_region_internal_plus_grid_row_mask_dp_switch_penalty"
        / "dp_lambda050_sw05_dynamic_topology"
        / "lst120s_backward_metrics"
        / "metric_summary"
        / "lst_dynamic_metric_summary.json",
    ),
    (
        "topk3_crit005",
        G60_RUN_ROOT / "eval_short" / "topk3_lam080_crit005_cnt003_lst120" / "lst_dynamic_metric_summary.json",
    ),
    (
        "topk3_crit003",
        G60_RUN_ROOT / "eval_short" / "topk3_lam080_crit003_cnt003_lst120" / "lst_dynamic_metric_summary.json",
    ),
    (
        "numpy_mlp_sw05",
        G60_RUN_ROOT
        / "motif0040_0056_region_internal_plus_grid_numpy_mlp_policy"
        / "numpy_mlp_smooth010_temp20_obj1_sw05_dynamic_topology"
        / "lst120s_backward_metrics"
        / "metric_summary"
        / "lst_dynamic_metric_summary.json",
    ),
    (
        "transition_value_mlp_rs1",
        G60_RUN_ROOT / "eval_short" / "tv_mlp_h160e80_metrics" / "lst_dynamic_metric_summary.json",
    ),
    (
        "transition_value_mlp_rs2",
        G60_RUN_ROOT / "eval_short" / "tv_mlp_rs2_metrics" / "lst_dynamic_metric_summary.json",
    ),
    (
        "fitted_q_mlp_adv",
        G60_RUN_ROOT / "eval_short" / "fitted_q_adv_h192e120_metrics" / "lst_dynamic_metric_summary.json",
    ),
    (
        "knn_teacher_sw02",
        G60_RUN_ROOT
        / "motif0040_0056_region_internal_plus_grid_lst_teacher_knn_imitation"
        / "hamming_sw05_teacher"
        / "knn_teacher_sw02_dynamic_topology"
        / "lst120s_backward_metrics"
        / "metric_summary"
        / "lst_dynamic_metric_summary.json",
    ),
    (
        "numpy_mlp_sw02",
        G60_RUN_ROOT
        / "motif0040_0056_region_internal_plus_grid_numpy_mlp_policy"
        / "numpy_mlp_smooth010_temp20_obj1_sw02_dynamic_topology"
        / "lst120s_backward_metrics"
        / "metric_summary"
        / "lst_dynamic_metric_summary.json",
    ),
    (
        "static_000056",
        G60_RUN_ROOT
        / "motif0040_0056_region_internal_plus_grid_pure_advantage_policy"
        / "pure_000056_all_day_dynamic_topology"
        / "lst120s_backward_topology_only"
        / "metric_summary"
        / "lst_dynamic_metric_summary.json",
    ),
]


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_row(policy: str, path: Path) -> dict[str, Any]:
    data = load_summary(path)
    pair = data["lst_active_hops_full_1s"]["pairs"]["china_europe"]
    delay = data["lst_active_delay_ms_sampled"]["pairs"]["china_europe"]["all_sampled"]
    link = data["link_setup"]
    return {
        "policy": policy,
        "mean_hops": float(pair["mean"]),
        "mean_delay_ms": float(delay["mean"]),
        "building_edges_mean": float(link["building_edges"]["mean"]),
        "active_dropped_mean": float(link["active_dropped_by_building"]["mean"]),
        "summary_json": str(path),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "policy",
        "mean_hops",
        "mean_delay_ms",
        "building_edges_mean",
        "active_dropped_mean",
        "summary_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "best_critical_count_k0": "#d62728",
        "l078_c005_cnt025": "#ff7f0e",
        "l078_c004_cnt025": "#ffbb78",
        "teacher_dp_sw05": "#1f77b4",
        "transition_value_mlp_rs1": "#9467bd",
        "transition_value_mlp_rs2": "#8c564b",
        "fitted_q_mlp_adv": "#17becf",
        "static_000056": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=180)
    for row in rows:
        label = row["policy"]
        ax.scatter(
            row["mean_hops"],
            row["mean_delay_ms"],
            s=58,
            color=colors.get(label, "#7f7f7f"),
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        ax.annotate(
            label,
            (row["mean_hops"], row["mean_delay_ms"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
        )

    ax.set_title("G60 000040/000056 fusion policies, region +grid, LST=120s")
    ax.set_xlabel("China-Europe mean hops, real LST active topology")
    ax.set_ylabel("China-Europe mean delay ms, sampled every 60s")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.margins(x=0.12, y=0.12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_zoom_plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    zoom_rows = [row for row in rows if row["policy"] != "static_000056"]
    offsets = {
        "l078_c005_cnt025": (6, -12),
        "best_critical_count_k0": (6, 7),
        "teacher_dp_sw05": (6, 9),
        "topk3_crit005": (6, -12),
        "l078_c004_cnt025": (6, 8),
        "topk3_crit003": (6, -12),
        "numpy_mlp_sw05": (6, 8),
        "transition_value_mlp_rs1": (6, -12),
        "transition_value_mlp_rs2": (6, 8),
        "fitted_q_mlp_adv": (6, -12),
        "knn_teacher_sw02": (6, -12),
        "numpy_mlp_sw02": (6, 8),
    }
    colors = {
        "best_critical_count_k0": "#d62728",
        "l078_c005_cnt025": "#ff7f0e",
        "l078_c004_cnt025": "#ffbb78",
        "teacher_dp_sw05": "#1f77b4",
        "transition_value_mlp_rs1": "#9467bd",
        "transition_value_mlp_rs2": "#8c564b",
        "fitted_q_mlp_adv": "#17becf",
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=180)
    for row in zoom_rows:
        label = row["policy"]
        ax.scatter(
            row["mean_hops"],
            row["mean_delay_ms"],
            s=64,
            color=colors.get(label, "#7f7f7f"),
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
        ax.annotate(
            label,
            (row["mean_hops"], row["mean_delay_ms"]),
            xytext=offsets.get(label, (6, 5)),
            textcoords="offset points",
            fontsize=7.5,
        )

    ax.set_title("Dynamic-policy cluster, real LST120")
    ax.set_xlabel("China-Europe mean hops")
    ax.set_ylabel("China-Europe mean delay ms")
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True, color="#dddddd", linewidth=0.7)
    ax.margins(x=0.18, y=0.18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    missing: list[tuple[str, Path]] = []
    for policy, path in POLICY_SUMMARIES:
        if path.exists():
            rows.append(extract_row(policy, path))
        else:
            missing.append((policy, path))

    rows.sort(key=lambda r: (r["mean_hops"], r["mean_delay_ms"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "mdp_learning_lst120_summary.csv"
    plot_path = args.out_dir / "mdp_learning_lst120_cluster.png"
    zoom_plot_path = args.out_dir / "mdp_learning_lst120_cluster_zoom.png"
    write_csv(rows, csv_path)
    write_plot(rows, plot_path)
    write_zoom_plot(rows, zoom_plot_path)

    print(f"[comparison] rows={len(rows)} csv={csv_path}")
    print(f"[comparison] plot={plot_path}")
    print(f"[comparison] zoom_plot={zoom_plot_path}")
    if missing:
        print("[comparison] missing summaries:")
        for policy, path in missing:
            print(f"  {policy}: {path}")
    print("[comparison] top rows:")
    for row in rows[:5]:
        print(
            f"  {row['policy']}: hops={row['mean_hops']:.9f} "
            f"delay={row['mean_delay_ms']:.9f} "
            f"building={row['building_edges_mean']:.3f} "
            f"drop={row['active_dropped_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
