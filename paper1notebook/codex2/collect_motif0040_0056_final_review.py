from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


G60_ROOT = Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60")
OUT_DIR = G60_ROOT / "motif0040_0056_region_internal_plus_grid_final_review"

REWARD_FULL_DIR = (
    G60_ROOT
    / "motif0040_0056_region_internal_plus_grid_row_mask_action_scorer_hard96"
    / "action_library_reward_table_full1437_merged"
)
LST_COMPARE_CSV = (
    G60_ROOT
    / "motif0040_0056_region_internal_plus_grid_mdp_learning_comparison"
    / "mdp_learning_lst120_summary.csv"
)
DYNAMIC_L078_SUMMARY = G60_ROOT / "eval_short" / "l078_c005_cnt025_metrics" / "lst_dynamic_metric_summary.json"
STATIC_56_SUMMARY = (
    G60_ROOT
    / "motif0040_0056_region_internal_plus_grid_pure_advantage_policy"
    / "pure_000056_all_day_dynamic_topology"
    / "lst120s_backward_topology_only"
    / "metric_summary"
    / "lst_dynamic_metric_summary.json"
)
STATIC_40_SUMMARY = (
    G60_ROOT
    / "motif0040_0056_region_internal_plus_grid_pure_advantage_policy"
    / "pure_000040_all_day_dynamic_topology"
    / "lst120s_backward_topology_only"
    / "metric_summary"
    / "lst_dynamic_metric_summary.json"
)
PLANE_Y_HARD8 = G60_ROOT / "motif0040_0056_region_internal_plus_grid_plane_y_window_search_hard8_seedrows"
PLANE_Y_SCREEN = G60_ROOT / "motif0040_0056_region_internal_plus_grid_plane_y_window_screen_s7200_seedrows"
PLANE_Y_SELECTED_FULL = G60_ROOT / "motif0040_0056_region_internal_plus_grid_plane_y_window_selected_full1437"
L078_TOPOLOGY = G60_ROOT / "eval_short" / "l078_c005_cnt025_topology"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_wide_csv(path: Path) -> tuple[list[int], list[str], np.ndarray]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    names = [key for key in rows[0] if key != "step"]
    steps = [int(float(row["step"])) for row in rows]
    values = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
    return steps, names, values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nested_get(raw: dict[str, Any], path: list[str], default: float = float("nan")) -> float:
    cur: Any = raw
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def collect_no_lst_metrics(out_tables: Path) -> list[dict[str, Any]]:
    steps, names_hop, hops = read_wide_csv(REWARD_FULL_DIR / "row_mask_action_library_mean_hops_wide.csv")
    steps_d, names_delay, delay = read_wide_csv(REWARD_FULL_DIR / "row_mask_action_library_mean_delay_ms_wide.csv")
    if steps != steps_d or names_hop != names_delay:
        raise ValueError("reward full hop/delay tables are not aligned")

    def col(name: str) -> int:
        return names_hop.index(name)

    rows = [
        {
            "scope": "no_lst_reward_table_full1437",
            "policy": "motif000056_action_000",
            "mean_hops": float(np.nanmean(hops[:, col("action_000")])),
            "mean_delay_ms": float(np.nanmean(delay[:, col("action_000")])),
            "note": "pure motif000056; region-internal +grid applied in metric",
        },
        {
            "scope": "no_lst_reward_table_full1437",
            "policy": "motif000040_action_040",
            "mean_hops": float(np.nanmean(hops[:, col("action_040")])),
            "mean_delay_ms": float(np.nanmean(delay[:, col("action_040")])),
            "note": "pure motif000040; region-internal +grid applied in metric",
        },
        {
            "scope": "no_lst_reward_table_full1437",
            "policy": "row_mask_114_envelope",
            "mean_hops": float(np.nanmean(np.nanmin(hops, axis=1))),
            "mean_delay_ms": float(np.nanmean(np.nanmin(delay, axis=1))),
            "note": "per-step lower envelope over 114 row-mask actions; not LST-feasible by itself",
        },
    ]
    write_csv(out_tables / "no_lst_static_and_envelope_metrics.csv", rows)
    return rows


def collect_lst_metrics(out_tables: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, path, note in (
        ("motif000056_static_lst120", STATIC_56_SUMMARY, "pure motif000056 after LST120 topology-only pipeline"),
        ("motif000040_static_lst120", STATIC_40_SUMMARY, "pure motif000040 after LST120 topology-only pipeline"),
        ("dynamic_l078_c005_cnt025_lst120", DYNAMIC_L078_SUMMARY, "best current row-mask dynamic policy after LST120"),
    ):
        raw = read_json(path)
        rows.append(
            {
                "scope": "lst120_true_active_topology",
                "policy": policy,
                "mean_hops": nested_get(raw, ["lst_active_hops_full_1s", "pairs", "china_europe", "mean"]),
                "mean_delay_ms": nested_get(
                    raw,
                    ["lst_active_delay_ms_sampled", "pairs", "china_europe", "all_sampled", "mean"],
                ),
                "building_edges_mean": nested_get(raw, ["link_setup", "building_edges", "mean"]),
                "active_dropped_mean": nested_get(raw, ["link_setup", "active_dropped_by_building", "mean"]),
                "note": note,
                "summary_json": str(path),
            }
        )
    write_csv(out_tables / "lst120_static_vs_dynamic_metrics.csv", rows)
    return rows


def collect_plane_y_diagnostics(out_tables: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, path, note in (
        ("plane_y_hard8_all753", PLANE_Y_HARD8 / "plane_y_window_search_summary.json", "hard8 diagnostic over 753 local candidates"),
        ("plane_y_screen_s7200_all753", PLANE_Y_SCREEN / "plane_y_window_search_summary.json", "12 sampled times over 753 local candidates"),
        ("plane_y_selected_full1437_10cand", PLANE_Y_SELECTED_FULL / "plane_y_window_search_summary.json", "full 1437-step eval of 10 candidates selected from hard8"),
    ):
        raw = read_json(path)
        rows.append(
            {
                "scope": "plane_y_no_lst_diagnostic",
                "policy": label,
                "candidate_count": int(raw.get("candidate_count", 0)),
                "step_count": len(raw.get("steps", [])),
                "baseline_mean_hops": float(raw.get("mean_baseline_hop_envelope", float("nan"))),
                "baseline_mean_delay_ms": float(raw.get("mean_baseline_delay_envelope_ms", float("nan"))),
                "plane_y_mean_hops": float(raw.get("mean_plane_y_hop_envelope", float("nan"))),
                "plane_y_mean_delay_ms": float(raw.get("mean_plane_y_delay_envelope_ms", float("nan"))),
                "gap_hops": float(raw.get("mean_gap_plane_y_to_rowmask_hop", float("nan"))),
                "gap_delay_ms": float(raw.get("mean_gap_plane_y_to_rowmask_delay_ms", float("nan"))),
                "better_hop_steps": int(raw.get("plane_y_better_hop_steps", 0)),
                "better_delay_steps": int(raw.get("plane_y_better_delay_steps", 0)),
                "note": note,
                "summary_json": str(path),
            }
        )
    write_csv(out_tables / "plane_y_local_window_diagnostics.csv", rows)
    return rows


def plot_metrics(out_plots: Path, lst_rows: list[dict[str, Any]], no_lst_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_plots.mkdir(parents=True, exist_ok=True)

    labels = [str(row["policy"]) for row in lst_rows]
    hops = [float(row["mean_hops"]) for row in lst_rows]
    delay = [float(row["mean_delay_ms"]) for row in lst_rows]
    building = [float(row.get("building_edges_mean", float("nan"))) for row in lst_rows]
    dropped = [float(row.get("active_dropped_mean", float("nan"))) for row in lst_rows]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.5), dpi=180)
    panels = [
        (axes[0, 0], hops, "Mean shortest hops"),
        (axes[0, 1], delay, "Mean shortest delay (ms)"),
        (axes[1, 0], building, "Mean building edges"),
        (axes[1, 1], dropped, "Mean active edges dropped by building"),
    ]
    colors = ["#6b7280", "#9ca3af", "#2563eb"]
    for ax, values, title in panels:
        ax.bar(range(len(labels)), values, color=colors[: len(labels)])
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.55)
        for idx, value in enumerate(values):
            if np.isfinite(value):
                ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("G60 China-Europe region-internal +grid: LST120 metric comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(out_plots / "lst120_static_vs_dynamic_metric_bars.png")
    plt.close(fig)

    labels2 = [str(row["policy"]) for row in no_lst_rows]
    hops2 = [float(row["mean_hops"]) for row in no_lst_rows]
    delay2 = [float(row["mean_delay_ms"]) for row in no_lst_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), dpi=180)
    for ax, values, title in (
        (axes[0], hops2, "No-LST mean shortest hops"),
        (axes[1], delay2, "No-LST mean shortest delay (ms)"),
    ):
        ax.bar(range(len(labels2)), values, color=["#6b7280", "#9ca3af", "#059669"])
        ax.set_title(title)
        ax.set_xticks(range(len(labels2)))
        ax.set_xticklabels(labels2, rotation=18, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.55)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("G60 China-Europe region-internal +grid: static motifs and row-mask envelope", y=1.02)
    fig.tight_layout()
    fig.savefig(out_plots / "no_lst_static_motif_and_rowmask_envelope_bars.png")
    plt.close(fig)


def copy_existing_plots(out_plots: Path) -> None:
    out_plots.mkdir(parents=True, exist_ok=True)
    copies = [
        PLANE_Y_HARD8 / "plane_y_window_hop_envelope.png",
        PLANE_Y_HARD8 / "plane_y_window_delay_envelope.png",
        PLANE_Y_SCREEN / "plane_y_window_hop_envelope.png",
        PLANE_Y_SCREEN / "plane_y_window_delay_envelope.png",
        PLANE_Y_SELECTED_FULL / "plane_y_window_hop_envelope.png",
        PLANE_Y_SELECTED_FULL / "plane_y_window_delay_envelope.png",
        L078_TOPOLOGY / "lst120" / "lst_summary.png",
        G60_ROOT / "motif0040_0056_region_internal_plus_grid_mdp_learning_comparison" / "mdp_learning_lst120_cluster.png",
        G60_ROOT / "motif0040_0056_region_internal_plus_grid_mdp_learning_comparison" / "mdp_learning_lst120_cluster_zoom.png",
    ]
    for src in copies:
        if src.exists():
            shutil.copy2(src, out_plots / src.name)


def write_readme(out_dir: Path) -> None:
    text = f"""# motif0040/0056 region-internal +grid final review

This folder collects the current G60 China-Europe results for motif000056, motif000040,
the best LST120 row-mask dynamic policy, and the new plane-y local-window diagnostics.

## Main tables

- `tables/no_lst_static_and_envelope_metrics.csv`
  Pure motif000056, pure motif000040, and the 114-action row-mask no-LST envelope.

- `tables/lst120_static_vs_dynamic_metrics.csv`
  True active topology metrics after LST120 for static motif000056, static motif000040,
  and the current best dynamic row-mask policy `l078_c005_cnt025`.

- `tables/plane_y_local_window_diagnostics.csv`
  Evidence that local plane-y windows can beat the row-mask envelope on hard/sampled
  points, but the 10-candidate hard8 transfer is not yet a final LST policy.

## Viewer

Interactive LST120 viewer script:

```powershell
& 'C:\\ProgramData\\miniconda3\\envs\\paper11\\python.exe' `
  'E:\\paper11\\generic\\paper1notebook\\codex2\\run_l078_lst120_topology_viewer.py' `
  --start 0 --end 86160
```

Blue dashed edges are links in `building` state. Black edges are active topology links.

## Source topology

- Dynamic topology: `{L078_TOPOLOGY}`
- LST120 topology: `{L078_TOPOLOGY / "lst120"}`
- Review output: `{out_dir}`
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    out_tables = OUT_DIR / "tables"
    out_plots = OUT_DIR / "plots"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    no_lst_rows = collect_no_lst_metrics(out_tables)
    lst_rows = collect_lst_metrics(out_tables)
    plane_y_rows = collect_plane_y_diagnostics(out_tables)
    all_rows = []
    all_rows.extend(no_lst_rows)
    all_rows.extend(lst_rows)
    all_rows.extend(plane_y_rows)
    write_csv(out_tables / "all_review_metrics.csv", all_rows)
    plot_metrics(out_plots, lst_rows, no_lst_rows)
    copy_existing_plots(out_plots)
    write_readme(OUT_DIR)
    print(
        json.dumps(
            {
                "out_dir": str(OUT_DIR),
                "tables": str(out_tables),
                "plots": str(out_plots),
                "all_review_metrics": str(out_tables / "all_review_metrics.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
