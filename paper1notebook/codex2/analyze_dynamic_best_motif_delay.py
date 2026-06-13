from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_w4h3_selected100_shortest_delay_t0_86164_stride60"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_w4h3_dynamic_best_selected100_t0_86164_stride60"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze time-varying best motif among selected W4H3 motifs."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--static-best", default=None, help="Default: best motif by full-period mean.")
    parser.add_argument("--top-winners", type=int, default=12)
    return parser.parse_args()


def motif_id_from_topology(name: str) -> int:
    return int(str(name).replace("motif_", ""))


def load_inputs(input_dir: Path):
    compare_path = input_dir / "compare_100motifs_gridplus_full_link.csv"
    summary_path = input_dir / "motif_summary_sorted.csv"
    selected_path = input_dir / "selected_motifs_original_rows.csv"
    if not compare_path.exists():
        raise FileNotFoundError(compare_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    compare = pd.read_csv(compare_path)
    summary = pd.read_csv(summary_path)
    selected = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()
    return compare, summary, selected


def build_motif_lookup(summary: pd.DataFrame, selected: pd.DataFrame) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for _, row in summary.iterrows():
        name = str(row["topology"])
        lookup[name] = {
            "motif_id": int(row["motif_id"]),
            "motif": str(row["motif"]),
            "edges": str(row.get("edges", "")),
            "full_period_mean_delay_ms": float(row["mean_delay_ms"]),
        }
    if not selected.empty:
        for _, row in selected.iterrows():
            name = f"motif_{int(row['motif_id']):06d}"
            lookup.setdefault(name, {})
            lookup[name].update(
                {
                    "motif_id": int(row["motif_id"]),
                    "motif": str(row.get("motif", lookup[name].get("motif", ""))),
                    "edges": str(row.get("edges", lookup[name].get("edges", ""))),
                }
            )
    return lookup


def contiguous_segments(best_names: np.ndarray, steps: np.ndarray, dynamic_values: np.ndarray, static_values: np.ndarray):
    rows = []
    if len(best_names) == 0:
        return rows
    start_idx = 0
    for idx in range(1, len(best_names) + 1):
        if idx == len(best_names) or best_names[idx] != best_names[start_idx]:
            segment_values = dynamic_values[start_idx:idx]
            segment_static = static_values[start_idx:idx]
            rows.append(
                {
                    "segment_id": len(rows),
                    "topology": str(best_names[start_idx]),
                    "motif_id": motif_id_from_topology(str(best_names[start_idx])),
                    "start_step": int(steps[start_idx]),
                    "end_step": int(steps[idx - 1]),
                    "num_points": int(idx - start_idx),
                    "duration_s_sampled": int(steps[idx - 1] - steps[start_idx]),
                    "mean_dynamic_delay_ms": float(np.mean(segment_values)),
                    "min_dynamic_delay_ms": float(np.min(segment_values)),
                    "max_dynamic_delay_ms": float(np.max(segment_values)),
                    "mean_improvement_vs_static_ms": float(np.mean(segment_static - segment_values)),
                }
            )
            start_idx = idx
    return rows


def write_plots(
    *,
    out_dir: Path,
    steps: np.ndarray,
    motif_matrix: np.ndarray,
    motif_cols: list[str],
    static_name: str,
    static_values: np.ndarray,
    dynamic_values: np.ndarray,
    gridplus: np.ndarray | None,
    full_link: np.ndarray | None,
    best_names: np.ndarray,
    winner_summary: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 5.3))
    for idx, name in enumerate(motif_cols):
        ax.plot(steps, motif_matrix[:, idx], color="#a0a7ad", alpha=0.18, linewidth=0.55)
    ax.plot(steps, static_values, color="#1f77b4", linewidth=1.4, label=f"static best {static_name}")
    ax.plot(steps, dynamic_values, color="#c1121f", linewidth=1.8, label="dynamic best among 100 motifs")
    if gridplus is not None:
        ax.plot(steps, gridplus, color="#ff7f0e", linewidth=1.2, label="gridplus baseline")
    if full_link is not None:
        ax.plot(steps, full_link, color="#2ca02c", linewidth=1.2, label="full_link lower-bound baseline")
    ax.set_xlabel("time step (s)")
    ax.set_ylabel("China-Europe mean shortest delay (ms)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dynamic_best_vs_static_gridplus_full_link.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.5, 3.5))
    improvement = static_values - dynamic_values
    ax.plot(steps, improvement, color="#c1121f", linewidth=1.1)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("time step (s)")
    ax.set_ylabel("static best - dynamic best (ms)")
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "dynamic_improvement_vs_static.png", dpi=180)
    plt.close(fig)

    top_names = winner_summary["topology"].head(12).tolist()
    name_to_rank = {name: rank for rank, name in enumerate(top_names)}
    y = np.array([name_to_rank.get(str(name), len(top_names)) for name in best_names], dtype=np.int32)
    labels = top_names + (["other"] if np.any(y == len(top_names)) else [])
    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    ax.scatter(steps, y, c=y, cmap="tab20", s=9, alpha=0.88, linewidths=0)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("time step (s)")
    ax.set_ylabel("winning motif")
    ax.grid(axis="x", alpha=0.22, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_dir / "dynamic_winner_timeline.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compare, summary, selected = load_inputs(input_dir)
    lookup = build_motif_lookup(summary, selected)
    motif_cols = [col for col in compare.columns if col.startswith("motif_")]
    if not motif_cols:
        raise ValueError("No motif columns found")

    steps = compare["step"].to_numpy(dtype=np.int64)
    motif_matrix = compare[motif_cols].to_numpy(dtype=np.float64)
    best_col_indices = np.nanargmin(motif_matrix, axis=1)
    best_names = np.asarray([motif_cols[idx] for idx in best_col_indices], dtype=object)
    dynamic_values = motif_matrix[np.arange(len(steps)), best_col_indices]

    static_name = str(args.static_best or summary.iloc[0]["topology"])
    if static_name not in compare.columns:
        raise ValueError(f"static best column not found: {static_name}")
    static_values = compare[static_name].to_numpy(dtype=np.float64)
    gridplus = compare["gridplus"].to_numpy(dtype=np.float64) if "gridplus" in compare.columns else None
    full_link = compare["full_link"].to_numpy(dtype=np.float64) if "full_link" in compare.columns else None

    dynamic_rows = []
    for row_idx, step in enumerate(steps):
        name = str(best_names[row_idx])
        info = lookup.get(name, {})
        item = {
            "step": int(step),
            "best_topology": name,
            "best_motif_id": int(info.get("motif_id", motif_id_from_topology(name))),
            "best_motif": str(info.get("motif", "")),
            "best_edges": str(info.get("edges", "")),
            "best_delay_ms": float(dynamic_values[row_idx]),
            "static_best_topology": static_name,
            "static_best_delay_ms": float(static_values[row_idx]),
            "improvement_vs_static_ms": float(static_values[row_idx] - dynamic_values[row_idx]),
        }
        if gridplus is not None:
            item["gridplus_delay_ms"] = float(gridplus[row_idx])
            item["dynamic_minus_gridplus_ms"] = float(dynamic_values[row_idx] - gridplus[row_idx])
        if full_link is not None:
            item["full_link_delay_ms"] = float(full_link[row_idx])
            item["dynamic_minus_full_link_ms"] = float(dynamic_values[row_idx] - full_link[row_idx])
        dynamic_rows.append(item)

    dynamic_path = out_dir / "dynamic_best_by_step.csv"
    pd.DataFrame(dynamic_rows).to_csv(dynamic_path, index=False, encoding="utf-8-sig")

    segment_rows = contiguous_segments(best_names, steps, dynamic_values, static_values)
    for row in segment_rows:
        info = lookup.get(str(row["topology"]), {})
        row["motif"] = str(info.get("motif", ""))
        row["edges"] = str(info.get("edges", ""))
    segments = pd.DataFrame(segment_rows)
    segments.to_csv(out_dir / "dynamic_best_segments.csv", index=False, encoding="utf-8-sig")

    winner_rows = []
    for name in sorted(set(str(x) for x in best_names)):
        mask = best_names == name
        info = lookup.get(name, {})
        winner_rows.append(
            {
                "topology": name,
                "motif_id": int(info.get("motif_id", motif_id_from_topology(name))),
                "motif": str(info.get("motif", "")),
                "edges": str(info.get("edges", "")),
                "winning_points": int(np.sum(mask)),
                "winning_fraction": float(np.mean(mask)),
                "winning_time_start": int(np.min(steps[mask])),
                "winning_time_end": int(np.max(steps[mask])),
                "mean_delay_when_winning_ms": float(np.mean(dynamic_values[mask])),
                "mean_full_period_delay_ms": float(np.mean(motif_matrix[:, motif_cols.index(name)])),
            }
        )
    winner_summary = pd.DataFrame(winner_rows).sort_values(
        ["winning_points", "mean_delay_when_winning_ms"], ascending=[False, True]
    )
    winner_summary.to_csv(out_dir / "winner_summary.csv", index=False, encoding="utf-8-sig")

    write_plots(
        out_dir=out_dir,
        steps=steps,
        motif_matrix=motif_matrix,
        motif_cols=motif_cols,
        static_name=static_name,
        static_values=static_values,
        dynamic_values=dynamic_values,
        gridplus=gridplus,
        full_link=full_link,
        best_names=best_names,
        winner_summary=winner_summary,
    )

    improvement = static_values - dynamic_values
    meta = {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "num_steps": int(len(steps)),
        "actual_start_step": int(steps[0]),
        "actual_end_step": int(steps[-1]),
        "step_stride_observed": int(np.median(np.diff(steps))) if len(steps) > 1 else None,
        "candidate_scope": "selected motif columns only; gridplus and full_link are reference baselines",
        "num_candidate_motifs": int(len(motif_cols)),
        "static_best_topology": static_name,
        "static_best_mean_delay_ms": float(np.mean(static_values)),
        "dynamic_best_mean_delay_ms": float(np.mean(dynamic_values)),
        "dynamic_best_min_delay_ms": float(np.min(dynamic_values)),
        "dynamic_best_max_delay_ms": float(np.max(dynamic_values)),
        "mean_improvement_vs_static_ms": float(np.mean(improvement)),
        "max_improvement_vs_static_ms": float(np.max(improvement)),
        "steps_dynamic_better_than_static": int(np.sum(improvement > 1e-9)),
        "fraction_dynamic_better_than_static": float(np.mean(improvement > 1e-9)),
        "num_unique_winning_motifs": int(winner_summary.shape[0]),
        "num_switch_segments": int(len(segment_rows)),
        "gridplus_mean_delay_ms": float(np.mean(gridplus)) if gridplus is not None else None,
        "full_link_mean_delay_ms": float(np.mean(full_link)) if full_link is not None else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[dynamic-best] input={input_dir}")
    print(f"[dynamic-best] out={out_dir}")
    print(
        f"[dynamic-best] static={static_name} mean={meta['static_best_mean_delay_ms']:.4f} ms | "
        f"dynamic mean={meta['dynamic_best_mean_delay_ms']:.4f} ms | "
        f"improvement={meta['mean_improvement_vs_static_ms']:.4f} ms"
    )
    print(
        f"[dynamic-best] winners={meta['num_unique_winning_motifs']} "
        f"segments={meta['num_switch_segments']} better_steps={meta['steps_dynamic_better_than_static']}/{meta['num_steps']}"
    )
    print("[dynamic-best] top winners:")
    print(winner_summary.head(int(args.top_winners)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
