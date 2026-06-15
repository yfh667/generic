from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_paper1_motif_shortest_hops import load_yaml, path_from, region_pair_specs


DEFAULT_CONFIG = THIS_DIR / "configs" / "g60_w4_h3_shortest_hops.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-style plots for paper1 motif shortest-hop runs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--pairs", nargs="*", default=None)
    return parser.parse_args()


def sanitize_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def read_compare_csv(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    names = [str(x) for x in header[1:]]
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64, encoding="utf-8-sig")
    data = np.atleast_2d(data)
    return data[:, 0].astype(np.int64), names, data[:, 1:].astype(np.float64)


def read_baseline_names(run_dir: str | Path, run_label: str) -> set[str]:
    library_path = Path(run_dir) / f"topology_library_{sanitize_label(run_label)}.csv"
    if not library_path.exists():
        return {"full_link", "gridplus"}
    out: set[str] = set()
    with library_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("baseline", "")).lower() in {"true", "1", "yes"}:
                out.add(str(row["topology"]))
    return out or {"full_link", "gridplus"}


def finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def write_dynamic_csv(
    path: str | Path,
    *,
    steps: np.ndarray,
    dynamic_names: np.ndarray,
    dynamic_values: np.ndarray,
    best_static_name: str,
    best_static_values: np.ndarray,
    full_link_values: np.ndarray,
    gridplus_values: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "step",
            "dynamic_best_topology",
            "dynamic_best_hops",
            "best_static_topology",
            "best_static_hops",
            "full_link_hops",
            "gridplus_hops",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(steps):
            writer.writerow(
                {
                    "step": int(step),
                    "dynamic_best_topology": str(dynamic_names[idx]),
                    "dynamic_best_hops": float(dynamic_values[idx]),
                    "best_static_topology": str(best_static_name),
                    "best_static_hops": float(best_static_values[idx]),
                    "full_link_hops": float(full_link_values[idx]),
                    "gridplus_hops": float(gridplus_values[idx]),
                }
            )


def plot_pair(
    *,
    run_dir: Path,
    pair_key: str,
    pair_label: str,
    run_label: str,
    baseline_names: set[str],
    constellation_name: str,
    motif_label: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe_label = sanitize_label(run_label)
    pair_dir = run_dir / str(pair_key)
    compare_path = pair_dir / f"compare_{safe_label}_{pair_key}_strict_reachable.csv"
    steps, names, matrix = read_compare_csv(compare_path)
    name_to_col = {name: idx for idx, name in enumerate(names)}
    if "full_link" not in name_to_col or "gridplus" not in name_to_col:
        raise ValueError(f"{pair_key}: compare CSV must contain full_link and gridplus")

    candidate_indices = [idx for idx, name in enumerate(names) if name not in baseline_names]
    candidate_names = [names[idx] for idx in candidate_indices]
    candidate_matrix = matrix[:, candidate_indices]

    all_step_complete = np.all(np.isfinite(candidate_matrix), axis=0)
    if np.any(all_step_complete):
        means = np.full(len(candidate_names), np.nan, dtype=np.float64)
        means[all_step_complete] = np.nanmean(candidate_matrix[:, all_step_complete], axis=0)
        best_static_pool = "all_steps_complete"
    else:
        means = np.nanmean(candidate_matrix, axis=0)
        best_static_pool = "partial_complete_fallback"
    best_static_idx = int(np.nanargmin(means))
    best_static_name = candidate_names[best_static_idx]
    best_static_values = candidate_matrix[:, best_static_idx]

    finite_valid = np.isfinite(candidate_matrix)
    has_valid = np.any(finite_valid, axis=1)
    dynamic_idx = np.full(candidate_matrix.shape[0], -1, dtype=np.int32)
    dynamic_values = np.full(candidate_matrix.shape[0], np.nan, dtype=np.float64)
    if np.any(has_valid):
        safe = np.where(finite_valid, candidate_matrix, np.inf)
        dynamic_idx[has_valid] = np.argmin(safe[has_valid], axis=1)
        dynamic_values[has_valid] = safe[np.where(has_valid)[0], dynamic_idx[has_valid]]
    dynamic_names = np.asarray(
        [candidate_names[int(idx)] if int(idx) >= 0 else "" for idx in dynamic_idx],
        dtype=object,
    )

    full_link_values = matrix[:, name_to_col["full_link"]]
    gridplus_values = matrix[:, name_to_col["gridplus"]]
    violation_mask = np.isfinite(dynamic_values) & np.isfinite(full_link_values) & (dynamic_values < full_link_values - 1e-9)

    dynamic_csv = pair_dir / "paper_style_dynamic_best_mean_shortest_hops.csv"
    write_dynamic_csv(
        dynamic_csv,
        steps=steps,
        dynamic_names=dynamic_names,
        dynamic_values=dynamic_values,
        best_static_name=best_static_name,
        best_static_values=best_static_values,
        full_link_values=full_link_values,
        gridplus_values=gridplus_values,
    )

    summary = {
        "pair_key": str(pair_key),
        "pair_label": str(pair_label),
        "num_steps": int(steps.size),
        "num_motifs": int(len(candidate_names)),
        "num_all_steps_complete_motifs": int(np.count_nonzero(all_step_complete)),
        "num_incomplete_motifs": int(len(candidate_names) - np.count_nonzero(all_step_complete)),
        "best_static_pool": best_static_pool,
        "best_static_topology": str(best_static_name),
        "best_static_mean_hops": finite_mean(best_static_values),
        "dynamic_best_mean_hops": finite_mean(dynamic_values),
        "dynamic_unique_topologies": int(len(set(str(x) for x in dynamic_names if str(x)))),
        "dynamic_steps_without_complete_candidate": int(np.count_nonzero(~has_valid)),
        "dynamic_less_than_full_link_steps": int(np.count_nonzero(violation_mask)),
        "dynamic_less_than_full_link_max_gap_hops": (
            float(np.nanmax(full_link_values[violation_mask] - dynamic_values[violation_mask]))
            if np.any(violation_mask)
            else 0.0
        ),
        "full_link_mean_hops": finite_mean(full_link_values),
        "gridplus_mean_hops": finite_mean(gridplus_values),
        "dynamic_csv": str(dynamic_csv),
    }
    (pair_dir / "paper_style_hops_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    x_hours = steps.astype(np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(16, 7.3), dpi=180)
    for col_idx in range(candidate_matrix.shape[1]):
        ax.plot(x_hours, candidate_matrix[:, col_idx], color="#64748b", alpha=0.035, linewidth=0.45)
    ax.plot(x_hours, best_static_values, color="#2563eb", linewidth=2.1, label=f"best static motif: {best_static_name}")
    ax.plot(x_hours, dynamic_values, color="#111827", linestyle="--", linewidth=2.0, label="dynamic best motif")
    ax.plot(x_hours, full_link_values, color="#dc2626", linewidth=1.8, label="full_link")
    ax.plot(x_hours, gridplus_values, color="#16a34a", linewidth=1.8, label="+grid")
    ax.set_title(f"{constellation_name} {pair_label} mean shortest hops, {motif_label}")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(f"{pair_label} mean shortest path length (hops)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    plot_path = pair_dir / "paper_style_mean_shortest_hops.png"
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def motif_label_from_config(raw: dict[str, Any]) -> str:
    library_raw = raw.get("motif_library", {}) if isinstance(raw.get("motif_library", {}), dict) else {}
    min_w = library_raw.get("min_w")
    max_w = library_raw.get("max_w")
    min_h = library_raw.get("min_h")
    max_h = library_raw.get("max_h")
    if min_w == max_w and min_h == max_h and max_w is not None and max_h is not None:
        return f"w={max_w} h={max_h} motifs"
    if max_w is not None and max_h is not None:
        return f"w<={max_w} h<={max_h} motifs"
    return "motifs"


def main() -> int:
    args = parse_args()
    raw = load_yaml(args.config)
    paths = raw.get("paths", {}) if isinstance(raw.get("paths", {}), dict) else {}
    run_raw = raw.get("run", {}) if isinstance(raw.get("run", {}), dict) else {}
    run_dir = Path(args.run_dir) if args.run_dir is not None else path_from(paths, "out_dir")
    run_label = str(run_raw.get("label", "paper1_motif_shortest_hops"))
    pairs = region_pair_specs(raw, subset=args.pairs)
    baseline_names = read_baseline_names(run_dir, run_label)
    constellation_raw = raw.get("constellation", {}) if isinstance(raw.get("constellation", {}), dict) else {}
    constellation_name = str(constellation_raw.get("name", "constellation"))
    motif_label = motif_label_from_config(raw)
    plots = [
        str(
            plot_pair(
                run_dir=run_dir,
                pair_key=str(pair.key),
                pair_label=str(pair.label),
                run_label=run_label,
                baseline_names=baseline_names,
                constellation_name=constellation_name,
                motif_label=motif_label,
            )
        )
        for pair in pairs
    ]
    print(json.dumps({"plots": plots}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
