from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2].parent
DEFAULT_SUPPORT_GRIDPLUS_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_motif_gridplus_shortest_delay_parallel_t0_86164"
DEFAULT_FULL_LINK_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_full_link_shortest_delay_t0_86164"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_shortest_delay_compare_support_gridplus_full_link_t0_86164"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine G60 shortest-delay result folders into one comparison.")
    parser.add_argument("--support-gridplus-dir", type=Path, default=DEFAULT_SUPPORT_GRIDPLUS_DIR)
    parser.add_argument("--full-link-dir", type=Path, default=DEFAULT_FULL_LINK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def load_steps(path: Path) -> np.ndarray:
    return np.load(path / "time_indices.npy")


def load_means(path: Path) -> np.ndarray:
    return np.load(path / "mean_shortest_delay_ms.npy")


def main() -> int:
    args = parse_args()
    support_dir = Path(args.support_gridplus_dir) / "support_motif_DAD_Cxx"
    gridplus_dir = Path(args.support_gridplus_dir) / "gridplus"
    full_link_dir = Path(args.full_link_dir) / "full_link"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = load_steps(support_dir)
    grid_steps = load_steps(gridplus_dir)
    full_steps = load_steps(full_link_dir)
    if not (np.array_equal(steps, grid_steps) and np.array_equal(steps, full_steps)):
        raise ValueError("Input time_indices.npy files are not aligned.")

    series = {
        "support_motif_DAD_Cxx": load_means(support_dir),
        "gridplus": load_means(gridplus_dir),
        "full_link": load_means(full_link_dir),
    }

    with (out_dir / "compare_step_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "step",
            "support_motif_DAD_Cxx_mean_delay_ms",
            "gridplus_mean_delay_ms",
            "full_link_mean_delay_ms",
            "support_minus_gridplus_ms",
            "support_minus_full_link_ms",
            "gridplus_minus_full_link_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(steps):
            support = float(series["support_motif_DAD_Cxx"][idx])
            gridplus = float(series["gridplus"][idx])
            full_link = float(series["full_link"][idx])
            writer.writerow(
                {
                    "step": int(step),
                    "support_motif_DAD_Cxx_mean_delay_ms": support,
                    "gridplus_mean_delay_ms": gridplus,
                    "full_link_mean_delay_ms": full_link,
                    "support_minus_gridplus_ms": support - gridplus,
                    "support_minus_full_link_ms": support - full_link,
                    "gridplus_minus_full_link_ms": gridplus - full_link,
                }
            )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4.8))
        for name, values in series.items():
            ax.plot(steps, values, linewidth=1.0, label=name)
        ax.set_xlabel("time step (s)")
        ax.set_ylabel("China-Europe mean shortest delay (ms)")
        ax.grid(alpha=0.28, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "mean_shortest_delay_timeseries.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[combine-shortest-delay] plot skipped: {exc}", flush=True)

    stats = {}
    for name, values in series.items():
        finite = values[np.isfinite(values)]
        stats[name] = {
            "min_ms": float(np.min(finite)),
            "max_ms": float(np.max(finite)),
            "mean_ms": float(np.mean(finite)),
        }
    diff_support_grid = series["support_motif_DAD_Cxx"] - series["gridplus"]
    diff_grid_full = series["gridplus"] - series["full_link"]
    meta = {
        "num_steps": int(steps.size),
        "start": int(steps[0]),
        "end": int(steps[-1]),
        "inputs": {
            "support_gridplus_dir": str(Path(args.support_gridplus_dir)),
            "full_link_dir": str(Path(args.full_link_dir)),
        },
        "stats": stats,
        "support_better_than_gridplus_steps": int(np.sum(diff_support_grid < 0)),
        "gridplus_better_than_support_steps": int(np.sum(diff_support_grid > 0)),
        "gridplus_better_than_full_link_steps": int(np.sum(diff_grid_full < 0)),
        "full_link_better_than_gridplus_steps": int(np.sum(diff_grid_full > 0)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[combine-shortest-delay] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
