from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GENERIC_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
PAPER2_SRC = THIS_DIR / "src"
for path in (GENERIC_ROOT, THIS_DIR, PAPER2_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from g60_paper2_config import DATA_ROOT, build_paper2_gw_config
from offset_sweep_metrics import (
    build_hop_distance,
    build_station_visibility_from_intervals,
    discover_offset_inputs,
    group_station_specs,
    load_visibility_intervals_mat,
    pair_key,
    read_csv_rows,
    region_visible_union,
    write_csv,
)
from plot_g60_china_europe_shortest_metrics import build_topology_edge_table


def default_g60_hops_csv(source_group: str, target_group: str) -> Path:
    return (
        DATA_ROOT
        / "outputs"
        / "paper2_static_hop_metrics"
        / "G60"
        / "plus_grid"
        / "t0_86164_stride1"
        / "pairs"
        / pair_key(source_group, target_group)
        / "timeseries.csv"
    )


def load_g60_hops_for_steps(path: Path, steps: np.ndarray) -> np.ndarray:
    wanted = {int(step): idx for idx, step in enumerate(steps)}
    values = np.full(steps.size, np.nan, dtype=np.float32)
    for row in read_csv_rows(path):
        step = int(float(row["step"]))
        idx = wanted.get(step)
        if idx is not None:
            values[idx] = float(row["mean_shortest_hops"])
    missing = int(np.count_nonzero(~np.isfinite(values)))
    if missing:
        raise ValueError(f"{path} misses {missing} G60 hop values for requested steps")
    return values


def compute_offset_hops(
    *,
    offset_input,
    config,
    source_station_ids: list[int],
    target_station_ids: list[int],
    steps: np.ndarray,
    hop_dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intervals = load_visibility_intervals_mat(offset_input.visibility_intervals_mat)
    source_visible_sats, source_visible_counts = build_station_visibility_from_intervals(
        intervals=intervals,
        station_ids=source_station_ids,
        total_sats=int(config.total_sats),
        steps=steps,
    )
    target_visible_sats, target_visible_counts = build_station_visibility_from_intervals(
        intervals=intervals,
        station_ids=target_station_ids,
        total_sats=int(config.total_sats),
        steps=steps,
    )

    values = np.full(steps.size, np.nan, dtype=np.float32)
    source_counts = np.zeros(steps.size, dtype=np.int16)
    target_counts = np.zeros(steps.size, dtype=np.int16)
    for idx in range(steps.size):
        left = region_visible_union(source_visible_sats, source_visible_counts, idx)
        right = region_visible_union(target_visible_sats, target_visible_counts, idx)
        source_counts[idx] = int(left.size)
        target_counts[idx] = int(right.size)
        if left.size and right.size:
            values[idx] = float(np.mean(hop_dist[np.ix_(left, right)], dtype=np.float64))
    return values, source_counts, target_counts


def plot_outputs(*, summary_csv: Path, out_dir: Path, source_group: str, target_group: str) -> dict[str, Path]:
    rows = sorted(read_csv_rows(summary_csv), key=lambda row: float(row["offset_deg"]))
    offsets = np.asarray([float(row["offset_deg"]) for row in rows], dtype=np.float64)
    g60 = np.asarray([float(row["g60_mean_hops"]) for row in rows], dtype=np.float64)
    gw = np.asarray([float(row["gw_offset_mean_hops"]) for row in rows], dtype=np.float64)
    dual = np.asarray([float(row["dual_min_mean_hops"]) for row in rows], dtype=np.float64)
    best_idx = int(np.nanargmin(dual))

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    label = f"{source_group.title()}-{target_group.title()}"

    fig, ax = plt.subplots(figsize=(12.0, 5.6), dpi=180)
    ax.axhline(float(np.nanmean(g60)), color="#616161", linestyle=":", linewidth=1.1, label=f"G60 baseline {np.nanmean(g60):.3f}")
    ax.plot(offsets, gw, color="#1E88E5", linewidth=1.2, marker="o", markersize=2.7, label="GW offset only")
    ax.plot(offsets, dual, color="#D81B60", linewidth=1.5, marker="o", markersize=2.9, label="dual min(G60, GW_offset)")
    ax.scatter([offsets[best_idx]], [dual[best_idx]], color="black", s=30, zorder=5)
    ax.annotate(
        f"best {offsets[best_idx]:g} deg\n{dual[best_idx]:.4f}",
        xy=(offsets[best_idx], dual[best_idx]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set_title(f"{label} RAAN offset sweep, hops only")
    ax.set_xlabel("GW RAAN offset (deg)")
    ax.set_ylabel("mean shortest hops")
    ax.grid(True, alpha=0.24, linestyle="--", linewidth=0.55)
    ax.legend(loc="best")
    fig.tight_layout()
    curve_path = fig_dir / "offset_sweep_mean_shortest_hops.png"
    fig.savefig(curve_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    ax.hist(dual, bins=32, color="#D81B60", alpha=0.78, edgecolor="white", label="dual min mean hops")
    ax.axvline(float(dual[best_idx]), color="black", linestyle="--", linewidth=1.0, label=f"best {dual[best_idx]:.4f}")
    ax.set_title(f"{label} dual mean shortest hops distribution")
    ax.set_xlabel("dual mean shortest hops")
    ax.set_ylabel("offset count")
    ax.grid(True, axis="y", alpha=0.22, linestyle="--", linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    hist_path = fig_dir / "offset_sweep_dual_hops_histogram.png"
    fig.savefig(hist_path)
    plt.close(fig)

    return {"curve": curve_path, "histogram": hist_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute real GW RAAN offset sweep for region-average hops only.")
    parser.add_argument("--source-group", type=str, default="china")
    parser.add_argument("--target-group", type=str, default="europe")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--gw-offset-root", type=Path, default=DATA_ROOT / "GW_raan_frome_0")
    parser.add_argument("--gw-offset-pattern", type=str, default="raan_*")
    parser.add_argument("--g60-hops-csv", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=DATA_ROOT / "outputs" / "paper2_real_raan_offset_sweep" / "GW_raan_frome_0")
    parser.add_argument("--out-tag", type=str, default=None, help="Optional suffix for the output directory name.")
    parser.add_argument(
        "--no-timeseries",
        action="store_true",
        help="Only write per-offset summary and figures; useful for dense 1s sweeps.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_group = str(args.source_group)
    target_group = str(args.target_group)
    pkey = pair_key(source_group, target_group)
    steps = np.arange(int(args.start), int(args.end) + 1, int(args.stride), dtype=np.int64)
    out_name = f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}_hops_only"
    if args.out_tag:
        out_name = f"{out_name}_{str(args.out_tag).strip()}"
    out_dir = Path(args.out_root) / pkey / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    offset_inputs = discover_offset_inputs(root=args.gw_offset_root, pattern=str(args.gw_offset_pattern))
    if not offset_inputs:
        raise FileNotFoundError(f"No offset inputs found under {args.gw_offset_root}")

    g60_hops_csv = Path(args.g60_hops_csv) if args.g60_hops_csv else default_g60_hops_csv(source_group, target_group)
    g60_hops = load_g60_hops_for_steps(g60_hops_csv, steps)

    config = build_paper2_gw_config()
    edge_table = build_topology_edge_table(config, "plus_grid")
    hop_dist = build_hop_distance(np.asarray(edge_table.src, dtype=np.int32), np.asarray(edge_table.dst, dtype=np.int32), int(config.total_sats))
    source_ids = [int(spec.xml_station_id) for spec in group_station_specs(source_group)]
    target_ids = [int(spec.xml_station_id) for spec in group_station_specs(target_group)]

    summary_rows: list[dict] = []
    timeseries_rows: list[dict] = []
    t0 = time.time()
    for idx, item in enumerate(offset_inputs, 1):
        gw_hops, source_counts, target_counts = compute_offset_hops(
            offset_input=item,
            config=config,
            source_station_ids=source_ids,
            target_station_ids=target_ids,
            steps=steps,
            hop_dist=hop_dist,
        )
        dual = np.minimum(g60_hops, gw_hops)
        summary_rows.append(
            {
                "pair_key": pkey,
                "source_group": source_group,
                "target_group": target_group,
                "offset_deg": float(item.offset_deg),
                "offset_label": item.label,
                "samples": int(steps.size),
                "g60_mean_hops": float(np.nanmean(g60_hops)),
                "gw_offset_mean_hops": float(np.nanmean(gw_hops)),
                "dual_min_mean_hops": float(np.nanmean(dual)),
            }
        )
        if not args.no_timeseries:
            for row_i, step in enumerate(steps):
                timeseries_rows.append(
                    {
                        "offset_deg": float(item.offset_deg),
                        "offset_label": item.label,
                        "step": int(step),
                        "hour": float(int(step) / 3600.0),
                        "source_region_visible_sats": int(source_counts[row_i]),
                        "target_region_visible_sats": int(target_counts[row_i]),
                        "g60_hops": float(g60_hops[row_i]),
                        "gw_offset_hops": float(gw_hops[row_i]) if np.isfinite(gw_hops[row_i]) else None,
                        "dual_min_hops": float(dual[row_i]) if np.isfinite(dual[row_i]) else None,
                    }
                )
        if idx % int(args.progress_every) == 0 or idx == len(offset_inputs):
            print(f"[hops-only] completed {idx}/{len(offset_inputs)} elapsed={time.time() - t0:.1f}s", flush=True)

    summary_csv = write_csv(out_dir / "offset_sweep_hops_summary.csv", summary_rows)
    timeseries_csv = None
    if not args.no_timeseries:
        timeseries_csv = write_csv(out_dir / "offset_sweep_hops_timeseries.csv", timeseries_rows)
    figures = plot_outputs(summary_csv=summary_csv, out_dir=out_dir, source_group=source_group, target_group=target_group)
    best = min(summary_rows, key=lambda row: float(row["dual_min_mean_hops"]))
    print(f"[hops-only] summary={summary_csv}", flush=True)
    if timeseries_csv is not None:
        print(f"[hops-only] timeseries={timeseries_csv}", flush=True)
    print(f"[hops-only] curve={figures['curve']}", flush=True)
    print(f"[hops-only] histogram={figures['histogram']}", flush=True)
    print(f"[hops-only] best_offset_deg={best['offset_deg']} best_dual_min_mean_hops={best['dual_min_mean_hops']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
