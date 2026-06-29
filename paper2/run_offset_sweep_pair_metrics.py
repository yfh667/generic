from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    PairMetricColumns,
    discover_offset_inputs,
    pair_key,
    read_or_compute_offset_metrics,
    summarize_offset_sweep,
)


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


def default_g60_delay_csv(source_group: str, target_group: str) -> Path:
    return (
        DATA_ROOT
        / "outputs"
        / "paper2_ground_link_metrics"
        / "G60"
        / "plus_grid"
        / "t0_86163_stride1"
        / "pairs"
        / pair_key(source_group, target_group)
        / "timeseries.csv"
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def plot_offset_curves(*, summary_csv: Path, out_dir: Path, source_group: str, target_group: str) -> dict[str, Path]:
    rows = sorted(read_csv_rows(summary_csv), key=lambda row: float(row["offset_deg"]))
    if not rows:
        raise ValueError(f"No rows in {summary_csv}")
    offsets = np.asarray([float(row["offset_deg"]) for row in rows], dtype=np.float64)
    g60_hops = np.asarray([float(row["g60_mean_hops"]) for row in rows], dtype=np.float64)
    gw_hops = np.asarray([float(row["gw_offset_mean_hops"]) for row in rows], dtype=np.float64)
    dual_hops = np.asarray([float(row["dual_min_mean_hops"]) for row in rows], dtype=np.float64)
    g60_delay = np.asarray([float(row["g60_mean_delay_ms_with_ground"]) for row in rows], dtype=np.float64)
    gw_delay = np.asarray([float(row["gw_offset_mean_delay_ms_with_ground"]) for row in rows], dtype=np.float64)
    dual_delay = np.asarray([float(row["dual_min_mean_delay_ms_with_ground"]) for row in rows], dtype=np.float64)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    label = f"{source_group.title()}-{target_group.title()}"

    def draw(y_g60: np.ndarray, y_gw: np.ndarray, y_dual: np.ndarray, ylabel: str, filename: str) -> Path:
        best_idx = int(np.nanargmin(y_dual))
        fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=180)
        ax.axhline(float(np.nanmean(y_g60)), color="#616161", linewidth=1.0, linestyle=":", label="G60 baseline")
        ax.plot(offsets, y_gw, color="#1E88E5", linewidth=1.1, marker="o", markersize=2.8, label="GW offset only")
        ax.plot(offsets, y_dual, color="#D81B60", linewidth=1.4, marker="o", markersize=3.0, label="dual min(G60, GW_offset)")
        ax.scatter([offsets[best_idx]], [y_dual[best_idx]], color="#000000", s=26, zorder=5)
        ax.annotate(
            f"best {offsets[best_idx]:g} deg\n{y_dual[best_idx]:.4f}",
            xy=(offsets[best_idx], y_dual[best_idx]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
        )
        ax.set_title(f"{label} offset sweep")
        ax.set_xlabel("GW RAAN offset (deg)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.24, linestyle="--", linewidth=0.55)
        ax.legend(loc="best")
        fig.tight_layout()
        path = fig_dir / filename
        fig.savefig(path)
        plt.close(fig)
        return path

    return {
        "hops": draw(
            g60_hops,
            gw_hops,
            dual_hops,
            "mean shortest hops, satellite region average",
            "offset_sweep_mean_shortest_hops.png",
        ),
        "delay": draw(
            g60_delay,
            gw_delay,
            dual_delay,
            "mean shortest delay with ground propagation (ms)",
            "offset_sweep_mean_shortest_delay_ms.png",
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each real GW RAAN-offset data directory, compute pair metrics and draw "
            "offset-vs-dual-envelope curves against a fixed G60 baseline."
        )
    )
    parser.add_argument("--source-group", type=str, default="china")
    parser.add_argument("--target-group", type=str, default="europe")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--gw-offset-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--gw-offset-pattern", type=str, default="raan_*")
    parser.add_argument("--out-root", type=Path, default=DATA_ROOT / "outputs" / "paper2_real_raan_offset_sweep")
    parser.add_argument("--g60-hops-csv", type=Path, default=None)
    parser.add_argument("--g60-delay-csv", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-envelope-timeseries", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument(
        "--offset-workers",
        type=int,
        default=1,
        help="Number of RAAN offset directories to compute concurrently.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_group = str(args.source_group)
    target_group = str(args.target_group)
    pkey = pair_key(source_group, target_group)
    out_dir = Path(args.out_root) / pkey / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    g60_hops_csv = Path(args.g60_hops_csv) if args.g60_hops_csv else default_g60_hops_csv(source_group, target_group)
    g60_delay_csv = Path(args.g60_delay_csv) if args.g60_delay_csv else default_g60_delay_csv(source_group, target_group)
    for path in (g60_hops_csv, g60_delay_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing G60 baseline CSV: {path}\n"
                "Pass --g60-hops-csv/--g60-delay-csv, or compute the baseline first."
            )

    offset_inputs = discover_offset_inputs(root=args.gw_offset_root, pattern=str(args.gw_offset_pattern))
    if not offset_inputs:
        raise FileNotFoundError(
            f"No offset data directories found under {args.gw_offset_root} matching {args.gw_offset_pattern!r}"
        )

    print(
        f"[offset-sweep] pair={pkey} offsets={len(offset_inputs)} "
        f"range={offset_inputs[0].offset_deg:g}..{offset_inputs[-1].offset_deg:g}",
        flush=True,
    )
    config = build_paper2_gw_config()
    metric_csvs: list[Path | None] = [None] * len(offset_inputs)

    def compute_one(index: int) -> tuple[int, Path]:
        item = offset_inputs[index]
        path = read_or_compute_offset_metrics(
            offset_input=item,
            config=config,
            source_group=source_group,
            target_group=target_group,
            start=int(args.start),
            end=int(args.end),
            stride=int(args.stride),
            out_dir=out_dir,
            force=bool(args.force),
            progress_every=int(args.progress_every),
        )
        return index, path

    workers = max(1, int(args.offset_workers))
    if workers == 1:
        for idx in range(len(offset_inputs)):
            out_idx, path = compute_one(idx)
            metric_csvs[out_idx] = path
    else:
        print(f"[offset-sweep] offset_workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {executor.submit(compute_one, idx): idx for idx in range(len(offset_inputs))}
            completed = 0
            for future in as_completed(future_to_idx):
                out_idx, path = future.result()
                metric_csvs[out_idx] = path
                completed += 1
                print(
                    f"[offset-sweep] completed offsets {completed}/{len(offset_inputs)}: "
                    f"{offset_inputs[out_idx].label}",
                    flush=True,
                )
    if any(path is None for path in metric_csvs):
        raise RuntimeError("Some offset metric CSVs were not produced.")

    summary_csv, envelope_csv = summarize_offset_sweep(
        offset_inputs=offset_inputs,
        offset_metric_csvs=[Path(path) for path in metric_csvs if path is not None],
        g60_hops_csv=g60_hops_csv,
        g60_delay_csv=g60_delay_csv,
        source_group=source_group,
        target_group=target_group,
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
        out_dir=out_dir,
        metric_columns=PairMetricColumns(),
        write_timeseries=not bool(args.no_envelope_timeseries),
    )
    figures = plot_offset_curves(
        summary_csv=summary_csv,
        out_dir=out_dir,
        source_group=source_group,
        target_group=target_group,
    )
    print(f"[offset-sweep] summary={summary_csv}", flush=True)
    if envelope_csv is not None:
        print(f"[offset-sweep] envelope_timeseries={envelope_csv}", flush=True)
    print(f"[offset-sweep] hops_figure={figures['hops']}", flush=True)
    print(f"[offset-sweep] delay_figure={figures['delay']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
