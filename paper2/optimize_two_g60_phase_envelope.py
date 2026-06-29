from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


GENERIC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
THIS_DIR = Path(__file__).resolve().parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from g60_paper2_config import DATA_ROOT


DEFAULT_METRICS_DIR = (
    DATA_ROOT
    / "outputs"
    / "paper2_shortest_metrics"
    / "G60"
    / "plus_grid"
    / "t0_86400_stride60"
)
DEFAULT_OUT_DIR = DATA_ROOT / "outputs" / "paper2_two_constellation_phase_envelope"
METRIC_COLUMNS = ("mean_shortest_hops", "mean_shortest_delay_ms")


@dataclass(frozen=True)
class PairSeries:
    pair_key: str
    pair_label: str
    steps: np.ndarray
    values_by_metric: dict[str, np.ndarray]


def read_all_pairs_timeseries(path: Path) -> dict[str, PairSeries]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            grouped.setdefault(row["pair_key"], []).append(row)

    out: dict[str, PairSeries] = {}
    for pair_key, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(float(item["step"])))
        steps = np.asarray([int(float(row["step"])) for row in rows], dtype=np.int64)
        metric_values: dict[str, np.ndarray] = {}
        for metric in METRIC_COLUMNS:
            metric_values[metric] = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        out[pair_key] = PairSeries(
            pair_key=pair_key,
            pair_label=str(rows[0].get("pair_label") or pair_key),
            steps=steps,
            values_by_metric=metric_values,
        )
    return out


def read_single_pair_timeseries(path: Path, *, pair_key: str, pair_label: str) -> dict[str, PairSeries]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows.extend(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")

    rows = sorted(rows, key=lambda item: int(float(item["step"])))
    steps = np.asarray([int(float(row["step"])) for row in rows], dtype=np.int64)
    metric_values: dict[str, np.ndarray] = {}
    for metric in METRIC_COLUMNS:
        if metric not in rows[0]:
            raise ValueError(f"Missing metric column {metric!r} in {path}")
        metric_values[metric] = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return {
        str(pair_key): PairSeries(
            pair_key=str(pair_key),
            pair_label=str(pair_label),
            steps=steps,
            values_by_metric=metric_values,
        )
    }


def infer_period_seconds(steps: np.ndarray, period_seconds: int | None) -> int:
    if period_seconds is not None:
        return int(period_seconds)
    if steps.size < 2:
        raise ValueError("Need at least two steps or pass --period-seconds")
    if int(steps[0]) == 0 and int(steps[-1]) > int(steps[-2]):
        return int(steps[-1])
    return int(steps[-1] - steps[0] + int(np.median(np.diff(steps))))


def drop_duplicate_period_sample(steps: np.ndarray, values: np.ndarray, period_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    if steps.size >= 2 and int(steps[0]) == 0 and int(steps[-1]) == int(period_seconds):
        return steps[:-1], values[:-1]
    return steps, values


def optimize_one_series(values: np.ndarray, *, shift_chunk_size: int = 256) -> tuple[np.ndarray, int, float, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contains NaN or inf")
    means = np.empty(values.size, dtype=np.float64)
    n = int(values.size)
    chunk = max(1, int(shift_chunk_size))
    base = np.arange(n, dtype=np.int32)
    work_values = values.astype(np.float32, copy=False)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        shifts = np.arange(start, end, dtype=np.int32)
        idx = (base[None, :] + shifts[:, None]) % np.int32(n)
        shifted = work_values[idx]
        means[start:end] = np.minimum(work_values[None, :], shifted).mean(axis=1, dtype=np.float64)
    best_shift = int(np.argmin(means))
    envelope = np.minimum(values, np.roll(values, -best_shift))
    return means, best_shift, float(means[best_shift]), envelope


def optimize_combined(series: list[np.ndarray], *, shift_chunk_size: int = 256) -> tuple[np.ndarray, int, float, np.ndarray]:
    if not series:
        raise ValueError("series is empty")
    arrays = [np.asarray(values, dtype=np.float64) for values in series]
    if len({arr.size for arr in arrays}) != 1:
        raise ValueError("all series must have the same length")
    if not all(np.all(np.isfinite(arr)) for arr in arrays):
        raise ValueError("series contains NaN or inf")
    sweeps = []
    for arr in arrays:
        sweep, _best_shift, _best_mean, _envelope = optimize_one_series(
            arr,
            shift_chunk_size=int(shift_chunk_size),
        )
        sweeps.append(sweep)
    means = np.mean(np.vstack(sweeps), axis=0)
    best_shift = int(np.argmin(means))
    envelope = np.vstack([np.minimum(arr, np.roll(arr, -best_shift)) for arr in arrays])
    return means, best_shift, float(means[best_shift]), envelope


def shift_to_record(*, shift: int, stride_seconds: int, period_seconds: int, mean_value: float) -> dict[str, Any]:
    offset_seconds = int(shift) * int(stride_seconds)
    angle_deg = 360.0 * float(offset_seconds) / float(period_seconds)
    complement_deg = (360.0 - angle_deg) % 360.0
    canonical_deg = min(angle_deg, complement_deg)
    return {
        "shift_steps": int(shift),
        "offset_seconds": int(offset_seconds),
        "offset_hours": float(offset_seconds / 3600.0),
        "offset_angle_deg": float(angle_deg),
        "equivalent_reverse_angle_deg": float(complement_deg),
        "canonical_angle_deg": float(canonical_deg),
        "envelope_mean": float(mean_value),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_series(path: Path, *, title: str, steps: np.ndarray, original: np.ndarray, envelope: np.ndarray, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(steps, dtype=np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(16.2, 6.8), dpi=180)
    ax.plot(x, original, label="single G60", linewidth=1.0, alpha=0.78)
    ax.plot(x, envelope, label="two shifted G60 envelope", linewidth=1.35)
    ax.set_title(title)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sweep(path: Path, *, title: str, sweep: np.ndarray, stride_seconds: int, period_seconds: int, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(sweep.size, dtype=np.float64) * float(stride_seconds) / float(period_seconds) * 360.0
    fig, ax = plt.subplots(figsize=(14.5, 5.4), dpi=180)
    ax.plot(x, sweep, linewidth=1.25)
    ax.set_title(title)
    ax.set_xlabel("phase offset (degree)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize two shifted G60 envelopes from group-pair shortest-metric time series."
    )
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--all-pairs-csv", type=Path, default=None)
    parser.add_argument("--single-pair-csv", type=Path, default=None)
    parser.add_argument("--single-pair-key", type=str, default="single_pair")
    parser.add_argument("--single-pair-label", type=str, default="Single pair")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR / "G60_plus_grid_t0_86400_stride60")
    parser.add_argument("--pairs", nargs="*", default=None, help="Optional pair keys to include.")
    parser.add_argument("--period-seconds", type=int, default=86400)
    parser.add_argument("--shift-chunk-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.single_pair_csv is not None:
        all_pairs_csv = Path(args.single_pair_csv)
        pair_series = read_single_pair_timeseries(
            all_pairs_csv,
            pair_key=str(args.single_pair_key),
            pair_label=str(args.single_pair_label),
        )
    else:
        all_pairs_csv = (
            Path(args.all_pairs_csv)
            if args.all_pairs_csv is not None
            else Path(args.metrics_dir) / "all_pairs_timeseries.csv"
        )
        pair_series = read_all_pairs_timeseries(all_pairs_csv)
    selected = list(pair_series)
    if args.pairs:
        requested = {str(x) for x in args.pairs}
        selected = [key for key in pair_series if key in requested]
        missing = sorted(requested - set(selected))
        if missing:
            raise ValueError(f"Unknown pair keys: {missing}; available={sorted(pair_series)}")
    if not selected:
        raise ValueError("No pairs selected")

    first_steps = pair_series[selected[0]].steps
    period_seconds = infer_period_seconds(first_steps, args.period_seconds)
    step_diffs = np.diff(first_steps)
    if step_diffs.size == 0:
        raise ValueError("Need at least two samples")
    stride_seconds = int(np.median(step_diffs))
    summary_rows: list[dict[str, Any]] = []
    combined_inputs: dict[str, list[np.ndarray]] = {metric: [] for metric in METRIC_COLUMNS}
    common_steps = None

    for pair_key in selected:
        item = pair_series[pair_key]
        for metric in METRIC_COLUMNS:
            steps, values = drop_duplicate_period_sample(item.steps, item.values_by_metric[metric], period_seconds)
            if common_steps is None:
                common_steps = steps
            elif not np.array_equal(common_steps, steps):
                raise ValueError(f"Pair {pair_key} steps do not match the first pair")
            sweep, best_shift, best_mean, envelope = optimize_one_series(
                values,
                shift_chunk_size=int(args.shift_chunk_size),
            )
            baseline_mean = float(np.mean(values))
            record = {
                "scope": "pair",
                "pair_key": pair_key,
                "pair_label": item.pair_label,
                "metric": metric,
                "single_g60_mean": baseline_mean,
                "improvement_abs": float(baseline_mean - best_mean),
                "improvement_pct": float((baseline_mean - best_mean) / baseline_mean * 100.0)
                if baseline_mean
                else None,
                **shift_to_record(
                    shift=best_shift,
                    stride_seconds=stride_seconds,
                    period_seconds=period_seconds,
                    mean_value=best_mean,
                ),
            }
            summary_rows.append(record)
            combined_inputs[metric].append(values)

            pair_metric_dir = out_dir / "pairs" / pair_key / metric
            write_csv(
                pair_metric_dir / "offset_sweep.csv",
                [
                    {
                        "shift_steps": int(shift),
                        "offset_seconds": int(shift) * int(stride_seconds),
                        "offset_angle_deg": 360.0 * int(shift) * int(stride_seconds) / float(period_seconds),
                        "envelope_mean": float(value),
                    }
                    for shift, value in enumerate(sweep)
                ],
            )
            write_csv(
                pair_metric_dir / "optimized_envelope_timeseries.csv",
                [
                    {
                        "step": int(step),
                        "hour": float(int(step) / 3600.0),
                        "single_g60": float(original),
                        "two_g60_envelope": float(env_value),
                    }
                    for step, original, env_value in zip(steps, values, envelope)
                ],
            )
            plot_series(
                pair_metric_dir / "optimized_envelope.png",
                title=f"{item.pair_label} {metric}: two shifted G60 envelope",
                steps=steps,
                original=values,
                envelope=envelope,
                ylabel=metric,
            )
            plot_sweep(
                pair_metric_dir / "offset_sweep.png",
                title=f"{item.pair_label} {metric}: envelope mean vs phase offset",
                sweep=sweep,
                stride_seconds=stride_seconds,
                period_seconds=period_seconds,
                ylabel=f"mean envelope {metric}",
            )

    assert common_steps is not None
    for metric in METRIC_COLUMNS:
        sweep, best_shift, best_mean, envelope_matrix = optimize_combined(
            combined_inputs[metric],
            shift_chunk_size=int(args.shift_chunk_size),
        )
        baseline_matrix = np.vstack(combined_inputs[metric])
        baseline_mean = float(np.mean(baseline_matrix))
        summary_rows.append(
            {
                "scope": "combined_pairs",
                "pair_key": "ALL",
                "pair_label": "Selected pairs average",
                "metric": metric,
                "single_g60_mean": baseline_mean,
                "improvement_abs": float(baseline_mean - best_mean),
                "improvement_pct": float((baseline_mean - best_mean) / baseline_mean * 100.0)
                if baseline_mean
                else None,
                **shift_to_record(
                    shift=best_shift,
                    stride_seconds=stride_seconds,
                    period_seconds=period_seconds,
                    mean_value=best_mean,
                ),
            }
        )
        combined_dir = out_dir / "combined_pairs" / metric
        write_csv(
            combined_dir / "offset_sweep.csv",
            [
                {
                    "shift_steps": int(shift),
                    "offset_seconds": int(shift) * int(stride_seconds),
                    "offset_angle_deg": 360.0 * int(shift) * int(stride_seconds) / float(period_seconds),
                    "envelope_mean": float(value),
                }
                for shift, value in enumerate(sweep)
            ],
        )
        original_mean_series = np.mean(baseline_matrix, axis=0)
        envelope_mean_series = np.mean(envelope_matrix, axis=0)
        write_csv(
            combined_dir / "optimized_envelope_timeseries.csv",
            [
                {
                    "step": int(step),
                    "hour": float(int(step) / 3600.0),
                    "single_g60_selected_pair_mean": float(original),
                    "two_g60_envelope_selected_pair_mean": float(env_value),
                }
                for step, original, env_value in zip(common_steps, original_mean_series, envelope_mean_series)
            ],
        )
        plot_series(
            combined_dir / "optimized_envelope.png",
            title=f"Selected pairs {metric}: two shifted G60 envelope",
            steps=common_steps,
            original=original_mean_series,
            envelope=envelope_mean_series,
            ylabel=metric,
        )
        plot_sweep(
            combined_dir / "offset_sweep.png",
            title=f"Selected pairs {metric}: envelope mean vs phase offset",
            sweep=sweep,
            stride_seconds=stride_seconds,
            period_seconds=period_seconds,
            ylabel=f"mean envelope {metric}",
        )

    write_csv(out_dir / "phase_envelope_summary.csv", summary_rows)
    write_json(
        out_dir / "phase_envelope_meta.json",
        {
            "input_all_pairs_csv": str(all_pairs_csv),
            "selected_pairs": selected,
            "metrics": list(METRIC_COLUMNS),
            "period_seconds": int(period_seconds),
            "stride_seconds": int(stride_seconds),
            "angle_resolution_deg": 360.0 * float(stride_seconds) / float(period_seconds),
            "envelope_rule": "two_g60_envelope(t, shift) = min(metric(t), metric(t + shift mod period))",
            "summary_csv": str(out_dir / "phase_envelope_summary.csv"),
        },
    )
    print(f"[phase-envelope] out_dir={out_dir}", flush=True)
    for row in summary_rows:
        if row["scope"] == "combined_pairs":
            print(
                f"[phase-envelope] combined metric={row['metric']} "
                f"angle={row['offset_angle_deg']:.3f}deg "
                f"canonical={row['canonical_angle_deg']:.3f}deg "
                f"mean={row['envelope_mean']:.6f} "
                f"improvement={row['improvement_pct']:.2f}%",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
