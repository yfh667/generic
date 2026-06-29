from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io_utils import read_csv_rows, write_csv, write_json


@dataclass(frozen=True)
class MetricSeries:
    key: str
    label: str
    steps: np.ndarray
    values: np.ndarray
    metric: str


@dataclass(frozen=True)
class PhaseEnvelopeResult:
    metric: str
    best_shift_steps: int
    offset_seconds: int
    offset_angle_deg: float
    equivalent_reverse_angle_deg: float
    canonical_angle_deg: float
    single_mean: float
    envelope_mean: float
    improvement_abs: float
    improvement_pct: float
    sweep_means: np.ndarray
    envelope_values: np.ndarray


def read_metric_series_csv(
    path: str | Path,
    *,
    metric_column: str,
    key: str = "metric",
    label: str = "metric",
) -> MetricSeries:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in {path}")
    if metric_column not in rows[0]:
        raise ValueError(f"Missing metric column {metric_column!r} in {path}")
    rows = sorted(rows, key=lambda row: int(float(row["step"])))
    return MetricSeries(
        key=str(key),
        label=str(label),
        metric=str(metric_column),
        steps=np.asarray([int(float(row["step"])) for row in rows], dtype=np.int64),
        values=np.asarray([float(row[metric_column]) for row in rows], dtype=np.float64),
    )


def drop_duplicate_period_sample(
    steps: np.ndarray,
    values: np.ndarray,
    *,
    period_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    if steps.size >= 2 and int(steps[0]) == 0 and int(steps[-1]) == int(period_seconds):
        return steps[:-1], values[:-1]
    return steps, values


def optimize_phase_envelope_values(
    values: Sequence[float] | np.ndarray,
    *,
    shift_chunk_size: int = 256,
) -> tuple[np.ndarray, int, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contains NaN or inf")
    sweep = np.empty(values.size, dtype=np.float64)
    n = int(values.size)
    chunk = max(1, int(shift_chunk_size))
    base = np.arange(n, dtype=np.int32)
    work_values = values.astype(np.float32, copy=False)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        shifts = np.arange(start, end, dtype=np.int32)
        idx = (base[None, :] + shifts[:, None]) % np.int32(n)
        shifted = work_values[idx]
        sweep[start:end] = np.minimum(work_values[None, :], shifted).mean(axis=1, dtype=np.float64)
    best_shift = int(np.argmin(sweep))
    envelope = np.minimum(values, np.roll(values, -best_shift))
    return sweep, best_shift, envelope


def phase_envelope_summary(
    *,
    metric: str,
    values: np.ndarray,
    sweep_means: np.ndarray,
    best_shift_steps: int,
    envelope_values: np.ndarray,
    stride_seconds: int,
    period_seconds: int,
) -> PhaseEnvelopeResult:
    offset_seconds = int(best_shift_steps) * int(stride_seconds)
    angle = 360.0 * float(offset_seconds) / float(period_seconds)
    reverse = (360.0 - angle) % 360.0
    canonical = min(angle, reverse)
    single_mean = float(np.mean(values))
    envelope_mean = float(np.mean(envelope_values))
    improvement = single_mean - envelope_mean
    return PhaseEnvelopeResult(
        metric=str(metric),
        best_shift_steps=int(best_shift_steps),
        offset_seconds=int(offset_seconds),
        offset_angle_deg=float(angle),
        equivalent_reverse_angle_deg=float(reverse),
        canonical_angle_deg=float(canonical),
        single_mean=single_mean,
        envelope_mean=envelope_mean,
        improvement_abs=float(improvement),
        improvement_pct=float(improvement / single_mean * 100.0) if single_mean else 0.0,
        sweep_means=np.asarray(sweep_means, dtype=np.float64),
        envelope_values=np.asarray(envelope_values, dtype=np.float64),
    )


def optimize_metric_series(
    series: MetricSeries,
    *,
    period_seconds: int = 86400,
    shift_chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, PhaseEnvelopeResult]:
    steps, values = drop_duplicate_period_sample(series.steps, series.values, period_seconds=int(period_seconds))
    if steps.size < 2:
        raise ValueError("Need at least two samples")
    stride_seconds = int(np.median(np.diff(steps)))
    sweep, best_shift, envelope = optimize_phase_envelope_values(
        values,
        shift_chunk_size=int(shift_chunk_size),
    )
    summary = phase_envelope_summary(
        metric=series.metric,
        values=values,
        sweep_means=sweep,
        best_shift_steps=best_shift,
        envelope_values=envelope,
        stride_seconds=stride_seconds,
        period_seconds=int(period_seconds),
    )
    return steps, values, summary


def write_phase_envelope_outputs(
    *,
    out_dir: str | Path,
    series: MetricSeries,
    steps: np.ndarray,
    original_values: np.ndarray,
    result: PhaseEnvelopeResult,
    period_seconds: int = 86400,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    stride_seconds = int(np.median(np.diff(steps))) if steps.size >= 2 else int(period_seconds)
    sweep_rows = [
        {
            "shift_steps": int(shift),
            "offset_seconds": int(shift) * stride_seconds,
            "offset_angle_deg": 360.0 * int(shift) * stride_seconds / float(period_seconds),
            "envelope_mean": float(value),
        }
        for shift, value in enumerate(result.sweep_means)
    ]
    envelope_rows = [
        {
            "step": int(step),
            "hour": float(int(step) / 3600.0),
            "single_value": float(original),
            "two_constellation_envelope": float(envelope),
        }
        for step, original, envelope in zip(steps, original_values, result.envelope_values)
    ]
    summary_row = {
        "series_key": series.key,
        "series_label": series.label,
        "metric": result.metric,
        "single_mean": result.single_mean,
        "envelope_mean": result.envelope_mean,
        "improvement_abs": result.improvement_abs,
        "improvement_pct": result.improvement_pct,
        "best_shift_steps": result.best_shift_steps,
        "offset_seconds": result.offset_seconds,
        "offset_hours": float(result.offset_seconds / 3600.0),
        "offset_angle_deg": result.offset_angle_deg,
        "equivalent_reverse_angle_deg": result.equivalent_reverse_angle_deg,
        "canonical_angle_deg": result.canonical_angle_deg,
    }
    return {
        "offset_sweep": write_csv(out_dir / "offset_sweep.csv", sweep_rows),
        "envelope_timeseries": write_csv(out_dir / "optimized_envelope_timeseries.csv", envelope_rows),
        "summary": write_csv(out_dir / "phase_envelope_summary.csv", [summary_row]),
        "meta": write_json(out_dir / "phase_envelope_meta.json", dict(meta or {})),
    }
