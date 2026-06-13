from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def infer_sample_seconds(steps: np.ndarray) -> float:
    steps = np.asarray(steps, dtype=np.float64)
    if steps.size < 2:
        return 1.0
    diffs = np.diff(steps)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if diffs.size else 1.0


def build_segment_cost_tables(value_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return best per-segment cost and argmin topology for every [i, j)."""

    values = np.asarray(value_matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("value_matrix must be 2D")
    num_steps, num_candidates = values.shape
    if num_steps == 0 or num_candidates == 0:
        raise ValueError("value_matrix must be non-empty")

    prefix = np.vstack(
        [
            np.zeros((1, num_candidates), dtype=np.float64),
            np.cumsum(values, axis=0),
        ]
    )
    seg_cost = np.full((num_steps + 1, num_steps + 1), np.inf, dtype=np.float64)
    seg_arg = np.full((num_steps + 1, num_steps + 1), -1, dtype=np.int32)
    for start in range(num_steps):
        sums = prefix[start + 1 :] - prefix[start]
        arg = np.argmin(sums, axis=1)
        cost = sums[np.arange(sums.shape[0]), arg]
        seg_cost[start, start + 1 :] = cost
        seg_arg[start, start + 1 :] = arg
    return seg_cost, seg_arg


def optimize_min_dwell(
    value_matrix: np.ndarray,
    min_len: int,
    seg_cost: np.ndarray | None = None,
    seg_arg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find the best piecewise-constant topology sequence with segment length >= min_len."""

    values = np.asarray(value_matrix, dtype=np.float64)
    num_steps = values.shape[0]
    min_len = int(min_len)
    if min_len <= 0:
        raise ValueError("min_len must be positive")
    if seg_cost is None or seg_arg is None:
        seg_cost, seg_arg = build_segment_cost_tables(values)

    dp = np.full(num_steps + 1, np.inf, dtype=np.float64)
    prev = np.full(num_steps + 1, -1, dtype=np.int32)
    dp[0] = 0.0
    for end in range(min_len, num_steps + 1):
        starts = np.arange(0, end - min_len + 1, dtype=np.int32)
        valid = np.isfinite(dp[starts])
        if not np.any(valid):
            continue
        starts = starts[valid]
        costs = dp[starts] + seg_cost[starts, end]
        best_pos = int(np.argmin(costs))
        dp[end] = float(costs[best_pos])
        prev[end] = int(starts[best_pos])

    if not np.isfinite(dp[num_steps]):
        raise RuntimeError(f"No feasible schedule for min_len={min_len}")

    topology_by_index = np.full(num_steps, -1, dtype=np.int32)
    end = num_steps
    while end > 0:
        start = int(prev[end])
        if start < 0:
            raise RuntimeError(f"Broken DP path for min_len={min_len}, end={end}")
        topology_by_index[start:end] = int(seg_arg[start, end])
        end = start

    chosen_values = values[np.arange(num_steps), topology_by_index]
    return topology_by_index, chosen_values, float(dp[num_steps])


def per_step_oracle(value_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(value_matrix, dtype=np.float64)
    topology_by_index = np.argmin(values, axis=1).astype(np.int32)
    chosen_values = values[np.arange(values.shape[0]), topology_by_index]
    return topology_by_index, chosen_values


def compress_segments(
    *,
    steps: Sequence[int],
    topology_by_index: np.ndarray,
    values_by_step: np.ndarray,
    topology_names: Sequence[str],
    topology_meta: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    steps_arr = np.asarray(steps, dtype=np.int64)
    topo = np.asarray(topology_by_index, dtype=np.int32)
    values = np.asarray(values_by_step, dtype=np.float64)
    meta_by_name = topology_meta or {}
    rows: list[dict[str, Any]] = []
    if steps_arr.size == 0:
        return rows

    start_idx = 0
    for idx in range(1, int(steps_arr.size) + 1):
        if idx < int(steps_arr.size) and int(topo[idx]) == int(topo[start_idx]):
            continue
        name = str(topology_names[int(topo[start_idx])])
        segment_values = values[start_idx:idx]
        meta = meta_by_name.get(name, {})
        rows.append(
            {
                "segment_id": len(rows),
                "start_index": int(start_idx),
                "end_index_exclusive": int(idx),
                "start_step": int(steps_arr[start_idx]),
                "end_step": int(steps_arr[idx - 1]),
                "num_points": int(idx - start_idx),
                "topology": name,
                "mean_value": float(np.mean(segment_values)),
                "min_value": float(np.min(segment_values)),
                "max_value": float(np.max(segment_values)),
                **{f"meta_{key}": value for key, value in meta.items()},
            }
        )
        start_idx = idx
    return rows


def read_compare_csv(
    path: str | Path,
    *,
    topology_prefix: str = "motif_",
    step_column: str = "step",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {path}")
        topology_columns = [name for name in reader.fieldnames if str(name).startswith(str(topology_prefix))]
        if not topology_columns:
            raise ValueError(f"No topology columns start with {topology_prefix!r}: {path}")
        steps: list[int] = []
        rows: list[list[float]] = []
        for item in reader:
            steps.append(int(float(item[step_column])))
            row_values = []
            for column in topology_columns:
                value = item.get(column, "")
                row_values.append(float(value) if value not in ("", None) else math.inf)
            rows.append(row_values)
    return np.asarray(steps, dtype=np.int64), topology_columns, np.asarray(rows, dtype=np.float64)


def write_csv_rows(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_schedule_by_step(
    path: str | Path,
    *,
    steps: Sequence[int],
    topology_by_index: np.ndarray,
    values_by_step: np.ndarray,
    topology_names: Sequence[str],
    reference_columns: dict[str, np.ndarray] | None = None,
) -> None:
    refs = reference_columns or {}
    rows = []
    for idx, step in enumerate(steps):
        row = {
            "step": int(step),
            "topology": str(topology_names[int(topology_by_index[idx])]),
            "value": float(values_by_step[idx]),
        }
        for name, values in refs.items():
            row[str(name)] = float(values[idx])
        rows.append(row)
    write_csv_rows(path, rows)


def write_dynamic_schedule_outputs(
    *,
    compare_csv: str | Path,
    out_dir: str | Path,
    topology_prefix: str = "motif_",
    min_dwell_minutes: Sequence[float] = (10.0, 30.0, 60.0, 120.0),
    metric_name: str = "value",
    reference_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Build oracle and min-dwell dynamic schedules from a comparison CSV."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps, topology_names, value_matrix = read_compare_csv(compare_csv, topology_prefix=topology_prefix)
    sample_seconds = infer_sample_seconds(steps)
    seg_cost, seg_arg = build_segment_cost_tables(value_matrix)

    oracle_idx, oracle_values = per_step_oracle(value_matrix)
    static_idx = int(np.argmin(np.mean(value_matrix, axis=0)))
    static_values = value_matrix[:, static_idx]

    summary_rows = [
        {
            "scheme": "static_best",
            "min_dwell_minutes": "",
            "mean_value": float(np.mean(static_values)),
            "min_value": float(np.min(static_values)),
            "max_value": float(np.max(static_values)),
            "num_segments": 1,
            "num_unique_topologies": 1,
            "topology": str(topology_names[static_idx]),
        },
        {
            "scheme": "per_step_oracle",
            "min_dwell_minutes": 0.0,
            "mean_value": float(np.mean(oracle_values)),
            "min_value": float(np.min(oracle_values)),
            "max_value": float(np.max(oracle_values)),
            "num_segments": len(compress_segments(steps=steps, topology_by_index=oracle_idx, values_by_step=oracle_values, topology_names=topology_names)),
            "num_unique_topologies": int(len(np.unique(oracle_idx))),
            "topology": "",
        },
    ]
    write_schedule_by_step(
        out_dir / "per_step_oracle_by_step.csv",
        steps=steps,
        topology_by_index=oracle_idx,
        values_by_step=oracle_values,
        topology_names=topology_names,
    )
    write_csv_rows(
        out_dir / "per_step_oracle_segments.csv",
        compress_segments(steps=steps, topology_by_index=oracle_idx, values_by_step=oracle_values, topology_names=topology_names),
    )

    schedule_lines = {"static_best": static_values, "per_step_oracle": oracle_values}
    for dwell_minutes in min_dwell_minutes:
        min_len = max(1, int(math.ceil(float(dwell_minutes) * 60.0 / sample_seconds)))
        topo_idx, chosen_values, total_cost = optimize_min_dwell(value_matrix, min_len, seg_cost, seg_arg)
        label = f"min_dwell_{float(dwell_minutes):g}min"
        schedule_lines[label] = chosen_values
        segments = compress_segments(
            steps=steps,
            topology_by_index=topo_idx,
            values_by_step=chosen_values,
            topology_names=topology_names,
        )
        write_schedule_by_step(
            out_dir / f"{label}_by_step.csv",
            steps=steps,
            topology_by_index=topo_idx,
            values_by_step=chosen_values,
            topology_names=topology_names,
        )
        write_csv_rows(out_dir / f"{label}_segments.csv", segments)
        summary_rows.append(
            {
                "scheme": label,
                "min_dwell_minutes": float(dwell_minutes),
                "mean_value": float(np.mean(chosen_values)),
                "min_value": float(np.min(chosen_values)),
                "max_value": float(np.max(chosen_values)),
                "num_segments": int(len(segments)),
                "num_unique_topologies": int(len(np.unique(topo_idx))),
                "topology": "",
                "total_cost": float(total_cost),
                "mean_gain_vs_static": float(np.mean(static_values - chosen_values)),
                "mean_gap_to_oracle": float(np.mean(chosen_values - oracle_values)),
            }
        )

    write_csv_rows(out_dir / "dynamic_schedule_summary.csv", summary_rows)
    plot_dynamic_schedules(
        out_dir / "dynamic_schedules_vs_references.png",
        steps=steps,
        lines=schedule_lines,
        metric_name=metric_name,
    )
    meta = {
        "compare_csv": str(Path(compare_csv)),
        "out_dir": str(out_dir),
        "metric_name": str(metric_name),
        "topology_prefix": str(topology_prefix),
        "num_steps": int(len(steps)),
        "num_candidate_topologies": int(len(topology_names)),
        "sample_seconds": float(sample_seconds),
        "min_dwell_minutes": [float(x) for x in min_dwell_minutes],
        "summary_csv": str(out_dir / "dynamic_schedule_summary.csv"),
    }
    (out_dir / "dynamic_schedule_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def plot_dynamic_schedules(
    path: str | Path,
    *,
    steps: Sequence[int],
    lines: dict[str, np.ndarray],
    metric_name: str = "value",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_hours = np.asarray(steps, dtype=np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(15, 7), dpi=180)
    colors = ["#2f5597", "#c00000", "#7030a0", "#00a2e8", "#70ad47", "#f4b183"]
    for idx, (label, values) in enumerate(lines.items()):
        linestyle = "--" if label == "per_step_oracle" else "-"
        ax.plot(x_hours, values, linewidth=1.5, color=colors[idx % len(colors)], linestyle=linestyle, label=label)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(str(metric_name))
    ax.set_title("Dynamic topology schedule under minimum dwell constraints")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
