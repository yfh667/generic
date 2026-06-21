from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _write_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            text = str(key)
            if text not in seen:
                fieldnames.append(text)
                seen.add(text)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def schedule_names_from_arrays(arrays: np.lib.npyio.NpzFile) -> tuple[str, ...]:
    return tuple(str(name)[len("schedule_") :] for name in arrays.files if str(name).startswith("schedule_"))


def setup_counts_from_selected_actions(
    selected_action: np.ndarray,
    transition_counts: np.ndarray,
    *,
    warm_start: bool = True,
) -> np.ndarray:
    """Count newly requested edges from a selected topology action sequence."""

    selected = np.asarray(selected_action, dtype=np.int32)
    transition = np.asarray(transition_counts, dtype=np.float64)
    counts = np.zeros(selected.shape[0], dtype=np.int32)
    if selected.size == 0:
        return counts
    if not bool(warm_start):
        counts[0] = int(round(float(transition[int(selected[0]), int(selected[0])])))
    for row in range(1, int(selected.size)):
        counts[row] = int(round(float(transition[int(selected[row - 1]), int(selected[row])])))
    return counts


def build_schedule_segment_rows(
    *,
    steps: np.ndarray,
    topology_names: Sequence[str],
    selected_action: np.ndarray,
    stage_cost: np.ndarray,
    transition_counts: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    steps = np.asarray(steps, dtype=np.int64)
    selected = np.asarray(selected_action, dtype=np.int32)
    stage = np.asarray(stage_cost, dtype=np.float64)
    transition = np.asarray(transition_counts, dtype=np.float64)
    if selected.shape[0] != steps.shape[0]:
        raise ValueError("selected_action length differs from steps")
    if stage.shape[0] != steps.shape[0]:
        raise ValueError("stage_cost row count differs from steps")
    if selected.size == 0:
        return [], [], {}

    setup_counts = setup_counts_from_selected_actions(selected, transition, warm_start=True)
    step_stride = int(steps[1] - steps[0]) if steps.size > 1 else 0
    chosen_stage = stage[np.arange(selected.shape[0]), selected]

    segment_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    start = 0
    segment_id = 0
    for row in range(1, int(selected.size) + 1):
        is_end = row == int(selected.size) or int(selected[row]) != int(selected[start])
        if not is_end:
            continue
        end_exclusive = row
        action = int(selected[start])
        prev_action = int(selected[start - 1]) if start > 0 else action
        segment_stage = chosen_stage[start:end_exclusive]
        duration_steps = int(end_exclusive - start)
        duration_s = int(steps[end_exclusive - 1] - steps[start] + step_stride) if step_stride > 0 else duration_steps
        setup_at_start = int(setup_counts[start])
        row_payload = {
            "segment_id": int(segment_id),
            "start_step": int(steps[start]),
            "end_step": int(steps[end_exclusive - 1]),
            "duration_steps": duration_steps,
            "duration_s": duration_s,
            "duration_h": float(duration_s / 3600.0),
            "action_idx": action,
            "topology": str(topology_names[action]),
            "previous_action_idx": prev_action if start > 0 else "",
            "previous_topology": str(topology_names[prev_action]) if start > 0 else "",
            "setup_commands_at_start": setup_at_start,
            "mean_stage_cost": float(np.mean(segment_stage)),
            "max_stage_cost": float(np.max(segment_stage)),
            "min_stage_cost": float(np.min(segment_stage)),
        }
        segment_rows.append(row_payload)
        if start > 0:
            switch_rows.append(
                {
                    "switch_id": int(len(switch_rows)),
                    "step": int(steps[start]),
                    "from_action_idx": prev_action,
                    "from_topology": str(topology_names[prev_action]),
                    "to_action_idx": action,
                    "to_topology": str(topology_names[action]),
                    "setup_commands": setup_at_start,
                    "previous_duration_steps": int(segment_rows[-2]["duration_steps"]) if len(segment_rows) >= 2 else "",
                    "previous_duration_s": int(segment_rows[-2]["duration_s"]) if len(segment_rows) >= 2 else "",
                    "next_segment_id": int(segment_id),
                }
            )
        start = row
        segment_id += 1

    nonzero_setup = setup_counts[setup_counts > 0]
    summary = {
        "num_steps": int(steps.size),
        "start_step": int(steps[0]),
        "end_step": int(steps[-1]),
        "step_stride_s": int(step_stride),
        "num_segments": int(len(segment_rows)),
        "num_switches": int(max(0, len(segment_rows) - 1)),
        "num_unique_topologies": int(len(set(int(x) for x in selected.tolist()))),
        "total_setup_commands": int(np.sum(setup_counts, dtype=np.int64)),
        "max_setup_commands_per_step": int(np.max(setup_counts)) if setup_counts.size else 0,
        "mean_setup_commands_per_step": float(np.mean(setup_counts)) if setup_counts.size else 0.0,
        "mean_nonzero_setup_commands": float(np.mean(nonzero_setup)) if nonzero_setup.size else 0.0,
        "min_segment_duration_steps": int(min(row["duration_steps"] for row in segment_rows)),
        "max_segment_duration_steps": int(max(row["duration_steps"] for row in segment_rows)),
        "mean_segment_duration_steps": float(np.mean([row["duration_steps"] for row in segment_rows])),
        "min_segment_duration_h": float(min(row["duration_h"] for row in segment_rows)),
        "max_segment_duration_h": float(max(row["duration_h"] for row in segment_rows)),
        "mean_stage_cost": float(np.mean(chosen_stage)),
        "max_stage_cost": float(np.max(chosen_stage)),
        "setup_commands_histogram": {
            str(int(value)): int(np.count_nonzero(setup_counts == int(value)))
            for value in sorted(set(int(x) for x in setup_counts.tolist()))
        },
    }
    if switch_rows:
        peak = max(switch_rows, key=lambda item: int(item["setup_commands"]))
        summary["peak_switch"] = dict(peak)
    else:
        summary["peak_switch"] = None
    if not math.isfinite(summary["mean_stage_cost"]):
        summary["mean_stage_cost"] = None
    return segment_rows, switch_rows, summary


def write_schedule_segment_report(
    *,
    selector_dir: str | Path,
    schedules: Sequence[str],
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    selector_dir = Path(selector_dir)
    arrays = np.load(selector_dir / "selector_arrays.npz", allow_pickle=False)
    available = set(schedule_names_from_arrays(arrays))
    if not schedules:
        schedules = tuple(sorted(available))
    out_path = Path(out_dir) if out_dir is not None else selector_dir / "schedule_segment_reports"
    out_path.mkdir(parents=True, exist_ok=True)

    steps = np.asarray(arrays["steps"], dtype=np.int64)
    topology_names = tuple(str(x) for x in np.asarray(arrays["topology_names"]))
    stage_cost = np.asarray(arrays["stage_cost"], dtype=np.float64)
    transition_counts = np.asarray(arrays["transition_counts"], dtype=np.float64)

    summary_rows: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, str]] = {}
    for schedule in schedules:
        schedule = str(schedule)
        key = f"schedule_{schedule}"
        if key not in arrays.files:
            raise KeyError(f"schedule {schedule!r} not found. Available examples: {sorted(available)[:8]}")
        selected = np.asarray(arrays[key], dtype=np.int32)
        segment_rows, switch_rows, summary = build_schedule_segment_rows(
            steps=steps,
            topology_names=topology_names,
            selected_action=selected,
            stage_cost=stage_cost,
            transition_counts=transition_counts,
        )
        schedule_dir = out_path / schedule
        segments_csv = schedule_dir / "segments.csv"
        switches_csv = schedule_dir / "switches.csv"
        summary_json = schedule_dir / "summary.json"
        _write_rows(segments_csv, segment_rows)
        _write_rows(switches_csv, switch_rows)
        summary_payload = {"schedule": schedule, **summary}
        summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(summary_payload)
        outputs[schedule] = {
            "segments_csv": str(segments_csv),
            "switches_csv": str(switches_csv),
            "summary_json": str(summary_json),
        }

    combined_csv = out_path / "schedule_segment_summary.csv"
    _write_rows(combined_csv, summary_rows)
    payload = {
        "selector_dir": str(selector_dir),
        "out_dir": str(out_path),
        "schedules": [str(x) for x in schedules],
        "combined_summary_csv": str(combined_csv),
        "outputs": outputs,
    }
    (out_path / "schedule_segment_report_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
