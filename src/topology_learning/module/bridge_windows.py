from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ScheduleSegment:
    segment_id: int
    start_row: int
    end_row: int
    start_step: int
    end_step: int
    action_idx: int
    topology: str
    length_rows: int


@dataclass(frozen=True)
class BridgeWindow:
    middle_segment_id: int
    left_row: int
    right_row: int
    left_step: int
    right_step: int
    prev_action_idx: int
    middle_action_idx: int
    next_action_idx: int
    prev_topology: str
    middle_topology: str
    next_topology: str
    middle_length_rows: int
    entry_setup: int
    exit_setup: int
    bridge_setup: int
    direct_setup: int
    direct_saving: int
    max_bridge_setup: int


def compress_action_segments(
    *,
    selected: Sequence[int],
    steps: Sequence[int],
    topology_names: Sequence[str],
) -> list[ScheduleSegment]:
    selected_arr = np.asarray(selected, dtype=np.int32)
    steps_arr = np.asarray(steps, dtype=np.int64)
    names = np.asarray(topology_names)
    if selected_arr.ndim != 1:
        raise ValueError("selected must be 1D")
    if steps_arr.shape != selected_arr.shape:
        raise ValueError("steps and selected must have the same shape")
    if selected_arr.size == 0:
        return []

    out: list[ScheduleSegment] = []
    start = 0
    for row in range(1, selected_arr.size + 1):
        if row == selected_arr.size or int(selected_arr[row]) != int(selected_arr[start]):
            action = int(selected_arr[start])
            out.append(
                ScheduleSegment(
                    segment_id=len(out),
                    start_row=int(start),
                    end_row=int(row - 1),
                    start_step=int(steps_arr[start]),
                    end_step=int(steps_arr[row - 1]),
                    action_idx=action,
                    topology=str(names[action]),
                    length_rows=int(row - start),
                )
            )
            start = row
    return out


def bridge_windows_from_segments(
    *,
    segments: Sequence[ScheduleSegment],
    transition_counts: np.ndarray,
) -> list[BridgeWindow]:
    transition = np.asarray(transition_counts, dtype=np.float64)
    out: list[BridgeWindow] = []
    for seg_idx in range(1, len(segments) - 1):
        prev = segments[seg_idx - 1]
        middle = segments[seg_idx]
        nxt = segments[seg_idx + 1]
        entry = int(round(float(transition[int(prev.action_idx), int(middle.action_idx)])))
        exit_ = int(round(float(transition[int(middle.action_idx), int(nxt.action_idx)])))
        direct = int(round(float(transition[int(prev.action_idx), int(nxt.action_idx)])))
        bridge = int(entry + exit_)
        out.append(
            BridgeWindow(
                middle_segment_id=int(middle.segment_id),
                left_row=int(middle.start_row),
                right_row=int(middle.end_row),
                left_step=int(middle.start_step),
                right_step=int(middle.end_step),
                prev_action_idx=int(prev.action_idx),
                middle_action_idx=int(middle.action_idx),
                next_action_idx=int(nxt.action_idx),
                prev_topology=str(prev.topology),
                middle_topology=str(middle.topology),
                next_topology=str(nxt.topology),
                middle_length_rows=int(middle.length_rows),
                entry_setup=int(entry),
                exit_setup=int(exit_),
                bridge_setup=int(bridge),
                direct_setup=int(direct),
                direct_saving=int(bridge - direct),
                max_bridge_setup=int(max(entry, exit_)),
            )
        )
    return out


def bridge_windows_from_schedule(
    *,
    selected: Sequence[int],
    steps: Sequence[int],
    topology_names: Sequence[str],
    transition_counts: np.ndarray,
) -> tuple[list[ScheduleSegment], list[BridgeWindow]]:
    segments = compress_action_segments(
        selected=selected,
        steps=steps,
        topology_names=topology_names,
    )
    return segments, bridge_windows_from_segments(
        segments=segments,
        transition_counts=transition_counts,
    )


def write_dataclass_rows(path: str | Path, rows: Sequence[object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dict_rows = [asdict(row) for row in rows]
    if not dict_rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dict_rows)
