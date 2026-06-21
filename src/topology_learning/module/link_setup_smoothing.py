from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.link_delay.module.edge_options import EdgeTable


@dataclass(frozen=True)
class StaggeredSetupResult:
    active_mask: np.ndarray
    building_mask: np.ndarray
    setup_command_mask: np.ndarray
    command_target_row: np.ndarray


@dataclass(frozen=True)
class _SetupEvent:
    edge: int
    activation_row: int
    earliest_command_row: int
    latest_command_row: int
    priority: float


def inter_port_keys_by_edge(edge_table: EdgeTable) -> list[tuple[tuple[int, str], ...]]:
    """Return physical inter-plane port keys occupied by each edge.

    For the current G60-style motif tables, inter-plane edges are stored from
    the lower plane to the higher plane. The lower-plane endpoint uses its
    right inter-plane port and the higher-plane endpoint uses its left
    inter-plane port. Intra-plane y-ring edges do not occupy these inter-plane
    ports and therefore return an empty tuple.
    """

    out: list[tuple[tuple[int, str], ...]] = []
    for idx in range(int(edge_table.num_edges)):
        option = int(edge_table.option[idx])
        if option < 0:
            out.append(())
            continue
        out.append(((int(edge_table.src[idx]), "right"), (int(edge_table.dst[idx]), "left")))
    return out


def _setup_lag_rows(steps: np.ndarray, setup_time_seconds: int | float) -> int:
    if float(setup_time_seconds) <= 0 or steps.size <= 1:
        return 0
    stride = int(steps[1] - steps[0])
    if stride <= 0:
        raise ValueError("steps must be strictly increasing")
    return int(math.ceil(float(setup_time_seconds) / float(stride)))


def _ready_row_for_command(steps: np.ndarray, command_row: int, setup_time_seconds: int | float) -> int:
    ready_time = float(steps[int(command_row)]) + float(setup_time_seconds)
    return int(np.searchsorted(steps, ready_time, side="left"))


def _target_run_start_rows(target: np.ndarray, edge: int) -> np.ndarray:
    col = np.asarray(target[:, int(edge)], dtype=bool)
    if col.size == 0:
        return np.asarray([], dtype=np.int32)
    previous_off = np.concatenate(([True], ~col[:-1]))
    return np.flatnonzero(col & previous_off).astype(np.int32, copy=False)


def _command_window_for_activation(
    *,
    steps: np.ndarray,
    activation_row: int,
    setup_time_seconds: int | float,
    schedule_window_rows: int | None,
) -> tuple[int, int]:
    """Return inclusive command-row window for a future activation row."""

    if float(setup_time_seconds) <= 0:
        return int(activation_row), int(activation_row)
    deadline_time = float(steps[int(activation_row)]) - float(setup_time_seconds)
    latest = int(np.searchsorted(steps, deadline_time, side="right") - 1)
    if latest < 0:
        latest = 0
    latest = min(latest, max(0, int(activation_row) - 1))
    if schedule_window_rows is None or int(schedule_window_rows) <= 0:
        earliest = 0
    else:
        earliest = max(0, latest - int(schedule_window_rows) + 1)
    return int(earliest), int(latest)


def _interval_overlaps(existing: list[tuple[int, int]], start: int, end: int) -> bool:
    if int(end) <= int(start):
        return False
    for left, right in existing:
        if int(start) < int(right) and int(left) < int(end):
            return True
    return False


def _target_port_owner_for_row(
    target_row: np.ndarray,
    port_keys: list[tuple[tuple[int, str], ...]],
) -> dict[tuple[int, str], int]:
    owner: dict[tuple[int, str], int] = {}
    for edge in np.flatnonzero(target_row):
        for port in port_keys[int(edge)]:
            owner[port] = int(edge)
    return owner


def build_staggered_setup_masks(
    *,
    steps: Sequence[int],
    target_mask: np.ndarray,
    edge_table: EdgeTable,
    setup_time_seconds: int | float,
    max_commands_per_step: int,
    spread_rows: int,
    edge_priority: np.ndarray | None = None,
    warm_start: bool = True,
) -> StaggeredSetupResult:
    """Build edge-level staggered setup masks for a desired topology sequence.

    ``target_mask`` is still the desired by-step topology. Unlike the basic
    reactive LST state machine, a topology change does not force every newly
    desired edge to start setup in the same row. New edges are ordered by
    ``edge_priority`` and spread across later rows with a per-row command cap.
    Old inter-plane edges may keep carrying traffic until a ready new edge
    claims the same left/right port.
    """

    steps_arr = np.asarray(steps, dtype=np.int64)
    target = np.asarray(target_mask, dtype=bool)
    if target.ndim != 2:
        raise ValueError("target_mask must be 2D")
    if steps_arr.shape[0] != target.shape[0]:
        raise ValueError("steps and target_mask row count differ")
    if target.shape[1] != int(edge_table.num_edges):
        raise ValueError("target_mask columns must match edge_table.num_edges")
    if int(max_commands_per_step) <= 0:
        raise ValueError("max_commands_per_step must be positive")
    if int(spread_rows) <= 0:
        raise ValueError("spread_rows must be positive")

    n_rows, n_edges = target.shape
    if n_rows == 0:
        empty = np.zeros_like(target, dtype=bool)
        return StaggeredSetupResult(
            active_mask=empty,
            building_mask=empty,
            setup_command_mask=empty,
            command_target_row=np.full(n_edges, -1, dtype=np.int32),
        )
    stride = int(steps_arr[1] - steps_arr[0]) if n_rows > 1 else 1
    lag_rows = int(math.ceil(float(setup_time_seconds) / float(stride))) if float(setup_time_seconds) > 0 else 0

    priority = np.zeros(n_edges, dtype=np.float32) if edge_priority is None else np.asarray(edge_priority, dtype=np.float32)
    if priority.shape[0] != n_edges:
        raise ValueError("edge_priority length must match edge_table.num_edges")

    command = np.zeros_like(target, dtype=bool)
    command_target_row = np.full(n_edges, -1, dtype=np.int32)
    command_load = np.zeros(n_rows, dtype=np.int32)

    previous = np.zeros(n_edges, dtype=bool)
    if bool(warm_start):
        previous = target[0].copy()
    else:
        added0 = np.flatnonzero(target[0])
        for edge in added0:
            if command_load[0] < int(max_commands_per_step):
                command[0, edge] = True
                command_load[0] += 1
                command_target_row[edge] = 0

    for row in range(1, n_rows):
        added = np.flatnonzero(target[row] & ~target[row - 1])
        if added.size == 0:
            continue
        ordered = added[np.argsort(-priority[added], kind="stable")]
        for edge in ordered:
            last = min(n_rows - 1, row + int(spread_rows) - 1)
            placed = False
            for cmd_row in range(row, last + 1):
                if not bool(target[cmd_row, edge]):
                    break
                if command_load[cmd_row] < int(max_commands_per_step):
                    command[cmd_row, edge] = True
                    command_load[cmd_row] += 1
                    command_target_row[edge] = row
                    placed = True
                    break
            if not placed:
                # Keep the cap as a hard control constraint. If a low-priority
                # edge cannot be scheduled inside the requested spread window,
                # it is skipped and the old active edge keeps carrying traffic.
                continue

    port_keys = inter_port_keys_by_edge(edge_table)
    port_to_edges: dict[tuple[int, str], list[int]] = {}
    for edge_idx, keys in enumerate(port_keys):
        for key in keys:
            port_to_edges.setdefault(key, []).append(edge_idx)

    active = np.zeros_like(target, dtype=bool)
    building = np.zeros_like(target, dtype=bool)
    ready_time = np.full(n_edges, np.nan, dtype=np.float64)
    current_active = target[0].copy() if bool(warm_start) else np.zeros(n_edges, dtype=bool)

    for row, step in enumerate(steps_arr):
        started = np.flatnonzero(command[row])
        for edge in started:
            ready_time[int(edge)] = float(step) + float(setup_time_seconds)

        pending = np.isfinite(ready_time) & (steps_arr[row] < ready_time)
        building[row, pending] = True

        ready_edges = np.flatnonzero(np.isfinite(ready_time) & (steps_arr[row] >= ready_time))
        for edge in ready_edges:
            edge = int(edge)
            if not bool(target[row, edge]):
                ready_time[edge] = np.nan
                continue
            for port in port_keys[edge]:
                for old_edge in port_to_edges.get(port, []):
                    if old_edge != edge:
                        current_active[old_edge] = False
            current_active[edge] = True
            ready_time[edge] = np.nan

        # Edges that are no longer desired and do not occupy an inter-plane
        # port are removed immediately. Inter-plane edges are removed when a
        # replacement claims the same port, preserving make-before-break
        # behavior during delayed setup.
        no_longer_target = current_active & ~target[row]
        for edge in np.flatnonzero(no_longer_target):
            if not port_keys[int(edge)]:
                current_active[int(edge)] = False

        active[row] = current_active

    return StaggeredSetupResult(
        active_mask=active,
        building_mask=building,
        setup_command_mask=command,
        command_target_row=command_target_row,
    )


def build_deadline_aware_setup_masks(
    *,
    steps: Sequence[int],
    target_mask: np.ndarray,
    edge_table: EdgeTable,
    setup_time_seconds: int | float,
    max_commands_per_step: int,
    schedule_window_rows: int | None,
    edge_priority: np.ndarray | None = None,
    warm_start: bool = True,
    enforce_building_port_exclusion: bool = True,
    placement_mode: str = "balanced",
) -> StaggeredSetupResult:
    """Build a just-in-time link setup schedule from a target topology sequence.

    The input ``target_mask`` is the ideal topology at each sampled row. This
    function treats each future target-active run as a deadline: a setup command
    only has to be issued early enough for the edge to become ready by that run.
    Within the allowed command window, ``placement_mode`` controls how commands
    are placed:

    * ``balanced`` keeps the historical behavior: prefer low-load command rows,
      with later rows used as a tie breaker.
    * ``latest`` places each command as late as possible before its deadline,
      only moving earlier when later rows are full or blocked by port setup
      exclusion.
    * ``latest_safe`` first builds the historical balanced feasible schedule,
      then greedily shifts scheduled commands right while preserving command
      caps and port setup exclusion. This keeps the target-active runs covered
      while avoiding unnecessarily early setup.

    Building edges are recorded but are not routable; an already active
    inter-plane edge keeps carrying traffic until its ready replacement claims
    the same physical left/right port.
    """

    steps_arr = np.asarray(steps, dtype=np.int64)
    target = np.asarray(target_mask, dtype=bool)
    if target.ndim != 2:
        raise ValueError("target_mask must be 2D")
    if steps_arr.shape[0] != target.shape[0]:
        raise ValueError("steps and target_mask row count differ")
    if target.shape[1] != int(edge_table.num_edges):
        raise ValueError("target_mask columns must match edge_table.num_edges")
    if int(max_commands_per_step) <= 0:
        raise ValueError("max_commands_per_step must be positive")
    placement = str(placement_mode or "balanced").lower()
    if placement not in {"balanced", "latest", "latest_safe"}:
        raise ValueError("placement_mode must be 'balanced', 'latest', or 'latest_safe'")

    n_rows, n_edges = target.shape
    empty_target_row = np.zeros(n_edges, dtype=bool)
    if n_rows == 0:
        empty = np.zeros_like(target, dtype=bool)
        return StaggeredSetupResult(
            active_mask=empty,
            building_mask=empty,
            setup_command_mask=empty,
            command_target_row=np.full(n_edges, -1, dtype=np.int32),
        )
    if n_rows > 1 and np.any(np.diff(steps_arr) <= 0):
        raise ValueError("steps must be strictly increasing")

    priority = np.zeros(n_edges, dtype=np.float32) if edge_priority is None else np.asarray(edge_priority, dtype=np.float32)
    if priority.shape[0] != n_edges:
        raise ValueError("edge_priority length must match edge_table.num_edges")

    if float(setup_time_seconds) <= 0:
        command = np.zeros_like(target, dtype=bool)
        if bool(warm_start):
            command[1:] = target[1:] & ~target[:-1]
        else:
            command[0] = target[0]
            command[1:] = target[1:] & ~target[:-1]
        return StaggeredSetupResult(
            active_mask=target.copy(),
            building_mask=np.zeros_like(target, dtype=bool),
            setup_command_mask=command,
            command_target_row=np.full(n_edges, -1, dtype=np.int32),
        )

    events: list[_SetupEvent] = []
    for edge in range(n_edges):
        for start in _target_run_start_rows(target, edge):
            if int(start) == 0 and bool(warm_start):
                continue
            earliest, latest = _command_window_for_activation(
                steps=steps_arr,
                activation_row=int(start),
                setup_time_seconds=float(setup_time_seconds),
                schedule_window_rows=schedule_window_rows,
            )
            events.append(
                _SetupEvent(
                    edge=int(edge),
                    activation_row=int(start),
                    earliest_command_row=int(earliest),
                    latest_command_row=int(latest),
                    priority=float(priority[int(edge)]),
                )
            )

    port_keys = inter_port_keys_by_edge(edge_table)
    command = np.zeros_like(target, dtype=bool)
    building = np.zeros_like(target, dtype=bool)
    command_target_row = np.full(n_edges, -1, dtype=np.int32)
    command_load = np.zeros(n_rows, dtype=np.int32)
    port_build_intervals: dict[tuple[int, str], list[tuple[int, int]]] = {}
    scheduled_by_ready_row: list[list[_SetupEvent]] = [[] for _ in range(n_rows)]
    scheduled_records: list[dict[str, object]] = []

    # Earliest deadlines are scheduled first. Within the same deadline, more
    # important edges win scarce command slots.
    events.sort(key=lambda item: (item.latest_command_row, item.activation_row, -item.priority, item.edge))
    for event in events:
        rows = range(int(event.earliest_command_row), int(event.latest_command_row) + 1)
        candidates: list[tuple[int, int]] = []
        for row in rows:
            if command_load[int(row)] >= int(max_commands_per_step):
                continue
            ready_row = _ready_row_for_command(steps_arr, int(row), float(setup_time_seconds))
            interval_end = min(int(ready_row), int(n_rows))
            if bool(enforce_building_port_exclusion):
                blocked = False
                for port in port_keys[int(event.edge)]:
                    if _interval_overlaps(port_build_intervals.get(port, []), int(row), interval_end):
                        blocked = True
                        break
                if blocked:
                    continue
            candidates.append((int(command_load[int(row)]), int(row)))
        if not candidates:
            continue

        if placement == "latest":
            # Prefer the latest feasible command row. This encodes the
            # deadline/JIT interpretation: a link that is only needed in the
            # future should not consume setup state earlier than necessary.
            _load, cmd_row = max(candidates, key=lambda item: item[1])
        else:
            # Prefer the least loaded command row; for equal load, stay as late
            # as possible so future-only links are not built immediately.
            _load, cmd_row = min(candidates, key=lambda item: (item[0], -item[1]))
        ready_row = _ready_row_for_command(steps_arr, int(cmd_row), float(setup_time_seconds))
        command[int(cmd_row), int(event.edge)] = True
        command_load[int(cmd_row)] += 1
        command_target_row[int(event.edge)] = int(event.activation_row)
        if int(cmd_row) < n_rows:
            building[int(cmd_row) : min(int(ready_row), int(n_rows)), int(event.edge)] = True
        if int(ready_row) < n_rows:
            scheduled_by_ready_row[int(ready_row)].append(event)
        if bool(enforce_building_port_exclusion):
            interval = (int(cmd_row), min(int(ready_row), int(n_rows)))
            for port in port_keys[int(event.edge)]:
                port_build_intervals.setdefault(port, []).append(interval)
        scheduled_records.append(
            {
                "event": event,
                "cmd_row": int(cmd_row),
                "ready_row": int(ready_row),
            }
        )

    def add_record_interval(
        intervals: dict[tuple[int, str], list[tuple[int, int]]],
        event: _SetupEvent,
        cmd_row: int,
        ready_row: int,
    ) -> None:
        interval = (int(cmd_row), min(int(ready_row), int(n_rows)))
        for port in port_keys[int(event.edge)]:
            intervals.setdefault(port, []).append(interval)

    def remove_record_interval(
        intervals: dict[tuple[int, str], list[tuple[int, int]]],
        event: _SetupEvent,
        cmd_row: int,
        ready_row: int,
    ) -> None:
        interval = (int(cmd_row), min(int(ready_row), int(n_rows)))
        for port in port_keys[int(event.edge)]:
            items = intervals.get(port)
            if not items:
                continue
            try:
                items.remove(interval)
            except ValueError:
                pass

    if placement == "latest_safe" and scheduled_records:
        shifted_load = np.zeros(n_rows, dtype=np.int32)
        shifted_intervals: dict[tuple[int, str], list[tuple[int, int]]] = {}
        for record in scheduled_records:
            event = record["event"]
            cmd_row = int(record["cmd_row"])
            ready_row = int(record["ready_row"])
            shifted_load[cmd_row] += 1
            if bool(enforce_building_port_exclusion):
                add_record_interval(shifted_intervals, event, cmd_row, ready_row)

        ordered_records = sorted(
            scheduled_records,
            key=lambda record: (
                int(record["event"].latest_command_row),
                int(record["event"].activation_row),
                float(record["event"].priority),
                int(record["event"].edge),
            ),
            reverse=True,
        )
        for record in ordered_records:
            event = record["event"]
            old_cmd = int(record["cmd_row"])
            old_ready = int(record["ready_row"])
            shifted_load[old_cmd] -= 1
            if bool(enforce_building_port_exclusion):
                remove_record_interval(shifted_intervals, event, old_cmd, old_ready)

            best_cmd = old_cmd
            best_ready = old_ready
            for row in range(int(event.latest_command_row), int(event.earliest_command_row) - 1, -1):
                if shifted_load[int(row)] >= int(max_commands_per_step):
                    continue
                ready_row = _ready_row_for_command(steps_arr, int(row), float(setup_time_seconds))
                interval_end = min(int(ready_row), int(n_rows))
                if bool(enforce_building_port_exclusion):
                    blocked = False
                    for port in port_keys[int(event.edge)]:
                        if _interval_overlaps(shifted_intervals.get(port, []), int(row), interval_end):
                            blocked = True
                            break
                    if blocked:
                        continue
                best_cmd = int(row)
                best_ready = int(ready_row)
                break

            record["cmd_row"] = int(best_cmd)
            record["ready_row"] = int(best_ready)
            shifted_load[int(best_cmd)] += 1
            if bool(enforce_building_port_exclusion):
                add_record_interval(shifted_intervals, event, best_cmd, best_ready)

    command.fill(False)
    building.fill(False)
    command_target_row.fill(-1)
    scheduled_by_ready_row = [[] for _ in range(n_rows)]
    for record in scheduled_records:
        event = record["event"]
        cmd_row = int(record["cmd_row"])
        ready_row = int(record["ready_row"])
        command[cmd_row, int(event.edge)] = True
        command_target_row[int(event.edge)] = int(event.activation_row)
        building[cmd_row : min(int(ready_row), int(n_rows)), int(event.edge)] = True
        if int(ready_row) < n_rows:
            scheduled_by_ready_row[int(ready_row)].append(event)

    port_to_edges: dict[tuple[int, str], list[int]] = {}
    for edge_idx, keys in enumerate(port_keys):
        for key in keys:
            port_to_edges.setdefault(key, []).append(edge_idx)

    active = np.zeros_like(target, dtype=bool)
    current_active = target[0].copy() if bool(warm_start) else np.zeros(n_edges, dtype=bool)
    prepared_by_activation_row: list[list[_SetupEvent]] = [[] for _ in range(n_rows)]

    for row in range(n_rows):
        target_row = target[row] if row < n_rows else empty_target_row
        ready_to_activate: list[int] = []

        for event in scheduled_by_ready_row[row]:
            edge = int(event.edge)
            if bool(target_row[edge]) and int(row) >= int(event.activation_row):
                ready_to_activate.append(edge)
            else:
                activation_row = int(event.activation_row)
                if 0 <= activation_row < n_rows:
                    prepared_by_activation_row[activation_row].append(event)

        for event in prepared_by_activation_row[row]:
            edge = int(event.edge)
            if bool(target_row[edge]):
                ready_to_activate.append(edge)

        ready_edges = sorted(set(ready_to_activate), key=lambda edge: (-float(priority[edge]), edge))
        for edge in ready_edges:
            for port in port_keys[int(edge)]:
                for old_edge in port_to_edges.get(port, []):
                    if int(old_edge) != int(edge):
                        current_active[int(old_edge)] = False
            current_active[int(edge)] = True

        target_port_owner = _target_port_owner_for_row(target_row, port_keys)
        for edge in np.flatnonzero(current_active & ~target_row):
            edge = int(edge)
            keys = port_keys[edge]
            if not keys:
                current_active[edge] = False
                continue
            keep_until_replacement_ready = False
            for port in keys:
                owner = target_port_owner.get(port)
                if owner is not None and int(owner) != edge and not bool(current_active[int(owner)]):
                    keep_until_replacement_ready = True
                    break
            if not keep_until_replacement_ready:
                current_active[edge] = False

        active[row] = current_active

    return StaggeredSetupResult(
        active_mask=active,
        building_mask=building,
        setup_command_mask=command,
        command_target_row=command_target_row,
    )
