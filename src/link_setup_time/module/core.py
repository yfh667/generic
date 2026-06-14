from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable, OPTION_DELTAS, build_full_option_edges, write_edges_csv
from src.topology_metrics.module.group_states import group_nodes_for_step

from src.topology_workflow.module.region_constraints import (
    apply_region_internal_option_constraint,
    build_region_internal_option_edges,
    edge_side_for_node,
    edge_records_from_table,
    make_edge_table_subset,
)


__all__ = [
    "P_GRID_ACTIVE",
    "P_GRID_BUILDING",
    "P_MOTIF_ACTIVE",
    "P_MOTIF_BUILDING",
    "LinkSetupTimeStepStats",
    "DynamicRegionConstraintSeries",
    "expand_group_data_for_link_setup_time",
    "compute_link_state_masks",
    "build_dynamic_region_internal_option_constraint_series",
    "write_dynamic_region_constraint_series",
    "_all_group_ids",
    "_build_group_bits_from_state",
    "_compute_active_and_internal_keys_for_state",
    "_edge_key",
    "_edge_ports",
    "_edge_table_from_key_records",
    "_edge_table_key_set",
    "_future_building_edge_keys_by_step",
    "_group_nodes_mapping_for_step",
    "_group_state_key_for_step",
    "_record_nodes",
    "_record_side_for_node",
    "_records_by_key",
    "_remove_active_edges_conflicting_with_building",
    "_remove_active_key_conflicts_with_building",
    "_step_payload",
    "_validate_steps",
]


P_GRID_ACTIVE = 100
P_GRID_BUILDING = 50
P_MOTIF_ACTIVE = 20
P_MOTIF_BUILDING = 10


@dataclass(frozen=True)
class LinkSetupTimeStepStats:
    step: int
    active_edges: int
    building_edges: int
    forced_internal_option_edges: int
    dropped_edges: int
    building_conflict_dropped_edges: int
    actual_nodes_by_group: dict[str, int]


@dataclass(frozen=True)
class DynamicRegionConstraintSeries:
    """Viewer-ready dynamic topology generated from a static base topology.

    ``edge_table`` is the union of all edges that appear in any time slice.
    ``edge_active_mask[row, col]`` says whether union edge ``col`` is active at
    ``steps[row]``. ``edge_building_mask[row, col]`` says whether the edge is
    being prepared for a future region-internal constraint but is not yet
    available for routing at ``steps[row]``.
    """

    steps: tuple[int, ...]
    edge_table: EdgeTable
    edge_active_mask: np.ndarray
    edge_building_mask: np.ndarray
    stats: tuple[LinkSetupTimeStepStats, ...]


def _step_payload(group_data: Mapping, step: int) -> Mapping:
    if not group_data:
        return {}
    payload = group_data.get(int(step), None)
    if payload is None:
        payload = group_data.get(str(int(step)), {})
    return payload if isinstance(payload, Mapping) else {}


def _all_group_ids(group_data: Mapping, steps: Sequence[int], constrained_groups: Iterable[int]) -> set[int]:
    group_ids = {int(x) for x in constrained_groups}
    for step in steps:
        payload = _step_payload(group_data, int(step))
        groups = payload.get("groups", {}) if isinstance(payload, Mapping) else {}
        if isinstance(groups, Mapping):
            group_ids.update(int(gid) for gid in groups.keys())
    return group_ids


def _validate_steps(steps: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(x) for x in steps)
    if not normalized:
        raise ValueError("steps must be non-empty")
    if any(normalized[idx] > normalized[idx + 1] for idx in range(len(normalized) - 1)):
        raise ValueError("steps must be sorted in ascending order")
    return normalized


def expand_group_data_for_link_setup_time(
    *,
    group_data: Mapping,
    steps: Sequence[int],
    constrained_groups: Iterable[int],
    setup_time_seconds: int | float,
) -> dict[int, dict]:
    """Apply node-level link-setup lookahead to selected region groups.

    At time ``t``, a selected group contains the union of nodes that will belong
    to that same group at any sampled step ``t2`` satisfying
    ``0 <= t2 - t <= setup_time_seconds``. Non-selected groups are copied from
    the original time step for reference only.

    This helper is kept for diagnostics. Dynamic link construction below uses
    edge-level lookahead, because a union of future nodes can create false
    internal edges between nodes that never appear in the same group at the same
    future step.
    """

    steps_tuple = _validate_steps(steps)
    setup_time_seconds = float(setup_time_seconds)
    if setup_time_seconds < 0:
        raise ValueError("setup_time_seconds must be >= 0")

    constrained = tuple(int(x) for x in constrained_groups)
    all_group_ids = _all_group_ids(group_data, steps_tuple, constrained)
    expanded: dict[int, dict] = {
        int(step): {"groups": {int(gid): set() for gid in all_group_ids}, "all_mentioned": set()}
        for step in steps_tuple
    }

    for step in steps_tuple:
        for gid in all_group_ids:
            expanded[int(step)]["groups"][int(gid)] = set(group_nodes_for_step(group_data, int(step), int(gid)))

    for gid in constrained:
        counts: dict[int, int] = {}
        right = 0
        for left, step in enumerate(steps_tuple):
            while right < len(steps_tuple) and int(steps_tuple[right]) - int(step) <= setup_time_seconds:
                for node in group_nodes_for_step(group_data, int(steps_tuple[right]), int(gid)):
                    counts[int(node)] = counts.get(int(node), 0) + 1
                right += 1

            expanded[int(step)]["groups"][int(gid)] = set(counts.keys())

            for node in group_nodes_for_step(group_data, int(step), int(gid)):
                current = counts.get(int(node), 0) - 1
                if current <= 0:
                    counts.pop(int(node), None)
                else:
                    counts[int(node)] = current

    for step in steps_tuple:
        all_mentioned: set[int] = set()
        for nodes in expanded[int(step)]["groups"].values():
            all_mentioned.update(int(node) for node in nodes)
        expanded[int(step)]["all_mentioned"] = all_mentioned
    return expanded


def _edge_key(src: int, dst: int, total_nodes: int) -> int:
    a, b = (int(src), int(dst)) if int(src) <= int(dst) else (int(dst), int(src))
    return a * int(total_nodes) + b


def _edge_table_key_set(edge_table: EdgeTable, total_nodes: int) -> set[int]:
    return {
        _edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]), int(total_nodes))
        for idx in range(int(edge_table.num_edges))
    }


def _records_by_key(edge_table: EdgeTable, total_nodes: int) -> dict[int, tuple[int, int, int, int, int]]:
    out: dict[int, tuple[int, int, int, int, int]] = {}
    for idx, record in enumerate(edge_records_from_table(edge_table)):
        key = _edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]), int(total_nodes))
        out.setdefault(key, record)
    return out


def _record_nodes(record: tuple[int, int, int, int, int], n: int) -> tuple[int, int]:
    src_plane, src_y, dst_plane, dst_y, _option = record
    return int(src_plane) * int(n) + int(src_y), int(dst_plane) * int(n) + int(dst_y)


def _record_side_for_node(
    record: tuple[int, int, int, int, int],
    node: int,
    *,
    n: int,
    p: int,
    wrap_planes: bool,
) -> int | None:
    src_plane, src_y, dst_plane, dst_y, option = record
    if int(option) == -1:
        return None
    src = int(src_plane) * int(n) + int(src_y)
    dst = int(dst_plane) * int(n) + int(dst_y)
    if int(node) not in (src, dst):
        return None

    if int(option) in OPTION_DELTAS:
        dp, _dy = OPTION_DELTAS[int(option)]
        if bool(wrap_planes):
            src_to_dst = (int(src_plane) + int(dp)) % int(p) == int(dst_plane)
            dst_to_src = (int(dst_plane) + int(dp)) % int(p) == int(src_plane)
        else:
            src_to_dst = int(src_plane) + int(dp) == int(dst_plane)
            dst_to_src = int(dst_plane) + int(dp) == int(src_plane)
        if src_to_dst and not dst_to_src:
            return 1 if int(node) == src else 0
        if dst_to_src and not src_to_dst:
            return 1 if int(node) == dst else 0

    if int(node) == src:
        return 1 if int(dst_plane) > int(src_plane) else 0
    return 0 if int(src_plane) < int(dst_plane) else 1


def _edge_ports(
    record: tuple[int, int, int, int, int],
    *,
    n: int,
    p: int,
    wrap_planes: bool,
) -> tuple[tuple[int, int], ...]:
    if int(record[4]) == -1:
        return tuple()
    src, dst = _record_nodes(record, int(n))
    ports: list[tuple[int, int]] = []
    for node in (int(src), int(dst)):
        side = _record_side_for_node(
            record,
            int(node),
            n=int(n),
            p=int(p),
            wrap_planes=bool(wrap_planes),
        )
        if side is not None:
            ports.append((int(node), int(side)))
    return tuple(ports)


def compute_link_state_masks(
    *,
    steps_tuple: Sequence[int],
    state_ids: np.ndarray,
    state_payloads: Sequence[Mapping[str, object]],
    all_records_by_key: Mapping[int, tuple[int, int, int, int, int]],
    key_to_col: Mapping[int, int],
    n: int,
    p: int,
    wrap_planes: bool,
    setup_time_seconds: int | float,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply LST to a target topology sequence.

    The input target topology for each step is already region-constrained. This
    state machine then distinguishes:

    * active links, usable for routing;
    * building links, visible and terminal-occupying but not routable;
    * inactive links.

    New region-internal +grid links are anticipatory: they start occupying ports
    before their target activation time. New motif links are reactive: once they
    re-enter the target topology, they spend one LST interval in building state
    before becoming active. Step 0 is treated as a warm start.
    """

    times = [int(x) for x in steps_tuple]
    n_steps = len(times)
    setup_time_seconds = float(setup_time_seconds)
    if setup_time_seconds < 0:
        raise ValueError("setup_time_seconds must be >= 0")

    target_inter: list[set[int]] = [set() for _ in range(n_steps)]
    grid_inter: list[set[int]] = [set() for _ in range(n_steps)]
    intra: list[set[int]] = [set() for _ in range(n_steps)]
    ports_of: dict[int, tuple[tuple[int, int], ...]] = {}

    for row in range(n_steps):
        payload = state_payloads[int(state_ids[row])]
        internal_keys = set(int(k) for k in payload["internal_keys"])
        for raw_key in payload["active_keys"]:
            key = int(raw_key)
            record = all_records_by_key[int(key)]
            if int(record[4]) == -1:
                intra[row].add(key)
            else:
                target_inter[row].add(key)
                if key not in ports_of:
                    ports_of[key] = _edge_ports(
                        record,
                        n=int(n),
                        p=int(p),
                        wrap_planes=bool(wrap_planes),
                    )
        grid_inter[row] = internal_keys & target_inter[row]

    run_start: list[dict[int, int]] = [dict() for _ in range(n_steps)]
    for row in range(n_steps):
        for key in target_inter[row]:
            if row > 0 and key in target_inter[row - 1]:
                run_start[row][key] = int(run_start[row - 1][key])
            else:
                run_start[row][key] = int(row)

    entered_as_grid: dict[tuple[int, int], bool] = {}
    for row in range(n_steps):
        for key in target_inter[row]:
            start = int(run_start[row][key])
            entered_as_grid.setdefault((start, int(key)), int(key) in grid_inter[start])

    anticipatory_grid_build: list[set[int]] = [set() for _ in range(n_steps)]
    for start in range(1, n_steps):
        for key in grid_inter[start]:
            if key in target_inter[start - 1]:
                continue
            for row in range(start - 1, 0, -1):
                if times[start] - times[row] > setup_time_seconds:
                    break
                if key in target_inter[row]:
                    continue
                anticipatory_grid_build[row].add(int(key))

    active_mask = np.zeros((n_steps, int(n_cols)), dtype=bool)
    building_mask = np.zeros((n_steps, int(n_cols)), dtype=bool)

    for row in range(n_steps):
        target_active: set[int] = set()
        target_building: set[int] = set()
        for key in target_inter[row]:
            start = int(run_start[row][key])
            if (
                start == 0
                or entered_as_grid[(start, int(key))]
                or times[row] - times[start] >= setup_time_seconds
            ):
                target_active.add(int(key))
            else:
                target_building.add(int(key))

        port_owner: dict[tuple[int, int], tuple[int, int, bool]] = {}

        def claim(port: tuple[int, int], priority: int, key: int, is_active: bool) -> None:
            current = port_owner.get(port)
            if current is None or int(priority) > int(current[0]):
                port_owner[port] = (int(priority), int(key), bool(is_active))

        for key in anticipatory_grid_build[row]:
            for port in ports_of.get(int(key), tuple()):
                claim(port, P_GRID_BUILDING, int(key), False)
        for key in target_active:
            priority = P_GRID_ACTIVE if int(key) in grid_inter[row] else P_MOTIF_ACTIVE
            for port in ports_of.get(int(key), tuple()):
                claim(port, priority, int(key), True)
        for key in target_building:
            for port in ports_of.get(int(key), tuple()):
                claim(port, P_MOTIF_BUILDING, int(key), False)

        held_states: dict[int, list[bool]] = {}
        for _port, (_priority, key, is_active) in port_owner.items():
            held_states.setdefault(int(key), []).append(bool(is_active))

        for key, states in held_states.items():
            needed_ports = len(ports_of.get(int(key), tuple()))
            col = key_to_col.get(int(key))
            if col is None or needed_ports == 0 or len(states) != needed_ports:
                continue
            if all(states):
                active_mask[row, int(col)] = True
            elif not any(states):
                building_mask[row, int(col)] = True

        for key in intra[row]:
            col = key_to_col.get(int(key))
            if col is not None:
                active_mask[row, int(col)] = True

    return active_mask, building_mask


def _group_nodes_mapping_for_step(
    group_data: Mapping,
    step: int,
    constrained_groups: Iterable[int],
) -> dict[int, set[int]]:
    return {
        int(gid): set(group_nodes_for_step(group_data, int(step), int(gid)))
        for gid in constrained_groups
    }


def _group_state_key_for_step(
    group_data: Mapping,
    step: int,
    constrained_groups: Iterable[int],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(group_nodes_for_step(group_data, int(step), int(gid)))
        for gid in constrained_groups
    )


def _build_group_bits_from_state(
    state_key: tuple[tuple[int, ...], ...],
    total_nodes: int,
) -> np.ndarray:
    bits = np.zeros(int(total_nodes), dtype=np.uint64)
    for bit_idx, nodes in enumerate(state_key):
        bit = np.uint64(1 << int(bit_idx))
        for node in nodes:
            if 0 <= int(node) < int(total_nodes):
                bits[int(node)] |= bit
    return bits


def _compute_active_and_internal_keys_for_state(
    *,
    state_key: tuple[tuple[int, ...], ...],
    total_nodes: int,
    n: int,
    p: int,
    forced_option: int,
    wrap_planes: bool,
    base_records_by_key: Mapping[int, tuple[int, int, int, int, int]],
    forced_records_by_key: Mapping[int, tuple[int, int, int, int, int]],
) -> tuple[set[int], set[int]]:
    bits = _build_group_bits_from_state(state_key, int(total_nodes))

    internal_keys: set[int] = set()
    allowed_neighbors: dict[tuple[int, int], set[int]] = {}
    for key, record in forced_records_by_key.items():
        src, dst = _record_nodes(record, int(n))
        if int(bits[src] & bits[dst]) == 0:
            continue
        internal_keys.add(int(key))
        for node, neighbor in ((src, dst), (dst, src)):
            side = _record_side_for_node(
                record,
                int(node),
                n=int(n),
                p=int(p),
                wrap_planes=bool(wrap_planes),
            )
            if side is not None:
                allowed_neighbors.setdefault((int(node), int(side)), set()).add(int(neighbor))

    active_keys: set[int] = set()
    for key, record in base_records_by_key.items():
        src, dst = _record_nodes(record, int(n))
        option = int(record[4])
        if option != -1:
            same_selected_region = int(bits[src] & bits[dst]) != 0
            if same_selected_region and option != int(forced_option):
                continue

            conflict = False
            for node, neighbor in ((src, dst), (dst, src)):
                if int(bits[node]) == 0:
                    continue
                side = _record_side_for_node(
                    record,
                    int(node),
                    n=int(n),
                    p=int(p),
                    wrap_planes=bool(wrap_planes),
                )
                if side is None:
                    continue
                allowed = allowed_neighbors.get((int(node), int(side)))
                if allowed and int(neighbor) not in allowed and int(key) not in internal_keys:
                    conflict = True
                    break
            if conflict:
                continue
        active_keys.add(int(key))

    active_keys.update(internal_keys)
    return active_keys, internal_keys


def _remove_active_key_conflicts_with_building(
    *,
    active_keys: set[int],
    building_keys: set[int],
    records_by_key: Mapping[int, tuple[int, int, int, int, int]],
    total_nodes: int,
    n: int,
    p: int,
    wrap_planes: bool,
) -> tuple[set[int], int]:
    if not building_keys:
        return set(active_keys), 0

    allowed_neighbors: dict[tuple[int, int], set[int]] = {}
    for key in building_keys:
        record = records_by_key.get(int(key))
        if record is None:
            continue
        src, dst = _record_nodes(record, int(n))
        for node, neighbor in ((src, dst), (dst, src)):
            side = _record_side_for_node(
                record,
                int(node),
                n=int(n),
                p=int(p),
                wrap_planes=bool(wrap_planes),
            )
            if side is not None:
                allowed_neighbors.setdefault((int(node), int(side)), set()).add(int(neighbor))

    kept: set[int] = set()
    dropped = 0
    for key in active_keys:
        record = records_by_key.get(int(key))
        if record is None:
            continue
        if int(record[4]) == -1:
            kept.add(int(key))
            continue
        src, dst = _record_nodes(record, int(n))
        conflict = False
        for node, neighbor in ((src, dst), (dst, src)):
            side = _record_side_for_node(
                record,
                int(node),
                n=int(n),
                p=int(p),
                wrap_planes=bool(wrap_planes),
            )
            if side is None:
                continue
            allowed = allowed_neighbors.get((int(node), int(side)))
            if allowed and int(neighbor) not in allowed:
                conflict = True
                break
        if conflict:
            dropped += 1
        else:
            kept.add(int(key))
    return kept, int(dropped)


def _future_building_edge_keys_by_step(
    *,
    config: ViewerConfig,
    group_data: Mapping,
    steps: Sequence[int],
    constrained_groups: Iterable[int],
    setup_time_seconds: int | float,
    forced_option: int,
    wrap_planes: bool,
) -> tuple[list[set[int]], dict[int, tuple[int, int, int, int, int]]]:
    """Return per-step future setup edge keys and the records needed to draw them.

    A building edge at time ``t`` is an option edge that will be forced by the
    region-internal rule at some future sampled time ``t2`` where
    ``t < t2 <= t + setup_time_seconds``. It is deliberately not an active link
    at ``t``.
    """

    total_nodes = int(config.total_sats)
    setup_time_seconds = float(setup_time_seconds)
    if setup_time_seconds < 0:
        raise ValueError("setup_time_seconds must be >= 0")
    future_key_sets: list[set[int]] = []
    records: dict[int, tuple[int, int, int, int, int]] = {}
    future_internal_by_row: list[tuple[set[int], dict[int, tuple[int, int, int, int, int]]]] = []

    for step in steps:
        group_nodes = _group_nodes_mapping_for_step(group_data, int(step), constrained_groups)
        internal_edges = build_region_internal_option_edges(
            config=config,
            group_nodes=group_nodes,
            constrained_groups=constrained_groups,
            option=int(forced_option),
            wrap_planes=bool(wrap_planes),
        )
        key_set = _edge_table_key_set(internal_edges, total_nodes)
        record_map = _records_by_key(internal_edges, total_nodes)
        future_internal_by_row.append((key_set, record_map))
        records.update(record_map)

    right = 0
    counts: dict[int, int] = {}
    for left, step in enumerate(steps):
        while right < len(steps) and int(steps[right]) - int(step) <= setup_time_seconds:
            if right > left:
                keys, _record_map = future_internal_by_row[right]
                for key in keys:
                    counts[int(key)] = counts.get(int(key), 0) + 1
            right += 1

        future_key_sets.append(set(counts.keys()))

        if left + 1 < len(steps):
            keys_to_remove, _record_map = future_internal_by_row[left + 1]
            for key in keys_to_remove:
                current = counts.get(int(key), 0) - 1
                if current <= 0:
                    counts.pop(int(key), None)
                else:
                    counts[int(key)] = current

    return future_key_sets, records


def _edge_table_from_key_records(
    *,
    config: ViewerConfig,
    keys: Iterable[int],
    records_by_key: Mapping[int, tuple[int, int, int, int, int]],
    sat_ids: list[str],
) -> EdgeTable:
    from src.topology_workflow.module.edge_tables import make_edge_table_from_records

    records = [records_by_key[int(key)] for key in keys if int(key) in records_by_key]
    return make_edge_table_from_records(
        p=int(config.P),
        n=int(config.N),
        records=records,
        sat_ids=list(sat_ids),
    )


def _remove_active_edges_conflicting_with_building(
    *,
    active_edge_table: EdgeTable,
    building_edge_table: EdgeTable,
    total_nodes: int,
    p: int,
    wrap_planes: bool,
) -> tuple[EdgeTable, int]:
    """Remove active inter-plane edges whose terminal side is occupied by setup.

    A building edge is not usable for routing yet, but it reserves the same
    left/right terminal side that the future active edge will need. Therefore an
    active edge on that same side must be dropped unless it connects to the same
    neighbor as the building edge.
    """

    if int(building_edge_table.num_edges) == 0:
        return active_edge_table, 0

    allowed_neighbors = [[set(), set()] for _ in range(int(total_nodes))]
    for idx in range(int(building_edge_table.num_edges)):
        src = int(building_edge_table.src[idx])
        dst = int(building_edge_table.dst[idx])
        for node, neighbor in ((src, dst), (dst, src)):
            side = edge_side_for_node(
                building_edge_table,
                idx,
                int(node),
                p=int(p),
                wrap_planes=bool(wrap_planes),
            )
            if side is not None:
                allowed_neighbors[int(node)][int(side)].add(int(neighbor))

    keep = np.ones(int(active_edge_table.num_edges), dtype=bool)
    for idx in range(int(active_edge_table.num_edges)):
        if int(active_edge_table.option[idx]) == -1:
            continue
        src = int(active_edge_table.src[idx])
        dst = int(active_edge_table.dst[idx])
        for node, neighbor in ((src, dst), (dst, src)):
            side = edge_side_for_node(
                active_edge_table,
                idx,
                int(node),
                p=int(p),
                wrap_planes=bool(wrap_planes),
            )
            if side is None:
                continue
            allowed = allowed_neighbors[int(node)][int(side)]
            if allowed and int(neighbor) not in allowed:
                keep[idx] = False
                break

    dropped = int(np.count_nonzero(~keep))
    if dropped == 0:
        return active_edge_table, 0
    return make_edge_table_subset(active_edge_table, keep), dropped


def build_dynamic_region_internal_option_constraint_series(
    *,
    config: ViewerConfig,
    base_edge_table: EdgeTable,
    group_data: Mapping,
    steps: Sequence[int],
    constrained_groups: Iterable[int],
    setup_time_seconds: int | float,
    forced_option: int = 0,
    wrap_planes: bool = False,
    group_names: Mapping[int, str] | None = None,
) -> DynamicRegionConstraintSeries:
    """Build a dynamic topology with region-internal links prepared in advance.

    The base topology is normally a tiled motif plus intra-y links. For each
    time step, active edges are built from the actual group state at that time.
    Building edges are the forced region-internal option edges needed by a
    future group state within ``setup_time_seconds``; they are exported and can
    be drawn as dashed links, but they are not active routing edges.
    """

    steps_tuple = _validate_steps(steps)
    constrained = tuple(int(x) for x in constrained_groups)
    total_nodes = int(config.total_sats)

    from src.topology_workflow.module.edge_tables import make_edge_table_from_records

    forced_option_edges = build_full_option_edges(
        config,
        options=(int(forced_option),),
        sat_ids=list(base_edge_table.sat_ids),
        wrap_planes=bool(wrap_planes),
    )
    base_records_by_key = _records_by_key(base_edge_table, total_nodes)
    forced_records_by_key = _records_by_key(forced_option_edges, total_nodes)
    all_records_by_key = dict(base_records_by_key)
    all_records_by_key.update(forced_records_by_key)

    union_edge_table = make_edge_table_from_records(
        p=int(config.P),
        n=int(config.N),
        records=list(all_records_by_key.values()),
        sat_ids=list(base_edge_table.sat_ids),
    )
    union_key_to_col = {
        _edge_key(int(union_edge_table.src[idx]), int(union_edge_table.dst[idx]), total_nodes): int(idx)
        for idx in range(int(union_edge_table.num_edges))
    }
    active_mask = np.zeros((len(steps_tuple), int(union_edge_table.num_edges)), dtype=bool)
    building_mask = np.zeros((len(steps_tuple), int(union_edge_table.num_edges)), dtype=bool)

    state_to_id: dict[tuple[tuple[int, ...], ...], int] = {}
    state_payloads: list[dict[str, object]] = []
    state_ids = np.empty(len(steps_tuple), dtype=np.int32)
    for row, step in enumerate(steps_tuple):
        state_key = _group_state_key_for_step(group_data, int(step), constrained)
        state_id = state_to_id.get(state_key)
        if state_id is None:
            active_keys, internal_keys = _compute_active_and_internal_keys_for_state(
                state_key=state_key,
                total_nodes=total_nodes,
                n=int(config.N),
                p=int(config.P),
                forced_option=int(forced_option),
                wrap_planes=bool(wrap_planes),
                base_records_by_key=base_records_by_key,
                forced_records_by_key=forced_records_by_key,
            )
            internal_new_count = len(set(internal_keys) - set(base_records_by_key))
            region_dropped = int(base_edge_table.num_edges + internal_new_count - len(active_keys))
            state_id = len(state_payloads)
            state_to_id[state_key] = state_id
            state_payloads.append(
                {
                    "state_key": state_key,
                    "active_keys": active_keys,
                    "active_cols": np.asarray(
                        [union_key_to_col[int(key)] for key in active_keys if int(key) in union_key_to_col],
                        dtype=np.int32,
                    ),
                    "internal_keys": internal_keys,
                    "region_dropped": region_dropped,
                    "actual_nodes_by_group": {
                        str(gid): int(len(state_key[pos]))
                        for pos, gid in enumerate(constrained)
                    },
                }
            )
        state_ids[row] = int(state_id)

    active_mask, building_mask = compute_link_state_masks(
        steps_tuple=steps_tuple,
        state_ids=state_ids,
        state_payloads=state_payloads,
        all_records_by_key=all_records_by_key,
        key_to_col=union_key_to_col,
        n=int(config.N),
        p=int(config.P),
        wrap_planes=bool(wrap_planes),
        setup_time_seconds=float(setup_time_seconds),
        n_cols=int(union_edge_table.num_edges),
    )

    stats: list[LinkSetupTimeStepStats] = []
    for left, step in enumerate(steps_tuple):
        payload = state_payloads[int(state_ids[left])]
        active_count = int(np.count_nonzero(active_mask[left]))
        building_count = int(np.count_nonzero(building_mask[left]))
        target_count = int(len(payload["active_keys"]))

        stats.append(
            LinkSetupTimeStepStats(
                step=int(step),
                active_edges=active_count,
                building_edges=building_count,
                forced_internal_option_edges=int(len(payload["internal_keys"])),
                dropped_edges=int(payload["region_dropped"]),
                building_conflict_dropped_edges=int(max(0, target_count - active_count)),
                actual_nodes_by_group=dict(payload["actual_nodes_by_group"]),
            )
        )

    return DynamicRegionConstraintSeries(
        steps=steps_tuple,
        edge_table=union_edge_table,
        edge_active_mask=active_mask,
        edge_building_mask=building_mask,
        stats=tuple(stats),
    )


def write_dynamic_region_constraint_series(
    series: DynamicRegionConstraintSeries,
    out_dir: str | Path,
    *,
    meta: Mapping | None = None,
    write_edge_state_csv: bool = True,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "steps.npy", np.asarray(series.steps, dtype=np.int64))
    np.save(out_dir / "edge_active_mask.npy", np.asarray(series.edge_active_mask, dtype=bool))
    np.save(out_dir / "edge_building_mask.npy", np.asarray(series.edge_building_mask, dtype=bool))
    write_edges_csv(series.edge_table, out_dir / "union_edges.csv")

    with (out_dir / "step_stats.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "step",
            "active_edges",
            "building_edges",
            "forced_internal_option_edges",
            "dropped_edges",
            "building_conflict_dropped_edges",
            "actual_nodes_by_group",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in series.stats:
            payload = asdict(row)
            payload["actual_nodes_by_group"] = json.dumps(payload["actual_nodes_by_group"], ensure_ascii=False)
            writer.writerow(payload)

    if write_edge_state_csv:
        with (out_dir / "edge_states.csv").open("w", encoding="utf-8-sig", newline="") as f:
            fieldnames = [
                "step",
                "edge_idx",
                "state",
                "src_node",
                "dst_node",
                "src_plane",
                "src_y",
                "dst_plane",
                "dst_y",
                "option",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row, step in enumerate(series.steps):
                active_cols = np.where(series.edge_active_mask[row])[0]
                building_cols = np.where(series.edge_building_mask[row])[0]
                for state, cols in (("active", active_cols), ("building", building_cols)):
                    for edge_idx in cols:
                        idx = int(edge_idx)
                        writer.writerow(
                            {
                                "step": int(step),
                                "edge_idx": idx,
                                "state": state,
                                "src_node": int(series.edge_table.src[idx]),
                                "dst_node": int(series.edge_table.dst[idx]),
                                "src_plane": int(series.edge_table.src_plane[idx]),
                                "src_y": int(series.edge_table.src_y[idx]),
                                "dst_plane": int(series.edge_table.dst_plane[idx]),
                                "dst_y": int(series.edge_table.dst_y[idx]),
                                "option": int(series.edge_table.option[idx]),
                            }
                        )

    meta_payload = dict(meta or {})
    meta_payload.update(
        {
            "num_steps": int(len(series.steps)),
            "union_edges": int(series.edge_table.num_edges),
            "active_mask_shape": [int(x) for x in series.edge_active_mask.shape],
            "building_mask_shape": [int(x) for x in series.edge_building_mask.shape],
            "outputs": {
                "steps": str(out_dir / "steps.npy"),
                "edge_active_mask": str(out_dir / "edge_active_mask.npy"),
                "edge_building_mask": str(out_dir / "edge_building_mask.npy"),
                "union_edges": str(out_dir / "union_edges.csv"),
                "step_stats": str(out_dir / "step_stats.csv"),
                "edge_states": str(out_dir / "edge_states.csv") if write_edge_state_csv else None,
            },
        }
    )
    (out_dir / "meta.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
