from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_motif0056_to_0061_link_setup as one_way  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import write_edges_csv  # noqa: E402
from src.topology_workflow.module.edge_tables import INTRA_OPTION, make_edge_table_from_records  # noqa: E402


DEFAULT_USAGE_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_switch_056_061_056_china_europe_1s_delay"
    r"\t0_86160_stride1\switch36000_54000\setup600_delay\tracked_usage"
)
DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\guarded_t2_switch_056_061_china_europe"
)


@dataclass(frozen=True)
class UsageStore:
    topology: one_way.TopologyData
    steps: np.ndarray
    counts: np.ndarray
    key_to_col: dict[tuple[int, int], int]


@dataclass(frozen=True)
class GuardedEvent:
    transition: str
    owner: int
    old: one_way.RightLink | None
    new: one_way.RightLink
    plan_start: int
    plan_end: int
    target_interval_start: int
    target_interval_end: int
    skipped_intervals: int
    blocker_summary: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a guarded T2 switch plan for motif000056 -> motif000061. "
            "The switch boundary is hard: no target link becomes active before T2. "
            "A changed link switches only when the pure motif000056 old links "
            "that would be released have zero edge betweenness for the whole "
            "LST setup window."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--segment-end", type=int, default=54000)
    parser.add_argument("--lst", type=int, default=60)
    parser.add_argument("--source-motif-id", type=int, default=56)
    parser.add_argument("--target-motif-id", type=int, default=61)
    parser.add_argument("--usage-root", type=Path, default=DEFAULT_USAGE_ROOT)
    parser.add_argument("--working-threshold", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--write-edge-state-csv", action="store_true")
    return parser.parse_args()


def build_static_topology(motif_id: int) -> one_way.TopologyData:
    rows = one_way.read_motif_rows(one_way.MOTIF_LIBRARY_CSV)
    spec = one_way.build_topology_spec(int(motif_id), rows[int(motif_id)])
    right_by_owner, left_by_right = one_way.build_right_links(spec.edge_table)
    return one_way.TopologyData(
        spec=spec,
        values=np.empty((0, int(spec.edge_table.num_edges)), dtype=np.float32),
        right_by_owner=right_by_owner,
        left_by_right=left_by_right,
        edge_key_to_idx=one_way.build_edge_key_index(spec.edge_table),
    )


def parse_edge_key_text(text: str) -> tuple[int, int]:
    left, right = str(text).strip().split("-", 1)
    return one_way.edge_key(int(left), int(right))


def read_usage_store(root: Path, topology: one_way.TopologyData) -> UsageStore:
    store_dir = Path(root) / topology.spec.name
    steps_path = store_dir / "time_indices.npy"
    counts_path = store_dir / "tracked_edge_usage_counts.npy"
    edges_path = store_dir / "tracked_edges.csv"
    if not steps_path.exists() or not counts_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"missing tracked usage store under {store_dir}")

    key_to_col: dict[tuple[int, int], int] = {}
    with edges_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key_to_col[parse_edge_key_text(row["edge_key"])] = int(row["tracked_col"])

    steps = np.load(steps_path)
    counts = np.load(counts_path, mmap_mode="r")
    if counts.shape[0] != steps.size:
        raise ValueError(f"usage row mismatch under {store_dir}: {counts.shape[0]} != {steps.size}")
    return UsageStore(
        topology=topology,
        steps=np.asarray(steps, dtype=np.int64),
        counts=counts,
        key_to_col=key_to_col,
    )


def slice_usage_store(store: UsageStore, *, start: int, end: int, stride: int) -> UsageStore:
    mask = (
        (store.steps >= int(start))
        & (store.steps <= int(end))
        & (((store.steps - int(start)) % int(stride)) == 0)
    )
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        raise ValueError(f"no rows in usage store for start={start}, end={end}, stride={stride}")
    return UsageStore(
        topology=store.topology,
        steps=np.asarray(store.steps[rows], dtype=np.int64),
        counts=np.asarray(store.counts[rows, :]),
        key_to_col=dict(store.key_to_col),
    )


def union_edge_table(source: one_way.TopologyData, target: one_way.TopologyData):
    records: list[tuple[int, int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for edge_table in (source.spec.edge_table, target.spec.edge_table):
        for idx in range(edge_table.num_edges):
            src = int(edge_table.src[idx])
            dst = int(edge_table.dst[idx])
            key = one_way.edge_key(src, dst)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                (
                    int(edge_table.src_plane[idx]),
                    int(edge_table.src_y[idx]),
                    int(edge_table.dst_plane[idx]),
                    int(edge_table.dst_y[idx]),
                    int(edge_table.option[idx]),
                )
            )
    return make_edge_table_from_records(p=int(G60_CONFIG.P), n=int(G60_CONFIG.N), records=records)


def edge_key_index(edge_table) -> dict[tuple[int, int], int]:
    return {
        one_way.edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])): int(idx)
        for idx in range(edge_table.num_edges)
    }


def all_intra_keys(edge_table) -> set[tuple[int, int]]:
    return {
        one_way.edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]))
        for idx in range(edge_table.num_edges)
        if int(edge_table.option[idx]) == INTRA_OPTION
    }


def interval_rows(steps: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.flatnonzero((steps >= int(start)) & (steps < int(end)))


def usage_values_for_link(store: UsageStore, link: one_way.RightLink | None) -> np.ndarray | None:
    if link is None:
        return None
    col = store.key_to_col.get(link.edge_key)
    if col is None:
        return None
    return np.asarray(store.counts[:, int(col)], dtype=np.float32)


def max_usage_in_window(
    store: UsageStore,
    link: one_way.RightLink | None,
    *,
    start: int,
    end: int,
) -> float:
    values = usage_values_for_link(store, link)
    if values is None:
        return 0.0
    rows = interval_rows(store.steps, int(start), int(end))
    if rows.size == 0:
        return 0.0
    return float(np.nanmax(values[rows]))


def working_intervals(
    store: UsageStore,
    link: one_way.RightLink,
    *,
    start: int,
    end: int,
    threshold: float,
) -> list[tuple[int, int, float]]:
    values = usage_values_for_link(store, link)
    if values is None:
        return []
    rows = interval_rows(store.steps, int(start), int(end))
    if rows.size == 0:
        return []
    working = np.asarray(values[rows] > float(threshold), dtype=bool)
    hits = np.flatnonzero(working)
    if hits.size == 0:
        return []

    stride = int(store.steps[1] - store.steps[0]) if store.steps.size > 1 else 1
    intervals: list[tuple[int, int, float]] = []
    begin_local = int(hits[0])
    prev_local = int(hits[0])
    for local in hits[1:]:
        local = int(local)
        if local != prev_local + 1:
            start_row = int(rows[begin_local])
            end_row = int(rows[prev_local])
            intervals.append(
                (
                    int(store.steps[start_row]),
                    int(store.steps[end_row]) + stride,
                    float(np.nanmax(values[start_row : end_row + 1])),
                )
            )
            begin_local = local
        prev_local = local
    start_row = int(rows[begin_local])
    end_row = int(rows[prev_local])
    intervals.append(
        (
            int(store.steps[start_row]),
            int(store.steps[end_row]) + stride,
            float(np.nanmax(values[start_row : end_row + 1])),
        )
    )
    return intervals


def link_ports(link: one_way.RightLink | None) -> tuple[tuple[str, int], ...]:
    if link is None:
        return tuple()
    return (("R", int(link.owner)), ("L", int(link.right)))


def source_port_occupants_for_new_link(
    source_topology: one_way.TopologyData,
    new_link: one_way.RightLink,
) -> list[tuple[str, one_way.RightLink]]:
    occupants: list[tuple[str, one_way.RightLink]] = []
    owner_right = source_topology.right_by_owner.get(int(new_link.owner))
    if owner_right is not None and owner_right.edge_key != new_link.edge_key:
        occupants.append(("source_owner_right", owner_right))
    target_left = source_topology.left_by_right.get(int(new_link.right))
    if target_left is not None and target_left.edge_key != new_link.edge_key:
        occupants.append(("source_target_left", target_left))
    return occupants


def windows_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return int(a_start) < int(b_end) and int(b_start) < int(a_end)


def event_by_owner(events: Sequence[GuardedEvent]) -> dict[int, GuardedEvent]:
    return {int(event.owner): event for event in events}


def blocker_conflicts(
    *,
    owner: int,
    new_link: one_way.RightLink,
    source: UsageStore,
    scheduled: Sequence[GuardedEvent],
    setup_start: int,
    setup_end: int,
    threshold: float,
) -> list[str]:
    conflicts: list[str] = []
    scheduled_by_owner = event_by_owner(scheduled)
    owner_old = source.topology.right_by_owner.get(int(owner))
    old_max = max_usage_in_window(source, owner_old, start=int(setup_start), end=int(setup_end))
    if old_max > float(threshold):
        conflicts.append(f"owner_old_working:{owner_old.edge_key_text if owner_old is not None else ''}:{old_max:g}")

    left_blocker = source.topology.left_by_right.get(int(new_link.right))
    if left_blocker is not None and int(left_blocker.owner) != int(owner):
        blocker_event = scheduled_by_owner.get(int(left_blocker.owner))
        blocker_already_released = blocker_event is not None and int(blocker_event.plan_end) <= int(setup_start)
        if not blocker_already_released:
            blocker_max = max_usage_in_window(
                source,
                left_blocker,
                start=int(setup_start),
                end=int(setup_end),
            )
            if blocker_max > float(threshold):
                conflicts.append(f"left_blocker_working:{left_blocker.edge_key_text}:{blocker_max:g}")

    new_ports = set(link_ports(new_link))
    for event in scheduled:
        if not windows_overlap(setup_start, setup_end, event.plan_start, event.plan_end):
            continue
        if new_ports.intersection(link_ports(event.new)):
            conflicts.append(f"building_port_reserved:{event.owner}->{event.new.right}")
    return conflicts


def find_right_link_changes(source: one_way.TopologyData, target: one_way.TopologyData) -> list[one_way.RightLinkChange]:
    changes: list[one_way.RightLinkChange] = []
    for owner in sorted(set(source.right_by_owner) | set(target.right_by_owner)):
        old = source.right_by_owner.get(int(owner))
        new = target.right_by_owner.get(int(owner))
        if (None if old is None else old.edge_key) == (None if new is None else new.edge_key):
            continue
        changes.append(
            one_way.RightLinkChange(
                owner=int(owner),
                old=old,
                new=new,
                target_usage_at_switch=0.0,
                target_usage_peak_after_switch=0.0,
            )
        )
    return changes


def build_guarded_events(
    *,
    source: UsageStore,
    target: UsageStore,
    switch_step: int,
    segment_end: int,
    lst: int,
    threshold: float,
) -> tuple[list[GuardedEvent], list[dict[str, object]], list[dict[str, object]]]:
    changes = find_right_link_changes(source.topology, target.topology)
    skipped_rows: list[dict[str, object]] = []
    no_work_rows: list[dict[str, object]] = []

    events: list[GuardedEvent] = []
    for change in changes:
        owner = int(change.owner)
        if change.new is None:
            no_work_rows.append(
                {
                    "owner": owner,
                    "old_edge": "" if change.old is None else change.old.edge_key_text,
                    "new_edge": "",
                    "reason": "target_has_no_right_link",
                }
            )
            continue

        protected_old_links: list[tuple[str, one_way.RightLink]] = []
        if change.old is not None:
            protected_old_links.append(("owner_old", change.old))
        left_blocker = source.topology.left_by_right.get(int(change.new.right))
        if left_blocker is not None and all(left_blocker.edge_key != link.edge_key for _name, link in protected_old_links):
            protected_old_links.append(("left_blocker_old", left_blocker))

        prebuild_start = int(switch_step) - int(lst)
        candidate_starts: list[int] = []
        if prebuild_start >= int(source.steps[0]):
            candidate_starts.append(prebuild_start)
        candidate_starts.extend(
            int(step)
            for step in source.steps
            if int(switch_step) <= int(step) <= int(segment_end) - int(lst)
        )

        chosen: tuple[int, int] | None = None
        for candidate_start in candidate_starts:
            candidate_end = int(candidate_start) + int(lst)
            blockers: list[dict[str, object]] = []
            if int(candidate_start) < int(switch_step):
                source_occupants = source_port_occupants_for_new_link(source.topology, change.new)
                if source_occupants:
                    skipped_rows.append(
                        {
                            "owner": owner,
                            "old_edge": "" if change.old is None else change.old.edge_key_text,
                            "new_edge": change.new.edge_key_text,
                            "new_symbol": change.new.symbol,
                            "setup_start": int(candidate_start),
                            "setup_end": int(candidate_end),
                            "reason": "source_port_occupied_before_switch_keep_000056",
                            "conflicts": "; ".join(
                                f"{kind}:{link.edge_key_text}" for kind, link in source_occupants
                            ),
                        }
                    )
                    continue
            for blocker_type, link in protected_old_links:
                max_value = max_usage_in_window(
                    source,
                    link,
                    start=int(candidate_start),
                    end=int(candidate_end),
                )
                if max_value > float(threshold):
                    blockers.append(
                        {
                            "blocker_type": blocker_type,
                            "blocker_owner": int(link.owner),
                            "blocker_right": int(link.right),
                            "blocker_symbol": link.symbol,
                            "blocker_edge": link.edge_key_text,
                            "max_source_usage_inside_setup": float(max_value),
                        }
                    )
            if blockers:
                if int(candidate_start) in {int(prebuild_start), int(switch_step)}:
                    skipped_rows.append(
                        {
                            "owner": owner,
                            "old_edge": "" if change.old is None else change.old.edge_key_text,
                            "new_edge": change.new.edge_key_text,
                            "new_symbol": change.new.symbol,
                            "setup_start": int(candidate_start),
                            "setup_end": int(candidate_end),
                            "reason": "source_old_edge_still_working_keep_000056",
                            "conflicts": "; ".join(
                                f"{row['blocker_type']}:{row['blocker_edge']}:{row['max_source_usage_inside_setup']:g}"
                                for row in blockers
                            ),
                        }
                    )
                continue
            chosen = (int(candidate_start), int(candidate_end))
            break

        if chosen is None:
            no_work_rows.append(
                {
                    "owner": owner,
                    "old_edge": "" if change.old is None else change.old.edge_key_text,
                    "new_edge": change.new.edge_key_text,
                    "new_symbol": change.new.symbol,
                    "reason": "no_zero_usage_setup_window_before_segment_end",
                    "protected_old_edges": "; ".join(f"{name}:{link.edge_key_text}" for name, link in protected_old_links),
                }
            )
            continue

        setup_start, setup_end = chosen

        event = GuardedEvent(
            transition=f"motif{int(source.topology.spec.motif_id):06d}_to_motif{int(target.topology.spec.motif_id):06d}",
            owner=owner,
            old=change.old,
            new=change.new,
            plan_start=int(setup_start),
            plan_end=int(setup_end),
            target_interval_start=int(setup_end),
            target_interval_end=int(segment_end),
            skipped_intervals=0,
            blocker_summary="; ".join(f"{name}:{link.edge_key_text}" for name, link in protected_old_links),
        )
        events.append(event)

    return sorted(events, key=lambda item: (item.plan_start, item.owner)), skipped_rows, no_work_rows


def write_dict_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def event_rows(events: Sequence[GuardedEvent]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event_id, event in enumerate(events, 1):
        old = event.old
        new = event.new
        rows.append(
            {
                "event_id": int(event_id),
                "transition": event.transition,
                "owner": int(event.owner),
                "owner_p": int(event.owner // G60_CONFIG.N),
                "owner_y": int(event.owner % G60_CONFIG.N),
                "old_right": "" if old is None else int(old.right),
                "old_symbol": "" if old is None else old.symbol,
                "old_edge": "" if old is None else old.edge_key_text,
                "new_right": int(new.right),
                "new_symbol": new.symbol,
                "new_edge": new.edge_key_text,
                "plan_start": int(event.plan_start),
                "plan_end": int(event.plan_end),
                "duration": int(event.plan_end) - int(event.plan_start),
                "target_interval_start": int(event.target_interval_start),
                "target_interval_end": int(event.target_interval_end),
                "skipped_intervals_before_schedule": int(event.skipped_intervals),
                "reason": "scheduled_after_guarded_conflict_check",
            }
        )
    return rows


def write_concurrency(path: Path, *, steps: np.ndarray, events: Sequence[GuardedEvent]) -> np.ndarray:
    counts = np.zeros(int(steps.size), dtype=np.int32)
    for event in events:
        rows = interval_rows(steps, int(event.plan_start), int(event.plan_end))
        counts[rows] += 1
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "hour", "building_count"])
        for step, count in zip(steps, counts):
            writer.writerow([int(step), f"{int(step) / 3600.0:.6f}", int(count)])
    return counts


def plot_concurrency(path: Path, *, steps: np.ndarray, counts: np.ndarray, switch_step: int, lst: int) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 4.8), dpi=170)
    ax.plot(steps / 3600.0, counts, color="#b91c1c", linewidth=1.25)
    ax.axvline(float(switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9, label="T2")
    ax.axvspan(
        (float(switch_step) - float(lst)) / 3600.0,
        float(switch_step) / 3600.0,
        color="#2563eb",
        alpha=0.08,
        label="initial LST window",
    )
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("building links")
    ax.set_title(f"Guarded T2 switch setup load, LST={int(lst)}s")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def resolve_active_links(
    *,
    raw_active: dict[int, tuple[one_way.RightLink, int]],
    building: dict[int, one_way.RightLink],
) -> dict[int, one_way.RightLink]:
    occupied = set()
    for link in building.values():
        occupied.update(link_ports(link))

    active: dict[int, one_way.RightLink] = {}
    for owner, (link, priority) in sorted(raw_active.items(), key=lambda item: (-int(item[1][1]), int(item[0]))):
        ports = set(link_ports(link))
        if ports.intersection(occupied):
            continue
        active[int(owner)] = link
        occupied.update(ports)
    return active


def current_links_at_step(
    *,
    step: int,
    source: one_way.TopologyData,
    target: one_way.TopologyData,
    events_by_owner: dict[int, GuardedEvent],
) -> tuple[dict[int, one_way.RightLink], dict[int, one_way.RightLink]]:
    raw_active: dict[int, tuple[one_way.RightLink, int]] = {}
    building: dict[int, one_way.RightLink] = {}
    owners = sorted(set(source.right_by_owner) | set(target.right_by_owner) | set(events_by_owner))
    for owner in owners:
        event = events_by_owner.get(int(owner))
        if event is None:
            link = source.right_by_owner.get(int(owner))
            if link is not None:
                raw_active[int(owner)] = (link, 10)
            continue
        if int(step) < int(event.plan_start):
            if event.old is not None:
                raw_active[int(owner)] = (event.old, 10)
        elif int(event.plan_start) <= int(step) < int(event.plan_end):
            building[int(owner)] = event.new
        else:
            raw_active[int(owner)] = (event.new, 20)
    return resolve_active_links(raw_active=raw_active, building=building), building


def usage_value_for_key(
    *,
    row: int,
    key: tuple[int, int],
    source: UsageStore,
    target: UsageStore,
) -> float:
    col = target.key_to_col.get(key)
    if col is not None:
        return float(target.counts[int(row), int(col)])
    col = source.key_to_col.get(key)
    if col is not None:
        return float(source.counts[int(row), int(col)])
    return 0.0


def build_viewer_arrays(
    *,
    steps: np.ndarray,
    edge_table,
    source: UsageStore,
    target: UsageStore,
    events: Sequence[GuardedEvent],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_to_idx = edge_key_index(edge_table)
    intra_keys = all_intra_keys(edge_table)
    n_rows = int(steps.size)
    active_mask = np.zeros((n_rows, int(edge_table.num_edges)), dtype=bool)
    building_mask = np.zeros_like(active_mask, dtype=bool)
    values = np.zeros((n_rows, int(edge_table.num_edges)), dtype=np.float32)

    for key in intra_keys:
        idx = key_to_idx.get(key)
        if idx is not None:
            active_mask[:, int(idx)] = True

    for link in source.topology.right_by_owner.values():
        idx = key_to_idx.get(link.edge_key)
        if idx is None:
            continue
        active_mask[:, int(idx)] = True
        col = source.key_to_col.get(link.edge_key)
        if col is not None:
            values[:, int(idx)] = np.asarray(source.counts[:, int(col)], dtype=np.float32)

    def row_ge(step: int) -> int:
        return int(np.searchsorted(steps, int(step), side="left"))

    def rows_between_steps(start: int, end: int) -> slice:
        left = int(np.searchsorted(steps, int(start), side="left"))
        right = int(np.searchsorted(steps, int(end), side="left"))
        return slice(left, right)

    # Phase 1: every scheduled setup reserves its future ports, so old source
    # links on those ports are removed from the setup start onward.
    for event in events:
        start_row = row_ge(int(event.plan_start))
        if event.old is not None:
            old_idx = key_to_idx.get(event.old.edge_key)
            if old_idx is not None:
                active_mask[start_row:, int(old_idx)] = False
        blocker = source.topology.left_by_right.get(int(event.new.right))
        if blocker is not None and blocker.edge_key != event.new.edge_key:
            blocker_idx = key_to_idx.get(blocker.edge_key)
            if blocker_idx is not None:
                active_mask[start_row:, int(blocker_idx)] = False

    # Phase 2: add building and ready target edges. This is separated from
    # removals so a link removed as somebody's old edge can still reappear as
    # its own target edge after its setup finishes.
    for event in events:
        new_idx = key_to_idx.get(event.new.edge_key)
        if new_idx is None:
            continue
        building_rows = rows_between_steps(int(event.plan_start), int(event.plan_end))
        building_mask[building_rows, int(new_idx)] = True
        ready_row = row_ge(int(event.plan_end))
        active_mask[ready_row:, int(new_idx)] = True
        col = target.key_to_col.get(event.new.edge_key)
        if col is not None:
            values[ready_row:, int(new_idx)] = np.asarray(target.counts[ready_row:, int(col)], dtype=np.float32)
    return active_mask, building_mask, values


def write_edge_state_csv(
    path: Path,
    *,
    steps: np.ndarray,
    source: UsageStore,
    target: UsageStore,
    events: Sequence[GuardedEvent],
    threshold: float,
) -> None:
    events_by_owner = event_by_owner(events)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "hour",
                "status",
                "owner",
                "owner_p",
                "owner_y",
                "right",
                "right_p",
                "right_y",
                "symbol",
                "edge",
                "usage",
                "working",
            ]
        )
        for row, step in enumerate(steps):
            active, building = current_links_at_step(
                step=int(step),
                source=source.topology,
                target=target.topology,
                events_by_owner=events_by_owner,
            )
            for status, links in (("active", active), ("building", building)):
                for owner, link in sorted(links.items()):
                    usage = usage_value_for_key(row=int(row), key=link.edge_key, source=source, target=target)
                    writer.writerow(
                        [
                            int(step),
                            f"{int(step) / 3600.0:.6f}",
                            status,
                            int(owner),
                            int(owner // G60_CONFIG.N),
                            int(owner % G60_CONFIG.N),
                            int(link.right),
                            int(link.right_p),
                            int(link.right_y),
                            link.symbol,
                            link.edge_key_text,
                            f"{usage:.6g}",
                            int(usage > float(threshold)),
                        ]
                    )


def verify_result(
    *,
    steps: np.ndarray,
    switch_step: int,
    source: UsageStore,
    target: UsageStore,
    events: Sequence[GuardedEvent],
    active_mask: np.ndarray,
    building_mask: np.ndarray,
    edge_table,
) -> dict[str, object]:
    key_to_idx = edge_key_index(edge_table)
    new_keys = {event.new.edge_key for event in events}
    before_rows = np.flatnonzero(steps < int(switch_step))
    early_active_new = 0
    for key in new_keys:
        col = key_to_idx.get(key)
        if col is not None and before_rows.size:
            early_active_new += int(np.count_nonzero(active_mask[before_rows, int(col)]))

    first_building_step = None
    if np.any(building_mask):
        first_building_step = int(steps[int(np.flatnonzero(np.any(building_mask, axis=1))[0])])

    sample_steps = {int(steps[0]), int(steps[-1]), int(switch_step)}
    for event in events:
        for step in (event.plan_start, event.plan_end, event.target_interval_start, event.target_interval_end):
            if int(steps[0]) <= int(step) <= int(steps[-1]):
                sample_steps.add(int(step))

    max_left = 0
    max_right = 0
    events_by_owner = event_by_owner(events)
    for step in sorted(sample_steps):
        active, building = current_links_at_step(
            step=int(step),
            source=source.topology,
            target=target.topology,
            events_by_owner=events_by_owner,
        )
        counts_r: dict[int, int] = {}
        counts_l: dict[int, int] = {}
        for link in list(active.values()) + list(building.values()):
            counts_r[int(link.owner)] = counts_r.get(int(link.owner), 0) + 1
            counts_l[int(link.right)] = counts_l.get(int(link.right), 0) + 1
        max_right = max(max_right, max(counts_r.values(), default=0))
        max_left = max(max_left, max(counts_l.values(), default=0))

    return {
        "no_target_active_before_switch": int(early_active_new) == 0,
        "target_active_entries_before_switch": int(early_active_new),
        "first_building_step": first_building_step,
        "first_allowed_initial_building_step": int(switch_step) - int(
            max((event.plan_end - event.plan_start for event in events), default=0)
        ),
        "max_right_port_occupancy": int(max_right),
        "max_left_port_occupancy": int(max_left),
        "port_occupancy_ok": int(max_right) <= 1 and int(max_left) <= 1,
    }


def main() -> int:
    args = parse_args()
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    if int(args.lst) <= 0:
        raise ValueError("--lst must be positive")
    if int(args.switch_step) < int(args.start) or int(args.switch_step) > int(args.end):
        raise ValueError("--switch-step must be inside the requested time axis")
    if int(args.segment_end) <= int(args.switch_step):
        raise ValueError("--segment-end must be greater than --switch-step")

    source_topology = build_static_topology(int(args.source_motif_id))
    target_topology = build_static_topology(int(args.target_motif_id))
    source_usage = slice_usage_store(
        read_usage_store(Path(args.usage_root), source_topology),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    target_usage = slice_usage_store(
        read_usage_store(Path(args.usage_root), target_topology),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    if not np.array_equal(source_usage.steps, target_usage.steps):
        raise ValueError("source and target usage stores have different time axes")

    events, skipped_rows, no_work_rows = build_guarded_events(
        source=source_usage,
        target=target_usage,
        switch_step=int(args.switch_step),
        segment_end=int(args.segment_end),
        lst=int(args.lst),
        threshold=float(args.working_threshold),
    )

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT
        / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
        / f"switch{int(args.switch_step)}_segend{int(args.segment_end)}_lst{int(args.lst):03d}_oldzero_delay"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_table = union_edge_table(source_topology, target_topology)
    active_mask, building_mask, values = build_viewer_arrays(
        steps=source_usage.steps,
        edge_table=edge_table,
        source=source_usage,
        target=target_usage,
        events=events,
    )
    write_edges_csv(edge_table, out_dir / "union_edges.csv")
    np.save(out_dir / "steps.npy", source_usage.steps.astype(np.int64))
    np.save(out_dir / "edge_active_mask.npy", active_mask.astype(bool))
    np.save(out_dir / "edge_building_mask.npy", building_mask.astype(bool))
    np.save(out_dir / "edge_usage_values.npy", values.astype(np.float32))

    write_dict_rows(out_dir / "guarded_switch_events.csv", event_rows(events))
    write_dict_rows(out_dir / "guarded_switch_skipped_intervals.csv", skipped_rows)
    write_dict_rows(out_dir / "guarded_switch_unscheduled_or_not_needed.csv", no_work_rows)
    counts = write_concurrency(out_dir / "building_concurrency_by_step.csv", steps=source_usage.steps, events=events)
    np.save(out_dir / "building_concurrency.npy", counts)
    plot_concurrency(
        out_dir / "building_concurrency.png",
        steps=source_usage.steps,
        counts=counts,
        switch_step=int(args.switch_step),
        lst=int(args.lst),
    )
    if bool(args.write_edge_state_csv):
        write_edge_state_csv(
            out_dir / "edge_state_by_step.csv.gz",
            steps=source_usage.steps,
            source=source_usage,
            target=target_usage,
            events=events,
            threshold=float(args.working_threshold),
        )

    verification = verify_result(
        steps=source_usage.steps,
        switch_step=int(args.switch_step),
        source=source_usage,
        target=target_usage,
        events=events,
        active_mask=active_mask,
        building_mask=building_mask,
        edge_table=edge_table,
    )
    summary = {
        "source_motif_id": int(args.source_motif_id),
        "source_motif": source_topology.spec.motif,
        "target_motif_id": int(args.target_motif_id),
        "target_motif": target_topology.spec.motif,
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "switch_step": int(args.switch_step),
        "segment_end": int(args.segment_end),
        "lst": int(args.lst),
        "working_mode": "delay_tracked_usage_counts_oldzero_release",
        "working_threshold": float(args.working_threshold),
        "changed_right_ports": int(len(find_right_link_changes(source_topology, target_topology))),
        "scheduled_events": int(len(events)),
        "skipped_intervals": int(len(skipped_rows)),
        "unscheduled_or_not_needed": int(len(no_work_rows)),
        "max_building_concurrency": int(np.max(counts)) if counts.size else 0,
        "nonzero_building_steps": int(np.count_nonzero(counts)),
        "verification": verification,
        "outputs": {
            "events": "guarded_switch_events.csv",
            "skipped_intervals": "guarded_switch_skipped_intervals.csv",
            "unscheduled_or_not_needed": "guarded_switch_unscheduled_or_not_needed.csv",
            "union_edges": "union_edges.csv",
            "steps": "steps.npy",
            "edge_active_mask": "edge_active_mask.npy",
            "edge_building_mask": "edge_building_mask.npy",
            "edge_usage_values": "edge_usage_values.npy",
            "building_concurrency": "building_concurrency_by_step.csv",
            "building_plot": "building_concurrency.png",
        },
        "runner": str(THIS_FILE),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"out_dir={out_dir}")
    print(
        f"scheduled={len(events)} skipped_intervals={len(skipped_rows)} "
        f"unscheduled_or_not_needed={len(no_work_rows)}"
    )
    print(
        f"max_building_concurrency={int(np.max(counts)) if counts.size else 0} "
        f"first_building_step={verification['first_building_step']} "
        f"no_target_active_before_switch={verification['no_target_active_before_switch']} "
        f"port_occupancy_ok={verification['port_occupancy_ok']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
