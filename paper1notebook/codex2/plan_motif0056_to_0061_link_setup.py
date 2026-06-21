from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable  # noqa: E402
from src.topology_metrics.module.stores import MetricStoreLayout  # noqa: E402
from src.topology_workflow.module.batch_shortest_hops import TopologySpec  # noqa: E402
from src.topology_workflow.module.edge_tables import INTRA_OPTION, build_motif_text_edge_table  # noqa: E402


MOTIF_LIBRARY_CSV = Path(
    r"E:\paper11\data\satnet_experiments\libraries\motif\exact_box"
    r"\w_le_4_h_le_3\combined_w_le4_h_le3_808.csv"
)
EDGE_USAGE_CACHE_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3"
    r"\dual_edge_usage_056_061_china_europe\edge_betweenness_cache"
)
OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\switch_setup\m0056_to_m0061_china_europe"
)

PAIR_KEY = "china_europe"
SOURCE_MOTIF_ID = 56
TARGET_MOTIF_ID = 61
OPTION_SYMBOL = {
    -1: "I",
    0: "A",
    1: "B",
    2: "D",
    4: "C",
}


@dataclass(frozen=True)
class RightLink:
    owner: int
    right: int
    edge_key: tuple[int, int]
    edge_idx: int
    option: int
    symbol: str

    @property
    def owner_p(self) -> int:
        return int(self.owner // G60_CONFIG.N)

    @property
    def owner_y(self) -> int:
        return int(self.owner % G60_CONFIG.N)

    @property
    def right_p(self) -> int:
        return int(self.right // G60_CONFIG.N)

    @property
    def right_y(self) -> int:
        return int(self.right % G60_CONFIG.N)

    @property
    def edge_key_text(self) -> str:
        return f"{self.edge_key[0]}-{self.edge_key[1]}"


@dataclass(frozen=True)
class RightLinkChange:
    owner: int
    old: RightLink | None
    new: RightLink | None
    target_usage_at_switch: float
    target_usage_peak_after_switch: float


@dataclass(frozen=True)
class ScheduledEvent:
    event_id: int
    component_id: int
    change: RightLinkChange
    plan_start: int
    plan_end: int
    deadline_step: int
    old_usage_max_during_setup: float
    old_usage_mean_during_setup: float
    fallback_reason: str


@dataclass(frozen=True)
class TopologyData:
    spec: TopologySpec
    values: np.ndarray
    right_by_owner: dict[int, RightLink]
    left_by_right: dict[int, RightLink]
    edge_key_to_idx: dict[tuple[int, int], int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan gradual right-link setup when switching G60 motif000056 to motif000061. "
            "The work flag is read from precomputed China-Europe edge betweenness caches."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--source-motif-id", type=int, default=SOURCE_MOTIF_ID)
    parser.add_argument("--target-motif-id", type=int, default=TARGET_MOTIF_ID)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--setup-duration", type=int, default=600)
    parser.add_argument("--lookback-seconds", type=int, default=21600)
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--work-threshold", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-all-node-state", action="store_true")
    return parser.parse_args()


def read_motif_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["motif_id"])] = dict(row)
    return rows


def build_topology_spec(motif_id: int, row: dict[str, str]) -> TopologySpec:
    motif_text = str(row["motif"]).strip()
    edge_table = build_motif_text_edge_table(
        motif_text=motif_text,
        config=G60_CONFIG,
        allow_vertical_overlap=True,
        allow_clipped_right=True,
        wrap_planes=False,
        add_intra_ring=True,
    )
    return TopologySpec(
        name=f"combined_motif_{int(motif_id):06d}",
        edge_table=edge_table,
        library="combined_motif",
        motif_id=int(motif_id),
        motif=motif_text,
        source_w=int(row.get("source_w") or 0) or None,
        source_h=int(row.get("source_h") or 0) or None,
        edge_count_local=int(row.get("edge_count") or 0) or None,
        support=str(row.get("support", "")),
        baseline=False,
        meta={k: v for k, v in row.items() if k != "motif"},
    )


def edge_key(src: int, dst: int) -> tuple[int, int]:
    src = int(src)
    dst = int(dst)
    return (src, dst) if src < dst else (dst, src)


def build_edge_key_index(edge_table: EdgeTable) -> dict[tuple[int, int], int]:
    return {
        edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])): int(idx)
        for idx in range(edge_table.num_edges)
    }


def build_right_links(edge_table: EdgeTable) -> tuple[dict[int, RightLink], dict[int, RightLink]]:
    right_by_owner: dict[int, RightLink] = {}
    left_by_right: dict[int, RightLink] = {}
    for idx in range(edge_table.num_edges):
        option = int(edge_table.option[idx])
        if option == INTRA_OPTION:
            continue

        src = int(edge_table.src[idx])
        dst = int(edge_table.dst[idx])
        src_p = int(edge_table.src_plane[idx])
        dst_p = int(edge_table.dst_plane[idx])
        if src_p == dst_p:
            continue
        if src_p < dst_p:
            owner = src
            right = dst
        else:
            owner = dst
            right = src

        link = RightLink(
            owner=owner,
            right=right,
            edge_key=edge_key(src, dst),
            edge_idx=int(idx),
            option=option,
            symbol=OPTION_SYMBOL.get(option, str(option)),
        )
        if owner in right_by_owner:
            raise ValueError(f"right port has multiple inter links: owner={owner}")
        if right in left_by_right:
            raise ValueError(f"left port has multiple inter links: right={right}")
        right_by_owner[owner] = link
        left_by_right[right] = link
    return right_by_owner, left_by_right


def expanded_edge_usage_values(cache_dir: Path) -> np.ndarray:
    layout = MetricStoreLayout(cache_dir)
    if not layout.unique_state_values_npy.exists() or not layout.state_ids_npy.exists():
        raise FileNotFoundError(f"missing edge-usage cache files under {cache_dir}")
    unique_values = np.load(layout.unique_state_values_npy, mmap_mode="r")
    state_ids = np.load(layout.state_ids_npy, mmap_mode="r")
    return np.asarray(unique_values[np.asarray(state_ids, dtype=np.int64), :], dtype=np.float32)


def load_topology_data(spec: TopologySpec, *, start: int, end: int, stride: int) -> TopologyData:
    cache_dir = (
        EDGE_USAGE_CACHE_ROOT
        / f"t{int(start)}_{int(end)}_stride{int(stride)}"
        / PAIR_KEY
        / spec.name
    )
    values = expanded_edge_usage_values(cache_dir)
    if values.shape[1] != spec.edge_table.num_edges:
        raise ValueError(
            f"cache edge count mismatch for {spec.name}: values={values.shape[1]}, "
            f"edge_table={spec.edge_table.num_edges}"
        )
    right_by_owner, left_by_right = build_right_links(spec.edge_table)
    return TopologyData(
        spec=spec,
        values=values,
        right_by_owner=right_by_owner,
        left_by_right=left_by_right,
        edge_key_to_idx=build_edge_key_index(spec.edge_table),
    )


def find_right_link_changes(
    source: TopologyData,
    target: TopologyData,
    *,
    switch_idx: int,
    switch_idx_end: int,
) -> list[RightLinkChange]:
    changes: list[RightLinkChange] = []
    owners = sorted(set(source.right_by_owner) | set(target.right_by_owner))
    for owner in owners:
        old = source.right_by_owner.get(owner)
        new = target.right_by_owner.get(owner)
        old_key = old.edge_key if old is not None else None
        new_key = new.edge_key if new is not None else None
        if old_key == new_key:
            continue
        target_usage_at_switch = 0.0
        target_usage_peak_after_switch = 0.0
        if new is not None:
            target_usage_at_switch = float(target.values[switch_idx, new.edge_idx])
            target_usage_peak_after_switch = float(np.nanmax(target.values[switch_idx:switch_idx_end, new.edge_idx]))
        changes.append(
            RightLinkChange(
                owner=int(owner),
                old=old,
                new=new,
                target_usage_at_switch=target_usage_at_switch,
                target_usage_peak_after_switch=target_usage_peak_after_switch,
            )
        )
    return changes


def setup_ports(link: RightLink | None) -> tuple[str, ...]:
    if link is None:
        return tuple()
    return (f"R:{link.owner}", f"L:{link.right}")


def interval_indices(steps_np: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.flatnonzero((steps_np >= int(start)) & (steps_np < int(end)))


def old_usage_stats(
    *,
    source: TopologyData,
    change: RightLinkChange,
    rows: np.ndarray,
) -> tuple[float, float]:
    if change.old is None or rows.size == 0:
        return 0.0, 0.0
    values = source.values[rows, change.old.edge_idx]
    return float(np.nanmax(values)), float(np.nanmean(values))


def transition_dependency_groups(
    changes: list[RightLinkChange],
    source: TopologyData,
) -> tuple[list[list[RightLinkChange]], dict[int, set[int]]]:
    """Return SCC groups and precedence edges between them.

    An edge ``A -> B`` means A must release its old link no later than B starts
    building its new link, otherwise B would occupy a left port that is still
    held by A's old link.
    """

    change_by_owner = {change.owner: change for change in changes}
    owners = sorted(change_by_owner)
    adjacency: dict[int, list[int]] = {owner: [] for owner in owners}
    for change in changes:
        if change.new is None:
            continue
        blocker = source.left_by_right.get(change.new.right)
        if blocker is not None and blocker.owner != change.owner and blocker.owner in change_by_owner:
            adjacency[blocker.owner].append(change.owner)

    index = 0
    stack: list[int] = []
    on_stack: set[int] = set()
    indices: dict[int, int] = {}
    lowlinks: dict[int, int] = {}
    sccs: list[list[int]] = []

    def strongconnect(owner: int) -> None:
        nonlocal index
        indices[owner] = index
        lowlinks[owner] = index
        index += 1
        stack.append(owner)
        on_stack.add(owner)

        for nxt in adjacency[owner]:
            if nxt not in indices:
                strongconnect(nxt)
                lowlinks[owner] = min(lowlinks[owner], lowlinks[nxt])
            elif nxt in on_stack:
                lowlinks[owner] = min(lowlinks[owner], indices[nxt])

        if lowlinks[owner] == indices[owner]:
            component: list[int] = []
            while True:
                nxt = stack.pop()
                on_stack.remove(nxt)
                component.append(nxt)
                if nxt == owner:
                    break
            sccs.append(sorted(component))

    for owner in owners:
        if owner not in indices:
            strongconnect(owner)

    owner_to_group: dict[int, int] = {}
    groups: list[list[RightLinkChange]] = []
    for group_idx, owners_in_group in enumerate(sccs):
        groups.append([change_by_owner[owner] for owner in sorted(owners_in_group)])
        for owner in owners_in_group:
            owner_to_group[owner] = group_idx

    successors: dict[int, set[int]] = {group_idx: set() for group_idx in range(len(groups))}
    for owner, targets in adjacency.items():
        src_group = owner_to_group[owner]
        for target_owner in targets:
            dst_group = owner_to_group[target_owner]
            if src_group != dst_group:
                successors[src_group].add(dst_group)

    return groups, successors


def component_usage_stats(
    *,
    source: TopologyData,
    component: list[RightLinkChange],
    rows: np.ndarray,
) -> tuple[float, float]:
    if rows.size == 0:
        return 0.0, 0.0
    values: list[np.ndarray] = []
    for change in component:
        if change.old is None:
            continue
        values.append(source.values[rows, change.old.edge_idx])
    if not values:
        return 0.0, 0.0
    merged = np.concatenate(values)
    return float(np.nanmax(merged)), float(np.nanmean(merged))


def schedule_changes(
    changes: list[RightLinkChange],
    *,
    source: TopologyData,
    steps: list[int],
    switch_step: int,
    setup_duration: int,
    lookback_seconds: int,
    max_concurrent: int,
    work_threshold: float,
    earliest_setup_step: int | None = None,
    schedule_preference: str = "latest",
    usage_hold_until_deadline: bool = False,
) -> tuple[list[ScheduledEvent], np.ndarray, list[dict[str, object]]]:
    steps_np = np.asarray(steps, dtype=np.int64)
    lookback_start = max(int(steps[0]), int(switch_step) - int(lookback_seconds))
    if earliest_setup_step is not None:
        lookback_start = max(int(lookback_start), int(earliest_setup_step))
    latest_start = int(switch_step) - int(setup_duration)
    candidate_starts = [
        int(step)
        for step in steps
        if lookback_start <= int(step) <= latest_start
    ]
    schedule_preference = str(schedule_preference).lower()
    if schedule_preference not in {"latest", "earliest"}:
        raise ValueError("schedule_preference must be 'latest' or 'earliest'")
    if schedule_preference == "latest":
        candidate_starts.reverse()
    if not candidate_starts:
        raise ValueError("no candidate setup start times; enlarge lookback or shorten setup duration")

    components, successors = transition_dependency_groups(changes, source)
    component_building_counts = {
        component_id: sum(1 for change in component if change.new is not None)
        for component_id, component in enumerate(components, 1)
    }
    largest_component = max(component_building_counts.values(), default=0)
    if largest_component > int(max_concurrent):
        raise ValueError(
            f"max_concurrent={max_concurrent} is smaller than the largest conflict-safe transition "
            f"batch ({largest_component}); use a larger threshold or implement ordered sub-batching."
        )

    # Topological order over component ids. Components are SCCs, so inter-component edges form a DAG.
    indegree = {idx: 0 for idx in range(len(components))}
    for src_idx, dst_indices in successors.items():
        for dst_idx in dst_indices:
            indegree[dst_idx] += 1
    ready = sorted([idx for idx, degree in indegree.items() if degree == 0])
    topo: list[int] = []
    while ready:
        idx = ready.pop(0)
        topo.append(idx)
        for dst_idx in sorted(successors[idx]):
            indegree[dst_idx] -= 1
            if indegree[dst_idx] == 0:
                ready.append(dst_idx)
                ready.sort()
    if len(topo) != len(components):
        raise ValueError("transition component graph still has a cycle after SCC compression")

    concurrency = np.zeros(len(steps), dtype=np.int32)
    events: list[ScheduledEvent] = []
    unscheduled: list[dict[str, object]] = []
    event_id = 0
    scheduled_start_by_component: dict[int, int] = {}

    for component_idx in reversed(topo):
        component_id = int(component_idx) + 1
        component = components[component_idx]
        building_count = sum(1 for change in component if change.new is not None)
        successor_starts = [
            scheduled_start_by_component[int(successor_idx)]
            for successor_idx in successors[component_idx]
            if int(successor_idx) in scheduled_start_by_component
        ]
        dependency_deadline = min(successor_starts) if successor_starts else int(switch_step)

        if building_count <= 0:
            candidate_deadline = min(int(dependency_deadline), int(switch_step))
            point_candidates = [int(step) for step in steps if lookback_start <= int(step) <= candidate_deadline]
            if schedule_preference == "latest":
                point_candidates.reverse()
            chosen_drop: tuple[int, np.ndarray, float, float, str] | None = None
            fallback_drop: tuple[int, np.ndarray, float, float, str] | None = None
            for candidate_start in point_candidates:
                rows = np.flatnonzero(steps_np == int(candidate_start))
                usage_rows = (
                    interval_indices(steps_np, candidate_start, switch_step)
                    if bool(usage_hold_until_deadline)
                    else rows
                )
                usage_max, usage_mean = component_usage_stats(source=source, component=component, rows=usage_rows)
                if usage_max <= float(work_threshold):
                    chosen_drop = (candidate_start, rows, usage_max, usage_mean, "drop_only_no_setup")
                    break
                if fallback_drop is None or (usage_max, usage_mean, -candidate_start) < (
                    fallback_drop[2],
                    fallback_drop[3],
                    -fallback_drop[0],
                ):
                    fallback_drop = (candidate_start, rows, usage_max, usage_mean, "old_link_still_working")
            chosen = chosen_drop or fallback_drop
            if chosen is None:
                for change in component:
                    unscheduled.append(
                        {
                            "component_id": int(component_id),
                            "owner": int(change.owner),
                            "old_edge": change.old.edge_key_text if change.old else "",
                            "new_edge": "",
                            "reason": "no_drop_slot_before_dependency_deadline",
                        }
                    )
                continue
            component_start, rows, usage_max, usage_mean, fallback_reason = chosen
        else:
            chosen: tuple[int, np.ndarray, float, float, str] | None = None
            fallback: tuple[int, np.ndarray, float, float, str] | None = None

            for candidate_start in candidate_starts:
                if int(candidate_start) > int(dependency_deadline):
                    continue
                candidate_end = int(candidate_start) + int(setup_duration)
                rows = interval_indices(steps_np, candidate_start, candidate_end)
                if rows.size == 0:
                    continue
                if np.any(concurrency[rows] + int(building_count) > int(max_concurrent)):
                    continue
                usage_rows = (
                    interval_indices(steps_np, candidate_start, switch_step)
                    if bool(usage_hold_until_deadline)
                    else rows
                )
                usage_max, usage_mean = component_usage_stats(source=source, component=component, rows=usage_rows)
                if usage_max <= float(work_threshold):
                    chosen = (candidate_start, rows, usage_max, usage_mean, "")
                    break
                if fallback is None or (usage_max, usage_mean, -candidate_start) < (
                    fallback[2],
                    fallback[3],
                    -fallback[0],
                ):
                    fallback = (candidate_start, rows, usage_max, usage_mean, "old_link_still_working")

            if chosen is None:
                chosen = fallback
            if chosen is None:
                for change in component:
                    unscheduled.append(
                        {
                            "component_id": int(component_id),
                            "owner": int(change.owner),
                            "old_edge": change.old.edge_key_text if change.old else "",
                            "new_edge": change.new.edge_key_text if change.new else "",
                            "reason": "no_batch_slot_under_concurrency",
                        }
                    )
                continue

            component_start, rows, usage_max, usage_mean, fallback_reason = chosen
            concurrency[rows] += int(building_count)

        scheduled_start_by_component[component_idx] = int(component_start)

        for change in component:
            event_id += 1
            if change.new is None:
                event_end = int(component_start)
                event_reason = "drop_only_no_setup" if fallback_reason == "" else fallback_reason
            else:
                event_end = int(component_start) + int(setup_duration)
                event_reason = str(fallback_reason)
            events.append(
                ScheduledEvent(
                    event_id=int(event_id),
                    component_id=int(component_id),
                    change=change,
                    plan_start=int(component_start),
                    plan_end=int(event_end),
                    deadline_step=int(switch_step),
                    old_usage_max_during_setup=float(usage_max),
                    old_usage_mean_during_setup=float(usage_mean),
                    fallback_reason=event_reason,
                )
            )

    return sorted(events, key=lambda event: event.event_id), concurrency, unscheduled


def write_change_candidates(path: Path, changes: list[RightLinkChange]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "owner",
                "owner_p",
                "owner_y",
                "old_right",
                "old_right_p",
                "old_right_y",
                "old_option",
                "old_symbol",
                "old_edge",
                "new_right",
                "new_right_p",
                "new_right_y",
                "new_option",
                "new_symbol",
                "new_edge",
                "target_usage_at_switch",
                "target_usage_peak_after_switch",
            ]
        )
        for change in changes:
            old = change.old
            new = change.new
            owner_p = int(change.owner // G60_CONFIG.N)
            owner_y = int(change.owner % G60_CONFIG.N)
            writer.writerow(
                [
                    int(change.owner),
                    owner_p,
                    owner_y,
                    "" if old is None else old.right,
                    "" if old is None else old.right_p,
                    "" if old is None else old.right_y,
                    "" if old is None else old.option,
                    "" if old is None else old.symbol,
                    "" if old is None else old.edge_key_text,
                    "" if new is None else new.right,
                    "" if new is None else new.right_p,
                    "" if new is None else new.right_y,
                    "" if new is None else new.option,
                    "" if new is None else new.symbol,
                    "" if new is None else new.edge_key_text,
                    f"{change.target_usage_at_switch:.6g}",
                    f"{change.target_usage_peak_after_switch:.6g}",
                ]
            )


def write_events(path: Path, events: list[ScheduledEvent]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_id",
                "component_id",
                "owner",
                "owner_p",
                "owner_y",
                "old_right",
                "old_right_p",
                "old_right_y",
                "old_option",
                "old_symbol",
                "old_edge",
                "new_right",
                "new_right_p",
                "new_right_y",
                "new_option",
                "new_symbol",
                "new_edge",
                "plan_start",
                "plan_end",
                "deadline_step",
                "duration",
                "target_usage_at_switch",
                "target_usage_peak_after_switch",
                "old_usage_max_during_setup",
                "old_usage_mean_during_setup",
                "fallback_reason",
            ]
        )
        for event in events:
            change = event.change
            old = change.old
            new = change.new
            owner_p = int(change.owner // G60_CONFIG.N)
            owner_y = int(change.owner % G60_CONFIG.N)
            writer.writerow(
                [
                    int(event.event_id),
                    int(event.component_id),
                    int(change.owner),
                    owner_p,
                    owner_y,
                    "" if old is None else old.right,
                    "" if old is None else old.right_p,
                    "" if old is None else old.right_y,
                    "" if old is None else old.option,
                    "" if old is None else old.symbol,
                    "" if old is None else old.edge_key_text,
                    "" if new is None else new.right,
                    "" if new is None else new.right_p,
                    "" if new is None else new.right_y,
                    "" if new is None else new.option,
                    "" if new is None else new.symbol,
                    "" if new is None else new.edge_key_text,
                    int(event.plan_start),
                    int(event.plan_end),
                    int(event.deadline_step),
                    int(event.plan_end) - int(event.plan_start),
                    f"{change.target_usage_at_switch:.6g}",
                    f"{change.target_usage_peak_after_switch:.6g}",
                    f"{event.old_usage_max_during_setup:.6g}",
                    f"{event.old_usage_mean_during_setup:.6g}",
                    event.fallback_reason,
                ]
            )


def event_by_owner(events: list[ScheduledEvent]) -> dict[int, ScheduledEvent]:
    return {event.change.owner: event for event in events}


def dynamic_right_links_at_step(
    *,
    step: int,
    source: TopologyData,
    target: TopologyData,
    events_by_owner: dict[int, ScheduledEvent],
) -> tuple[dict[int, RightLink], dict[int, RightLink]]:
    active_right_by_owner: dict[int, RightLink] = {}
    building_by_owner: dict[int, RightLink] = {}
    for owner in range(int(G60_CONFIG.total_sats)):
        status, active_link, building_link = active_right_link_for_owner(
            owner=owner,
            step=int(step),
            source_link=source.right_by_owner.get(owner),
            target_link=target.right_by_owner.get(owner),
            event_by_owner=events_by_owner,
        )
        if active_link is not None:
            active_right_by_owner[owner] = active_link
        if status == "building" and building_link is not None:
            building_by_owner[owner] = building_link
    return active_right_by_owner, building_by_owner


def count_port_conflicts(
    active_right_by_owner: dict[int, RightLink],
    building_by_owner: dict[int, RightLink],
) -> dict[str, int]:
    right_ports: dict[int, int] = {}
    left_ports: dict[int, int] = {}
    for links in (active_right_by_owner, building_by_owner):
        for link in links.values():
            right_ports[link.owner] = right_ports.get(link.owner, 0) + 1
            left_ports[link.right] = left_ports.get(link.right, 0) + 1
    return {
        "right_conflict_nodes": sum(1 for count in right_ports.values() if count > 1),
        "left_conflict_nodes": sum(1 for count in left_ports.values() if count > 1),
    }


def write_edge_state_by_step(
    path: Path,
    *,
    steps: list[int],
    switch_step: int,
    source: TopologyData,
    target: TopologyData,
    events: list[ScheduledEvent],
    work_threshold: float,
) -> None:
    events_by_owner = event_by_owner(events)
    event_id_by_owner = {event.change.owner: event.event_id for event in events}
    component_id_by_owner = {event.change.owner: event.component_id for event in events}
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "hour",
                "status",
                "event_id",
                "component_id",
                "owner",
                "owner_p",
                "owner_y",
                "right",
                "right_p",
                "right_y",
                "symbol",
                "option",
                "edge",
                "usage",
                "work",
                "metric_topology",
            ]
        )
        for step_idx, step in enumerate(steps):
            active_right_by_owner, building_by_owner = dynamic_right_links_at_step(
                step=int(step),
                source=source,
                target=target,
                events_by_owner=events_by_owner,
            )
            for status, links in (("active", active_right_by_owner), ("building", building_by_owner)):
                for owner, link in sorted(links.items()):
                    usage, metric_topology = link_usage_at_step(
                        step_idx=step_idx,
                        step=int(step),
                        switch_step=int(switch_step),
                        link=link,
                        source=source,
                        target=target,
                    )
                    writer.writerow(
                        [
                            int(step),
                            f"{int(step) / 3600:.6f}",
                            status,
                            event_id_by_owner.get(int(owner), ""),
                            component_id_by_owner.get(int(owner), ""),
                            int(owner),
                            link.owner_p,
                            link.owner_y,
                            int(link.right),
                            link.right_p,
                            link.right_y,
                            link.symbol,
                            int(link.option),
                            link.edge_key_text,
                            f"{usage:.6g}",
                            int(usage > float(work_threshold)),
                            metric_topology,
                        ]
                    )


def write_concurrency(path: Path, steps: list[int], concurrency: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "hour", "building_count"])
        for step, count in zip(steps, concurrency):
            writer.writerow([int(step), f"{int(step) / 3600:.6f}", int(count)])


def active_right_link_for_owner(
    *,
    owner: int,
    step: int,
    source_link: RightLink | None,
    target_link: RightLink | None,
    event_by_owner: dict[int, ScheduledEvent],
) -> tuple[str, RightLink | None, RightLink | None]:
    event = event_by_owner.get(int(owner))
    if event is None:
        return "active", source_link or target_link, None
    if int(step) < int(event.plan_start):
        return "active", source_link, None
    if int(event.plan_start) <= int(step) < int(event.plan_end):
        return "building", None, target_link
    return "active", target_link, None


def link_usage_at_step(
    *,
    step_idx: int,
    step: int,
    switch_step: int,
    link: RightLink | None,
    source: TopologyData,
    target: TopologyData,
) -> tuple[float, str]:
    if link is None:
        return 0.0, ""
    if int(step) < int(switch_step) and link.edge_key in source.edge_key_to_idx:
        return float(source.values[step_idx, source.edge_key_to_idx[link.edge_key]]), source.spec.name
    if link.edge_key in target.edge_key_to_idx:
        return float(target.values[step_idx, target.edge_key_to_idx[link.edge_key]]), target.spec.name
    if link.edge_key in source.edge_key_to_idx:
        return float(source.values[step_idx, source.edge_key_to_idx[link.edge_key]]), source.spec.name
    return 0.0, ""


def write_all_node_state(
    path: Path,
    *,
    steps: list[int],
    switch_step: int,
    source: TopologyData,
    target: TopologyData,
    events: list[ScheduledEvent],
    work_threshold: float,
) -> None:
    events_by_owner = event_by_owner(events)
    total_nodes = int(G60_CONFIG.total_sats)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "hour",
                "node",
                "node_p",
                "node_y",
                "status",
                "left_neighbor",
                "left_neighbor_p",
                "left_neighbor_y",
                "left_symbol",
                "left_edge",
                "right_neighbor",
                "right_neighbor_p",
                "right_neighbor_y",
                "right_symbol",
                "right_edge",
                "right_edge_usage",
                "right_edge_work",
                "building_target_neighbor",
                "building_target_p",
                "building_target_y",
                "building_symbol",
                "building_edge",
                "building_target_usage_at_deadline",
                "metric_topology",
                "is_changed_owner",
            ]
        )
        for step_idx, step in enumerate(steps):
            active_right_by_owner, building_by_owner = dynamic_right_links_at_step(
                step=int(step),
                source=source,
                target=target,
                events_by_owner=events_by_owner,
            )
            active_left_by_right = {link.right: link for link in active_right_by_owner.values()}
            for node in range(total_nodes):
                node_p = int(node // G60_CONFIG.N)
                node_y = int(node % G60_CONFIG.N)
                right_link = active_right_by_owner.get(node)
                left_link = active_left_by_right.get(node)
                building_link = building_by_owner.get(node)
                status = "building" if building_link is not None else ("active" if right_link is not None else "empty")
                usage, metric_topology = link_usage_at_step(
                    step_idx=step_idx,
                    step=int(step),
                    switch_step=int(switch_step),
                    link=right_link,
                    source=source,
                    target=target,
                )
                building_usage = 0.0
                if building_link is not None and building_link.edge_key in target.edge_key_to_idx:
                    target_idx = target.edge_key_to_idx[building_link.edge_key]
                    deadline_idx = min(step_idx + 1, target.values.shape[0] - 1)
                    building_usage = float(target.values[deadline_idx, target_idx])

                writer.writerow(
                    [
                        int(step),
                        f"{int(step) / 3600:.6f}",
                        int(node),
                        node_p,
                        node_y,
                        status,
                        "" if left_link is None else left_link.owner,
                        "" if left_link is None else left_link.owner_p,
                        "" if left_link is None else left_link.owner_y,
                        "" if left_link is None else left_link.symbol,
                        "" if left_link is None else left_link.edge_key_text,
                        "" if right_link is None else right_link.right,
                        "" if right_link is None else right_link.right_p,
                        "" if right_link is None else right_link.right_y,
                        "" if right_link is None else right_link.symbol,
                        "" if right_link is None else right_link.edge_key_text,
                        f"{usage:.6g}",
                        int(usage > float(work_threshold)),
                        "" if building_link is None else building_link.right,
                        "" if building_link is None else building_link.right_p,
                        "" if building_link is None else building_link.right_y,
                        "" if building_link is None else building_link.symbol,
                        "" if building_link is None else building_link.edge_key_text,
                        f"{building_usage:.6g}",
                        metric_topology,
                        int(node in events_by_owner),
                    ]
                )


def plot_concurrency(path: Path, steps: list[int], concurrency: np.ndarray, *, switch_step: int, max_concurrent: int) -> None:
    hours = np.asarray(steps, dtype=np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(hours, concurrency, color="#b32134", linewidth=1.8)
    ax.axvline(float(switch_step) / 3600.0, color="#222222", linestyle="--", linewidth=1.0, label="switch deadline")
    ax.axhline(int(max_concurrent), color="#555555", linestyle=":", linewidth=1.0, label="max concurrent")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("building links")
    ax.set_title("Motif000056 -> motif000061 link setup concurrency")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_event_timeline(path: Path, events: list[ScheduledEvent], *, switch_step: int) -> None:
    non_drop = [event for event in events if event.plan_end > event.plan_start]
    if not non_drop:
        return
    y = np.arange(len(non_drop))
    starts = np.asarray([event.plan_start for event in non_drop], dtype=np.float64) / 3600.0
    widths = np.asarray([event.plan_end - event.plan_start for event in non_drop], dtype=np.float64) / 3600.0
    old_usage = np.asarray([event.old_usage_max_during_setup for event in non_drop], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, max(5.0, min(13.0, len(non_drop) * 0.035))))
    colors = plt.cm.Reds(np.clip(old_usage / max(float(np.nanmax(old_usage)), 1.0), 0.15, 1.0))
    ax.barh(y, widths, left=starts, height=0.8, color=colors, edgecolor="none")
    ax.axvline(float(switch_step) / 3600.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("scheduled link setup event")
    ax.set_title("Setup windows before the 10h topology switch")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def edge_key_set(links: dict[int, RightLink]) -> set[tuple[int, int]]:
    return {link.edge_key for link in links.values()}


def verify_plan(
    *,
    steps: list[int],
    switch_step: int,
    max_concurrent: int,
    work_threshold: float,
    source: TopologyData,
    target: TopologyData,
    events: list[ScheduledEvent],
    concurrency: np.ndarray,
    unscheduled: list[dict[str, object]],
) -> dict[str, object]:
    events_by_owner = event_by_owner(events)
    max_left_conflicts = 0
    max_right_conflicts = 0
    first_conflict_step: int | None = None
    building_at_or_after_switch = 0
    for step in steps:
        active_right_by_owner, building_by_owner = dynamic_right_links_at_step(
            step=int(step),
            source=source,
            target=target,
            events_by_owner=events_by_owner,
        )
        conflicts = count_port_conflicts(active_right_by_owner, building_by_owner)
        if conflicts["left_conflict_nodes"] or conflicts["right_conflict_nodes"]:
            if first_conflict_step is None:
                first_conflict_step = int(step)
        max_left_conflicts = max(max_left_conflicts, int(conflicts["left_conflict_nodes"]))
        max_right_conflicts = max(max_right_conflicts, int(conflicts["right_conflict_nodes"]))
        if int(step) >= int(switch_step):
            building_at_or_after_switch += len(building_by_owner)

    start_active, start_building = dynamic_right_links_at_step(
        step=int(steps[0]),
        source=source,
        target=target,
        events_by_owner=events_by_owner,
    )
    switch_active, switch_building = dynamic_right_links_at_step(
        step=int(switch_step),
        source=source,
        target=target,
        events_by_owner=events_by_owner,
    )
    source_edges = edge_key_set(source.right_by_owner)
    target_edges = edge_key_set(target.right_by_owner)
    start_edges = edge_key_set(start_active)
    switch_edges = edge_key_set(switch_active)
    events_finish_by_switch = all(int(event.plan_end) <= int(switch_step) for event in events)
    no_old_work_fallback = all(
        event.fallback_reason not in {"old_link_still_working"}
        and float(event.old_usage_max_during_setup) <= float(work_threshold)
        for event in events
    )
    max_concurrent_observed = int(np.max(concurrency)) if concurrency.size else 0

    checks = {
        "no_unscheduled_events": len(unscheduled) == 0,
        "max_concurrent_ok": max_concurrent_observed <= int(max_concurrent),
        "events_finish_by_switch": events_finish_by_switch,
        "no_building_at_or_after_switch": building_at_or_after_switch == 0,
        "no_left_port_conflicts": max_left_conflicts == 0,
        "no_right_port_conflicts": max_right_conflicts == 0,
        "start_matches_source_topology": start_edges == source_edges and len(start_building) == 0,
        "switch_matches_target_topology": switch_edges == target_edges and len(switch_building) == 0,
        "old_links_changed_only_when_no_work": no_old_work_fallback,
    }
    return {
        "passed": all(bool(value) for value in checks.values()),
        "checks": checks,
        "max_concurrent_observed": max_concurrent_observed,
        "max_left_conflict_nodes": int(max_left_conflicts),
        "max_right_conflict_nodes": int(max_right_conflicts),
        "first_conflict_step": first_conflict_step,
        "building_at_or_after_switch": int(building_at_or_after_switch),
        "source_right_edges": int(len(source_edges)),
        "target_right_edges": int(len(target_edges)),
        "start_active_right_edges": int(len(start_edges)),
        "switch_active_right_edges": int(len(switch_edges)),
        "source_missing_at_start": [f"{a}-{b}" for a, b in sorted(source_edges - start_edges)[:20]],
        "source_extra_at_start": [f"{a}-{b}" for a, b in sorted(start_edges - source_edges)[:20]],
        "target_missing_at_switch": [f"{a}-{b}" for a, b in sorted(target_edges - switch_edges)[:20]],
        "target_extra_at_switch": [f"{a}-{b}" for a, b in sorted(switch_edges - target_edges)[:20]],
    }


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    source: TopologyData,
    target: TopologyData,
    changes: list[RightLinkChange],
    events: list[ScheduledEvent],
    concurrency: np.ndarray,
    unscheduled: list[dict[str, object]],
    verification: dict[str, object],
) -> None:
    payload = {
        "source_motif_id": int(args.source_motif_id),
        "source_motif": source.spec.motif,
        "target_motif_id": int(args.target_motif_id),
        "target_motif": target.spec.motif,
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "switch_step": int(args.switch_step),
        "switch_hour": int(args.switch_step) / 3600,
        "setup_duration": int(args.setup_duration),
        "lookback_seconds": int(args.lookback_seconds),
        "max_concurrent_limit": int(args.max_concurrent),
        "max_concurrent_observed": int(np.max(concurrency)) if concurrency.size else 0,
        "source_inter_right_links": int(len(source.right_by_owner)),
        "target_inter_right_links": int(len(target.right_by_owner)),
        "changed_right_ports": int(len(changes)),
        "scheduled_events": int(len(events)),
        "unscheduled_events": int(len(unscheduled)),
        "fallback_events_old_link_still_working": int(
            sum(1 for event in events if event.fallback_reason == "old_link_still_working")
        ),
        "drop_only_events": int(sum(1 for event in events if event.fallback_reason == "drop_only_no_setup")),
        "transition_components": int(len({event.component_id for event in events})),
        "largest_component_events": int(
            max(
                (
                    sum(1 for event in events if event.component_id == component_id)
                    for component_id in {event.component_id for event in events}
                ),
                default=0,
            )
        ),
        "largest_component_building_links": int(
            max(
                (
                    sum(1 for event in events if event.component_id == component_id and event.change.new is not None)
                    for component_id in {event.component_id for event in events}
                ),
                default=0,
            )
        ),
        "work_threshold": float(args.work_threshold),
        "verification_passed": bool(verification.get("passed", False)),
        "runner": str(THIS_FILE),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if int(args.switch_step) not in set(steps):
        raise ValueError("--switch-step must be on the requested time axis")
    switch_idx = steps.index(int(args.switch_step))
    switch_idx_end = min(len(steps), switch_idx + max(2, int(3600 / int(args.stride))))

    motif_rows = read_motif_rows(MOTIF_LIBRARY_CSV)
    source_spec = build_topology_spec(int(args.source_motif_id), motif_rows[int(args.source_motif_id)])
    target_spec = build_topology_spec(int(args.target_motif_id), motif_rows[int(args.target_motif_id)])
    source = load_topology_data(source_spec, start=int(args.start), end=int(args.end), stride=int(args.stride))
    target = load_topology_data(target_spec, start=int(args.start), end=int(args.end), stride=int(args.stride))

    if source.values.shape[0] != len(steps) or target.values.shape[0] != len(steps):
        raise ValueError(
            f"cache time length mismatch: steps={len(steps)}, "
            f"source={source.values.shape[0]}, target={target.values.shape[0]}"
        )

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            OUT_ROOT
            / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
            / f"switch{int(args.switch_step)}_dur{int(args.setup_duration)}_c{int(args.max_concurrent)}"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    changes = find_right_link_changes(
        source,
        target,
        switch_idx=switch_idx,
        switch_idx_end=switch_idx_end,
    )
    events, concurrency, unscheduled = schedule_changes(
        changes,
        source=source,
        steps=steps,
        switch_step=int(args.switch_step),
        setup_duration=int(args.setup_duration),
        lookback_seconds=int(args.lookback_seconds),
        max_concurrent=int(args.max_concurrent),
        work_threshold=float(args.work_threshold),
    )

    write_change_candidates(out_dir / "right_link_change_candidates.csv", changes)
    write_events(out_dir / "link_setup_events.csv", events)
    write_concurrency(out_dir / "link_setup_concurrency_by_step.csv", steps, concurrency)
    if unscheduled:
        with (out_dir / "unscheduled_events.json").open("w", encoding="utf-8") as f:
            json.dump(unscheduled, f, ensure_ascii=False, indent=2)
    write_edge_state_by_step(
        out_dir / "edge_state_by_step.csv.gz",
        steps=steps,
        switch_step=int(args.switch_step),
        source=source,
        target=target,
        events=events,
        work_threshold=float(args.work_threshold),
    )
    if not bool(args.skip_all_node_state):
        write_all_node_state(
            out_dir / "right_link_state_all_nodes.csv.gz",
            steps=steps,
            switch_step=int(args.switch_step),
            source=source,
            target=target,
            events=events,
            work_threshold=float(args.work_threshold),
        )
    plot_concurrency(
        out_dir / "link_setup_concurrency.png",
        steps,
        concurrency,
        switch_step=int(args.switch_step),
        max_concurrent=int(args.max_concurrent),
    )
    plot_event_timeline(out_dir / "link_setup_event_timeline.png", events, switch_step=int(args.switch_step))
    verification = verify_plan(
        steps=steps,
        switch_step=int(args.switch_step),
        max_concurrent=int(args.max_concurrent),
        work_threshold=float(args.work_threshold),
        source=source,
        target=target,
        events=events,
        concurrency=concurrency,
        unscheduled=unscheduled,
    )
    (out_dir / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(
        out_dir / "summary.json",
        args=args,
        source=source,
        target=target,
        changes=changes,
        events=events,
        concurrency=concurrency,
        unscheduled=unscheduled,
        verification=verification,
    )

    print(f"out_dir={out_dir}")
    print(f"source={source.spec.motif_id:06d} {source.spec.motif}")
    print(f"target={target.spec.motif_id:06d} {target.spec.motif}")
    print(f"changed_right_ports={len(changes)} scheduled_events={len(events)} unscheduled={len(unscheduled)}")
    print(f"max_concurrent_observed={int(np.max(concurrency)) if concurrency.size else 0}")
    print(f"fallback_old_working={sum(1 for event in events if event.fallback_reason == 'old_link_still_working')}")
    print(f"verification_passed={bool(verification.get('passed', False))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
