from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from dataclasses import replace
from pathlib import Path

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


OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\switch_setup\m0056_to_m0061_to_m0056_china_europe"
)
DEFAULT_DELAY_USAGE_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\delay_edge_usage_056_061_china_europe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan round-trip setup: motif000056 -> motif000061 at 10h, then motif000061 -> motif000056 at 15h."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--return-switch-step", type=int, default=54000)
    parser.add_argument("--setup-duration", type=int, default=600)
    parser.add_argument("--lookback-seconds", type=int, default=21600)
    parser.add_argument("--max-concurrent", type=int, default=20)
    parser.add_argument("--work-threshold", type=float, default=0.0)
    parser.add_argument("--schedule-preference", choices=("latest", "earliest"), default="latest")
    parser.add_argument("--usage-mode", choices=("hop", "hop-delay"), default="hop")
    parser.add_argument("--usage-window", choices=("setup", "until-switch"), default="setup")
    parser.add_argument("--delay-usage-root", type=Path, default=DEFAULT_DELAY_USAGE_ROOT)
    parser.add_argument("--second-earliest-start", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-all-node-state", action="store_true")
    return parser.parse_args()


def build_topologies() -> tuple[one_way.TopologyData, one_way.TopologyData]:
    rows = one_way.read_motif_rows(one_way.MOTIF_LIBRARY_CSV)
    spec56 = one_way.build_topology_spec(56, rows[56])
    spec61 = one_way.build_topology_spec(61, rows[61])
    topo56 = one_way.load_topology_data(spec56, start=0, end=86160, stride=60)
    topo61 = one_way.load_topology_data(spec61, start=0, end=86160, stride=60)
    return topo56, topo61


def load_delay_usage_values(
    topo: one_way.TopologyData,
    *,
    delay_usage_root: Path,
    start: int,
    end: int,
    stride: int,
) -> np.ndarray:
    path = (
        Path(delay_usage_root)
        / f"t{int(start)}_{int(end)}_stride{int(stride)}"
        / topo.spec.name
        / "china_europe"
        / "edge_betweenness.npy"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"missing delay edge-usage cache: {path}\n"
            "Run compute_delay_edge_usage_0056_0061_china_europe.py first."
        )
    values = np.load(path, mmap_mode="r")
    if values.shape != topo.values.shape:
        raise ValueError(f"delay usage shape mismatch for {topo.spec.name}: {values.shape} != {topo.values.shape}")
    return np.asarray(values, dtype=np.float32)


def with_combined_usage(
    topo: one_way.TopologyData,
    *,
    delay_usage_root: Path,
    start: int,
    end: int,
    stride: int,
) -> one_way.TopologyData:
    delay_values = load_delay_usage_values(
        topo,
        delay_usage_root=delay_usage_root,
        start=int(start),
        end=int(end),
        stride=int(stride),
    )
    combined = np.maximum(np.asarray(topo.values, dtype=np.float32), delay_values)
    return replace(topo, values=combined)


def planned_transition(
    *,
    label: str,
    source: one_way.TopologyData,
    target: one_way.TopologyData,
    steps: list[int],
    switch_step: int,
    setup_duration: int,
    lookback_seconds: int,
    max_concurrent: int,
    work_threshold: float,
    earliest_setup_step: int | None,
    schedule_preference: str,
    event_id_offset: int,
    component_id_offset: int,
    usage_hold_until_deadline: bool,
) -> tuple[list[one_way.ScheduledEvent], np.ndarray, list[dict[str, object]], list[one_way.RightLinkChange]]:
    switch_idx = steps.index(int(switch_step))
    switch_idx_end = min(len(steps), switch_idx + max(2, int(3600 / int(steps[1] - steps[0]))))
    changes = one_way.find_right_link_changes(source, target, switch_idx=switch_idx, switch_idx_end=switch_idx_end)
    events, concurrency, unscheduled = one_way.schedule_changes(
        changes,
        source=source,
        steps=steps,
        switch_step=int(switch_step),
        setup_duration=int(setup_duration),
        lookback_seconds=int(lookback_seconds),
        max_concurrent=int(max_concurrent),
        work_threshold=float(work_threshold),
        earliest_setup_step=earliest_setup_step,
        schedule_preference=str(schedule_preference),
        usage_hold_until_deadline=bool(usage_hold_until_deadline),
    )
    adjusted = [
        replace(
            event,
            event_id=int(event.event_id) + int(event_id_offset),
            component_id=int(event.component_id) + int(component_id_offset),
        )
        for event in events
    ]
    for item in unscheduled:
        item["transition"] = str(label)
    return adjusted, concurrency, unscheduled, changes


def event_by_owner(events: list[one_way.ScheduledEvent]) -> dict[int, one_way.ScheduledEvent]:
    return {int(event.change.owner): event for event in events}


def active_and_building_at_step(
    *,
    step: int,
    topo56: one_way.TopologyData,
    topo61: one_way.TopologyData,
    first_events: dict[int, one_way.ScheduledEvent],
    second_events: dict[int, one_way.ScheduledEvent],
) -> tuple[dict[int, one_way.RightLink], dict[int, one_way.RightLink]]:
    active: dict[int, one_way.RightLink] = {}
    building: dict[int, one_way.RightLink] = {}
    owners = range(int(G60_CONFIG.total_sats))
    for owner in owners:
        e1 = first_events.get(int(owner))
        e2 = second_events.get(int(owner))
        link56 = topo56.right_by_owner.get(int(owner))
        link61 = topo61.right_by_owner.get(int(owner))

        if e1 is not None and int(step) < int(e1.plan_end):
            if int(step) < int(e1.plan_start):
                if link56 is not None:
                    active[int(owner)] = link56
            else:
                if e1.change.new is not None:
                    building[int(owner)] = e1.change.new
            continue

        if e2 is not None:
            if int(step) < int(e2.plan_start):
                if link61 is not None:
                    active[int(owner)] = link61
            elif int(e2.plan_start) <= int(step) < int(e2.plan_end):
                if e2.change.new is not None:
                    building[int(owner)] = e2.change.new
            else:
                if link56 is not None:
                    active[int(owner)] = link56
            continue

        # No pending event in the current transition; use the phase implied by already finished e1.
        if e1 is not None and int(step) >= int(e1.plan_end):
            if link61 is not None:
                active[int(owner)] = link61
        else:
            # Unchanged links are identical, but this also gives a sane default before first setup.
            link = link56 or link61
            if link is not None:
                active[int(owner)] = link
    return active, building


def edge_key_set(links: dict[int, one_way.RightLink]) -> set[tuple[int, int]]:
    return {link.edge_key for link in links.values()}


def link_usage(
    *,
    row: int,
    step: int,
    link: one_way.RightLink | None,
    topo56: one_way.TopologyData,
    topo61: one_way.TopologyData,
    return_switch_step: int,
) -> tuple[float, str]:
    if link is None:
        return 0.0, ""
    if int(step) >= int(return_switch_step) and link.edge_key in topo56.edge_key_to_idx:
        return float(topo56.values[row, topo56.edge_key_to_idx[link.edge_key]]), topo56.spec.name
    if link.edge_key in topo61.edge_key_to_idx:
        return float(topo61.values[row, topo61.edge_key_to_idx[link.edge_key]]), topo61.spec.name
    if link.edge_key in topo56.edge_key_to_idx:
        return float(topo56.values[row, topo56.edge_key_to_idx[link.edge_key]]), topo56.spec.name
    return 0.0, ""


def write_events(path: Path, *, first_events: list[one_way.ScheduledEvent], second_events: list[one_way.ScheduledEvent]) -> None:
    rows: list[tuple[str, str, str, one_way.ScheduledEvent]] = []
    rows.extend(("switch_056_to_061", "combined_motif_000056", "combined_motif_000061", event) for event in first_events)
    rows.extend(("switch_061_to_056", "combined_motif_000061", "combined_motif_000056", event) for event in second_events)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "transition",
                "source_topology",
                "target_topology",
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
        for transition, source_name, target_name, event in rows:
            change = event.change
            old = change.old
            new = change.new
            writer.writerow(
                [
                    transition,
                    source_name,
                    target_name,
                    int(event.event_id),
                    int(event.component_id),
                    int(change.owner),
                    int(change.owner // G60_CONFIG.N),
                    int(change.owner % G60_CONFIG.N),
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


def write_concurrency(path: Path, steps: list[int], first: np.ndarray, second: np.ndarray) -> np.ndarray:
    total = np.asarray(first, dtype=np.int32) + np.asarray(second, dtype=np.int32)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "hour", "building_count", "building_056_to_061", "building_061_to_056"])
        for step, total_count, first_count, second_count in zip(steps, total, first, second):
            writer.writerow([int(step), f"{int(step) / 3600:.6f}", int(total_count), int(first_count), int(second_count)])
    return total


def write_edge_state(
    path: Path,
    *,
    steps: list[int],
    topo56: one_way.TopologyData,
    topo61: one_way.TopologyData,
    first_by_owner: dict[int, one_way.ScheduledEvent],
    second_by_owner: dict[int, one_way.ScheduledEvent],
    return_switch_step: int,
    work_threshold: float,
) -> None:
    event_id_by_owner = {**{owner: event.event_id for owner, event in first_by_owner.items()}, **{owner: event.event_id for owner, event in second_by_owner.items()}}
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "hour",
                "status",
                "event_id",
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
        for row, step in enumerate(steps):
            active, building = active_and_building_at_step(
                step=int(step),
                topo56=topo56,
                topo61=topo61,
                first_events=first_by_owner,
                second_events=second_by_owner,
            )
            for status, links in (("active", active), ("building", building)):
                for owner, link in sorted(links.items()):
                    usage, metric_topology = link_usage(
                        row=row,
                        step=int(step),
                        link=link,
                        topo56=topo56,
                        topo61=topo61,
                        return_switch_step=int(return_switch_step),
                    )
                    writer.writerow(
                        [
                            int(step),
                            f"{int(step) / 3600:.6f}",
                            status,
                            event_id_by_owner.get(int(owner), ""),
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


def write_node_state(
    path: Path,
    *,
    steps: list[int],
    topo56: one_way.TopologyData,
    topo61: one_way.TopologyData,
    first_by_owner: dict[int, one_way.ScheduledEvent],
    second_by_owner: dict[int, one_way.ScheduledEvent],
    return_switch_step: int,
    work_threshold: float,
) -> None:
    changed_owners = set(first_by_owner) | set(second_by_owner)
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
                "left_symbol",
                "left_edge",
                "right_neighbor",
                "right_symbol",
                "right_edge",
                "right_edge_usage",
                "right_edge_work",
                "building_target_neighbor",
                "building_symbol",
                "building_edge",
                "metric_topology",
                "is_changed_owner",
            ]
        )
        for row, step in enumerate(steps):
            active, building = active_and_building_at_step(
                step=int(step),
                topo56=topo56,
                topo61=topo61,
                first_events=first_by_owner,
                second_events=second_by_owner,
            )
            active_left = {link.right: link for link in active.values()}
            for node in range(int(G60_CONFIG.total_sats)):
                right = active.get(node)
                left = active_left.get(node)
                build = building.get(node)
                usage, metric_topology = link_usage(
                    row=row,
                    step=int(step),
                    link=right,
                    topo56=topo56,
                    topo61=topo61,
                    return_switch_step=int(return_switch_step),
                )
                writer.writerow(
                    [
                        int(step),
                        f"{int(step) / 3600:.6f}",
                        int(node),
                        int(node // G60_CONFIG.N),
                        int(node % G60_CONFIG.N),
                        "building" if build is not None else ("active" if right is not None else "empty"),
                        "" if left is None else left.owner,
                        "" if left is None else left.symbol,
                        "" if left is None else left.edge_key_text,
                        "" if right is None else right.right,
                        "" if right is None else right.symbol,
                        "" if right is None else right.edge_key_text,
                        f"{usage:.6g}",
                        int(usage > float(work_threshold)),
                        "" if build is None else build.right,
                        "" if build is None else build.symbol,
                        "" if build is None else build.edge_key_text,
                        metric_topology,
                        int(node in changed_owners),
                    ]
                )


def verify_roundtrip(
    *,
    steps: list[int],
    switch_step: int,
    return_switch_step: int,
    max_concurrent: int,
    total_concurrency: np.ndarray,
    topo56: one_way.TopologyData,
    topo61: one_way.TopologyData,
    first_by_owner: dict[int, one_way.ScheduledEvent],
    second_by_owner: dict[int, one_way.ScheduledEvent],
    unscheduled: list[dict[str, object]],
    work_threshold: float,
) -> dict[str, object]:
    max_left = 0
    max_right = 0
    first_conflict_step = None
    building_at_switches = {}
    for step in steps:
        active, building = active_and_building_at_step(
            step=int(step),
            topo56=topo56,
            topo61=topo61,
            first_events=first_by_owner,
            second_events=second_by_owner,
        )
        conflicts = one_way.count_port_conflicts(active, building)
        if conflicts["left_conflict_nodes"] or conflicts["right_conflict_nodes"]:
            if first_conflict_step is None:
                first_conflict_step = int(step)
        max_left = max(max_left, int(conflicts["left_conflict_nodes"]))
        max_right = max(max_right, int(conflicts["right_conflict_nodes"]))
        if int(step) in (int(switch_step), int(return_switch_step)):
            building_at_switches[str(int(step))] = int(len(building))

    active_10h, building_10h = active_and_building_at_step(
        step=int(switch_step),
        topo56=topo56,
        topo61=topo61,
        first_events=first_by_owner,
        second_events=second_by_owner,
    )
    active_15h, building_15h = active_and_building_at_step(
        step=int(return_switch_step),
        topo56=topo56,
        topo61=topo61,
        first_events=first_by_owner,
        second_events=second_by_owner,
    )
    source_edges = edge_key_set(topo56.right_by_owner)
    target_edges = edge_key_set(topo61.right_by_owner)
    checks = {
        "no_unscheduled_events": len(unscheduled) == 0,
        "max_concurrent_ok": int(np.max(total_concurrency)) <= int(max_concurrent),
        "first_events_finish_by_10h": all(int(event.plan_end) <= int(switch_step) for event in first_by_owner.values()),
        "second_events_finish_by_15h": all(int(event.plan_end) <= int(return_switch_step) for event in second_by_owner.values()),
        "topology_at_10h_is_061": edge_key_set(active_10h) == target_edges and len(building_10h) == 0,
        "topology_at_15h_is_056": edge_key_set(active_15h) == source_edges and len(building_15h) == 0,
        "no_left_port_conflicts": max_left == 0,
        "no_right_port_conflicts": max_right == 0,
        "old_links_changed_only_when_no_work": all(
            event.fallback_reason != "old_link_still_working"
            and float(event.old_usage_max_during_setup) <= float(work_threshold)
            for event in list(first_by_owner.values()) + list(second_by_owner.values())
        ),
    }
    return {
        "passed": all(bool(value) for value in checks.values()),
        "checks": checks,
        "max_concurrent_observed": int(np.max(total_concurrency)) if total_concurrency.size else 0,
        "max_left_conflict_nodes": int(max_left),
        "max_right_conflict_nodes": int(max_right),
        "first_conflict_step": first_conflict_step,
        "building_at_switches": building_at_switches,
        "edges_at_10h": int(len(active_10h)),
        "edges_at_15h": int(len(active_15h)),
    }


def plot_concurrency(path: Path, steps: list[int], first: np.ndarray, second: np.ndarray, total: np.ndarray, *, switch_step: int, return_switch_step: int, max_concurrent: int) -> None:
    hours = np.asarray(steps, dtype=np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(12.5, 4.9), dpi=180)
    ax.plot(hours, total, color="#b32134", linewidth=1.8, label="total building")
    ax.plot(hours, first, color="#d97706", linewidth=1.0, alpha=0.7, label="056->061")
    ax.plot(hours, second, color="#1d4ed8", linewidth=1.0, alpha=0.7, label="061->056")
    ax.axvline(float(switch_step) / 3600.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.axvline(float(return_switch_step) / 3600.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.axhline(int(max_concurrent), color="#555555", linestyle=":", linewidth=1.0)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("building links")
    ax.set_title("Round-trip link setup concurrency: motif000056 -> motif000061 -> motif000056")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if int(args.switch_step) not in set(steps) or int(args.return_switch_step) not in set(steps):
        raise ValueError("switch steps must be on the requested time axis")
    second_earliest = args.second_earliest_start
    if second_earliest is None:
        second_earliest = int(args.switch_step) + int(args.setup_duration)

    topo56, topo61 = build_topologies()
    if str(args.usage_mode) == "hop-delay":
        topo56 = with_combined_usage(
            topo56,
            delay_usage_root=Path(args.delay_usage_root),
            start=int(args.start),
            end=int(args.end),
            stride=int(args.stride),
        )
        topo61 = with_combined_usage(
            topo61,
            delay_usage_root=Path(args.delay_usage_root),
            start=int(args.start),
            end=int(args.end),
            stride=int(args.stride),
        )
    usage_hold_until_deadline = str(args.usage_window) == "until-switch"
    first_events, first_conc, first_unscheduled, first_changes = planned_transition(
        label="switch_056_to_061",
        source=topo56,
        target=topo61,
        steps=steps,
        switch_step=int(args.switch_step),
        setup_duration=int(args.setup_duration),
        lookback_seconds=int(args.lookback_seconds),
        max_concurrent=int(args.max_concurrent),
        work_threshold=float(args.work_threshold),
        earliest_setup_step=None,
        schedule_preference=str(args.schedule_preference),
        event_id_offset=0,
        component_id_offset=0,
        usage_hold_until_deadline=usage_hold_until_deadline,
    )
    second_events, second_conc, second_unscheduled, second_changes = planned_transition(
        label="switch_061_to_056",
        source=topo61,
        target=topo56,
        steps=steps,
        switch_step=int(args.return_switch_step),
        setup_duration=int(args.setup_duration),
        lookback_seconds=int(args.lookback_seconds),
        max_concurrent=int(args.max_concurrent),
        work_threshold=float(args.work_threshold),
        earliest_setup_step=int(second_earliest),
        schedule_preference=str(args.schedule_preference),
        event_id_offset=len(first_events),
        component_id_offset=max((event.component_id for event in first_events), default=0),
        usage_hold_until_deadline=usage_hold_until_deadline,
    )

    out_dir = args.out_dir
    if out_dir is None:
        suffix_parts = [f"switch{int(args.switch_step)}_{int(args.return_switch_step)}"]
        suffix_parts.append(f"dur{int(args.setup_duration)}")
        suffix_parts.append(f"c{int(args.max_concurrent)}")
        suffix_parts.append(str(args.schedule_preference))
        if str(args.usage_mode) != "hop":
            suffix_parts.append(str(args.usage_mode))
        if str(args.usage_window) != "setup":
            suffix_parts.append(str(args.usage_window))
        out_dir = (
            OUT_ROOT
            / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
            / "_".join(suffix_parts)
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_conc = write_concurrency(out_dir / "link_setup_concurrency_by_step.csv", steps, first_conc, second_conc)
    write_events(out_dir / "link_setup_events.csv", first_events=first_events, second_events=second_events)
    first_by_owner = event_by_owner(first_events)
    second_by_owner = event_by_owner(second_events)
    write_edge_state(
        out_dir / "edge_state_by_step.csv.gz",
        steps=steps,
        topo56=topo56,
        topo61=topo61,
        first_by_owner=first_by_owner,
        second_by_owner=second_by_owner,
        return_switch_step=int(args.return_switch_step),
        work_threshold=float(args.work_threshold),
    )
    if not bool(args.skip_all_node_state):
        write_node_state(
            out_dir / "right_link_state_all_nodes.csv.gz",
            steps=steps,
            topo56=topo56,
            topo61=topo61,
            first_by_owner=first_by_owner,
            second_by_owner=second_by_owner,
            return_switch_step=int(args.return_switch_step),
            work_threshold=float(args.work_threshold),
        )
    unscheduled = first_unscheduled + second_unscheduled
    if unscheduled:
        (out_dir / "unscheduled_events.json").write_text(json.dumps(unscheduled, ensure_ascii=False, indent=2), encoding="utf-8")
    verification = verify_roundtrip(
        steps=steps,
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
        max_concurrent=int(args.max_concurrent),
        total_concurrency=total_conc,
        topo56=topo56,
        topo61=topo61,
        first_by_owner=first_by_owner,
        second_by_owner=second_by_owner,
        unscheduled=unscheduled,
        work_threshold=float(args.work_threshold),
    )
    (out_dir / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "source_motif": topo56.spec.motif,
        "middle_motif": topo61.spec.motif,
        "switch_step": int(args.switch_step),
        "return_switch_step": int(args.return_switch_step),
        "setup_duration": int(args.setup_duration),
        "max_concurrent_limit": int(args.max_concurrent),
        "max_concurrent_observed": int(np.max(total_conc)) if total_conc.size else 0,
        "schedule_preference": str(args.schedule_preference),
        "usage_mode": str(args.usage_mode),
        "usage_window": str(args.usage_window),
        "first_changed_right_ports": int(len(first_changes)),
        "second_changed_right_ports": int(len(second_changes)),
        "first_scheduled_events": int(len(first_events)),
        "second_scheduled_events": int(len(second_events)),
        "unscheduled_events": int(len(unscheduled)),
        "fallback_events_old_link_still_working": int(
            sum(1 for event in first_events + second_events if event.fallback_reason == "old_link_still_working")
        ),
        "verification_passed": bool(verification.get("passed", False)),
        "runner": str(THIS_FILE),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_concurrency(
        out_dir / "link_setup_concurrency.png",
        steps,
        first_conc,
        second_conc,
        total_conc,
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
        max_concurrent=int(args.max_concurrent),
    )

    print(f"out_dir={out_dir}")
    print(f"first 056->061 events={len(first_events)} unscheduled={len(first_unscheduled)}")
    print(f"second 061->056 events={len(second_events)} unscheduled={len(second_unscheduled)}")
    print(f"max_concurrent_observed={int(np.max(total_conc)) if total_conc.size else 0}")
    print(f"fallback_old_working={summary['fallback_events_old_link_still_working']}")
    print(f"verification_passed={verification.get('passed', False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
