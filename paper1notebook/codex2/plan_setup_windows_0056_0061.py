from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import write_edges_csv  # noqa: E402

from plan_guarded_t2_switch_0056_0061 import (  # noqa: E402
    DEFAULT_USAGE_ROOT,
    GuardedEvent,
    UsageStore,
    all_intra_keys,
    build_static_topology,
    build_viewer_arrays,
    edge_key_index,
    event_by_owner,
    find_right_link_changes,
    read_usage_store,
    slice_usage_store,
    source_port_occupants_for_new_link,
    union_edge_table,
)


DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\setup_windows_056_061_china_europe"
)


@dataclass(frozen=True)
class SetupWindow:
    event: GuardedEvent | None
    owner: int
    old_edge: str
    new_edge: str
    new_symbol: str
    protected_edges: str
    source_ports_occupied_before_switch: str
    target_first_work_step: int | None
    window_left: int | None
    window_right: int | None
    window_feasible_before_target_work: bool
    plan_start: int | None
    plan_end: int | None
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-node setup windows for the motif000056 -> motif000061 switch. "
            "The actual setup start is chosen as the left endpoint of the computed window."
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
    return parser.parse_args()


def usage_values_for_link(store: UsageStore, link) -> np.ndarray:
    if link is None:
        return np.zeros(int(store.steps.size), dtype=np.float32)
    col = store.key_to_col.get(link.edge_key)
    if col is None:
        return np.zeros(int(store.steps.size), dtype=np.float32)
    return np.asarray(store.counts[:, int(col)], dtype=np.float32)


def combined_work_mask(store: UsageStore, links: Sequence, *, threshold: float) -> np.ndarray:
    mask = np.zeros(int(store.steps.size), dtype=bool)
    for link in links:
        mask |= usage_values_for_link(store, link) > float(threshold)
    return mask


def source_incident_links_for_nodes(store: UsageStore, nodes: Sequence[int]) -> list:
    """Return source-topology right links touching any node in ``nodes``."""
    node_set = {int(node) for node in nodes}
    links = []
    seen: set[tuple[int, int]] = set()
    for link in store.topology.right_by_owner.values():
        if int(link.owner) not in node_set and int(link.right) not in node_set:
            continue
        if link.edge_key in seen:
            continue
        seen.add(link.edge_key)
        links.append(link)
    return links


def first_work_step(store: UsageStore, link, *, start: int, end: int, threshold: float) -> int | None:
    values = usage_values_for_link(store, link)
    rows = np.flatnonzero((store.steps >= int(start)) & (store.steps < int(end)))
    if rows.size == 0:
        return None
    hits = rows[np.flatnonzero(values[rows] > float(threshold))]
    if hits.size == 0:
        return None
    return int(store.steps[int(hits[0])])


def setup_lag_rows(steps: np.ndarray, lst: int) -> int:
    if int(lst) <= 0:
        return 0
    if steps.size <= 1:
        return 1
    stride = int(steps[1] - steps[0])
    if stride <= 0:
        raise ValueError("steps must be strictly increasing")
    return int(math.ceil(float(lst) / float(stride)))


def row_for_step_left(steps: np.ndarray, step: int) -> int:
    return int(np.searchsorted(steps, int(step), side="left"))


def contiguous_right_from_left(
    feasible: np.ndarray,
    *,
    left_row: int,
    right_limit_row: int,
) -> int | None:
    if int(left_row) > int(right_limit_row):
        return None
    row = int(left_row)
    last = None
    while row <= int(right_limit_row) and row < feasible.size and bool(feasible[row]):
        last = row
        row += 1
    return last


def compute_one_window(
    *,
    source: UsageStore,
    target: UsageStore,
    change,
    switch_step: int,
    segment_end: int,
    lst: int,
    threshold: float,
) -> SetupWindow:
    owner = int(change.owner)
    old = change.old
    new = change.new
    if new is None:
        return SetupWindow(
            event=None,
            owner=owner,
            old_edge="" if old is None else old.edge_key_text,
            new_edge="",
            new_symbol="",
            protected_edges="",
            source_ports_occupied_before_switch="",
            target_first_work_step=None,
            window_left=None,
            window_right=None,
            window_feasible_before_target_work=False,
            plan_start=None,
            plan_end=None,
            reason="target_has_no_right_link",
        )

    occupants = source_port_occupants_for_new_link(source.topology, new)
    protected_links = [link for _kind, link in occupants]
    changed_nodes = {int(owner), int(new.right)}
    if old is not None:
        changed_nodes.add(int(old.right))
    for _kind, link in occupants:
        changed_nodes.add(int(link.owner))
        changed_nodes.add(int(link.right))
    protected_links.extend(source_incident_links_for_nodes(source, sorted(changed_nodes)))
    protected_text = "; ".join(f"{kind}:{link.edge_key_text}" for kind, link in occupants)
    if changed_nodes:
        node_text = ",".join(str(int(node)) for node in sorted(changed_nodes))
        protected_text = f"{protected_text}; node_work_guard:{node_text}" if protected_text else f"node_work_guard:{node_text}"
    target_first = first_work_step(
        target,
        new,
        start=int(switch_step),
        end=int(segment_end),
        threshold=float(threshold),
    )

    lag = setup_lag_rows(source.steps, int(lst))
    if lag <= 0:
        lag = 1
    if source.steps.size <= lag:
        raise ValueError("time axis is shorter than setup lag")

    work_mask = combined_work_mask(source, protected_links, threshold=float(threshold))
    prefix = np.concatenate(([0], np.cumsum(work_mask.astype(np.int32))))
    feasible = np.zeros(int(source.steps.size), dtype=bool)
    max_start_row = int(source.steps.size) - lag
    if max_start_row >= 0:
        rows = np.arange(0, max_start_row + 1, dtype=np.int64)
        feasible[rows] = (prefix[rows + lag] - prefix[rows]) == 0

    prebuild_allowed = len(occupants) == 0
    if prebuild_allowed:
        candidate_mask = source.steps <= int(switch_step) - int(lst)
    else:
        candidate_mask = source.steps >= int(switch_step)
    candidate_mask &= source.steps <= int(segment_end) - int(lst)
    candidate_rows = np.flatnonzero(candidate_mask & feasible)

    if candidate_rows.size == 0:
        return SetupWindow(
            event=None,
            owner=owner,
            old_edge="" if old is None else old.edge_key_text,
            new_edge=new.edge_key_text,
            new_symbol=new.symbol,
            protected_edges=protected_text,
            source_ports_occupied_before_switch=protected_text,
            target_first_work_step=target_first,
            window_left=None,
            window_right=None,
            window_feasible_before_target_work=False,
            plan_start=None,
            plan_end=None,
            reason="no_setup_window_found",
        )

    left_row = int(candidate_rows[0])
    left_step = int(source.steps[left_row])
    latest_before_target = (int(target_first) - int(lst)) if target_first is not None else int(segment_end) - int(lst)
    right_limit_row = min(row_for_step_left(source.steps, latest_before_target), max_start_row)
    right_row = contiguous_right_from_left(feasible, left_row=left_row, right_limit_row=right_limit_row)
    right_step = None if right_row is None else int(source.steps[int(right_row)])
    feasible_before_target = right_row is not None and int(left_step) <= int(latest_before_target)

    if not feasible_before_target:
        return SetupWindow(
            event=None,
            owner=owner,
            old_edge="" if old is None else old.edge_key_text,
            new_edge=new.edge_key_text,
            new_symbol=new.symbol,
            protected_edges=protected_text,
            source_ports_occupied_before_switch=protected_text if occupants else "",
            target_first_work_step=target_first,
            window_left=left_step,
            window_right=right_step,
            window_feasible_before_target_work=False,
            plan_start=None,
            plan_end=None,
            reason="no_setup_window_before_target_first_work",
        )

    event = GuardedEvent(
        transition=f"motif{int(source.topology.spec.motif_id):06d}_to_motif{int(target.topology.spec.motif_id):06d}",
        owner=owner,
        old=old,
        new=new,
        plan_start=int(left_step),
        plan_end=int(left_step) + int(lst),
        target_interval_start=int(target_first) if target_first is not None else int(segment_end),
        target_interval_end=int(segment_end),
        skipped_intervals=0,
        blocker_summary=protected_text,
    )
    return SetupWindow(
        event=event,
        owner=owner,
        old_edge="" if old is None else old.edge_key_text,
        new_edge=new.edge_key_text,
        new_symbol=new.symbol,
        protected_edges=protected_text,
        source_ports_occupied_before_switch=protected_text if occupants else "",
        target_first_work_step=target_first,
        window_left=left_step,
        window_right=right_step,
        window_feasible_before_target_work=True,
        plan_start=int(event.plan_start),
        plan_end=int(event.plan_end),
        reason="scheduled_at_window_left",
    )


def window_to_row(item: SetupWindow, event_id: int | None) -> dict[str, object]:
    return {
        "event_id": "" if event_id is None else int(event_id),
        "owner": int(item.owner),
        "owner_p": int(item.owner // G60_CONFIG.N),
        "owner_y": int(item.owner % G60_CONFIG.N),
        "old_edge": item.old_edge,
        "new_edge": item.new_edge,
        "new_symbol": item.new_symbol,
        "protected_edges": item.protected_edges,
        "source_ports_occupied_before_switch": item.source_ports_occupied_before_switch,
        "target_first_work_step": "" if item.target_first_work_step is None else int(item.target_first_work_step),
        "window_left": "" if item.window_left is None else int(item.window_left),
        "window_right": "" if item.window_right is None else int(item.window_right),
        "window_feasible_before_target_work": bool(item.window_feasible_before_target_work),
        "plan_start": "" if item.plan_start is None else int(item.plan_start),
        "plan_end": "" if item.plan_end is None else int(item.plan_end),
        "reason": item.reason,
    }


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
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


def write_concurrency(path: Path, steps: np.ndarray, events: Sequence[GuardedEvent]) -> np.ndarray:
    counts = np.zeros(int(steps.size), dtype=np.int32)
    for event in events:
        rows = np.flatnonzero((steps >= int(event.plan_start)) & (steps < int(event.plan_end)))
        counts[rows] += 1
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "hour", "building_count"])
        for step, count in zip(steps, counts):
            writer.writerow([int(step), f"{int(step) / 3600.0:.6f}", int(count)])
    return counts


def plot_concurrency(path: Path, steps: np.ndarray, counts: np.ndarray, *, switch_step: int) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 4.8), dpi=170)
    ax.plot(steps / 3600.0, counts, color="#b91c1c", linewidth=1.25)
    ax.axvline(float(switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("building links")
    ax.set_title("Setup-window-left schedule: motif000056 -> motif000061")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    source_topology = build_static_topology(int(args.source_motif_id))
    target_topology = build_static_topology(int(args.target_motif_id))
    source = slice_usage_store(
        read_usage_store(Path(args.usage_root), source_topology),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    target = slice_usage_store(
        read_usage_store(Path(args.usage_root), target_topology),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    if not np.array_equal(source.steps, target.steps):
        raise ValueError("source and target usage stores have different time axes")

    changes = find_right_link_changes(source.topology, target.topology)
    windows = [
        compute_one_window(
            source=source,
            target=target,
            change=change,
            switch_step=int(args.switch_step),
            segment_end=int(args.segment_end),
            lst=int(args.lst),
            threshold=float(args.working_threshold),
        )
        for change in changes
    ]
    events = [item.event for item in windows if item.event is not None]

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT
        / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
        / f"switch{int(args.switch_step)}_segend{int(args.segment_end)}_lst{int(args.lst):03d}_leftedge_delay"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    next_event_id = 1
    for item in sorted(windows, key=lambda x: (10**12 if x.plan_start is None else int(x.plan_start), int(x.owner))):
        event_id = None
        if item.event is not None:
            event_id = next_event_id
            next_event_id += 1
        rows.append(window_to_row(item, event_id))
    write_rows(out_dir / "setup_windows.csv", rows)

    edge_table = union_edge_table(source.topology, target.topology)
    active_mask, building_mask, values = build_viewer_arrays(
        steps=source.steps,
        edge_table=edge_table,
        source=source,
        target=target,
        events=events,
    )
    write_edges_csv(edge_table, out_dir / "union_edges.csv")
    np.save(out_dir / "steps.npy", source.steps.astype(np.int64))
    np.save(out_dir / "edge_active_mask.npy", active_mask.astype(bool))
    np.save(out_dir / "edge_building_mask.npy", building_mask.astype(bool))
    np.save(out_dir / "edge_usage_values.npy", values.astype(np.float32))

    counts = write_concurrency(out_dir / "building_concurrency_by_step.csv", source.steps, events)
    np.save(out_dir / "building_concurrency.npy", counts)
    plot_concurrency(out_dir / "building_concurrency.png", source.steps, counts, switch_step=int(args.switch_step))

    feasible_count = sum(1 for item in windows if item.window_feasible_before_target_work)
    missed_count = sum(1 for item in windows if item.event is not None and not item.window_feasible_before_target_work)
    unscheduled_count = sum(1 for item in windows if item.event is None)
    summary = {
        "source_motif_id": int(args.source_motif_id),
        "source_motif": source.topology.spec.motif,
        "target_motif_id": int(args.target_motif_id),
        "target_motif": target.topology.spec.motif,
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "switch_step": int(args.switch_step),
        "segment_end": int(args.segment_end),
        "lst": int(args.lst),
        "changed_right_ports": int(len(changes)),
        "scheduled_events": int(len(events)),
        "feasible_before_target_first_work": int(feasible_count),
        "scheduled_but_misses_target_first_work": int(missed_count),
        "unscheduled": int(unscheduled_count),
        "max_building_concurrency": int(np.max(counts)) if counts.size else 0,
        "first_building_step": None if not np.any(counts) else int(source.steps[int(np.flatnonzero(counts > 0)[0])]),
        "outputs": {
            "setup_windows": "setup_windows.csv",
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
        f"changed={len(changes)} scheduled={len(events)} feasible_before_target={feasible_count} "
        f"misses_target={missed_count} unscheduled={unscheduled_count}"
    )
    print(
        f"max_building_concurrency={summary['max_building_concurrency']} "
        f"first_building_step={summary['first_building_step']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
