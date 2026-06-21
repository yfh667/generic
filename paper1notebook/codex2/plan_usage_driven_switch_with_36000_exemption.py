from __future__ import annotations

import argparse
import csv
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
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_motif0056_0061_roundtrip_link_setup as roundtrip  # noqa: E402
import plan_motif0056_to_0061_link_setup as one_way  # noqa: E402
from build_ideal_splice_right_link_state_0056_0061 import DEFAULT_DELAY_USAGE_ROOT, load_delay_usage  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402


DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_switch_with_exemption_056_061_china_europe"
)


@dataclass(frozen=True)
class UsageBundle:
    topo: one_way.TopologyData
    hop: np.ndarray
    delay: np.ndarray
    any_usage: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan motif000056 -> motif000061 right-link setup with a one-cycle exemption for "
            "links that are required exactly at 36000s but cannot be built because their old ports "
            "are still working."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--return-switch-step", type=int, default=54000)
    parser.add_argument("--lst", type=int, default=60)
    parser.add_argument("--source-motif-id", type=int, default=56)
    parser.add_argument("--target-motif-id", type=int, default=61)
    parser.add_argument("--working-threshold", type=float, default=0.0)
    parser.add_argument("--delay-usage-root", type=Path, default=DEFAULT_DELAY_USAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def usage_bundle(
    motif_id: int,
    *,
    delay_usage_root: Path,
    start: int,
    end: int,
    stride: int,
) -> UsageBundle:
    rows = one_way.read_motif_rows(one_way.MOTIF_LIBRARY_CSV)
    spec = one_way.build_topology_spec(int(motif_id), rows[int(motif_id)])
    topo = one_way.load_topology_data(spec, start=int(start), end=int(end), stride=int(stride))
    hop = np.asarray(topo.values, dtype=np.float32)
    delay = load_delay_usage(
        topo,
        delay_usage_root=Path(delay_usage_root),
        start=int(start),
        end=int(end),
        stride=int(stride),
    )
    return UsageBundle(topo=topo, hop=hop, delay=delay, any_usage=np.maximum(hop, delay))


def edge_key_text(link: one_way.RightLink | None) -> str:
    if link is None:
        return ""
    return f"{int(link.edge_key[0])}-{int(link.edge_key[1])}"


def link_text(link: one_way.RightLink | None) -> str:
    if link is None:
        return ""
    return f"{int(link.owner)}->{int(link.right)}"


def rows_between(steps: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.flatnonzero((steps >= int(start)) & (steps < int(end)))


def working_intervals(
    *,
    steps: np.ndarray,
    usage: np.ndarray,
    edge_idx: int,
    start_step: int,
    end_step: int,
    threshold: float,
) -> list[tuple[int, int, int, int]]:
    rows = rows_between(steps, int(start_step), int(end_step))
    if rows.size == 0:
        return []
    mask = np.asarray(usage[rows, int(edge_idx)] > float(threshold), dtype=bool)
    hit = np.flatnonzero(mask)
    if hit.size == 0:
        return []
    intervals: list[tuple[int, int, int, int]] = []
    start_local = int(hit[0])
    prev_local = int(hit[0])
    for local in hit[1:]:
        local = int(local)
        if local != prev_local + 1:
            start_row = int(rows[start_local])
            end_row = int(rows[prev_local])
            intervals.append((start_row, end_row, int(steps[start_row]), int(steps[end_row])))
            start_local = local
        prev_local = local
    start_row = int(rows[start_local])
    end_row = int(rows[prev_local])
    intervals.append((start_row, end_row, int(steps[start_row]), int(steps[end_row])))
    return intervals


def usage_mode_at(bundle: UsageBundle, *, row: int, edge_idx: int, threshold: float) -> str:
    hop = float(bundle.hop[int(row), int(edge_idx)])
    delay = float(bundle.delay[int(row), int(edge_idx)])
    if hop > float(threshold) and delay > float(threshold):
        return "hop+delay"
    if hop > float(threshold):
        return "hop"
    if delay > float(threshold):
        return "delay"
    return "none"


def blocker_rows_for_window(
    *,
    source: UsageBundle,
    old: one_way.RightLink | None,
    new: one_way.RightLink,
    steps: np.ndarray,
    setup_start: int,
    setup_end: int,
    threshold: float,
) -> list[dict[str, object]]:
    setup_rows = rows_between(steps, int(setup_start), int(setup_end))
    blockers: list[dict[str, object]] = []
    if setup_rows.size == 0:
        return blockers

    candidates: list[tuple[str, one_way.RightLink | None]] = [("owner_right_old", old)]
    candidates.append(("target_left_old", source.topo.left_by_right.get(int(new.right))))
    for blocker_type, link in candidates:
        if link is None:
            continue
        values = np.asarray(source.any_usage[setup_rows, int(link.edge_idx)], dtype=np.float32)
        if not np.any(values > float(threshold)):
            continue
        hit_rows = setup_rows[np.flatnonzero(values > float(threshold))]
        first_row = int(hit_rows[0])
        blockers.append(
            {
                "blocker_type": blocker_type,
                "blocker_link": link_text(link),
                "blocker_symbol": str(link.symbol),
                "blocker_edge_key": edge_key_text(link),
                "blocker_owner": int(link.owner),
                "blocker_right": int(link.right),
                "blocker_first_step_inside_setup": int(steps[first_row]),
                "blocker_max_hop_inside_setup": float(np.nanmax(source.hop[setup_rows, int(link.edge_idx)])),
                "blocker_max_delay_inside_setup": float(np.nanmax(source.delay[setup_rows, int(link.edge_idx)])),
                "blocker_mode_at_first": usage_mode_at(
                    source,
                    row=first_row,
                    edge_idx=int(link.edge_idx),
                    threshold=float(threshold),
                ),
            }
        )
    return blockers


def plan_event_by_owner(plan_rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    events = {}
    for row in plan_rows:
        if bool(row.get("scheduled", False)):
            events[int(row["owner"])] = row
    return events


def current_link_for_owner(
    *,
    owner: int,
    step: int,
    source: UsageBundle,
    target: UsageBundle,
    event_by_owner: dict[int, dict[str, object]],
) -> tuple[str, one_way.RightLink | None]:
    event = event_by_owner.get(int(owner))
    if event is None:
        return "source", source.topo.right_by_owner.get(int(owner))
    if int(step) < int(event["plan_start"]):
        return "source", source.topo.right_by_owner.get(int(owner))
    if int(event["plan_start"]) <= int(step) < int(event["plan_end"]):
        return "building", None
    return "target", target.topo.right_by_owner.get(int(owner))


def usage_value_for_link_at_row(
    *,
    source: UsageBundle,
    target: UsageBundle,
    topology_id: str,
    link: one_way.RightLink,
    row: int,
) -> tuple[float, float, float]:
    if topology_id == "source":
        hop = float(source.hop[int(row), int(link.edge_idx)])
        delay = float(source.delay[int(row), int(link.edge_idx)])
    elif topology_id == "target":
        hop = float(target.hop[int(row), int(link.edge_idx)])
        delay = float(target.delay[int(row), int(link.edge_idx)])
    else:
        return 0.0, 0.0, 0.0
    return hop, delay, max(hop, delay)


def dynamic_blockers_for_window(
    *,
    source: UsageBundle,
    target: UsageBundle,
    event_by_owner: dict[int, dict[str, object]],
    owner: int,
    new: one_way.RightLink,
    steps: np.ndarray,
    setup_start: int,
    setup_end: int,
    threshold: float,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    setup_rows = rows_between(steps, int(setup_start), int(setup_end))
    if setup_rows.size == 0:
        return blockers

    for port_type in ("owner_right", "target_left"):
        best: dict[str, object] | None = None
        max_hop = 0.0
        max_delay = 0.0
        for row in setup_rows:
            step = int(steps[int(row)])
            if port_type == "owner_right":
                state, link = current_link_for_owner(
                    owner=int(owner),
                    step=step,
                    source=source,
                    target=target,
                    event_by_owner=event_by_owner,
                )
                if link is None:
                    continue
                # The new link being built owns this port; only another active
                # link on the same right port can block it.
                if int(link.owner) != int(owner) or link.edge_key == new.edge_key:
                    continue
            else:
                link = None
                state = ""
                for candidate_owner in set(source.topo.left_by_right) | set(target.topo.left_by_right):
                    # This loop walks right-node keys, not owners. It is easier
                    # and safer to inspect all current owners below instead.
                    del candidate_owner
                for candidate_owner in sorted(set(source.topo.right_by_owner) | set(target.topo.right_by_owner)):
                    cand_state, cand_link = current_link_for_owner(
                        owner=int(candidate_owner),
                        step=step,
                        source=source,
                        target=target,
                        event_by_owner=event_by_owner,
                    )
                    if cand_link is None:
                        continue
                    if int(cand_link.right) == int(new.right) and cand_link.edge_key != new.edge_key:
                        link = cand_link
                        state = cand_state
                        break
                if link is None:
                    continue

            hop, delay, usage = usage_value_for_link_at_row(
                source=source,
                target=target,
                topology_id=state,
                link=link,
                row=int(row),
            )
            max_hop = max(max_hop, hop)
            max_delay = max(max_delay, delay)
            if usage <= float(threshold):
                continue
            if best is None:
                if hop > float(threshold) and delay > float(threshold):
                    mode = "hop+delay"
                elif hop > float(threshold):
                    mode = "hop"
                elif delay > float(threshold):
                    mode = "delay"
                else:
                    mode = "none"
                best = {
                    "blocker_type": "owner_right_active" if port_type == "owner_right" else "target_left_active",
                    "blocker_state": state,
                    "blocker_link": link_text(link),
                    "blocker_symbol": str(link.symbol),
                    "blocker_edge_key": edge_key_text(link),
                    "blocker_owner": int(link.owner),
                    "blocker_right": int(link.right),
                    "blocker_first_step_inside_setup": step,
                    "blocker_mode_at_first": mode,
                }
        if best is not None:
            best["blocker_max_hop_inside_setup"] = float(max_hop)
            best["blocker_max_delay_inside_setup"] = float(max_delay)
            blockers.append(best)
    return blockers


def dynamic_conflicts_after_schedule(
    *,
    plan_rows: list[dict[str, object]],
    source: UsageBundle,
    target: UsageBundle,
    steps: np.ndarray,
    threshold: float,
) -> list[dict[str, object]]:
    events = plan_event_by_owner(plan_rows)
    conflicts: list[dict[str, object]] = []
    for row in plan_rows:
        if not bool(row.get("scheduled", False)):
            continue
        owner = int(row["owner"])
        new = target.topo.right_by_owner.get(owner)
        if new is None:
            continue
        blockers = dynamic_blockers_for_window(
            source=source,
            target=target,
            event_by_owner=events,
            owner=owner,
            new=new,
            steps=steps,
            setup_start=int(row["plan_start"]),
            setup_end=int(row["plan_end"]),
            threshold=float(threshold),
        )
        for blocker in blockers:
            payload = dict(row)
            payload.update(blocker)
            conflicts.append(payload)
    return conflicts


def plan_with_exemption(
    *,
    source: UsageBundle,
    target: UsageBundle,
    steps: np.ndarray,
    switch_step: int,
    return_switch_step: int,
    lst: int,
    threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    plan_rows: list[dict[str, object]] = []
    exempt_rows: list[dict[str, object]] = []

    switch_idx = int(np.where(steps == int(switch_step))[0][0])
    switch_idx_end = min(len(steps), switch_idx + max(2, int(3600 / max(1, int(steps[1] - steps[0])))))
    changes = one_way.find_right_link_changes(
        source.topo,
        target.topo,
        switch_idx=switch_idx,
        switch_idx_end=switch_idx_end,
    )

    for change in changes:
        old = change.old
        new = change.new
        owner = int(change.owner)
        if new is None:
            continue

        intervals = working_intervals(
            steps=steps,
            usage=target.any_usage,
            edge_idx=int(new.edge_idx),
            start_step=int(switch_step),
            end_step=int(return_switch_step),
            threshold=float(threshold),
        )
        if not intervals:
            plan_rows.append(
                {
                    "transition": "switch_056_to_061",
                    "owner": owner,
                    "owner_p": int(owner // G60_CONFIG.N),
                    "owner_y": int(owner % G60_CONFIG.N),
                    "old_link": link_text(old),
                    "old_symbol": "" if old is None else old.symbol,
                    "old_edge_key": edge_key_text(old),
                    "new_link": link_text(new),
                    "new_symbol": new.symbol,
                    "new_edge_key": edge_key_text(new),
                    "scheduled": False,
                    "exempted_first_interval": False,
                    "reason": "new_edge_never_working_in_target_segment",
                }
            )
            continue

        selected_interval_idx = 0
        first_start_row, first_end_row, first_start_step, first_end_step = intervals[0]
        first_setup_start = int(first_start_step) - int(lst)
        first_blockers = blocker_rows_for_window(
            source=source,
            old=old,
            new=new,
            steps=steps,
            setup_start=first_setup_start,
            setup_end=int(first_start_step),
            threshold=float(threshold),
        )
        exempted = False
        reason = "scheduled_before_first_working_interval"
        if int(first_start_step) == int(switch_step) and first_blockers:
            exempted = True
            exempt_rows.append(
                {
                    "transition": "switch_056_to_061",
                    "owner": owner,
                    "old_link": link_text(old),
                    "old_symbol": "" if old is None else old.symbol,
                    "new_link": link_text(new),
                    "new_symbol": new.symbol,
                    "first_interval_start": int(first_start_step),
                    "first_interval_end": int(first_end_step),
                    "first_setup_start": int(first_setup_start),
                    "first_setup_end": int(first_start_step),
                    "blocker_count": int(len(first_blockers)),
                    "blockers": "; ".join(
                        f"{b['blocker_type']}:{b['blocker_link']}:{b['blocker_mode_at_first']}"
                        for b in first_blockers
                    ),
                    "decision": "skip_first_working_interval_and_schedule_next_interval",
                }
            )
            selected_interval_idx = 1
            reason = "exempted_36000_conflict_scheduled_before_next_working_interval"

        if selected_interval_idx >= len(intervals):
            plan_rows.append(
                {
                    "transition": "switch_056_to_061",
                    "owner": owner,
                    "owner_p": int(owner // G60_CONFIG.N),
                    "owner_y": int(owner % G60_CONFIG.N),
                    "old_link": link_text(old),
                    "old_symbol": "" if old is None else old.symbol,
                    "old_edge_key": edge_key_text(old),
                    "new_link": link_text(new),
                    "new_symbol": new.symbol,
                    "new_edge_key": edge_key_text(new),
                    "scheduled": False,
                    "exempted_first_interval": bool(exempted),
                    "reason": "exempted_first_interval_but_no_next_working_interval",
                }
            )
            continue

        start_row, end_row, target_step, target_interval_end = intervals[selected_interval_idx]
        plan_start = int(target_step) - int(lst)
        plan_end = int(target_step)
        blockers = blocker_rows_for_window(
            source=source,
            old=old,
            new=new,
            steps=steps,
            setup_start=int(plan_start),
            setup_end=int(plan_end),
            threshold=float(threshold),
        )
        row = {
            "transition": "switch_056_to_061",
            "owner": owner,
            "owner_p": int(owner // G60_CONFIG.N),
            "owner_y": int(owner % G60_CONFIG.N),
            "old_link": link_text(old),
            "old_symbol": "" if old is None else old.symbol,
            "old_edge_key": edge_key_text(old),
            "new_link": link_text(new),
            "new_symbol": new.symbol,
            "new_edge_key": edge_key_text(new),
            "scheduled": True,
            "exempted_first_interval": bool(exempted),
            "skipped_interval_start": "" if not exempted else int(first_start_step),
            "skipped_interval_end": "" if not exempted else int(first_end_step),
            "target_interval_index": int(selected_interval_idx),
            "target_first_work_step": int(target_step),
            "target_interval_end_step": int(target_interval_end),
            "target_first_work_hop": float(target.hop[int(start_row), int(new.edge_idx)]),
            "target_first_work_delay": float(target.delay[int(start_row), int(new.edge_idx)]),
            "target_first_work_mode": usage_mode_at(
                target,
                row=int(start_row),
                edge_idx=int(new.edge_idx),
                threshold=float(threshold),
            ),
            "plan_start": int(plan_start),
            "plan_end": int(plan_end),
            "static_source_blocker_count": int(len(blockers)),
            "static_source_blockers": "; ".join(
                f"{b['blocker_type']}:{b['blocker_link']}:{b['blocker_mode_at_first']}"
                for b in blockers
            ),
            "reason": reason,
        }
        plan_rows.append(row)

    remaining_conflicts = dynamic_conflicts_after_schedule(
        plan_rows=plan_rows,
        source=source,
        target=target,
        steps=steps,
        threshold=float(threshold),
    )
    conflict_owners = {int(row["owner"]) for row in remaining_conflicts}
    for row in plan_rows:
        row["dynamic_conflict_after_exemption"] = int(row["owner"]) in conflict_owners
    return plan_rows, exempt_rows, remaining_conflicts


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_schedule(path: Path, *, steps: np.ndarray, rows: list[dict[str, object]], switch_step: int, return_switch_step: int) -> None:
    scheduled = [row for row in rows if bool(row.get("scheduled", False))]
    fig, axes = plt.subplots(2, 1, figsize=(15.8, 7.8), dpi=170, sharex=False)
    counts = np.zeros(len(steps), dtype=np.int32)
    for row in scheduled:
        idx = rows_between(steps, int(row["plan_start"]), int(row["plan_end"]))
        counts[idx] += 1

    axes[0].plot(steps / 3600.0, counts, color="#b91c1c", linewidth=1.45)
    axes[0].axvline(float(switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[0].axvline(float(return_switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[0].set_ylabel("building links")
    axes[0].set_title(f"Exempted usage-driven setup plan, max concurrency={int(np.max(counts)) if counts.size else 0}")
    axes[0].grid(True, alpha=0.25, linestyle="--", linewidth=0.55)

    if scheduled:
        starts = np.asarray([int(row["plan_start"]) for row in scheduled], dtype=np.int64)
        owners = np.asarray([int(row["owner"]) for row in scheduled], dtype=np.int32)
        exempted = np.asarray([bool(row.get("exempted_first_interval", False)) for row in scheduled], dtype=bool)
        colors = np.where(exempted, "#F97316", "#2563EB")
        axes[1].scatter(starts / 3600.0, owners, s=12, c=colors, alpha=0.72)
    axes[1].axvline(float(switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[1].axvline(float(return_switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("time (hour)")
    axes[1].set_ylabel("owner node")
    axes[1].grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    steps = np.asarray(list(range(int(args.start), int(args.end) + 1, int(args.stride))), dtype=np.int64)
    if int(args.switch_step) not in set(int(x) for x in steps):
        raise ValueError("--switch-step must be on the sampled step axis")
    if int(args.return_switch_step) not in set(int(x) for x in steps):
        raise ValueError("--return-switch-step must be on the sampled step axis")

    source = usage_bundle(
        int(args.source_motif_id),
        delay_usage_root=Path(args.delay_usage_root),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    target = usage_bundle(
        int(args.target_motif_id),
        delay_usage_root=Path(args.delay_usage_root),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )

    plan_rows, exempt_rows, remaining_conflicts = plan_with_exemption(
        source=source,
        target=target,
        steps=steps,
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
        lst=int(args.lst),
        threshold=float(args.working_threshold),
    )

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT
        / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
        / f"switch{int(args.switch_step)}_{int(args.return_switch_step)}"
        / f"lst{int(args.lst):03d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "exempted_usage_driven_switch_plan.csv", plan_rows)
    write_rows(out_dir / "exempted_links_at_36000.csv", exempt_rows)
    write_rows(out_dir / "remaining_conflicts_after_exemption.csv", remaining_conflicts)
    plot_schedule(
        out_dir / "exempted_usage_driven_switch_plan.png",
        steps=steps,
        rows=plan_rows,
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
    )

    scheduled = [row for row in plan_rows if bool(row.get("scheduled", False))]
    unscheduled = [row for row in plan_rows if not bool(row.get("scheduled", False))]
    counts = np.zeros(len(steps), dtype=np.int32)
    for row in scheduled:
        counts[rows_between(steps, int(row["plan_start"]), int(row["plan_end"]))] += 1
    summary = {
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "source_motif_id": int(args.source_motif_id),
        "target_motif_id": int(args.target_motif_id),
        "switch_step": int(args.switch_step),
        "return_switch_step": int(args.return_switch_step),
        "lst": int(args.lst),
        "working_threshold": float(args.working_threshold),
        "changed_links_with_target": int(len(plan_rows)),
        "scheduled": int(len(scheduled)),
        "unscheduled": int(len(unscheduled)),
        "exempted_36000_links": int(len(exempt_rows)),
        "remaining_conflict_rows": int(len(remaining_conflicts)),
        "remaining_conflict_owners": int(len({int(row['owner']) for row in remaining_conflicts})) if remaining_conflicts else 0,
        "max_concurrency_raw": int(np.max(counts)) if counts.size else 0,
        "outputs": {
            "plan": "exempted_usage_driven_switch_plan.csv",
            "exempted": "exempted_links_at_36000.csv",
            "remaining_conflicts": "remaining_conflicts_after_exemption.csv",
            "plot": "exempted_usage_driven_switch_plan.png",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"out_dir={out_dir}")
    print(
        f"changed_links_with_target={len(plan_rows)} scheduled={len(scheduled)} "
        f"unscheduled={len(unscheduled)} exempted_36000={len(exempt_rows)}"
    )
    print(
        f"remaining_conflict_rows={len(remaining_conflicts)} "
        f"remaining_conflict_owners={summary['remaining_conflict_owners']} "
        f"max_concurrency_raw={summary['max_concurrency_raw']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
