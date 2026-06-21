from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_motif0056_to_0061_link_setup as one_way  # noqa: E402
import plan_usage_driven_switch_with_36000_exemption as base  # noqa: E402
from build_ideal_splice_right_link_state_0056_0061 import DEFAULT_DELAY_USAGE_ROOT  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402


DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_switch_iterative_exemption_056_061_china_europe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan motif000056 -> motif000061 right-link setup with iterative usage exemptions. "
            "Seed exemptions are links blocked at 36000s; propagated exemptions skip later "
            "working intervals that are blocked by still-working active links."
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
    parser.add_argument("--max-iterations", type=int, default=100)
    return parser.parse_args()


def build_contexts(
    *,
    source: base.UsageBundle,
    target: base.UsageBundle,
    steps: np.ndarray,
    switch_step: int,
    return_switch_step: int,
    lst: int,
    threshold: float,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    switch_idx = int(np.where(steps == int(switch_step))[0][0])
    switch_idx_end = min(len(steps), switch_idx + max(2, int(3600 / max(1, int(steps[1] - steps[0])))))
    changes = one_way.find_right_link_changes(
        source.topo,
        target.topo,
        switch_idx=switch_idx,
        switch_idx_end=switch_idx_end,
    )
    contexts: dict[int, dict[str, object]] = {}
    seed_exemptions: list[dict[str, object]] = []
    for change in changes:
        old = change.old
        new = change.new
        if new is None:
            continue
        owner = int(change.owner)
        intervals = base.working_intervals(
            steps=steps,
            usage=target.any_usage,
            edge_idx=int(new.edge_idx),
            start_step=int(switch_step),
            end_step=int(return_switch_step),
            threshold=float(threshold),
        )
        if not intervals:
            contexts[owner] = {
                "owner": owner,
                "old": old,
                "new": new,
                "intervals": [],
                "selected_idx": None,
                "seed_exempted": False,
                "skipped": [],
                "unscheduled_reason": "new_edge_never_working_in_target_segment",
            }
            continue
        selected_idx = 0
        seed_exempted = False
        first_start_row, first_end_row, first_start_step, first_end_step = intervals[0]
        first_setup_start = int(first_start_step) - int(lst)
        first_blockers = base.blocker_rows_for_window(
            source=source,
            old=old,
            new=new,
            steps=steps,
            setup_start=first_setup_start,
            setup_end=int(first_start_step),
            threshold=float(threshold),
        )
        if int(first_start_step) == int(switch_step):
            # The first target use is exactly at the splice point. With 60s
            # sampled usage, a short LST window such as [35990, 36000) has no
            # sampled row inside it, so also inspect the splice sample itself.
            # This is the explicit "36000s exemption" rule: if the old working
            # link still owns either required right/left port at the splice
            # sample, the new edge skips this first target-use interval.
            first_blockers = first_blockers or blockers_at_step(
                source=source,
                old=old,
                new=new,
                steps=steps,
                step=int(switch_step),
                threshold=float(threshold),
            )
        skipped: list[dict[str, object]] = []
        if int(first_start_step) == int(switch_step) and first_blockers:
            seed_exempted = True
            selected_idx = 1
            skipped.append(
                {
                    "kind": "seed_36000",
                    "interval_index": 0,
                    "interval_start": int(first_start_step),
                    "interval_end": int(first_end_step),
                    "blockers": "; ".join(
                        f"{b['blocker_type']}:{b['blocker_link']}:{b['blocker_mode_at_first']}"
                        for b in first_blockers
                    ),
                }
            )
            seed_exemptions.append(
                {
                    "owner": owner,
                    "old_link": base.link_text(old),
                    "old_symbol": "" if old is None else old.symbol,
                    "new_link": base.link_text(new),
                    "new_symbol": new.symbol,
                    "skipped_interval_start": int(first_start_step),
                    "skipped_interval_end": int(first_end_step),
                    "blocker_count": int(len(first_blockers)),
                    "blockers": skipped[-1]["blockers"],
                }
            )
        contexts[owner] = {
            "owner": owner,
            "old": old,
            "new": new,
            "intervals": intervals,
            "selected_idx": selected_idx if selected_idx < len(intervals) else None,
            "seed_exempted": seed_exempted,
            "skipped": skipped,
            "unscheduled_reason": "" if selected_idx < len(intervals) else "seed_exempted_but_no_next_interval",
        }
    return contexts, seed_exemptions


def blockers_at_step(
    *,
    source: base.UsageBundle,
    old: object | None,
    new: object,
    steps: np.ndarray,
    step: int,
    threshold: float,
) -> list[dict[str, object]]:
    rows = np.flatnonzero(steps == int(step))
    if rows.size == 0:
        return []
    row = int(rows[0])
    blockers: list[dict[str, object]] = []
    candidates = [
        ("owner_right_old_at_switch", old),
        ("target_left_old_at_switch", source.topo.left_by_right.get(int(new.right))),
    ]
    for blocker_type, link in candidates:
        if link is None:
            continue
        hop = float(source.hop[row, int(link.edge_idx)])
        delay = float(source.delay[row, int(link.edge_idx)])
        if max(hop, delay) <= float(threshold):
            continue
        blockers.append(
            {
                "blocker_type": str(blocker_type),
                "blocker_state": "source",
                "blocker_link": base.link_text(link),
                "blocker_symbol": str(link.symbol),
                "blocker_edge_key": base.edge_key_text(link),
                "blocker_owner": int(link.owner),
                "blocker_right": int(link.right),
                "blocker_first_step_inside_setup": int(step),
                "blocker_max_hop_inside_setup": float(hop),
                "blocker_max_delay_inside_setup": float(delay),
                "blocker_mode_at_first": base.usage_mode_at(
                    source,
                    row=row,
                    edge_idx=int(link.edge_idx),
                    threshold=float(threshold),
                ),
            }
        )
    return blockers


def make_plan_rows(
    *,
    contexts: dict[int, dict[str, object]],
    target: base.UsageBundle,
    lst: int,
    threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for owner in sorted(contexts):
        ctx = contexts[owner]
        old = ctx["old"]
        new = ctx["new"]
        assert new is not None
        selected_idx = ctx["selected_idx"]
        common = {
            "transition": "switch_056_to_061",
            "owner": int(owner),
            "owner_p": int(owner // G60_CONFIG.N),
            "owner_y": int(owner % G60_CONFIG.N),
            "old_link": base.link_text(old),
            "old_symbol": "" if old is None else old.symbol,
            "old_edge_key": base.edge_key_text(old),
            "new_link": base.link_text(new),
            "new_symbol": new.symbol,
            "new_edge_key": base.edge_key_text(new),
            "seed_exempted_36000": bool(ctx["seed_exempted"]),
            "total_skipped_intervals": int(len(ctx["skipped"])),
            "skipped_intervals": json.dumps(ctx["skipped"], ensure_ascii=False),
        }
        if selected_idx is None:
            row = dict(common)
            row.update(
                {
                    "scheduled": False,
                    "target_interval_index": "",
                    "target_first_work_step": "",
                    "target_interval_end_step": "",
                    "plan_start": "",
                    "plan_end": "",
                    "reason": str(ctx.get("unscheduled_reason", "")),
                }
            )
            rows.append(row)
            continue
        intervals = ctx["intervals"]
        start_row, end_row, target_step, target_interval_end = intervals[int(selected_idx)]
        row = dict(common)
        row.update(
            {
                "scheduled": True,
                "target_interval_index": int(selected_idx),
                "target_first_work_step": int(target_step),
                "target_interval_end_step": int(target_interval_end),
                "target_first_work_hop": float(target.hop[int(start_row), int(new.edge_idx)]),
                "target_first_work_delay": float(target.delay[int(start_row), int(new.edge_idx)]),
                "target_first_work_mode": base.usage_mode_at(
                    target,
                    row=int(start_row),
                    edge_idx=int(new.edge_idx),
                    threshold=float(threshold),
                ),
                "plan_start": int(target_step) - int(lst),
                "plan_end": int(target_step),
                "reason": "iterative_exemption_schedule",
            }
        )
        rows.append(row)
    return rows


def apply_one_conflict_round(
    *,
    contexts: dict[int, dict[str, object]],
    conflicts: list[dict[str, object]],
) -> int:
    changed = 0
    for owner in sorted({int(row["owner"]) for row in conflicts}):
        ctx = contexts.get(owner)
        if ctx is None or ctx["selected_idx"] is None:
            continue
        intervals = ctx["intervals"]
        current_idx = int(ctx["selected_idx"])
        if current_idx + 1 >= len(intervals):
            ctx["selected_idx"] = None
            ctx["unscheduled_reason"] = "blocked_interval_but_no_later_working_interval"
            changed += 1
            continue
        start_row, end_row, start_step, end_step = intervals[current_idx]
        owner_conflicts = [row for row in conflicts if int(row["owner"]) == owner]
        ctx["skipped"].append(
            {
                "kind": "propagated",
                "interval_index": int(current_idx),
                "interval_start": int(start_step),
                "interval_end": int(end_step),
                "blockers": "; ".join(
                    f"{row['blocker_type']}:{row['blocker_link']}:{row['blocker_mode_at_first']}"
                    for row in owner_conflicts
                ),
            }
        )
        ctx["selected_idx"] = int(current_idx) + 1
        changed += 1
    return changed


def main() -> int:
    args = parse_args()
    steps = np.asarray(list(range(int(args.start), int(args.end) + 1, int(args.stride))), dtype=np.int64)
    source = base.usage_bundle(
        int(args.source_motif_id),
        delay_usage_root=Path(args.delay_usage_root),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    target = base.usage_bundle(
        int(args.target_motif_id),
        delay_usage_root=Path(args.delay_usage_root),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    contexts, seed_exemptions = build_contexts(
        source=source,
        target=target,
        steps=steps,
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
        lst=int(args.lst),
        threshold=float(args.working_threshold),
    )

    iteration_rows: list[dict[str, object]] = []
    final_conflicts: list[dict[str, object]] = []
    for iteration in range(int(args.max_iterations) + 1):
        plan_rows = make_plan_rows(contexts=contexts, target=target, lst=int(args.lst), threshold=float(args.working_threshold))
        conflicts = base.dynamic_conflicts_after_schedule(
            plan_rows=plan_rows,
            source=source,
            target=target,
            steps=steps,
            threshold=float(args.working_threshold),
        )
        iteration_rows.append(
            {
                "iteration": int(iteration),
                "scheduled": int(sum(1 for row in plan_rows if bool(row.get("scheduled", False)))),
                "unscheduled": int(sum(1 for row in plan_rows if not bool(row.get("scheduled", False)))),
                "conflict_rows": int(len(conflicts)),
                "conflict_owners": int(len({int(row["owner"]) for row in conflicts})) if conflicts else 0,
            }
        )
        if not conflicts:
            final_conflicts = []
            break
        changed = apply_one_conflict_round(contexts=contexts, conflicts=conflicts)
        final_conflicts = conflicts
        if changed == 0:
            break
    else:
        plan_rows = make_plan_rows(contexts=contexts, target=target, lst=int(args.lst), threshold=float(args.working_threshold))
        final_conflicts = base.dynamic_conflicts_after_schedule(
            plan_rows=plan_rows,
            source=source,
            target=target,
            steps=steps,
            threshold=float(args.working_threshold),
        )

    plan_rows = make_plan_rows(contexts=contexts, target=target, lst=int(args.lst), threshold=float(args.working_threshold))
    final_conflicts = base.dynamic_conflicts_after_schedule(
        plan_rows=plan_rows,
        source=source,
        target=target,
        steps=steps,
        threshold=float(args.working_threshold),
    )

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT
        / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
        / f"switch{int(args.switch_step)}_{int(args.return_switch_step)}"
        / f"lst{int(args.lst):03d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    base.write_rows(out_dir / "iterative_exempted_switch_plan.csv", plan_rows)
    base.write_rows(out_dir / "seed_exemptions_at_36000.csv", seed_exemptions)
    base.write_rows(out_dir / "remaining_conflicts.csv", final_conflicts)
    base.write_rows(out_dir / "iteration_summary.csv", iteration_rows)
    base.plot_schedule(
        out_dir / "iterative_exempted_switch_plan.png",
        steps=steps,
        rows=plan_rows,
        switch_step=int(args.switch_step),
        return_switch_step=int(args.return_switch_step),
    )
    scheduled = [row for row in plan_rows if bool(row.get("scheduled", False))]
    unscheduled = [row for row in plan_rows if not bool(row.get("scheduled", False))]
    propagated = [
        ctx for ctx in contexts.values()
        if int(len(ctx["skipped"])) > (1 if bool(ctx["seed_exempted"]) else 0)
    ]
    counts = np.zeros(len(steps), dtype=np.int32)
    for row in scheduled:
        counts[base.rows_between(steps, int(row["plan_start"]), int(row["plan_end"]))] += 1
    summary = {
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "switch_step": int(args.switch_step),
        "return_switch_step": int(args.return_switch_step),
        "lst": int(args.lst),
        "changed_links_with_target": int(len(plan_rows)),
        "scheduled": int(len(scheduled)),
        "unscheduled": int(len(unscheduled)),
        "seed_exemptions_36000": int(len(seed_exemptions)),
        "propagated_exemption_owners": int(len(propagated)),
        "remaining_conflict_rows": int(len(final_conflicts)),
        "remaining_conflict_owners": int(len({int(row['owner']) for row in final_conflicts})) if final_conflicts else 0,
        "max_concurrency_raw": int(np.max(counts)) if counts.size else 0,
        "outputs": {
            "plan": "iterative_exempted_switch_plan.csv",
            "seed_exemptions": "seed_exemptions_at_36000.csv",
            "remaining_conflicts": "remaining_conflicts.csv",
            "iteration_summary": "iteration_summary.csv",
            "plot": "iterative_exempted_switch_plan.png",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}")
    print(
        f"scheduled={len(scheduled)} unscheduled={len(unscheduled)} "
        f"seed_exemptions={len(seed_exemptions)} propagated_owners={len(propagated)}"
    )
    print(
        f"remaining_conflict_rows={len(final_conflicts)} "
        f"remaining_conflict_owners={summary['remaining_conflict_owners']} "
        f"max_concurrency_raw={summary['max_concurrency_raw']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
