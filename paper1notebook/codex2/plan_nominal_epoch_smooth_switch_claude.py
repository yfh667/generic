from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

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
    UsageStore,
    all_intra_keys,
    build_static_topology,
    edge_key_index,
    find_right_link_changes,
    read_usage_store,
    slice_usage_store,
    union_edge_table,
)


DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\nominal_epoch_smooth_switch_056_061"
)


@dataclass(frozen=True)
class PlanRow:
    event_id: int | None
    owner: int
    old_right: Optional[int]
    new_right: Optional[int]
    old_edge: str
    new_edge: str
    new_symbol: str
    tau: int
    rho_self: Optional[int]
    blocker: Optional[int]
    rho_blocker: Optional[int]
    rho_a: Optional[int]
    eta: Optional[int]
    s_max: Optional[int]
    s_a: Optional[int]
    ready: Optional[int]
    delta_lag: int
    conflict: bool
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nominal-epoch smooth motif switch planner. "
            "Implements the rho/eta model: rho is the earliest port-release "
            "time inferred from m- usage around tau; eta is the first time the "
            "new m+ link is needed after tau."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--source-motif-id", type=int, default=56)
    parser.add_argument("--target-motif-id", type=int, default=61)
    parser.add_argument("--nominal-switch-step", "--tau", dest="tau", type=int, default=36000)
    parser.add_argument("--lst", type=int, default=60)
    parser.add_argument(
        "--delta",
        type=int,
        default=0,
        help="Continuous idle window used when the old link is still working at tau-1. 0 means one time row.",
    )
    parser.add_argument("--usage-root", type=Path, default=DEFAULT_USAGE_ROOT)
    parser.add_argument("--working-threshold", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def edge_text(link) -> str:
    return "" if link is None else str(link.edge_key_text)


def link_right(link) -> int | None:
    return None if link is None else int(link.right)


def usage_values_for_link(store: UsageStore, link) -> np.ndarray:
    if link is None:
        return np.zeros(int(store.steps.size), dtype=np.float32)
    col = store.key_to_col.get(link.edge_key)
    if col is None:
        return np.zeros(int(store.steps.size), dtype=np.float32)
    return np.asarray(store.counts[:, int(col)], dtype=np.float32)


def stride_seconds(steps: np.ndarray) -> int:
    if steps.size <= 1:
        return 1
    stride = int(steps[1] - steps[0])
    if stride <= 0:
        raise ValueError("steps must be strictly increasing")
    return stride


def rho_release_step(
    steps: np.ndarray,
    z_old: np.ndarray,
    *,
    tau: int,
    delta: int,
    threshold: float,
) -> int:
    """Earliest release step for one old link under m- usage.

    Case A: if the link is working at the tau-1 snapshot, scan forward from tau
    and find the first continuous idle window.

    Case B: if the link is not working at tau-1, scan backward to the most
    recent working row and release immediately after it.
    """

    values = np.asarray(z_old, dtype=np.float32)
    if values.size != int(steps.size):
        raise ValueError(f"z_old shape {values.shape} does not match steps {steps.shape}")
    if not np.any(values > float(threshold)):
        return int(steps[0])

    working = values > float(threshold)
    snap_row = int(np.searchsorted(steps, int(tau), side="left")) - 1
    snap_row = max(0, min(snap_row, int(steps.size) - 1))

    if bool(working[snap_row]):
        stride = stride_seconds(steps)
        lag = max(1, int(math.ceil(float(delta) / float(stride)))) if int(delta) > 0 else 1
        tau_row = int(np.searchsorted(steps, int(tau), side="left"))
        tau_row = max(0, min(tau_row, int(steps.size) - 1))
        not_working = (~working).astype(np.int32)
        prefix = np.concatenate(([0], np.cumsum(not_working)))
        last_start = int(steps.size) - lag
        for row in range(tau_row, last_start + 1):
            if int(prefix[row + lag] - prefix[row]) == int(lag):
                return int(steps[row])
        return int(steps[-1])

    prior = np.flatnonzero(working[: snap_row + 1])
    if prior.size == 0:
        return int(steps[0])
    release_row = int(prior[-1]) + 1
    if release_row >= int(steps.size):
        return int(steps[-1])
    return int(steps[release_row])


def eta_first_needed_step(
    steps: np.ndarray,
    z_new: np.ndarray,
    *,
    tau: int,
    threshold: float,
) -> int | None:
    """First step at which a new m+ link is working after the nominal epoch."""

    values = np.asarray(z_new, dtype=np.float32)
    if values.size != int(steps.size):
        raise ValueError(f"z_new shape {values.shape} does not match steps {steps.shape}")
    tau_row = int(np.searchsorted(steps, int(tau), side="left"))
    if tau_row >= int(steps.size):
        return None
    hits = np.flatnonzero(values[tau_row:] > float(threshold))
    if hits.size == 0:
        return None
    return int(steps[tau_row + int(hits[0])])


def find_blocker_owner(r_minus: dict[int, int | None], target_node: int) -> int | None:
    for owner, right in r_minus.items():
        if right is not None and int(right) == int(target_node):
            return int(owner)
    return None


def right_neighbor_dict(store: UsageStore, total_nodes: int) -> dict[int, int | None]:
    out: dict[int, int | None] = {}
    for node in range(int(total_nodes)):
        link = store.topology.right_by_owner.get(int(node))
        out[int(node)] = None if link is None else int(link.right)
    return out


def z_by_owner(store: UsageStore, owner: int) -> np.ndarray:
    return usage_values_for_link(store, store.topology.right_by_owner.get(int(owner)))


def compute_plan_row(
    *,
    owner: int,
    source: UsageStore,
    target: UsageStore,
    r_minus: dict[int, int | None],
    r_plus: dict[int, int | None],
    z_minus: Callable[[int], np.ndarray],
    z_plus: Callable[[int], np.ndarray],
    tau: int,
    lst: int,
    delta: int,
    threshold: float,
) -> PlanRow:
    old_link = source.topology.right_by_owner.get(int(owner))
    new_link = target.topology.right_by_owner.get(int(owner))
    old_right = r_minus.get(int(owner))
    new_right = r_plus.get(int(owner))

    if new_right is None or new_link is None:
        rho_self = None
        if old_link is not None:
            rho_self = rho_release_step(
                source.steps,
                z_minus(int(owner)),
                tau=int(tau),
                delta=int(delta),
                threshold=float(threshold),
            )
        return PlanRow(
            event_id=None,
            owner=int(owner),
            old_right=old_right,
            new_right=None,
            old_edge=edge_text(old_link),
            new_edge="",
            new_symbol="",
            tau=int(tau),
            rho_self=rho_self,
            blocker=None,
            rho_blocker=None,
            rho_a=rho_self,
            eta=None,
            s_max=None,
            s_a=rho_self,
            ready=None,
            delta_lag=0,
            conflict=False,
            reason="target_no_right_link",
        )

    rho_self = rho_release_step(
        source.steps,
        z_minus(int(owner)),
        tau=int(tau),
        delta=int(delta),
        threshold=float(threshold),
    )
    blocker = find_blocker_owner(r_minus, int(new_right))
    if blocker is None:
        rho_blocker = None
        rho_a = int(rho_self)
    else:
        rho_blocker = rho_release_step(
            source.steps,
            z_minus(int(blocker)),
            tau=int(tau),
            delta=int(delta),
            threshold=float(threshold),
        )
        rho_a = max(int(rho_self), int(rho_blocker))

    eta = eta_first_needed_step(
        target.steps,
        z_plus(int(owner)),
        tau=int(tau),
        threshold=float(threshold),
    )
    s_max = None if eta is None else int(eta) - int(lst)

    if eta is None:
        s_a = int(rho_a)
        ready = int(s_a) + int(lst)
        reason = "new_link_never_needed"
        conflict = False
        delta_lag = 0
    elif int(rho_a) <= int(s_max):
        s_a = int(s_max)
        ready = int(s_a) + int(lst)
        reason = "lossless"
        conflict = False
        delta_lag = 0
    else:
        s_a = int(rho_a)
        ready = int(s_a) + int(lst)
        delta_lag = int(ready) - int(eta)
        reason = "conflict"
        conflict = True

    return PlanRow(
        event_id=None,
        owner=int(owner),
        old_right=old_right,
        new_right=new_right,
        old_edge=edge_text(old_link),
        new_edge=edge_text(new_link),
        new_symbol=str(new_link.symbol),
        tau=int(tau),
        rho_self=int(rho_self),
        blocker=blocker,
        rho_blocker=rho_blocker,
        rho_a=int(rho_a),
        eta=eta,
        s_max=s_max,
        s_a=int(s_a),
        ready=int(ready),
        delta_lag=int(delta_lag),
        conflict=bool(conflict),
        reason=reason,
    )


def row_for_csv(row: PlanRow) -> dict[str, object]:
    return {
        "event_id": "" if row.event_id is None else int(row.event_id),
        "owner": int(row.owner),
        "owner_p": int(row.owner // G60_CONFIG.N),
        "owner_y": int(row.owner % G60_CONFIG.N),
        "old_right": "" if row.old_right is None else int(row.old_right),
        "new_right": "" if row.new_right is None else int(row.new_right),
        "old_edge": row.old_edge,
        "new_edge": row.new_edge,
        "new_symbol": row.new_symbol,
        "tau": int(row.tau),
        "rho_self": "" if row.rho_self is None else int(row.rho_self),
        "blocker": "" if row.blocker is None else int(row.blocker),
        "rho_blocker": "" if row.rho_blocker is None else int(row.rho_blocker),
        "rho_a": "" if row.rho_a is None else int(row.rho_a),
        "eta": "" if row.eta is None else int(row.eta),
        "s_max": "" if row.s_max is None else int(row.s_max),
        "s_a": "" if row.s_a is None else int(row.s_a),
        "ready": "" if row.ready is None else int(row.ready),
        "delta_lag": int(row.delta_lag),
        "conflict": int(bool(row.conflict)),
        "reason": row.reason,
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


def write_concurrency(path: Path, steps: np.ndarray, building_mask: np.ndarray) -> np.ndarray:
    counts = np.asarray(building_mask, dtype=bool).sum(axis=1).astype(np.int32)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "hour", "building_count"])
        for step, count in zip(steps, counts):
            writer.writerow([int(step), f"{int(step) / 3600.0:.6f}", int(count)])
    return counts


def plot_concurrency(path: Path, steps: np.ndarray, counts: np.ndarray, *, tau: int) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 4.8), dpi=170)
    ax.plot(steps / 3600.0, counts, color="#b91c1c", linewidth=1.15)
    ax.axvline(float(tau) / 3600.0, color="#111827", linestyle="--", linewidth=0.9, label="nominal switch")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("building links")
    ax.set_title("Nominal-epoch smooth switch: rho/eta model")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_viewer_arrays_from_plan(
    *,
    steps: np.ndarray,
    edge_table,
    source: UsageStore,
    target: UsageStore,
    plan_rows: Sequence[PlanRow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_to_idx = edge_key_index(edge_table)
    intra_keys = all_intra_keys(edge_table)
    n_rows = int(steps.size)
    n_edges = int(edge_table.num_edges)
    active = np.zeros((n_rows, n_edges), dtype=bool)
    building = np.zeros_like(active, dtype=bool)
    values = np.zeros((n_rows, n_edges), dtype=np.float32)

    for key in intra_keys:
        idx = key_to_idx.get(key)
        if idx is not None:
            active[:, int(idx)] = True

    for link in source.topology.right_by_owner.values():
        idx = key_to_idx.get(link.edge_key)
        if idx is None:
            continue
        active[:, int(idx)] = True
        col = source.key_to_col.get(link.edge_key)
        if col is not None:
            values[:, int(idx)] = np.asarray(source.counts[:, int(col)], dtype=np.float32)

    def row_ge(step: int) -> int:
        return int(np.searchsorted(steps, int(step), side="left"))

    def rows_between(start: int, end: int) -> slice:
        left = int(np.searchsorted(steps, int(start), side="left"))
        right = int(np.searchsorted(steps, int(end), side="left"))
        return slice(left, right)

    for plan in plan_rows:
        if plan.s_a is None:
            continue
        start_row = row_ge(int(plan.s_a))
        old_link = source.topology.right_by_owner.get(int(plan.owner))
        if old_link is not None:
            old_idx = key_to_idx.get(old_link.edge_key)
            if old_idx is not None:
                active[start_row:, int(old_idx)] = False

        if plan.new_right is not None:
            blocker = source.topology.left_by_right.get(int(plan.new_right))
            if blocker is not None and int(blocker.owner) != int(plan.owner):
                blocker_idx = key_to_idx.get(blocker.edge_key)
                if blocker_idx is not None:
                    active[start_row:, int(blocker_idx)] = False

    for plan in plan_rows:
        if plan.s_a is None or plan.ready is None or plan.new_right is None:
            continue
        new_link = target.topology.right_by_owner.get(int(plan.owner))
        if new_link is None:
            continue
        new_idx = key_to_idx.get(new_link.edge_key)
        if new_idx is None:
            continue
        building[rows_between(int(plan.s_a), int(plan.ready)), int(new_idx)] = True
        ready_row = row_ge(int(plan.ready))
        active[ready_row:, int(new_idx)] = True
        col = target.key_to_col.get(new_link.edge_key)
        if col is not None:
            values[ready_row:, int(new_idx)] = np.asarray(target.counts[ready_row:, int(col)], dtype=np.float32)

    return active, building, values


def summarize(rows: Sequence[PlanRow], *, lst: int) -> dict[str, object]:
    conflicts = [row for row in rows if row.conflict]
    scheduled = [row for row in rows if row.s_a is not None and row.ready is not None and row.new_right is not None]
    return {
        "lst": int(lst),
        "changed_nodes": int(len(rows)),
        "scheduled_events": int(len(scheduled)),
        "lossless": int(sum(1 for row in rows if row.reason == "lossless")),
        "conflict": int(len(conflicts)),
        "new_link_never_needed": int(sum(1 for row in rows if row.reason == "new_link_never_needed")),
        "target_no_right_link": int(sum(1 for row in rows if row.reason == "target_no_right_link")),
        "max_lag": int(max((row.delta_lag for row in rows), default=0)),
        "mean_lag_over_conflicts": float(np.mean([row.delta_lag for row in conflicts])) if conflicts else 0.0,
    }


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

    r_minus = right_neighbor_dict(source, int(G60_CONFIG.total_sats))
    r_plus = right_neighbor_dict(target, int(G60_CONFIG.total_sats))
    changes = find_right_link_changes(source.topology, target.topology)
    changed_owners = [int(change.owner) for change in changes]

    rows: list[PlanRow] = []
    for owner in changed_owners:
        rows.append(
            compute_plan_row(
                owner=int(owner),
                source=source,
                target=target,
                r_minus=r_minus,
                r_plus=r_plus,
                z_minus=lambda node, _source=source: z_by_owner(_source, int(node)),
                z_plus=lambda node, _target=target: z_by_owner(_target, int(node)),
                tau=int(args.tau),
                lst=int(args.lst),
                delta=int(args.delta),
                threshold=float(args.working_threshold),
            )
        )

    event_id = 1
    with_ids: list[PlanRow] = []
    for row in sorted(rows, key=lambda item: (10**12 if item.s_a is None else int(item.s_a), int(item.owner))):
        if row.s_a is None or row.ready is None or row.new_right is None:
            with_ids.append(row)
            continue
        with_ids.append(
            PlanRow(
                event_id=event_id,
                owner=row.owner,
                old_right=row.old_right,
                new_right=row.new_right,
                old_edge=row.old_edge,
                new_edge=row.new_edge,
                new_symbol=row.new_symbol,
                tau=row.tau,
                rho_self=row.rho_self,
                blocker=row.blocker,
                rho_blocker=row.rho_blocker,
                rho_a=row.rho_a,
                eta=row.eta,
                s_max=row.s_max,
                s_a=row.s_a,
                ready=row.ready,
                delta_lag=row.delta_lag,
                conflict=row.conflict,
                reason=row.reason,
            )
        )
        event_id += 1

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT
        / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
        / f"tau{int(args.tau)}_lst{int(args.lst):03d}_delta{int(args.delta):03d}_rho_eta"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    write_rows(out_dir / "nominal_epoch_smooth_switch_plan.csv", [row_for_csv(row) for row in with_ids])

    edge_table = union_edge_table(source.topology, target.topology)
    active_mask, building_mask, values = build_viewer_arrays_from_plan(
        steps=source.steps,
        edge_table=edge_table,
        source=source,
        target=target,
        plan_rows=with_ids,
    )
    write_edges_csv(edge_table, out_dir / "union_edges.csv")
    np.save(out_dir / "steps.npy", source.steps.astype(np.int64))
    np.save(out_dir / "edge_active_mask.npy", active_mask.astype(bool))
    np.save(out_dir / "edge_building_mask.npy", building_mask.astype(bool))
    np.save(out_dir / "edge_usage_values.npy", values.astype(np.float32))

    counts = write_concurrency(out_dir / "building_concurrency_by_step.csv", source.steps, building_mask)
    np.save(out_dir / "building_concurrency.npy", counts)
    plot_concurrency(out_dir / "building_concurrency.png", source.steps, counts, tau=int(args.tau))

    summary = {
        "source_motif_id": int(args.source_motif_id),
        "source_motif": source.topology.spec.motif,
        "target_motif_id": int(args.target_motif_id),
        "target_motif": target.topology.spec.motif,
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "tau": int(args.tau),
        "lst": int(args.lst),
        "delta": int(args.delta),
        "working_threshold": float(args.working_threshold),
        **summarize(with_ids, lst=int(args.lst)),
        "max_building_concurrency": int(counts.max()) if counts.size else 0,
        "first_building_step": None if not np.any(counts > 0) else int(source.steps[int(np.flatnonzero(counts > 0)[0])]),
        "last_building_step": None if not np.any(counts > 0) else int(source.steps[int(np.flatnonzero(counts > 0)[-1])]),
        "files": {
            "plan_csv": "nominal_epoch_smooth_switch_plan.csv",
            "steps": "steps.npy",
            "union_edges": "union_edges.csv",
            "edge_active_mask": "edge_active_mask.npy",
            "edge_building_mask": "edge_building_mask.npy",
            "edge_usage_values": "edge_usage_values.npy",
            "building_concurrency_csv": "building_concurrency_by_step.csv",
            "building_concurrency_plot": "building_concurrency.png",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"out_dir={out_dir}", flush=True)
    print(
        f"changed={summary['changed_nodes']} scheduled={summary['scheduled_events']} "
        f"lossless={summary['lossless']} conflict={summary['conflict']} "
        f"max_lag={summary['max_lag']} max_building={summary['max_building_concurrency']} "
        f"first_building={summary['first_building_step']} last_building={summary['last_building_step']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
