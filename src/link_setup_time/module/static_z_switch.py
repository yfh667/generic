from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable, Mapping, Optional, Sequence

import numpy as np


RightMap = Mapping[int, Optional[int]]
ZSource = Callable[[int], np.ndarray] | Mapping[int, np.ndarray]


@dataclass(frozen=True)
class PlanRow:
    owner: int
    old_right: Optional[int]
    new_right: Optional[int]
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


@dataclass(frozen=True)
class TopologyProfile:
    """One static topology and static per-owner right-link usage series."""

    name: str
    right: RightMap
    z: ZSource


@dataclass(frozen=True)
class SwitchEvent:
    """Nominal switch from current topology to ``target`` at ``tau``."""

    tau: int
    target: str


@dataclass(frozen=True)
class NodeTransition:
    switch_index: int
    from_name: str
    to_name: str
    owner: int
    old_right: Optional[int]
    new_right: Optional[int]
    tau: int
    start: Optional[int]
    ready: Optional[int]
    conflict: bool
    delta_lag: int
    reason: str


@dataclass(frozen=True)
class AppliedSwitch:
    index: int
    from_name: str
    to_name: str
    tau: int
    rows: list[PlanRow]
    summary: dict


@dataclass(frozen=True)
class ChainPlan:
    initial: str
    final: str
    switches: list[AppliedSwitch]
    transitions: list[NodeTransition]
    warnings: list[str]


def _validate_steps(steps: Sequence[int] | np.ndarray) -> np.ndarray:
    out = np.asarray(steps, dtype=np.int64)
    if out.ndim != 1 or out.size == 0:
        raise ValueError("steps must be a non-empty 1-D sequence")
    if out.size > 1 and bool(np.any(np.diff(out) <= 0)):
        raise ValueError("steps must be strictly increasing")
    return out


def _stride(steps: np.ndarray) -> int:
    if steps.size <= 1:
        return 1
    stride = int(steps[1] - steps[0])
    if stride <= 0:
        raise ValueError("steps must be strictly increasing")
    return stride


def z_callable(z_source: ZSource, steps: Sequence[int] | np.ndarray) -> Callable[[int], np.ndarray]:
    steps_array = _validate_steps(steps)
    if callable(z_source):
        return z_source

    zero = np.zeros(int(steps_array.size), dtype=np.float32)

    def _z(owner: int) -> np.ndarray:
        values = z_source.get(int(owner), zero)  # type: ignore[union-attr]
        out = np.asarray(values, dtype=np.float32)
        if out.shape != zero.shape:
            raise ValueError(f"z[{owner}] shape {out.shape} != steps shape {zero.shape}")
        return out

    return _z


def rho_self(
    steps: Sequence[int] | np.ndarray,
    z: np.ndarray,
    *,
    tau: int,
    delta: int = 0,
    threshold: float = 0.0,
) -> int:
    """Earliest release time for the old right link using the tau-1 snapshot rule."""

    steps_array = _validate_steps(steps)
    values = np.asarray(z, dtype=np.float32)
    if values.shape != steps_array.shape:
        raise ValueError(f"z shape {values.shape} != steps shape {steps_array.shape}")
    if not np.any(values > float(threshold)):
        return int(steps_array[0])

    working = values > float(threshold)
    snap_row = int(np.searchsorted(steps_array, int(tau), side="left")) - 1
    snap_row = max(0, min(snap_row, int(steps_array.size) - 1))

    if bool(working[snap_row]):
        stride = _stride(steps_array)
        lag_rows = max(1, int(np.ceil(float(delta) / float(stride)))) if int(delta) > 0 else 1
        tau_row = int(np.searchsorted(steps_array, int(tau), side="left"))
        tau_row = max(0, min(tau_row, int(steps_array.size) - 1))
        not_working = (~working).astype(np.int32)
        prefix = np.concatenate(([0], np.cumsum(not_working)))
        last_start = int(steps_array.size) - int(lag_rows)
        for row in range(tau_row, last_start + 1):
            if int(prefix[row + lag_rows] - prefix[row]) == int(lag_rows):
                return int(steps_array[row])
        return int(steps_array[-1])

    prior = np.flatnonzero(working[: snap_row + 1])
    if prior.size == 0:
        return int(steps_array[0])
    release_row = int(prior[-1]) + 1
    if release_row >= int(steps_array.size):
        return int(steps_array[-1])
    return int(steps_array[release_row])


def eta_plus(
    steps: Sequence[int] | np.ndarray,
    z_new: np.ndarray,
    *,
    tau: int,
    threshold: float = 0.0,
) -> Optional[int]:
    """First time the new right link is used in the target static topology after tau."""

    steps_array = _validate_steps(steps)
    values = np.asarray(z_new, dtype=np.float32)
    if values.shape != steps_array.shape:
        raise ValueError(f"z_new shape {values.shape} != steps shape {steps_array.shape}")
    tau_row = int(np.searchsorted(steps_array, int(tau), side="left"))
    if tau_row >= int(steps_array.size):
        return None
    hits = np.flatnonzero(values[tau_row:] > float(threshold))
    if hits.size == 0:
        return None
    return int(steps_array[tau_row + int(hits[0])])


def find_blocker(r_minus: RightMap, target_node: int) -> Optional[int]:
    """Find owner b in m- whose right neighbor occupies target_node's left port."""

    for owner, right in r_minus.items():
        if right is not None and int(right) == int(target_node):
            return int(owner)
    return None


def plan_one_static_z(
    owner: int,
    *,
    steps: Sequence[int] | np.ndarray,
    r_minus: RightMap,
    r_plus: RightMap,
    z_minus: Callable[[int], np.ndarray],
    z_plus: Callable[[int], np.ndarray],
    tau: int,
    lst: int,
    delta: int = 0,
    threshold: float = 0.0,
) -> PlanRow:
    """Plan one owner's right-link transition under the static-z approximation."""

    steps_array = _validate_steps(steps)
    owner = int(owner)
    old_right = r_minus.get(owner)
    new_right = r_plus.get(owner)

    if old_right is None:
        release_self = int(steps_array[0])
    else:
        release_self = rho_self(
            steps_array,
            z_minus(owner),
            tau=int(tau),
            delta=int(delta),
            threshold=float(threshold),
        )

    if new_right is None:
        return PlanRow(
            owner=owner,
            old_right=old_right,
            new_right=None,
            tau=int(tau),
            rho_self=release_self,
            blocker=None,
            rho_blocker=None,
            rho_a=release_self,
            eta=None,
            s_max=None,
            s_a=release_self,
            ready=release_self,
            delta_lag=0,
            conflict=False,
            reason="remove_only",
        )

    blocker = find_blocker(r_minus, int(new_right))
    if blocker is None:
        release_blocker = None
        rho_a = int(release_self)
    else:
        release_blocker = rho_self(
            steps_array,
            z_minus(int(blocker)),
            tau=int(tau),
            delta=int(delta),
            threshold=float(threshold),
        )
        rho_a = max(int(release_self), int(release_blocker))

    eta = eta_plus(steps_array, z_plus(owner), tau=int(tau), threshold=float(threshold))
    s_max = None if eta is None else int(eta) - int(lst)

    if eta is None:
        s_a = int(rho_a)
        return PlanRow(
            owner=owner,
            old_right=old_right,
            new_right=new_right,
            tau=int(tau),
            rho_self=release_self,
            blocker=blocker,
            rho_blocker=release_blocker,
            rho_a=rho_a,
            eta=None,
            s_max=None,
            s_a=s_a,
            ready=s_a + int(lst),
            delta_lag=0,
            conflict=False,
            reason="new_link_never_needed",
        )

    if int(rho_a) <= int(s_max):
        s_a = int(s_max)
        return PlanRow(
            owner=owner,
            old_right=old_right,
            new_right=new_right,
            tau=int(tau),
            rho_self=release_self,
            blocker=blocker,
            rho_blocker=release_blocker,
            rho_a=rho_a,
            eta=eta,
            s_max=s_max,
            s_a=s_a,
            ready=s_a + int(lst),
            delta_lag=0,
            conflict=False,
            reason="lossless",
        )

    s_a = int(rho_a)
    ready = s_a + int(lst)
    return PlanRow(
        owner=owner,
        old_right=old_right,
        new_right=new_right,
        tau=int(tau),
        rho_self=release_self,
        blocker=blocker,
        rho_blocker=release_blocker,
        rho_a=rho_a,
        eta=eta,
        s_max=s_max,
        s_a=s_a,
        ready=ready,
        delta_lag=int(max(0, ready - int(eta))),
        conflict=ready > int(eta),
        reason="conflict",
    )


def plan_all_static_z(
    *,
    steps: Sequence[int] | np.ndarray,
    r_minus: RightMap,
    r_plus: RightMap,
    z_minus: Callable[[int], np.ndarray],
    z_plus: Callable[[int], np.ndarray],
    tau: int,
    lst: int,
    delta: int = 0,
    threshold: float = 0.0,
) -> list[PlanRow]:
    """Plan all owners whose right neighbor differs between m- and m+."""

    changed = [
        int(owner)
        for owner in set(int(x) for x in r_minus.keys()) | set(int(x) for x in r_plus.keys())
        if r_minus.get(int(owner)) != r_plus.get(int(owner))
    ]
    return [
        plan_one_static_z(
            owner,
            steps=steps,
            r_minus=r_minus,
            r_plus=r_plus,
            z_minus=z_minus,
            z_plus=z_plus,
            tau=int(tau),
            lst=int(lst),
            delta=int(delta),
            threshold=float(threshold),
        )
        for owner in sorted(changed)
    ]


def reschedule_plan_rows(
    rows: Sequence[PlanRow],
    *,
    lst: int,
    strategy: str = "latest",
    tau: Optional[int] = None,
) -> list[PlanRow]:
    """Choose concrete build starts from each row's feasible window."""

    if strategy not in {"latest", "earliest", "near_tau"}:
        raise ValueError("strategy must be one of: latest, earliest, near_tau")

    out: list[PlanRow] = []
    for row in rows:
        if row.rho_a is None or row.s_a is None:
            out.append(row)
            continue

        if row.new_right is None or row.reason == "remove_only":
            s_a = int(row.rho_a)
            out.append(replace(row, s_a=s_a, ready=s_a, delta_lag=0, conflict=False))
            continue

        rho_a = int(row.rho_a)
        if row.eta is None or row.s_max is None:
            if strategy == "earliest" or tau is None:
                s_a = rho_a
            else:
                s_a = max(rho_a, int(tau))
            out.append(replace(row, s_a=s_a, ready=s_a + int(lst), delta_lag=0, conflict=False))
            continue

        eta = int(row.eta)
        s_max = int(row.s_max)
        if rho_a <= s_max:
            if strategy == "earliest":
                s_a = rho_a
            elif strategy == "latest":
                s_a = s_max
            else:
                anchor = int(tau) if tau is not None else s_max
                s_a = min(max(anchor, rho_a), s_max)
            out.append(
                replace(
                    row,
                    s_a=s_a,
                    ready=s_a + int(lst),
                    delta_lag=0,
                    conflict=False,
                    reason="lossless",
                )
            )
            continue

        s_a = rho_a
        ready = s_a + int(lst)
        out.append(
            replace(
                row,
                s_a=s_a,
                ready=ready,
                delta_lag=int(max(0, ready - eta)),
                conflict=ready > eta,
                reason="conflict",
            )
        )
    return out


def summarize_plan_rows(rows: Sequence[PlanRow], *, lst: int) -> dict:
    conflicts = [row for row in rows if row.conflict]
    return {
        "lst": int(lst),
        "changed_nodes": int(len(rows)),
        "lossless": int(sum(1 for row in rows if row.reason == "lossless")),
        "conflict": int(len(conflicts)),
        "new_link_never_needed": int(sum(1 for row in rows if row.reason == "new_link_never_needed")),
        "remove_only": int(sum(1 for row in rows if row.reason == "remove_only")),
        "target_no_right_link": int(sum(1 for row in rows if row.new_right is None)),
        "max_lag": int(max((row.delta_lag for row in rows), default=0)),
        "mean_lag_over_conflicts": float(np.mean([row.delta_lag for row in conflicts])) if conflicts else 0.0,
    }


def rows_to_transitions(
    rows: Sequence[PlanRow],
    *,
    switch_index: int,
    from_name: str,
    to_name: str,
) -> list[NodeTransition]:
    return [
        NodeTransition(
            switch_index=int(switch_index),
            from_name=str(from_name),
            to_name=str(to_name),
            owner=int(row.owner),
            old_right=row.old_right,
            new_right=row.new_right,
            tau=int(row.tau),
            start=None if row.s_a is None else int(row.s_a),
            ready=None if row.ready is None else int(row.ready),
            conflict=bool(row.conflict),
            delta_lag=int(row.delta_lag),
            reason=str(row.reason),
        )
        for row in rows
    ]


def apply_switch_static_z(
    *,
    steps: Sequence[int] | np.ndarray,
    from_profile: TopologyProfile,
    to_profile: TopologyProfile,
    tau: int,
    lst: int,
    delta: int = 0,
    threshold: float = 0.0,
    strategy: str = "latest",
    switch_index: int = 1,
) -> AppliedSwitch:
    """Apply one nominal static-z switch and return its plan rows."""

    steps_array = _validate_steps(steps)
    rows = plan_all_static_z(
        steps=steps_array,
        r_minus=from_profile.right,
        r_plus=to_profile.right,
        z_minus=z_callable(from_profile.z, steps_array),
        z_plus=z_callable(to_profile.z, steps_array),
        tau=int(tau),
        lst=int(lst),
        delta=int(delta),
        threshold=float(threshold),
    )
    rows = reschedule_plan_rows(rows, lst=int(lst), strategy=strategy, tau=int(tau))
    summary = summarize_plan_rows(rows, lst=int(lst))
    summary.update(
        {
            "switch_index": int(switch_index),
            "from": str(from_profile.name),
            "to": str(to_profile.name),
            "tau": int(tau),
            "strategy": str(strategy),
        }
    )
    return AppliedSwitch(
        index=int(switch_index),
        from_name=str(from_profile.name),
        to_name=str(to_profile.name),
        tau=int(tau),
        rows=rows,
        summary=summary,
    )


def apply_switch_chain_static_z(
    *,
    steps: Sequence[int] | np.ndarray,
    profiles: Mapping[str, TopologyProfile],
    initial: str,
    events: Sequence[SwitchEvent],
    lst: int,
    delta: int = 0,
    threshold: float = 0.0,
    strategy: str = "latest",
    enforce_previous_ready: bool = True,
) -> ChainPlan:
    """Apply a sequence of nominal static-z switches in tau order."""

    steps_array = _validate_steps(steps)
    if initial not in profiles:
        raise KeyError(f"initial profile not found: {initial}")

    current = str(initial)
    switches: list[AppliedSwitch] = []
    transitions: list[NodeTransition] = []
    warnings: list[str] = []
    last_ready_by_owner: dict[int, int] = {}

    ordered_events = sorted(list(events), key=lambda event: int(event.tau))
    for index, event in enumerate(ordered_events, start=1):
        if event.target not in profiles:
            raise KeyError(f"target profile not found: {event.target}")
        applied = apply_switch_static_z(
            steps=steps_array,
            from_profile=profiles[current],
            to_profile=profiles[str(event.target)],
            tau=int(event.tau),
            lst=int(lst),
            delta=int(delta),
            threshold=float(threshold),
            strategy=strategy,
            switch_index=index,
        )
        adjusted_rows: list[PlanRow] = []
        for row in applied.rows:
            adjusted = row
            if row.s_a is None:
                adjusted_rows.append(adjusted)
                continue
            prev_ready = last_ready_by_owner.get(int(row.owner))
            if prev_ready is not None and int(row.s_a) < int(prev_ready):
                if bool(enforce_previous_ready):
                    new_start = int(prev_ready)
                    if row.new_right is None:
                        new_ready = new_start
                    else:
                        new_ready = new_start + int(lst)
                    lag = 0 if row.eta is None else int(max(0, new_ready - int(row.eta)))
                    adjusted = replace(
                        row,
                        s_a=new_start,
                        ready=new_ready,
                        delta_lag=lag,
                        conflict=bool(lag > 0),
                        reason="conflict" if lag > 0 else row.reason,
                    )
                    warnings.append(
                        f"owner {row.owner}: switch {index} start delayed from {row.s_a} "
                        f"to previous ready {prev_ready}."
                    )
                else:
                    warnings.append(
                        f"owner {row.owner}: switch {index} starts at {row.s_a}, "
                        f"before previous ready {prev_ready}; static-z approximation may be too coarse."
                    )
            if adjusted.ready is not None:
                last_ready_by_owner[int(adjusted.owner)] = int(adjusted.ready)
            adjusted_rows.append(adjusted)

        if adjusted_rows != applied.rows:
            summary = summarize_plan_rows(adjusted_rows, lst=int(lst))
            summary.update(
                {
                    "switch_index": int(index),
                    "from": applied.from_name,
                    "to": applied.to_name,
                    "tau": int(applied.tau),
                    "strategy": str(strategy),
                    "enforce_previous_ready": bool(enforce_previous_ready),
                }
            )
            applied = replace(applied, rows=adjusted_rows, summary=summary)

        switches.append(applied)
        transitions.extend(
            rows_to_transitions(
                applied.rows,
                switch_index=index,
                from_name=applied.from_name,
                to_name=applied.to_name,
            )
        )
        current = str(event.target)

    return ChainPlan(
        initial=str(initial),
        final=current,
        switches=switches,
        transitions=transitions,
        warnings=warnings,
    )


def materialize_chain_static_z(
    *,
    steps: Sequence[int] | np.ndarray,
    initial_right: RightMap,
    transitions: Sequence[NodeTransition],
    nodes: Optional[Sequence[int]] = None,
    dtype=np.int32,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Convert compact node transitions to per-owner active/building right-neighbor arrays."""

    steps_array = _validate_steps(steps)
    if nodes is None:
        node_set = {int(node) for node in initial_right.keys()}
        for transition in transitions:
            node_set.add(int(transition.owner))
            if transition.old_right is not None:
                node_set.add(int(transition.old_right))
            if transition.new_right is not None:
                node_set.add(int(transition.new_right))
        nodes = sorted(node_set)

    active: dict[int, np.ndarray] = {}
    building: dict[int, np.ndarray] = {}
    total_rows = int(steps_array.size)
    for node in nodes:
        owner = int(node)
        right = initial_right.get(owner)
        active[owner] = np.full(total_rows, -1 if right is None else int(right), dtype=dtype)
        building[owner] = np.full(total_rows, -1, dtype=dtype)

    ordered = sorted(
        transitions,
        key=lambda item: (
            10**30 if item.start is None else int(item.start),
            int(item.switch_index),
            int(item.owner),
        ),
    )
    for transition in ordered:
        if transition.start is None or transition.ready is None:
            continue
        owner = int(transition.owner)
        if owner not in active:
            active[owner] = np.full(total_rows, -1, dtype=dtype)
            building[owner] = np.full(total_rows, -1, dtype=dtype)

        start_row = int(np.searchsorted(steps_array, int(transition.start), side="left"))
        ready_row = int(np.searchsorted(steps_array, int(transition.ready), side="left"))
        start_row = max(0, min(total_rows, start_row))
        ready_row = max(0, min(total_rows, ready_row))

        if transition.new_right is None:
            active[owner][start_row:] = -1
            building[owner][start_row:] = -1
            continue

        if ready_row > start_row:
            active[owner][start_row:ready_row] = -1
            building[owner][start_row:ready_row] = int(transition.new_right)
        active[owner][ready_row:] = int(transition.new_right)
        building[owner][ready_row:] = -1

    return active, building


def chain_summary(chain: ChainPlan) -> list[dict]:
    return [dict(switch.summary) for switch in chain.switches]


def plan_row_to_dict(row: PlanRow) -> dict:
    return asdict(row)


def transition_to_dict(transition: NodeTransition) -> dict:
    return asdict(transition)
