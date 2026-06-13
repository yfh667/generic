from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


NodeSet = tuple[int, ...]
GroupState = tuple[NodeSet, NodeSet]


@dataclass(frozen=True)
class GroupStateIndex:
    """Unique group-pair states and the per-step state mapping."""

    unique_states: tuple[GroupState, ...]
    state_ids: np.ndarray

    @property
    def num_states(self) -> int:
        return len(self.unique_states)


def _lookup_group_map(step_payload: object) -> Mapping | None:
    if isinstance(step_payload, Mapping):
        groups = step_payload.get("groups")
        if isinstance(groups, Mapping):
            return groups
    return None


def group_nodes_for_step(group_data: Mapping, step: int, group_id: int) -> NodeSet:
    """Return sorted node ids for one group at one step.

    The expected input shape matches the existing region-group cache:
    `group_data[step]["groups"][group_id] -> iterable[node_id]`.
    Missing steps or groups return an empty tuple.
    """

    step_payload = group_data.get(int(step), {}) if group_data else {}
    groups = _lookup_group_map(step_payload)
    if groups is None:
        return tuple()
    raw_nodes = groups.get(int(group_id), set()) or set()
    return tuple(sorted(int(node) for node in raw_nodes))


def build_group_state_index(
    *,
    group_data: Mapping,
    steps: Sequence[int],
    source_group_id: int,
    target_group_id: int,
) -> GroupStateIndex:
    """Deduplicate source/target node sets over a requested time axis."""

    state_to_id: dict[GroupState, int] = {}
    unique_states: list[GroupState] = []
    state_ids = np.empty(len(steps), dtype=np.int32)

    for row, step in enumerate(steps):
        key: GroupState = (
            group_nodes_for_step(group_data, int(step), int(source_group_id)),
            group_nodes_for_step(group_data, int(step), int(target_group_id)),
        )
        state_id = state_to_id.get(key)
        if state_id is None:
            state_id = len(unique_states)
            state_to_id[key] = state_id
            unique_states.append(key)
        state_ids[row] = int(state_id)

    return GroupStateIndex(unique_states=tuple(unique_states), state_ids=state_ids)

