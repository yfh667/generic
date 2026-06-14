from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable

from .core import DynamicRegionConstraintSeries, build_dynamic_region_internal_option_constraint_series


@dataclass(frozen=True)
class RegionInternalLinkSetupPolicy:
    """Rule for applying link setup time around selected region groups.

    The input ``group_data`` is observational data and is not modified. The
    policy says which groups should force a specific internal option, and how
    long a new link must spend in building state before it becomes active.
    """

    constrained_groups: tuple[int, ...]
    setup_time_seconds: float
    forced_option: int = 0
    wrap_planes: bool = False


def apply_region_internal_link_setup_time(
    *,
    config: ViewerConfig,
    topology_edge_table: EdgeTable,
    group_data: Mapping,
    steps: Sequence[int],
    policy: RegionInternalLinkSetupPolicy,
    group_names: Mapping[int, str] | None = None,
) -> DynamicRegionConstraintSeries:
    """Apply a region-internal LST policy to an existing topology graph.

    Conceptually this is the high-level transform:

    ``topology graph + group data + selected groups + LST policy``
    -> ``dynamic topology states``.

    The returned series contains:

    * ``edge_table``: union of all edges that may appear;
    * ``edge_active_mask``: links usable for routing at each step;
    * ``edge_building_mask``: links occupying terminals while being established.

    ``group_data`` is deliberately treated as read-only. Expanding or rewriting
    groups loses edge-level timing information and can create false internal
    links between nodes that are never in the same group at the same future
    instant.
    """

    return build_dynamic_region_internal_option_constraint_series(
        config=config,
        base_edge_table=topology_edge_table,
        group_data=group_data,
        steps=steps,
        constrained_groups=tuple(int(x) for x in policy.constrained_groups),
        setup_time_seconds=float(policy.setup_time_seconds),
        forced_option=int(policy.forced_option),
        wrap_planes=bool(policy.wrap_planes),
        group_names=group_names,
    )


def make_region_internal_link_setup_policy(
    *,
    constrained_groups: Iterable[int],
    setup_time_seconds: int | float,
    forced_option: int = 0,
    wrap_planes: bool = False,
) -> RegionInternalLinkSetupPolicy:
    return RegionInternalLinkSetupPolicy(
        constrained_groups=tuple(int(x) for x in constrained_groups),
        setup_time_seconds=float(setup_time_seconds),
        forced_option=int(forced_option),
        wrap_planes=bool(wrap_planes),
    )
