from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable, OPTION_DELTAS, build_full_option_edges

from .edge_tables import INTRA_OPTION, make_edge_table_from_records


LEFT_SIDE = 0
RIGHT_SIDE = 1


@dataclass(frozen=True)
class RegionInternalOptionConstraintStats:
    base_edges: int
    forced_internal_option_edges: int
    combined_edges_before_filter: int
    kept_edges: int
    dropped_edges: int
    drop_by_reason: dict[str, int]
    drop_by_option: dict[str, int]
    option_counts_kept: dict[str, int]
    constrained_groups: list[dict]
    validation: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_group_nodes(group_nodes: Mapping[int, Iterable[int]]) -> dict[int, set[int]]:
    return {int(group_id): set(int(node) for node in nodes) for group_id, nodes in group_nodes.items()}


def make_edge_table_subset(edge_table: EdgeTable, mask: np.ndarray) -> EdgeTable:
    mask = np.asarray(mask, dtype=bool)
    return EdgeTable(
        src=np.asarray(edge_table.src[mask], dtype=np.int32),
        dst=np.asarray(edge_table.dst[mask], dtype=np.int32),
        option=np.asarray(edge_table.option[mask], dtype=np.int16),
        src_plane=np.asarray(edge_table.src_plane[mask], dtype=np.int16),
        src_y=np.asarray(edge_table.src_y[mask], dtype=np.int16),
        dst_plane=np.asarray(edge_table.dst_plane[mask], dtype=np.int16),
        dst_y=np.asarray(edge_table.dst_y[mask], dtype=np.int16),
        sat_ids=list(edge_table.sat_ids),
    )


def edge_records_from_table(edge_table: EdgeTable) -> list[tuple[int, int, int, int, int]]:
    return [
        (
            int(edge_table.src_plane[idx]),
            int(edge_table.src_y[idx]),
            int(edge_table.dst_plane[idx]),
            int(edge_table.dst_y[idx]),
            int(edge_table.option[idx]),
        )
        for idx in range(int(edge_table.num_edges))
    ]


def edge_side_for_node(
    edge_table: EdgeTable,
    edge_idx: int,
    node: int,
    *,
    p: int | None = None,
    wrap_planes: bool = False,
) -> int | None:
    src = int(edge_table.src[edge_idx])
    dst = int(edge_table.dst[edge_idx])
    src_plane = int(edge_table.src_plane[edge_idx])
    dst_plane = int(edge_table.dst_plane[edge_idx])
    option = int(edge_table.option[edge_idx])
    if option == int(INTRA_OPTION):
        return None

    if option in OPTION_DELTAS:
        dp, _dy = OPTION_DELTAS[option]
        if bool(wrap_planes):
            if p is None:
                raise ValueError("p is required when wrap_planes=True")
            src_to_dst = (src_plane + int(dp)) % int(p) == dst_plane
            dst_to_src = (dst_plane + int(dp)) % int(p) == src_plane
        else:
            src_to_dst = src_plane + int(dp) == dst_plane
            dst_to_src = dst_plane + int(dp) == src_plane

        if src_to_dst and not dst_to_src:
            if int(node) == src:
                return RIGHT_SIDE
            if int(node) == dst:
                return LEFT_SIDE
            return None
        if dst_to_src and not src_to_dst:
            if int(node) == dst:
                return RIGHT_SIDE
            if int(node) == src:
                return LEFT_SIDE
            return None

    if int(node) == src:
        return RIGHT_SIDE if dst_plane > src_plane else LEFT_SIDE
    if int(node) == dst:
        return LEFT_SIDE if src_plane < dst_plane else RIGHT_SIDE
    return None


def selected_group_bits(
    *,
    group_nodes: Mapping[int, Iterable[int]],
    constrained_groups: Iterable[int],
    total_nodes: int,
) -> np.ndarray:
    normalized = normalize_group_nodes(group_nodes)
    bits = np.zeros(int(total_nodes), dtype=np.uint64)
    for offset, group_id in enumerate(int(x) for x in constrained_groups):
        if int(offset) >= 63:
            raise ValueError("at most 63 constrained groups are supported by the bit-mask implementation")
        bit = np.uint64(1 << int(offset))
        for node in normalized.get(int(group_id), set()):
            if 0 <= int(node) < int(total_nodes):
                bits[int(node)] |= bit
    return bits


def build_region_internal_option_edges(
    *,
    config: ViewerConfig,
    group_nodes: Mapping[int, Iterable[int]],
    constrained_groups: Iterable[int],
    option: int = 0,
    wrap_planes: bool = False,
) -> EdgeTable:
    """Build full-grid option edges whose endpoints are inside the same selected group."""

    bits = selected_group_bits(
        group_nodes=group_nodes,
        constrained_groups=constrained_groups,
        total_nodes=int(config.total_sats),
    )
    option_edges = build_full_option_edges(config, options=(int(option),), wrap_planes=bool(wrap_planes))
    keep = np.zeros(int(option_edges.num_edges), dtype=bool)
    for idx in range(int(option_edges.num_edges)):
        src = int(option_edges.src[idx])
        dst = int(option_edges.dst[idx])
        keep[idx] = int(bits[src] & bits[dst]) != 0
    return make_edge_table_subset(option_edges, keep)


def apply_region_internal_option_constraint(
    *,
    base_edge_table: EdgeTable,
    internal_option_edge_table: EdgeTable,
    group_nodes: Mapping[int, Iterable[int]],
    constrained_groups: Iterable[int],
    total_nodes: int,
    p: int,
    n: int,
    forced_option: int = 0,
    intra_option: int = INTRA_OPTION,
    group_names: Mapping[int, str] | None = None,
    wrap_planes: bool = False,
) -> tuple[EdgeTable, RegionInternalOptionConstraintStats]:
    """Apply a side-aware selected-region internal-option constraint.

    The base topology can be any edge table. The internal option edge table is
    the set of edges to force inside selected groups, for example option-0
    grid links. If a selected-region node side has a forced internal edge, all
    other inter-plane edges on the same side are removed. Same selected-region
    inter-plane edges whose option differs from ``forced_option`` are removed.
    """

    constrained_groups = [int(x) for x in constrained_groups]
    normalized_nodes = normalize_group_nodes(group_nodes)
    group_bits = selected_group_bits(
        group_nodes=normalized_nodes,
        constrained_groups=constrained_groups,
        total_nodes=int(total_nodes),
    )
    combined = make_edge_table_from_records(
        p=int(p),
        n=int(n),
        records=edge_records_from_table(base_edge_table) + edge_records_from_table(internal_option_edge_table),
    )

    allowed_neighbors = [[set(), set()] for _ in range(int(total_nodes))]
    forced_key_set: set[tuple[int, int]] = set()
    for idx in range(int(internal_option_edge_table.num_edges)):
        src = int(internal_option_edge_table.src[idx])
        dst = int(internal_option_edge_table.dst[idx])
        forced_key_set.add((min(src, dst), max(src, dst)))
        src_side = edge_side_for_node(
            internal_option_edge_table,
            idx,
            src,
            p=int(p),
            wrap_planes=bool(wrap_planes),
        )
        dst_side = edge_side_for_node(
            internal_option_edge_table,
            idx,
            dst,
            p=int(p),
            wrap_planes=bool(wrap_planes),
        )
        if src_side is not None:
            allowed_neighbors[src][int(src_side)].add(dst)
        if dst_side is not None:
            allowed_neighbors[dst][int(dst_side)].add(src)

    keep = np.ones(int(combined.num_edges), dtype=bool)
    drop_by_reason = {
        "same_region_non_forced_option": 0,
        "selected_node_side_has_internal_forced_option": 0,
    }
    drop_by_option: dict[str, int] = {}

    for idx in range(int(combined.num_edges)):
        option = int(combined.option[idx])
        if option == int(intra_option):
            continue
        src = int(combined.src[idx])
        dst = int(combined.dst[idx])
        same_selected_region = int(group_bits[src] & group_bits[dst]) != 0
        edge_key = (min(src, dst), max(src, dst))

        if same_selected_region and option != int(forced_option):
            keep[idx] = False
            drop_by_reason["same_region_non_forced_option"] += 1
            drop_by_option[str(option)] = drop_by_option.get(str(option), 0) + 1
            continue

        conflict = False
        for node, neighbor in ((src, dst), (dst, src)):
            if int(group_bits[node]) == 0:
                continue
            side = edge_side_for_node(combined, idx, node, p=int(p), wrap_planes=bool(wrap_planes))
            if side is None:
                continue
            allowed = allowed_neighbors[node][int(side)]
            if allowed and int(neighbor) not in allowed:
                conflict = True
                break
        if conflict and edge_key not in forced_key_set:
            keep[idx] = False
            drop_by_reason["selected_node_side_has_internal_forced_option"] += 1
            drop_by_option[str(option)] = drop_by_option.get(str(option), 0) + 1

    constrained = make_edge_table_subset(combined, keep)
    validation_same_region_non_forced_option = 0
    validation_side_extra = 0
    for idx in range(int(constrained.num_edges)):
        option = int(constrained.option[idx])
        if option == int(intra_option):
            continue
        src = int(constrained.src[idx])
        dst = int(constrained.dst[idx])
        if int(group_bits[src] & group_bits[dst]) != 0 and option != int(forced_option):
            validation_same_region_non_forced_option += 1
        for node, neighbor in ((src, dst), (dst, src)):
            if int(group_bits[node]) == 0:
                continue
            side = edge_side_for_node(constrained, idx, node, p=int(p), wrap_planes=bool(wrap_planes))
            if side is None:
                continue
            allowed = allowed_neighbors[node][int(side)]
            if allowed and int(neighbor) not in allowed:
                validation_side_extra += 1

    option_counts = {
        str(option): int(np.count_nonzero(constrained.option == int(option)))
        for option in sorted(set(int(x) for x in constrained.option.tolist()))
    }
    names = {int(k): str(v) for k, v in (group_names or {}).items()}
    stats = RegionInternalOptionConstraintStats(
        base_edges=int(base_edge_table.num_edges),
        forced_internal_option_edges=int(internal_option_edge_table.num_edges),
        combined_edges_before_filter=int(combined.num_edges),
        kept_edges=int(constrained.num_edges),
        dropped_edges=int(combined.num_edges - constrained.num_edges),
        drop_by_reason=drop_by_reason,
        drop_by_option=drop_by_option,
        option_counts_kept=option_counts,
        constrained_groups=[
            {
                "id": int(group_id),
                "name": names.get(int(group_id), f"Group {int(group_id)}"),
                "nodes": int(len(normalized_nodes.get(int(group_id), set()))),
            }
            for group_id in constrained_groups
        ],
        validation={
            "same_selected_group_non_forced_option_edges_remaining": int(validation_same_region_non_forced_option),
            "selected_node_same_side_extra_edges_remaining": int(validation_side_extra),
        },
    )
    return constrained, stats
