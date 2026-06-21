from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.link_delay.module.edge_options import EdgeTable

from .edge_tables import INTRA_OPTION, make_edge_table_from_records


@dataclass(frozen=True)
class EdgeRecord:
    src_plane: int
    src_y: int
    dst_plane: int
    dst_y: int
    option: int

    @property
    def tuple(self) -> tuple[int, int, int, int, int]:
        return (self.src_plane, self.src_y, self.dst_plane, self.dst_y, self.option)


@dataclass(frozen=True)
class InterDegreeStats:
    total_edges: int
    inter_edges: int
    intra_edges: int
    max_out_degree: int
    max_in_degree: int
    out_degree_gt1_nodes: int
    in_degree_gt1_nodes: int
    zero_out_degree_nodes: int
    zero_in_degree_nodes: int

    @property
    def satisfies_one_right_one_left(self) -> bool:
        return self.max_out_degree <= 1 and self.max_in_degree <= 1


@dataclass(frozen=True)
class HybridEdgeTableResult:
    edge_table: EdgeTable
    added_patch_records: tuple[EdgeRecord, ...]
    removed_base_records: tuple[EdgeRecord, ...]
    kept_base_records: tuple[EdgeRecord, ...]
    degree_stats: InterDegreeStats


def cyclic_rows(start: int, end: int, *, n: int) -> tuple[int, ...]:
    """Return inclusive row ids on a cyclic y axis."""

    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    start = int(start) % n
    end = int(end) % n
    if start <= end:
        return tuple(range(start, end + 1))
    return tuple(list(range(start, n)) + list(range(0, end + 1)))


def node_id(plane: int, y: int, *, n: int) -> int:
    return int(plane) * int(n) + int(y)


def edge_key(src: int, dst: int) -> tuple[int, int]:
    src = int(src)
    dst = int(dst)
    return (src, dst) if src <= dst else (dst, src)


def edge_records_from_table(edge_table: EdgeTable) -> list[EdgeRecord]:
    return [
        EdgeRecord(
            src_plane=int(edge_table.src_plane[idx]),
            src_y=int(edge_table.src_y[idx]),
            dst_plane=int(edge_table.dst_plane[idx]),
            dst_y=int(edge_table.dst_y[idx]),
            option=int(edge_table.option[idx]),
        )
        for idx in range(edge_table.num_edges)
    ]


def edge_keys_from_table(edge_table: EdgeTable) -> set[tuple[int, int]]:
    return {edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])) for idx in range(edge_table.num_edges)}


def added_edge_count(previous: EdgeTable, next_table: EdgeTable) -> int:
    """Count new setup edges requested by switching from previous to next_table."""

    previous_keys = edge_keys_from_table(previous)
    return int(sum(1 for key in edge_keys_from_table(next_table) if key not in previous_keys))


def inter_degree_stats(edge_table: EdgeTable, *, total_sats: int) -> InterDegreeStats:
    out_degree: Counter[int] = Counter()
    in_degree: Counter[int] = Counter()
    inter_edges = 0
    intra_edges = 0
    for idx in range(edge_table.num_edges):
        option = int(edge_table.option[idx])
        if option == INTRA_OPTION:
            intra_edges += 1
            continue
        inter_edges += 1
        out_degree[int(edge_table.src[idx])] += 1
        in_degree[int(edge_table.dst[idx])] += 1

    total_sats = int(total_sats)
    return InterDegreeStats(
        total_edges=int(edge_table.num_edges),
        inter_edges=int(inter_edges),
        intra_edges=int(intra_edges),
        max_out_degree=int(max(out_degree.values(), default=0)),
        max_in_degree=int(max(in_degree.values(), default=0)),
        out_degree_gt1_nodes=int(sum(1 for value in out_degree.values() if value > 1)),
        in_degree_gt1_nodes=int(sum(1 for value in in_degree.values() if value > 1)),
        zero_out_degree_nodes=int(total_sats - len(out_degree)),
        zero_in_degree_nodes=int(total_sats - len(in_degree)),
    )


def _normalise_patch_options(patch_options: Iterable[int] | None) -> set[int] | None:
    if patch_options is None:
        return None
    return {int(option) for option in patch_options}


def build_y_band_hybrid_edge_table(
    *,
    base_edge_table: EdgeTable,
    patch_edge_table: EdgeTable,
    p: int,
    n: int,
    rows: Iterable[int],
    patch_options: Iterable[int] | None = None,
    remove_base_conflicts: bool = True,
    fail_on_degree_violation: bool = True,
    sat_ids: Sequence[str] | None = None,
) -> HybridEdgeTableResult:
    """Replace selected source-y inter links from a base topology with patch links.

    The operation is deliberately local and mechanical:

    - keep all intra-plane links from the base topology;
    - select patch inter links whose source row is in ``rows`` and whose option
      is in ``patch_options`` when provided;
    - remove base inter links that would share an outgoing source or incoming
      target with one of those patch links;
    - build a new ``EdgeTable`` and verify the one-right/one-left constraint.

    This function does not know about a paper, region pair, motif id, or metric.
    Those choices belong in callers and examples.
    """

    p = int(p)
    n = int(n)
    total_sats = p * n
    row_set = {int(row) % n for row in rows}
    allowed_options = _normalise_patch_options(patch_options)

    base_records = edge_records_from_table(base_edge_table)
    patch_candidates = edge_records_from_table(patch_edge_table)
    added_records = tuple(
        record
        for record in patch_candidates
        if record.option != INTRA_OPTION
        and int(record.src_y) in row_set
        and (allowed_options is None or int(record.option) in allowed_options)
    )

    patch_sources = {node_id(record.src_plane, record.src_y, n=n) for record in added_records}
    patch_targets = {node_id(record.dst_plane, record.dst_y, n=n) for record in added_records}

    kept_records: list[EdgeRecord] = []
    removed_records: list[EdgeRecord] = []
    for record in base_records:
        if record.option == INTRA_OPTION:
            kept_records.append(record)
            continue
        src = node_id(record.src_plane, record.src_y, n=n)
        dst = node_id(record.dst_plane, record.dst_y, n=n)
        conflicts = src in patch_sources or dst in patch_targets
        if bool(remove_base_conflicts) and conflicts:
            removed_records.append(record)
            continue
        kept_records.append(record)

    if sat_ids is None:
        sat_ids = base_edge_table.sat_ids
    edge_table = make_edge_table_from_records(
        p=p,
        n=n,
        records=[record.tuple for record in kept_records] + [record.tuple for record in added_records],
        sat_ids=list(sat_ids) if sat_ids is not None else None,
    )
    stats = inter_degree_stats(edge_table, total_sats=total_sats)
    if bool(fail_on_degree_violation) and not stats.satisfies_one_right_one_left:
        raise ValueError(f"hybrid edge table violates one-right/one-left constraint: {stats}")

    return HybridEdgeTableResult(
        edge_table=edge_table,
        added_patch_records=added_records,
        removed_base_records=tuple(removed_records),
        kept_base_records=tuple(kept_records),
        degree_stats=stats,
    )


def build_y_band_hybrid_edge_tables(
    *,
    base_edge_table: EdgeTable,
    patch_edge_table: EdgeTable,
    p: int,
    n: int,
    row_bands: Iterable[Iterable[int]],
    patch_options: Iterable[int] | None = None,
    remove_base_conflicts: bool = True,
    fail_on_degree_violation: bool = True,
    sat_ids: Sequence[str] | None = None,
) -> list[HybridEdgeTableResult]:
    return [
        build_y_band_hybrid_edge_table(
            base_edge_table=base_edge_table,
            patch_edge_table=patch_edge_table,
            p=p,
            n=n,
            rows=rows,
            patch_options=patch_options,
            remove_base_conflicts=remove_base_conflicts,
            fail_on_degree_violation=fail_on_degree_violation,
            sat_ids=sat_ids,
        )
        for rows in row_bands
    ]


def edge_table_mask(edge_table: EdgeTable, *, all_edge_keys: Sequence[tuple[int, int]] | None = None) -> np.ndarray:
    keys = tuple(sorted(edge_keys_from_table(edge_table)) if all_edge_keys is None else all_edge_keys)
    key_set = edge_keys_from_table(edge_table)
    return np.asarray([key in key_set for key in keys], dtype=bool)
