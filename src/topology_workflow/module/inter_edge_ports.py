from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class NormalizedInterEdge:
    edge_idx: int
    src: int
    dst: int
    right_owner: int
    left_owner: int
    plane_delta: int


@dataclass(frozen=True)
class InvalidInterEdge:
    edge_idx: int
    src: int
    dst: int
    reason: str


@dataclass(frozen=True)
class InterPortViolation:
    node: int
    plane: int
    y: int
    left_count: int
    right_count: int
    left_neighbors: tuple[int, ...]
    right_neighbors: tuple[int, ...]
    left_edge_indices: tuple[int, ...]
    right_edge_indices: tuple[int, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class InterPortCheckResult:
    p: int
    n: int
    total_input_edges: int
    checked_inter_edges: int
    duplicate_edges: int
    invalid_edges: tuple[InvalidInterEdge, ...]
    violations: tuple[InterPortViolation, ...]
    left_count_by_node: np.ndarray
    right_count_by_node: np.ndarray

    @property
    def ok(self) -> bool:
        return not self.invalid_edges and not self.violations

    @property
    def summary(self) -> dict:
        return {
            "ok": bool(self.ok),
            "p": int(self.p),
            "n": int(self.n),
            "total_nodes": int(self.p * self.n),
            "total_input_edges": int(self.total_input_edges),
            "checked_inter_edges": int(self.checked_inter_edges),
            "duplicate_edges": int(self.duplicate_edges),
            "invalid_edges": int(len(self.invalid_edges)),
            "violation_nodes": int(len(self.violations)),
            "left_gt1_nodes": int(sum(1 for item in self.violations if item.left_count > 1)),
            "right_gt1_nodes": int(sum(1 for item in self.violations if item.right_count > 1)),
            "max_left_count": int(np.max(self.left_count_by_node)) if self.left_count_by_node.size else 0,
            "max_right_count": int(np.max(self.right_count_by_node)) if self.right_count_by_node.size else 0,
        }


def node_plane_y(node: int, *, n: int) -> tuple[int, int]:
    node = int(node)
    return node // int(n), node % int(n)


def normalize_inter_edge(
    *,
    edge_idx: int,
    src: int,
    dst: int,
    p: int,
    n: int,
    allowed_plane_deltas: Sequence[int] | None = None,
) -> NormalizedInterEdge | InvalidInterEdge | None:
    """Normalize an undirected inter edge into right-owner and left-owner roles.

    The right side is the lower plane endpoint, and the left side is the higher
    plane endpoint. This deliberately rejects wrap-around seam edges when
    ``allowed_plane_deltas`` is set to the usual G60 values ``(1, 2)``.
    """

    p = int(p)
    n = int(n)
    total_nodes = p * n
    src = int(src)
    dst = int(dst)
    edge_idx = int(edge_idx)
    if src == dst:
        return InvalidInterEdge(edge_idx=edge_idx, src=src, dst=dst, reason="self_loop")
    if not (0 <= src < total_nodes and 0 <= dst < total_nodes):
        return InvalidInterEdge(edge_idx=edge_idx, src=src, dst=dst, reason="node_out_of_range")

    src_plane, _src_y = node_plane_y(src, n=n)
    dst_plane, _dst_y = node_plane_y(dst, n=n)
    if src_plane == dst_plane:
        return None

    if src_plane < dst_plane:
        right_owner = src
        left_owner = dst
        plane_delta = dst_plane - src_plane
    else:
        right_owner = dst
        left_owner = src
        plane_delta = src_plane - dst_plane

    if allowed_plane_deltas is not None:
        allowed = {int(delta) for delta in allowed_plane_deltas}
        if int(plane_delta) not in allowed:
            return InvalidInterEdge(
                edge_idx=edge_idx,
                src=src,
                dst=dst,
                reason=f"plane_delta_not_allowed:{plane_delta}",
            )

    right_plane, _right_y = node_plane_y(right_owner, n=n)
    left_plane, _left_y = node_plane_y(left_owner, n=n)
    if right_plane >= p - 1:
        return InvalidInterEdge(edge_idx=edge_idx, src=src, dst=dst, reason="last_plane_has_right_link")
    if left_plane <= 0:
        return InvalidInterEdge(edge_idx=edge_idx, src=src, dst=dst, reason="first_plane_has_left_link")

    return NormalizedInterEdge(
        edge_idx=edge_idx,
        src=src,
        dst=dst,
        right_owner=int(right_owner),
        left_owner=int(left_owner),
        plane_delta=int(plane_delta),
    )


def check_inter_edge_port_constraints(
    edges: Iterable[tuple[int, int]],
    *,
    p: int,
    n: int,
    allowed_plane_deltas: Sequence[int] | None = (1, 2),
    deduplicate: bool = True,
) -> InterPortCheckResult:
    """Check one-left/one-right constraints for an inter-edge set."""

    p = int(p)
    n = int(n)
    total_nodes = p * n
    left_neighbors: list[list[int]] = [[] for _ in range(total_nodes)]
    right_neighbors: list[list[int]] = [[] for _ in range(total_nodes)]
    left_edge_indices: list[list[int]] = [[] for _ in range(total_nodes)]
    right_edge_indices: list[list[int]] = [[] for _ in range(total_nodes)]
    invalid: list[InvalidInterEdge] = []
    seen: set[tuple[int, int]] = set()
    duplicate_edges = 0
    total_input = 0
    checked_inter = 0

    for edge_idx, raw in enumerate(edges):
        total_input += 1
        src, dst = int(raw[0]), int(raw[1])
        key = (src, dst) if src <= dst else (dst, src)
        if key in seen:
            duplicate_edges += 1
            if bool(deduplicate):
                continue
        seen.add(key)

        normalized = normalize_inter_edge(
            edge_idx=int(edge_idx),
            src=src,
            dst=dst,
            p=p,
            n=n,
            allowed_plane_deltas=allowed_plane_deltas,
        )
        if normalized is None:
            continue
        if isinstance(normalized, InvalidInterEdge):
            invalid.append(normalized)
            continue

        checked_inter += 1
        right_owner = int(normalized.right_owner)
        left_owner = int(normalized.left_owner)
        right_neighbors[right_owner].append(left_owner)
        right_edge_indices[right_owner].append(int(normalized.edge_idx))
        left_neighbors[left_owner].append(right_owner)
        left_edge_indices[left_owner].append(int(normalized.edge_idx))

    left_count = np.asarray([len(items) for items in left_neighbors], dtype=np.int16)
    right_count = np.asarray([len(items) for items in right_neighbors], dtype=np.int16)

    violations: list[InterPortViolation] = []
    for node in range(total_nodes):
        plane, y = node_plane_y(node, n=n)
        reasons: list[str] = []
        if int(left_count[node]) > 1:
            reasons.append("left_count_gt1")
        if int(right_count[node]) > 1:
            reasons.append("right_count_gt1")
        if int(plane) == 0 and int(left_count[node]) > 0:
            reasons.append("first_plane_left_count_gt0")
        if int(plane) == p - 1 and int(right_count[node]) > 0:
            reasons.append("last_plane_right_count_gt0")
        if not reasons:
            continue
        violations.append(
            InterPortViolation(
                node=int(node),
                plane=int(plane),
                y=int(y),
                left_count=int(left_count[node]),
                right_count=int(right_count[node]),
                left_neighbors=tuple(int(x) for x in sorted(left_neighbors[node])),
                right_neighbors=tuple(int(x) for x in sorted(right_neighbors[node])),
                left_edge_indices=tuple(int(x) for x in left_edge_indices[node]),
                right_edge_indices=tuple(int(x) for x in right_edge_indices[node]),
                reasons=tuple(reasons),
            )
        )

    return InterPortCheckResult(
        p=p,
        n=n,
        total_input_edges=int(total_input),
        checked_inter_edges=int(checked_inter),
        duplicate_edges=int(duplicate_edges),
        invalid_edges=tuple(invalid),
        violations=tuple(violations),
        left_count_by_node=left_count,
        right_count_by_node=right_count,
    )


def dataclass_rows(items: Iterable[object]) -> list[dict]:
    return [asdict(item) for item in items]
