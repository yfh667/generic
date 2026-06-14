from __future__ import annotations

from typing import Iterable

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable, build_full_option_edges


INTRA_OPTION = -1


def build_full_option_plus_intra_edges(
    config: ViewerConfig,
    *,
    inter_options: Iterable[int] = (0, 1, 2, 4),
    include_intra: bool = True,
    intra_option: int = INTRA_OPTION,
    sat_ids: list[str] | None = None,
    wrap_planes: bool = False,
) -> EdgeTable:
    """Build the undirected edge representative table used by the 2D G60-style topology.

    The inter-plane part follows the existing full-option convention. The intra-plane
    part adds the y-ring links (p, y) -- (p, y+1 mod N), including the wrap link
    (p, N-1) -- (p, 0).
    """

    inter = build_full_option_edges(
        config,
        options=tuple(int(x) for x in inter_options),
        sat_ids=sat_ids,
        wrap_planes=bool(wrap_planes),
    )
    if not include_intra:
        return inter

    src: list[int] = list(int(x) for x in inter.src)
    dst: list[int] = list(int(x) for x in inter.dst)
    option: list[int] = list(int(x) for x in inter.option)
    src_plane: list[int] = list(int(x) for x in inter.src_plane)
    src_y: list[int] = list(int(x) for x in inter.src_y)
    dst_plane: list[int] = list(int(x) for x in inter.dst_plane)
    dst_y: list[int] = list(int(x) for x in inter.dst_y)

    P = int(config.P)
    N = int(config.N)
    for p in range(P):
        for y in range(N):
            yy = (y + 1) % N
            src.append(p * N + y)
            dst.append(p * N + yy)
            option.append(int(intra_option))
            src_plane.append(p)
            src_y.append(y)
            dst_plane.append(p)
            dst_y.append(yy)

    return EdgeTable(
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        option=np.asarray(option, dtype=np.int16),
        src_plane=np.asarray(src_plane, dtype=np.int16),
        src_y=np.asarray(src_y, dtype=np.int16),
        dst_plane=np.asarray(dst_plane, dtype=np.int16),
        dst_y=np.asarray(dst_y, dtype=np.int16),
        sat_ids=inter.sat_ids,
    )


def build_undirected_adjacency(edge_table: EdgeTable, total_nodes: int) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(int(total_nodes))]
    for edge_idx in range(int(edge_table.num_edges)):
        u = int(edge_table.src[edge_idx])
        v = int(edge_table.dst[edge_idx])
        adjacency[u].append((v, edge_idx))
        adjacency[v].append((u, edge_idx))

    for neighbors in adjacency:
        neighbors.sort(key=lambda item: (item[0], item[1]))
    return adjacency
