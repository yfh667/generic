from __future__ import annotations

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable

from .tiling import TiledMotifResult


SYMBOL_TO_OPTION = {
    "A": 0,
    "B": 1,
    "C": 4,
    "D": 2,
}

DELTA_TO_OPTION = {
    (1, 0): 0,
    (1, -1): 1,
    (2, 0): 2,
    (1, 1): 4,
}


def option_from_symbol(symbol: str) -> int:
    symbol = str(symbol)
    if symbol not in SYMBOL_TO_OPTION:
        raise ValueError(f"unsupported motif edge symbol {symbol!r}")
    return int(SYMBOL_TO_OPTION[symbol])


def option_from_delta(dx: int, dy: int) -> int:
    key = (int(dx), int(dy))
    if key not in DELTA_TO_OPTION:
        raise ValueError(f"unsupported motif edge delta {key!r}")
    return int(DELTA_TO_OPTION[key])


def make_viewer_config(
    *,
    p: int,
    n: int,
    name: str = "motif_topology",
    station_groups: dict[int, dict] | None = None,
    group_colors: list[str] | None = None,
) -> ViewerConfig:
    """Build a minimal ViewerConfig for the 2D topology viewer."""

    return ViewerConfig(
        name=str(name),
        P=int(p),
        N=int(n),
        station_groups=station_groups or {},
        group_colors=group_colors or [],
    )


def tiled_result_to_edge_table(
    result: TiledMotifResult,
    *,
    sat_ids: list[str] | None = None,
) -> EdgeTable:
    """Convert a tiled motif result to the EdgeTable expected by the 2D viewer."""

    if sat_ids is None:
        sat_ids = [str(idx + 1) for idx in range(int(result.p) * int(result.n))]
    if len(sat_ids) != int(result.p) * int(result.n):
        raise ValueError(f"sat_ids length {len(sat_ids)} != p*n={int(result.p) * int(result.n)}")

    src: list[int] = []
    dst: list[int] = []
    option: list[int] = []
    src_plane: list[int] = []
    src_y: list[int] = []
    dst_plane: list[int] = []
    dst_y: list[int] = []

    for edge in result.placed_edges:
        src.append(int(edge.src_col) * int(result.n) + int(edge.src_row))
        dst.append(int(edge.dst_col) * int(result.n) + int(edge.dst_row))
        option.append(
            option_from_delta(
                int(edge.dst_col) - int(edge.src_col),
                int(edge.dst_row) - int(edge.src_row),
            )
        )
        src_plane.append(int(edge.src_col))
        src_y.append(int(edge.src_row))
        dst_plane.append(int(edge.dst_col))
        dst_y.append(int(edge.dst_row))

    return EdgeTable(
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        option=np.asarray(option, dtype=np.int16),
        src_plane=np.asarray(src_plane, dtype=np.int16),
        src_y=np.asarray(src_y, dtype=np.int16),
        dst_plane=np.asarray(dst_plane, dtype=np.int16),
        dst_y=np.asarray(dst_y, dtype=np.int16),
        sat_ids=sat_ids,
    )
