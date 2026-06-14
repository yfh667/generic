from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config.viewer_config import ViewerConfig


# Same option semantics as the full-option topology design:
# option 0: right neighbor, option 1/4: diagonal neighbors, option 2: skip-one-plane neighbor.
OPTION_DELTAS = {
    0: (1, 0),
    1: (1, -1),
    2: (2, 0),
    4: (1, 1),
}


@dataclass(frozen=True)
class EdgeTable:
    src: np.ndarray
    dst: np.ndarray
    option: np.ndarray
    src_plane: np.ndarray
    src_y: np.ndarray
    dst_plane: np.ndarray
    dst_y: np.ndarray
    sat_ids: list[str]

    @property
    def num_edges(self) -> int:
        return int(self.src.size)


def build_full_option_edges(
    config: ViewerConfig,
    *,
    options: Iterable[int] = (0, 1, 2, 4),
    sat_ids: list[str] | None = None,
    wrap_planes: bool = False,
) -> EdgeTable:
    """Build inter-plane option edges.

    ``wrap_planes=False`` keeps the old Walker-star/G60 behavior where the
    first and last planes are separated by a seam. ``wrap_planes=True`` is for
    Walker-delta constellations where plane ``P-1`` connects back to plane 0.
    """

    options = tuple(int(x) for x in options)
    bad = [x for x in options if x not in OPTION_DELTAS]
    if bad:
        raise ValueError(f"Unsupported options: {bad}; supported={sorted(OPTION_DELTAS)}")

    src: list[int] = []
    dst: list[int] = []
    opt_values: list[int] = []
    src_plane: list[int] = []
    src_y: list[int] = []
    dst_plane: list[int] = []
    dst_y: list[int] = []

    for p in range(int(config.P)):
        for y in range(int(config.N)):
            u = p * int(config.N) + y
            for option in options:
                dp, dy = OPTION_DELTAS[option]
                q = p + dp
                if bool(wrap_planes):
                    q = q % int(config.P)
                else:
                    if not (0 <= q < int(config.P)):
                        continue
                yy = (y + dy) % int(config.N)
                v = q * int(config.N) + yy
                if v == u:
                    continue
                src.append(u)
                dst.append(v)
                opt_values.append(option)
                src_plane.append(p)
                src_y.append(y)
                dst_plane.append(q)
                dst_y.append(yy)

    if sat_ids is None:
        sat_ids = [str(i + 1) for i in range(int(config.total_sats))]
    if len(sat_ids) != int(config.total_sats):
        raise ValueError(f"sat_ids length {len(sat_ids)} != total_sats {config.total_sats}")

    return EdgeTable(
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        option=np.asarray(opt_values, dtype=np.int16),
        src_plane=np.asarray(src_plane, dtype=np.int16),
        src_y=np.asarray(src_y, dtype=np.int16),
        dst_plane=np.asarray(dst_plane, dtype=np.int16),
        dst_y=np.asarray(dst_y, dtype=np.int16),
        sat_ids=sat_ids,
    )


def write_edges_csv(edge_table: EdgeTable, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "edge_idx",
                "src_node",
                "dst_node",
                "src_sat_id",
                "dst_sat_id",
                "src_plane",
                "src_y",
                "dst_plane",
                "dst_y",
                "option",
            ]
        )
        for idx in range(edge_table.num_edges):
            src = int(edge_table.src[idx])
            dst = int(edge_table.dst[idx])
            writer.writerow(
                [
                    idx,
                    src,
                    dst,
                    edge_table.sat_ids[src],
                    edge_table.sat_ids[dst],
                    int(edge_table.src_plane[idx]),
                    int(edge_table.src_y[idx]),
                    int(edge_table.dst_plane[idx]),
                    int(edge_table.dst_y[idx]),
                    int(edge_table.option[idx]),
                ]
            )
