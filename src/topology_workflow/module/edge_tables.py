from __future__ import annotations

from typing import Any

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable, build_full_option_edges
from src.motif_generator.module.exact_box import Motif
from src.motif_generator.module.support import motif_support_from_dict
from src.motif_generator.module.tiling import tile_motif_on_grid
from src.motif_generator.module.viewer_adapter import option_from_symbol


INTRA_OPTION = -1


def make_edge_table_from_records(
    *,
    p: int,
    n: int,
    records: list[tuple[int, int, int, int, int]],
    sat_ids: list[str] | None = None,
) -> EdgeTable:
    unique: dict[tuple[int, int], int] = {}
    for src_plane, src_y, dst_plane, dst_y, option in records:
        src = int(src_plane) * int(n) + int(src_y)
        dst = int(dst_plane) * int(n) + int(dst_y)
        if src == dst:
            continue
        a, b = (src, dst) if src < dst else (dst, src)
        unique.setdefault((a, b), int(option))

    src_values: list[int] = []
    dst_values: list[int] = []
    option_values: list[int] = []
    src_plane_values: list[int] = []
    src_y_values: list[int] = []
    dst_plane_values: list[int] = []
    dst_y_values: list[int] = []

    for src, dst in sorted(unique):
        src_values.append(src)
        dst_values.append(dst)
        option_values.append(int(unique[(src, dst)]))
        src_plane_values.append(src // int(n))
        src_y_values.append(src % int(n))
        dst_plane_values.append(dst // int(n))
        dst_y_values.append(dst % int(n))

    total_sats = int(p) * int(n)
    if sat_ids is None:
        sat_ids = [str(i + 1) for i in range(total_sats)]
    if len(sat_ids) != total_sats:
        raise ValueError(f"sat_ids length {len(sat_ids)} != p*n={total_sats}")

    return EdgeTable(
        src=np.asarray(src_values, dtype=np.int32),
        dst=np.asarray(dst_values, dtype=np.int32),
        option=np.asarray(option_values, dtype=np.int16),
        src_plane=np.asarray(src_plane_values, dtype=np.int16),
        src_y=np.asarray(src_y_values, dtype=np.int16),
        dst_plane=np.asarray(dst_plane_values, dtype=np.int16),
        dst_y=np.asarray(dst_y_values, dtype=np.int16),
        sat_ids=sat_ids,
    )


def add_intra_ring_records(records: list[tuple[int, int, int, int, int]], *, p: int, n: int) -> None:
    for plane in range(int(p)):
        for y in range(int(n)):
            records.append((plane, y, plane, (y + 1) % int(n), INTRA_OPTION))


def motif_text_to_matrix(motif_text: str) -> Motif:
    columns = [part.strip() for part in str(motif_text).split("|")]
    if not columns or any(part == "" for part in columns):
        raise ValueError(f"empty motif text: {motif_text!r}")

    height = len(columns[0])
    if height <= 0:
        raise ValueError(f"empty motif column in text: {motif_text!r}")
    if any(len(column) != height for column in columns):
        raise ValueError(f"inconsistent motif column heights: {motif_text!r}")

    matrix = []
    for column in columns:
        values = []
        for symbol in column:
            values.append(None if symbol == "-" else str(symbol))
        matrix.append(tuple(values))
    return tuple(matrix)


def build_motif_text_edge_table(
    *,
    motif_text: str,
    config: ViewerConfig,
    horizontal_step: int | None = None,
    allow_vertical_overlap: bool = True,
    allow_clipped_right: bool = True,
    wrap_planes: bool = False,
    add_intra_ring: bool = True,
) -> EdgeTable:
    motif = motif_text_to_matrix(motif_text)
    tiled = tile_motif_on_grid(
        p=int(config.P),
        n=int(config.N),
        motif=motif,
        horizontal_step=horizontal_step,
        allow_vertical_overlap=bool(allow_vertical_overlap),
        allow_clipped_right=bool(allow_clipped_right),
        wrap_cols=bool(wrap_planes),
    )

    records: list[tuple[int, int, int, int, int]] = []
    for edge in tiled.placed_edges:
        option = option_from_symbol(edge.symbol)
        records.append((int(edge.src_col), int(edge.src_row), int(edge.dst_col), int(edge.dst_row), int(option)))

    if bool(add_intra_ring):
        add_intra_ring_records(records, p=int(config.P), n=int(config.N))
    return make_edge_table_from_records(p=int(config.P), n=int(config.N), records=records)


def build_full_option_plus_intra_edge_table(
    *,
    config: ViewerConfig,
    options: tuple[int, ...] = (0, 1, 2, 4),
    add_intra_ring: bool = True,
    wrap_planes: bool = False,
) -> EdgeTable:
    inter_edges = build_full_option_edges(config, options=options, wrap_planes=bool(wrap_planes))
    records: list[tuple[int, int, int, int, int]] = []
    for edge_idx in range(inter_edges.num_edges):
        records.append(
            (
                int(inter_edges.src_plane[edge_idx]),
                int(inter_edges.src_y[edge_idx]),
                int(inter_edges.dst_plane[edge_idx]),
                int(inter_edges.dst_y[edge_idx]),
                int(inter_edges.option[edge_idx]),
            )
        )
    if bool(add_intra_ring):
        add_intra_ring_records(records, p=int(config.P), n=int(config.N))
    return make_edge_table_from_records(p=int(config.P), n=int(config.N), records=records)


def build_single_motif_edge_table(
    *,
    topology_raw: dict[str, Any],
    config: ViewerConfig,
) -> EdgeTable:
    motif_raw = topology_raw.get("motif")
    if not isinstance(motif_raw, dict):
        raise ValueError("topology.motif must be a mapping for kind=single_motif")
    tiling_raw = topology_raw.get("tiling", {})
    if not isinstance(tiling_raw, dict):
        raise ValueError("topology.tiling must be a mapping when present")

    motif_support = motif_support_from_dict({"motif": motif_raw})
    wrap_planes = bool(topology_raw.get("wrap_planes", False))
    tiled = tile_motif_on_grid(
        p=int(config.P),
        n=int(config.N),
        motif=motif_support,
        horizontal_step=tiling_raw.get("horizontal_step"),
        allow_vertical_overlap=bool(tiling_raw.get("allow_vertical_overlap", True)),
        allow_clipped_right=bool(tiling_raw.get("allow_clipped_right", True)),
        wrap_cols=wrap_planes,
    )

    records: list[tuple[int, int, int, int, int]] = []
    for edge in tiled.placed_edges:
        option = option_from_symbol(edge.symbol)
        records.append((int(edge.src_col), int(edge.src_row), int(edge.dst_col), int(edge.dst_row), int(option)))

    if bool(topology_raw.get("add_intra_ring", True)):
        add_intra_ring_records(records, p=int(config.P), n=int(config.N))
    return make_edge_table_from_records(p=int(config.P), n=int(config.N), records=records)


def build_edge_table_from_topology_config(
    *,
    topology_raw: dict[str, Any],
    config: ViewerConfig,
) -> EdgeTable:
    kind = str(topology_raw.get("kind", "single_motif"))
    if kind == "single_motif":
        return build_single_motif_edge_table(topology_raw=topology_raw, config=config)
    if kind == "motif_text":
        return build_motif_text_edge_table(
            motif_text=str(topology_raw["motif"]),
            config=config,
            horizontal_step=topology_raw.get("horizontal_step"),
            allow_vertical_overlap=bool(topology_raw.get("allow_vertical_overlap", True)),
            allow_clipped_right=bool(topology_raw.get("allow_clipped_right", True)),
            wrap_planes=bool(topology_raw.get("wrap_planes", False)),
            add_intra_ring=bool(topology_raw.get("add_intra_ring", True)),
        )
    if kind == "full_option_plus_intra":
        return build_full_option_plus_intra_edge_table(
            config=config,
            options=tuple(int(x) for x in topology_raw.get("options", (0, 1, 2, 4))),
            add_intra_ring=bool(topology_raw.get("add_intra_ring", True)),
            wrap_planes=bool(topology_raw.get("wrap_planes", False)),
        )
    raise NotImplementedError(
        f"Unsupported topology.kind={kind!r}. "
        "Planned extension points include motif_library and paper-specific baselines."
    )
