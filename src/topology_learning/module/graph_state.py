from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable


@dataclass(frozen=True)
class GraphArrays:
    """Numpy graph arrays that can be passed to a future PyTorch/PyG adapter."""

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    node_feature_names: tuple[str, ...]
    edge_feature_names: tuple[str, ...]


def build_node_features(
    config: ViewerConfig,
    *,
    group_nodes: Mapping[int, Sequence[int]] | None = None,
    group_ids: Sequence[int] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build simple per-satellite features independent of any ML framework."""

    total_nodes = int(config.P) * int(config.N)
    planes = np.repeat(np.arange(int(config.P), dtype=np.float32), int(config.N))
    y = np.tile(np.arange(int(config.N), dtype=np.float32), int(config.P))
    p_norm = planes / max(1.0, float(int(config.P) - 1))
    y_phase = y / max(1.0, float(int(config.N)))
    columns = [
        p_norm,
        np.sin(2.0 * np.pi * y_phase),
        np.cos(2.0 * np.pi * y_phase),
    ]
    names = ["plane_norm", "y_sin", "y_cos"]

    if group_nodes is not None:
        selected_group_ids = tuple(int(x) for x in (group_ids or sorted(int(k) for k in group_nodes.keys())))
        for gid in selected_group_ids:
            bit = np.zeros(total_nodes, dtype=np.float32)
            for node in group_nodes.get(int(gid), ()):
                if 0 <= int(node) < total_nodes:
                    bit[int(node)] = 1.0
            columns.append(bit)
            names.append(f"group_{int(gid)}")

    return np.column_stack(columns).astype(np.float32), tuple(names)


def _option_features(options: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    option_values = (-1, 0, 1, 2, 4)
    cols = [(options == value).astype(np.float32) for value in option_values]
    names = [f"option_{value}" for value in option_values]
    return cols, names


def build_edge_arrays(
    edge_table: EdgeTable,
    *,
    node_features: np.ndarray,
    active_mask: np.ndarray | None = None,
    building_mask: np.ndarray | None = None,
    delay_ms: np.ndarray | None = None,
    betweenness: np.ndarray | None = None,
    make_directed: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build edge index and edge features for a time-step graph.

    ``active_mask`` and ``building_mask`` should be 1D arrays aligned to
    ``edge_table``. If omitted, active defaults to one and building to zero.
    """

    num_edges = int(edge_table.num_edges)
    src = np.asarray(edge_table.src, dtype=np.int64)
    dst = np.asarray(edge_table.dst, dtype=np.int64)
    if src.size != num_edges or dst.size != num_edges:
        raise ValueError("edge_table arrays are inconsistent")

    active = np.ones(num_edges, dtype=np.float32) if active_mask is None else np.asarray(active_mask, dtype=np.float32)
    building = np.zeros(num_edges, dtype=np.float32) if building_mask is None else np.asarray(building_mask, dtype=np.float32)
    if active.shape[0] != num_edges or building.shape[0] != num_edges:
        raise ValueError("active/building masks must align with edge_table")

    options = np.asarray(edge_table.option, dtype=np.int16)
    option_cols, option_names = _option_features(options)
    columns = option_cols + [active, building]
    names = option_names + ["active", "building"]

    if delay_ms is not None:
        delay = np.asarray(delay_ms, dtype=np.float32)
        if delay.shape[0] != num_edges:
            raise ValueError("delay_ms must align with edge_table")
        finite = np.isfinite(delay)
        scale = float(np.nanmax(delay[finite])) if np.any(finite) else 1.0
        columns.append(np.where(finite, delay / max(scale, 1e-12), 0.0).astype(np.float32))
        names.append("delay_norm")

    if betweenness is not None:
        between = np.asarray(betweenness, dtype=np.float32)
        if between.shape[0] != num_edges:
            raise ValueError("betweenness must align with edge_table")
        scale = float(np.max(between)) if between.size else 1.0
        columns.append((between / max(scale, 1e-12)).astype(np.float32))
        names.append("betweenness_norm")

    edge_features = np.column_stack(columns).astype(np.float32)
    edge_index = np.vstack([src, dst]).astype(np.int64)
    if make_directed:
        edge_index = np.hstack([edge_index, np.vstack([dst, src])]).astype(np.int64)
        edge_features = np.vstack([edge_features, edge_features]).astype(np.float32)

    if node_features.shape[0] <= int(np.max(edge_index)):
        raise ValueError("node_features has fewer rows than referenced edge nodes")
    return edge_index, edge_features, tuple(names)

