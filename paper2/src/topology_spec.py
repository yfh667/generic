from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable
from src.topology_workflow.module.edge_tables import build_edge_table_from_topology_config


PLUS_GRID_MOTIF: dict[str, Any] = {
    "w": 2,
    "h": 1,
    "support": [
        (0, 0, "A"),
    ],
}


def normalize_topology_alias(value: str | Mapping[str, Any]) -> str | dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "+grid": "plus_grid",
        "grid+": "plus_grid",
        "grid_plus": "plus_grid",
        "plusgrid": "plus_grid",
    }
    return aliases.get(token, token)


def motif_topology_config(
    motif: Mapping[str, Any],
    *,
    add_intra_ring: bool = True,
    wrap_planes: bool = False,
    allow_vertical_overlap: bool = True,
    allow_clipped_right: bool = True,
    horizontal_step: int | None = None,
) -> dict[str, Any]:
    tiling: dict[str, Any] = {
        "allow_vertical_overlap": bool(allow_vertical_overlap),
        "allow_clipped_right": bool(allow_clipped_right),
    }
    if horizontal_step is not None:
        tiling["horizontal_step"] = int(horizontal_step)
    return {
        "kind": "single_motif",
        "motif": deepcopy(dict(motif)),
        "tiling": tiling,
        "add_intra_ring": bool(add_intra_ring),
        "wrap_planes": bool(wrap_planes),
    }


def topology_config_from_input(
    topology: str | Mapping[str, Any],
    *,
    add_intra_ring: bool = True,
) -> dict[str, Any]:
    normalized = normalize_topology_alias(topology)
    if isinstance(normalized, dict):
        if "kind" in normalized:
            return dict(normalized)
        return motif_topology_config(normalized, add_intra_ring=add_intra_ring)
    if normalized == "plus_grid":
        return motif_topology_config(PLUS_GRID_MOTIF, add_intra_ring=add_intra_ring)
    raise ValueError(
        f"Unsupported topology {topology!r}. Pass a motif dict or use alias '+grid'/'plus_grid'."
    )


def build_edge_table_from_paper2_topology(
    *,
    config: ViewerConfig,
    topology: str | Mapping[str, Any],
    add_intra_ring: bool = True,
) -> EdgeTable:
    topology_raw = topology_config_from_input(topology, add_intra_ring=add_intra_ring)
    return build_edge_table_from_topology_config(topology_raw=topology_raw, config=config)

