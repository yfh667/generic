from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config.viewer_config import ViewerConfig
from src.satellite_topology_viewer.module import (
    EdgeDelayViewerData,
    load_edge_delay_data_for_viewer,
    load_or_build_group_data,
)

from .position_data import PositionSeries, load_position_series


@dataclass(frozen=True)
class Synced2D3DInputs:
    """Prepared inputs for the synced 2D+3D viewer."""

    config: ViewerConfig
    delay_data: EdgeDelayViewerData
    position_series: PositionSeries
    group_data: dict[int, dict]
    store_dir: Path
    position_cache_dir: Path
    start: int
    end: int
    stride: int
    options: tuple[int, ...]
    groups_enabled: bool
    window_config: dict[str, Any]
    view3d_config: dict[str, Any]
    raw_config: dict[str, Any]


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_path(
    value: Any,
    *,
    base_dir: str | Path,
) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path


def load_station_groups(path: str | Path) -> tuple[dict[int, dict], list[str]]:
    raw = load_yaml_dict(path)
    groups = raw.get("station_group_scheme", {}).get("groups", [])
    station_groups: dict[int, dict] = {}
    colors_by_gid: dict[int, str] = {}
    for item in groups:
        gid = int(item["id"])
        station_groups[gid] = {
            "name": str(item.get("name", f"Group {gid}")),
            "stations": [int(x) for x in item.get("stations", [])],
        }
        colors_by_gid[gid] = str(item.get("color", "#64748b"))

    if not station_groups:
        return {}, []
    max_gid = max(station_groups)
    group_colors = [colors_by_gid.get(gid, "#64748b") for gid in range(max_gid + 1)]
    return station_groups, group_colors


def load_viewer_config(
    viewer_config_file: str | Path,
    station_group_scheme_file: str | Path | None = None,
) -> ViewerConfig:
    raw = load_yaml_dict(viewer_config_file)
    viewer_raw = raw.get("viewer_config", raw)
    if station_group_scheme_file is None:
        station_groups, group_colors = {}, []
    else:
        station_groups, group_colors = load_station_groups(station_group_scheme_file)
    return ViewerConfig(
        name=str(viewer_raw["name"]),
        P=int(viewer_raw["P"]),
        N=int(viewer_raw["N"]),
        station_groups=station_groups,
        group_colors=group_colors,
    )


def position_cache_from_delay_store(store_dir: str | Path) -> Path | None:
    meta_path = Path(store_dir) / "delay_meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    cache_dir = meta.get("signature", {}).get("cache_dir")
    return Path(cache_dir) if cache_dir else None


def load_synced_2d3d_inputs_from_yaml(
    config_file: str | Path,
    *,
    store_dir_override: str | Path | None = None,
    position_cache_dir_override: str | Path | None = None,
    start_override: int | None = None,
    end_override: int | None = None,
    stride_override: int | None = None,
    groups_enabled_override: bool | None = None,
    force_group_cache: bool = False,
) -> Synced2D3DInputs:
    raw = load_yaml_dict(config_file)
    config_dir = Path(config_file).resolve().parent

    viewer_config_file = resolve_path(raw.get("viewer_config_file"), base_dir=config_dir)
    group_scheme_file = resolve_path(raw.get("station_group_scheme_file"), base_dir=config_dir)
    if viewer_config_file is None:
        raise ValueError("viewer_config_file is required")

    config = load_viewer_config(viewer_config_file, group_scheme_file)

    edge_raw = raw.get("edge_data", {}) or {}
    time_raw = raw.get("time", {}) or {}
    group_raw = raw.get("groups", {}) or {}
    position_raw = raw.get("position_data", {}) or {}
    window_raw = raw.get("window", {}) or {}
    view3d_raw = raw.get("view3d", {}) or {}

    store_dir = Path(store_dir_override) if store_dir_override is not None else resolve_path(
        edge_raw.get("store_dir"),
        base_dir=config_dir,
    )
    if store_dir is None:
        raise ValueError("edge_data.store_dir is required")
    store_dir = Path(store_dir)

    start = int(start_override if start_override is not None else time_raw.get("start", 0))
    end = int(end_override if end_override is not None else time_raw.get("end", start))
    stride = int(stride_override if stride_override is not None else time_raw.get("stride", 1))
    options = tuple(int(x) for x in edge_raw.get("options", [0, 1, 2, 4]))

    delay_data = load_edge_delay_data_for_viewer(
        store_dir=store_dir,
        config=config,
        start=start,
        end=end,
        stride=stride,
        options=options,
    )

    position_cache_dir = (
        Path(position_cache_dir_override)
        if position_cache_dir_override is not None
        else resolve_path(position_raw.get("cache_dir"), base_dir=config_dir)
    )
    position_cache_root = resolve_path(position_raw.get("cache_root"), base_dir=config_dir)
    full_position_cache_dir = resolve_path(position_raw.get("full_cache_dir"), base_dir=config_dir)
    if position_cache_dir is None and full_position_cache_dir is None and position_cache_root is None:
        position_cache_dir = position_cache_from_delay_store(store_dir)
    if position_cache_dir is None and full_position_cache_dir is None and position_cache_root is None:
        raise ValueError(
            "position cache is required; set position_data.cache_dir/cache_root/full_cache_dir "
            "or provide a delay store with delay_meta.signature.cache_dir"
        )

    position_series = load_position_series(
        cache_dir=position_cache_dir,
        cache_root=position_cache_root,
        full_cache_dir=full_position_cache_dir,
        start=start,
        end=end,
        stride=stride,
    )

    groups_enabled = bool(group_raw.get("enabled", True))
    if groups_enabled_override is not None:
        groups_enabled = bool(groups_enabled_override)

    group_cache_dir = resolve_path(group_raw.get("cache_dir"), base_dir=config_dir) or (
        Path(store_dir).parent / "group_data_cache"
    )
    group_data = load_or_build_group_data(
        xml_file=resolve_path(group_raw.get("xml_file"), base_dir=config_dir),
        group_cache_dir=group_cache_dir,
        steps=delay_data.steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=stride,
        enabled=groups_enabled,
        force=bool(force_group_cache or group_raw.get("force_cache", False)),
    )

    return Synced2D3DInputs(
        config=config,
        delay_data=delay_data,
        position_series=position_series,
        group_data=group_data,
        store_dir=Path(store_dir),
        position_cache_dir=Path(position_series.cache_dir),
        start=start,
        end=end,
        stride=stride,
        options=options,
        groups_enabled=groups_enabled,
        window_config=dict(window_raw),
        view3d_config=dict(view3d_raw),
        raw_config=dict(raw),
    )
