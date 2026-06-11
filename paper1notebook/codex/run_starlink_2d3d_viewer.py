from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = GENERIC_ROOT.parent
STARLINK_CONFIG_NAME = "edge_delay_viewer_starlink_72_22_test100.yaml"

if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.config.viewer_config import ViewerConfig
from src.satellite_topology_viewer.module import load_edge_delay_data_for_viewer, load_or_build_group_data
from synced_2d3d_viewer import load_position_series, run_synced_2d3d_viewer


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def discover_default_config() -> Path:
    basic_root = PROJECT_ROOT / "data" / "basic_file"
    candidates = [
        basic_root / "Starlink_72_22" / "codex_yaml_test" / STARLINK_CONFIG_NAME,
        basic_root / "Starlink_72_22_1_550" / "codex_yaml_test" / STARLINK_CONFIG_NAME,
    ]
    candidates.extend(sorted(basic_root.glob(f"Starlink_72_22*/codex_yaml_test/{STARLINK_CONFIG_NAME}")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def recover_missing_basic_file_path(path: Path) -> Path:
    if path.exists():
        return path

    parts = list(path.parts)
    try:
        idx = parts.index("basic_file")
    except ValueError:
        return path
    if idx + 1 >= len(parts):
        return path

    basic_root = Path(*parts[: idx + 1])
    old_name = parts[idx + 1]
    tail = parts[idx + 2 :]
    if not basic_root.exists():
        return path

    for sibling in sorted(basic_root.glob(f"{old_name}*")):
        candidate = sibling.joinpath(*tail)
        if candidate.exists() or candidate.parent.exists():
            return candidate
    return path


def resolve_path(value: Any, *, base_dir: str | Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return recover_missing_basic_file_path(path)
    return recover_missing_basic_file_path(Path(base_dir) / path)


def load_station_groups(path: str | Path) -> tuple[dict[int, dict], list[str]]:
    raw = load_yaml_dict(path)
    groups = raw.get("station_group_scheme", {}).get("groups", [])
    station_groups: dict[int, dict] = {}
    colors: dict[int, str] = {}
    for item in groups:
        gid = int(item["id"])
        station_groups[gid] = {
            "name": str(item.get("name", f"Group {gid}")),
            "stations": [int(x) for x in item.get("stations", [])],
        }
        colors[gid] = str(item.get("color", "#64748b"))
    return station_groups, [colors.get(gid, "#64748b") for gid in sorted(station_groups)]


def load_viewer_config(viewer_config_file: str | Path, station_group_scheme_file: str | Path | None) -> ViewerConfig:
    raw = load_yaml_dict(viewer_config_file)
    viewer_raw = raw.get("viewer_config", {})
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
    return recover_missing_basic_file_path(Path(cache_dir)) if cache_dir else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synced 2D + 3D Starlink topology viewer.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--position-cache-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--link-stride", type=int, default=1)
    parser.add_argument("--no-3d-links", action="store_true")
    parser.add_argument("--no-orbits", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--timer-interval-ms", type=int, default=180)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.config is None:
        args.config = discover_default_config()
    raw = load_yaml_dict(args.config)
    config_dir = args.config.resolve().parent

    generic_root = resolve_path(raw.get("project", {}).get("generic_root"), base_dir=config_dir)
    if generic_root is not None and str(generic_root) not in sys.path:
        sys.path.insert(0, str(generic_root))

    if bool(args.offscreen or raw.get("window", {}).get("offscreen", False)):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    viewer_config_file = resolve_path(raw.get("viewer_config_file"), base_dir=config_dir)
    group_scheme_file = resolve_path(raw.get("station_group_scheme_file"), base_dir=config_dir)
    if viewer_config_file is None:
        raise ValueError("viewer_config_file is required")

    config = load_viewer_config(viewer_config_file, group_scheme_file)

    edge_raw = raw.get("edge_data", {}) or {}
    time_raw = raw.get("time", {}) or {}
    group_raw = raw.get("groups", {}) or {}
    window_raw = raw.get("window", {}) or {}

    store_dir = args.store_dir or resolve_path(edge_raw.get("store_dir"), base_dir=config_dir)
    if store_dir is None:
        raise ValueError("edge_data.store_dir is required")

    start = int(args.start if args.start is not None else time_raw.get("start", 0))
    end = int(args.end if args.end is not None else time_raw.get("end", start))
    stride = int(args.stride if args.stride is not None else time_raw.get("stride", 1))
    options = tuple(int(x) for x in edge_raw.get("options", [0, 1, 2, 4]))

    delay_data = load_edge_delay_data_for_viewer(
        store_dir=store_dir,
        config=config,
        start=start,
        end=end,
        stride=stride,
        options=options,
    )

    position_cache_dir = args.position_cache_dir or position_cache_from_delay_store(store_dir)
    if position_cache_dir is None:
        raise ValueError("position cache dir is required; pass --position-cache-dir")
    position_series = load_position_series(cache_dir=position_cache_dir, start=start, end=end, stride=stride)
    pos_shape = tuple(int(x) for x in position_series.positions_km.shape)

    groups_enabled = bool(group_raw.get("enabled", True)) and not bool(args.no_groups)
    group_data = load_or_build_group_data(
        xml_file=resolve_path(group_raw.get("xml_file"), base_dir=config_dir),
        group_cache_dir=resolve_path(group_raw.get("cache_dir"), base_dir=config_dir)
        or (Path(store_dir).parent / "group_data_cache"),
        steps=delay_data.steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=stride,
        enabled=groups_enabled,
        force=bool(args.force_group_cache or group_raw.get("force_cache", False)),
    )

    print(
        f"[starlink-2d3d] config={config.name} steps={len(delay_data.steps)} "
        f"edges={delay_data.edge_table.num_edges}",
        flush=True,
    )
    print(
        f"[starlink-2d3d] 3d_position_source=real_position_cache "
        f"cache={position_cache_dir} shape={pos_shape} "
        f"time={position_series.steps[0]}..{position_series.steps[-1]} stride={stride}",
        flush=True,
    )

    return run_synced_2d3d_viewer(
        config=config,
        delay_data=delay_data,
        position_series=position_series,
        group_data=group_data,
        width=int(args.width if args.width is not None else window_raw.get("width", 1600)),
        height=int(args.height if args.height is not None else window_raw.get("height", 900)),
        show_groups=groups_enabled,
        show_3d_links=not bool(args.no_3d_links),
        show_3d_orbits=not bool(args.no_orbits),
        link_stride=max(1, int(args.link_stride)),
        timer_interval_ms=int(args.timer_interval_ms),
        check_only=bool(args.check_only or window_raw.get("check_only", False)),
        offscreen=bool(args.offscreen or window_raw.get("offscreen", False)),
        screenshot=args.screenshot or resolve_path(window_raw.get("screenshot"), base_dir=config_dir),
    )


if __name__ == "__main__":
    raise SystemExit(main())
