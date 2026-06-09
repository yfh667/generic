from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


GENERIC_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = GENERIC_ROOT.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "g60_edge_delay_viewer.yaml"
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.satellite_topology_viewer.module import (
    load_edge_delay_data_for_viewer,
    load_or_build_group_data,
    run_edge_delay_viewer,
)


def load_yaml_dict(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the G60 edge-delay 2D topology viewer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw = load_yaml_dict(args.config)
    edge_cfg = raw.get("edge_data", {}) or {}
    time_cfg = raw.get("time", {}) or {}
    group_cfg = raw.get("groups", {}) or {}
    window_cfg = raw.get("window", {}) or {}

    store_dir = args.store_dir or resolve_path(edge_cfg.get("store_dir"))
    if store_dir is None:
        raise ValueError("edge_data.store_dir is required")
    options = tuple(int(x) for x in edge_cfg.get("options", [0, 1, 2, 4]))

    start = int(args.start if args.start is not None else time_cfg.get("start", 0))
    end = int(args.end if args.end is not None else time_cfg.get("end", 100))
    stride = int(args.stride if args.stride is not None else time_cfg.get("stride", 1))

    delay_data = load_edge_delay_data_for_viewer(
        store_dir=store_dir,
        config=G60_CONFIG,
        start=start,
        end=end,
        stride=stride,
        options=options,
    )

    groups_enabled = bool(group_cfg.get("enabled", True)) and not bool(args.no_groups)
    group_data = load_or_build_group_data(
        xml_file=resolve_path(group_cfg.get("xml_file")),
        group_cache_dir=resolve_path(group_cfg.get("cache_dir"))
        or (Path(store_dir).parent / "group_data_cache"),
        steps=delay_data.steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=stride,
        enabled=groups_enabled,
        force=bool(group_cfg.get("force_cache", False)) or bool(args.force_group_cache),
    )

    return run_edge_delay_viewer(
        config=G60_CONFIG,
        delay_data=delay_data,
        group_data=group_data,
        width=int(args.width if args.width is not None else window_cfg.get("width", 1200)),
        height=int(args.height if args.height is not None else window_cfg.get("height", 760)),
        show_groups=groups_enabled,
        check_only=bool(window_cfg.get("check_only", False)) or bool(args.check_only),
        offscreen=bool(window_cfg.get("offscreen", False)) or bool(args.offscreen),
        screenshot=args.screenshot or resolve_path(window_cfg.get("screenshot")),
    )


if __name__ == "__main__":
    raise SystemExit(main())
