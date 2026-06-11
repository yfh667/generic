from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = THIS_DIR / "configs" / "starlink_2d3d_viewer_test100.yaml"

if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.satellite_topology_viewer_3D.module import (  # noqa: E402
    load_synced_2d3d_inputs_from_yaml,
    run_synced_2d3d_viewer,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synced 2D + 3D Starlink topology viewer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--position-cache-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--link-stride", type=int, default=None)
    parser.add_argument("--no-3d-links", action="store_true")
    parser.add_argument("--no-orbits", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--timer-interval-ms", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = load_synced_2d3d_inputs_from_yaml(
        args.config,
        store_dir_override=args.store_dir,
        position_cache_dir_override=args.position_cache_dir,
        start_override=args.start,
        end_override=args.end,
        stride_override=args.stride,
        groups_enabled_override=False if args.no_groups else None,
        force_group_cache=args.force_group_cache,
    )

    window_cfg = inputs.window_config
    view3d_cfg = inputs.view3d_config

    width = int(args.width if args.width is not None else window_cfg.get("width", 1600))
    height = int(args.height if args.height is not None else window_cfg.get("height", 900))
    link_stride = int(args.link_stride if args.link_stride is not None else view3d_cfg.get("link_stride", 1))
    timer_interval_ms = int(
        args.timer_interval_ms if args.timer_interval_ms is not None else view3d_cfg.get("timer_interval_ms", 180)
    )
    check_only = bool(args.check_only or window_cfg.get("check_only", False))
    offscreen = bool(args.offscreen or window_cfg.get("offscreen", False))
    screenshot = args.screenshot or window_cfg.get("screenshot")

    show_3d_links = bool(view3d_cfg.get("show_links", True)) and not bool(args.no_3d_links)
    show_3d_orbits = bool(view3d_cfg.get("show_orbits", True)) and not bool(args.no_orbits)

    print(
        f"[starlink-2d3d] config={inputs.config.name} steps={len(inputs.delay_data.steps)} "
        f"edges={inputs.delay_data.edge_table.num_edges}",
        flush=True,
    )
    print(
        f"[starlink-2d3d] 3d_position_source=real_position_cache "
        f"cache={inputs.position_cache_dir} "
        f"shape={tuple(int(x) for x in inputs.position_series.positions_km.shape)} "
        f"time={inputs.position_series.steps[0]}..{inputs.position_series.steps[-1]} "
        f"stride={inputs.stride}",
        flush=True,
    )

    return run_synced_2d3d_viewer(
        config=inputs.config,
        delay_data=inputs.delay_data,
        position_series=inputs.position_series,
        group_data=inputs.group_data,
        width=width,
        height=height,
        show_groups=inputs.groups_enabled,
        show_3d_links=show_3d_links,
        show_3d_orbits=show_3d_orbits,
        link_stride=max(1, link_stride),
        timer_interval_ms=timer_interval_ms,
        check_only=check_only,
        offscreen=offscreen,
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
