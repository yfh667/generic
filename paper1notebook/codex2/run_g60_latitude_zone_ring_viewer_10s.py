from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.config.viewer_config import G60_CONFIG
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.examples.run_g60_motif_2x2_link_setup_viewer import (
    DEFAULT_GROUP_CACHE,
    DEFAULT_XML,
    MOTIF_2X2_CB,
    build_base_edge_table,
)

from latitude_zone_ring_viewer import DEFAULT_LATITUDE_ZONES, LatitudeZoneRingViewer, load_latitude_zone_data


DEFAULT_POSITION_CACHE = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "_position_cache"
    / "cache_0_86164_1s"
)
DEFAULT_LATITUDE_CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "runs"
    / "paper1"
    / "G60"
    / "latitude_zone_ring_demo"
    / "latitude_zone_cache"
)


def zone_signature() -> list[dict[str, object]]:
    return [
        {
            "name": zone.name,
            "min_abs_deg": float(zone.min_abs_deg),
            "max_abs_deg": float(zone.max_abs_deg),
            "color": zone.color,
        }
        for zone in DEFAULT_LATITUDE_ZONES
    ]


def default_latitude_cache_dir(start: int, end: int, stride: int) -> Path:
    return DEFAULT_LATITUDE_CACHE_ROOT / f"t{int(start)}_{int(end)}_stride{int(stride)}_abs_0_30_60_90"


def load_or_build_latitude_cache(
    *,
    position_cache_dir: Path,
    steps: list[int],
    cache_dir: Path | None,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_dir is None:
        return load_latitude_zone_data(
            position_cache_dir=position_cache_dir,
            steps=steps,
            zones=DEFAULT_LATITUDE_ZONES,
        )

    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    lat_path = cache_dir / "latitude_deg.npy"
    zone_path = cache_dir / "latitude_zone_ids.npy"
    expected = {
        "position_cache_dir": str(Path(position_cache_dir).resolve()),
        "steps_start": int(steps[0]),
        "steps_end": int(steps[-1]),
        "steps_stride": int(steps[1] - steps[0]) if len(steps) > 1 else 1,
        "num_steps": len(steps),
        "zones": zone_signature(),
    }
    if not force and meta_path.exists() and lat_path.exists() and zone_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if all(meta.get(key) == value for key, value in expected.items()):
            print(f"[latitude-zone-viewer] Reusing latitude cache: {cache_dir}", flush=True)
            return np.load(lat_path, mmap_mode="r"), np.load(zone_path, mmap_mode="r")

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[latitude-zone-viewer] Building latitude cache: {cache_dir}", flush=True)
    latitude_deg, latitude_zone_ids = load_latitude_zone_data(
        position_cache_dir=position_cache_dir,
        steps=steps,
        zones=DEFAULT_LATITUDE_ZONES,
    )
    np.save(lat_path, latitude_deg)
    np.save(zone_path, latitude_zone_ids)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(expected, f, ensure_ascii=False, indent=2)
    return np.load(lat_path, mmap_mode="r"), np.load(zone_path, mmap_mode="r")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental 2D topology viewer with latitude-zone rings around each satellite node."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--position-cache-dir", type=Path, default=DEFAULT_POSITION_CACHE)
    parser.add_argument("--latitude-cache-dir", type=Path, default=None)
    parser.add_argument("--force-latitude-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    edge_table = build_base_edge_table()
    latitude_cache_dir = args.latitude_cache_dir
    if latitude_cache_dir is None and int(args.end) - int(args.start) >= 1000:
        latitude_cache_dir = default_latitude_cache_dir(int(args.start), int(args.end), int(args.stride))
    latitude_deg, latitude_zone_ids = load_or_build_latitude_cache(
        position_cache_dir=Path(args.position_cache_dir),
        steps=steps,
        cache_dir=latitude_cache_dir,
        force=bool(args.force_latitude_cache),
    )

    print(
        f"[latitude-zone-viewer] motif={MOTIF_2X2_CB['name']} steps={steps[0]}..{steps[-1]} "
        f"rows={len(steps)} edges={edge_table.num_edges} position_cache={args.position_cache_dir}",
        flush=True,
    )
    for row, step in enumerate(steps[: min(5, len(steps))]):
        counts = np.bincount(
            np.maximum(latitude_zone_ids[row], 0),
            minlength=len(DEFAULT_LATITUDE_ZONES),
        )[: len(DEFAULT_LATITUDE_ZONES)]
        print(
            f"[latitude-zone-viewer] step={step} "
            + " ".join(f"{DEFAULT_LATITUDE_ZONES[idx].name}={int(counts[idx])}" for idx in range(len(counts))),
            flush=True,
        )

    if args.cache_only:
        if latitude_cache_dir is None:
            print("[latitude-zone-viewer] cache-only requested, but no latitude cache dir was used.", flush=True)
        else:
            print(f"[latitude-zone-viewer] latitude cache ready: {latitude_cache_dir}", flush=True)
        return 0

    group_data = {}
    if not args.no_groups:
        group_data = load_or_build_group_data(
            xml_file=Path(args.xml_file),
            group_cache_dir=Path(args.group_cache_dir),
            steps=steps,
            station_groups=G60_CONFIG.station_groups,
            total_sats=G60_CONFIG.total_sats,
            constellation_name=G60_CONFIG.name,
            stride=int(args.stride),
            enabled=True,
            force=bool(args.force_group_cache),
        )

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = LatitudeZoneRingViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        latitude_deg=latitude_deg,
        latitude_zone_ids=latitude_zone_ids,
        latitude_zones=DEFAULT_LATITUDE_ZONES,
        window_title=f"G60 latitude-zone rings + groups | {MOTIF_2X2_CB['name']} | {steps[0]}..{steps[-1]}s",
        group_data=group_data,
        show_groups=not args.no_groups,
        show_grid_lines=False,
        topology_edge_alpha=115,
    )
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        check_only=bool(args.check_only),
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
