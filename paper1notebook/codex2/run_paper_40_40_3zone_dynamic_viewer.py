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
REPRO_ROOT = Path("E:/") / "\u590d\u73b0" / "satnetwork.github.io"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"

if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPRO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(REPRO_SCRIPTS))

from evaluate_motif_timeseries import (  # noqa: E402
    CONSTELLATIONS,
    calibrate_orbit_model,
    generate_sat_positions,
    read_sat_positions,
)

from src.config.viewer_config import ViewerConfig  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402

from paper_3zone_dynamic_viewer import (  # noqa: E402
    Paper3ZoneDynamicViewer,
    load_or_build_paper_3zone_topology_cache,
    read_level_wise_best_motif,
)


PAPER_CONFIG_NAME = "40_40_53deg"
PAPER_VIEWER_CONFIG = ViewerConfig(
    name=PAPER_CONFIG_NAME,
    P=40,
    N=40,
    station_groups={},
    group_colors=[],
)
DEFAULT_AUTHOR_MOTIF_FILE = (
    REPRO_ROOT
    / "output_data"
    / "multi_motif_40_40_53deg_5014"
    / "level_wise_best_motif.txt"
)
DEFAULT_T0_SAT_POSITIONS = (
    REPRO_ROOT
    / "input_data"
    / "constellation_40_40_53deg"
    / "data_sat_position"
    / "sat_positions_0.txt"
)
DEFAULT_POSITION_CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "caches"
    / "paper_constellations"
    / PAPER_CONFIG_NAME
    / "position_cache"
)
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "data"
    / "satnet_experiments"
    / "runs"
    / "paper1"
    / "paper_constellations"
    / PAPER_CONFIG_NAME
    / "paper_3zone_dynamic_demo"
)
DEFAULT_TOPOLOGY_CACHE_ROOT = DEFAULT_RUN_ROOT / "cache"
DEFAULT_SCREENSHOT = DEFAULT_RUN_ROOT / "paper_40_40_3zone_dynamic_0_10min.png"


def default_position_cache_dir(start_minute: int, end_minute: int, stride_minute: int) -> Path:
    return DEFAULT_POSITION_CACHE_ROOT / (
        f"t{int(start_minute)}_{int(end_minute)}_stride{int(stride_minute)}min"
    )


def default_topology_cache_dir(
    *,
    start_minute: int,
    end_minute: int,
    stride_minute: int,
    author_motif_file: Path,
    max_isl_km: float | None,
    wrap_planes: bool,
) -> Path:
    range_tag = "range_none" if max_isl_km is None else f"range{float(max_isl_km):g}km"
    wrap_tag = "wrap1" if wrap_planes else "wrap0"
    motif_tag = Path(author_motif_file).parent.name
    return DEFAULT_TOPOLOGY_CACHE_ROOT / (
        f"{motif_tag}_t{int(start_minute)}_{int(end_minute)}_stride{int(stride_minute)}min_{range_tag}_{wrap_tag}"
    )


def build_paper_position_cache(
    *,
    minutes: list[int],
    cache_dir: Path,
    t0_sat_positions: Path,
    force: bool,
) -> Path:
    spec = CONSTELLATIONS[PAPER_CONFIG_NAME]
    expected = {
        "schema": "paper_constellation_position_cache_v1",
        "config": PAPER_CONFIG_NAME,
        "num_orbits": int(spec.num_orbits),
        "sats_per_orbit": int(spec.sats_per_orbit),
        "total_sats": int(spec.total_sats),
        "inclination_deg": float(spec.inclination_deg),
        "mean_motion_rev_per_day": float(spec.mean_motion_rev_per_day),
        "phase_diff": bool(spec.phase_diff),
        "source_t0_sat_positions": str(Path(t0_sat_positions).resolve()),
        "time_unit": "minute",
        "steps_start": int(minutes[0]),
        "steps_end": int(minutes[-1]),
        "steps_stride": int(minutes[1] - minutes[0]) if len(minutes) > 1 else 1,
        "num_steps": len(minutes),
    }

    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    pos_path = cache_dir / "positions_km.npy"
    times_path = cache_dir / "times_s.npy"
    if not force and meta_path.exists() and pos_path.exists() and times_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if all(meta.get(key) == value for key, value in expected.items()):
            print(f"[paper-40x40] Reusing position cache: {cache_dir}", flush=True)
            return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    t0_sats = read_sat_positions(Path(t0_sat_positions))
    theta0_rad, altitude_km = calibrate_orbit_model(spec, t0_sats)
    positions = np.empty((len(minutes), int(spec.total_sats), 3), dtype=np.float32)
    for row, minute in enumerate(minutes):
        sats = generate_sat_positions(
            spec,
            minute=int(minute),
            theta0_rad=float(theta0_rad),
            altitude_km=float(altitude_km),
        )
        positions[row] = np.asarray([sat["xyz"] for sat in sats], dtype=np.float32)

    expected["theta0_rad"] = float(theta0_rad)
    expected["altitude_km"] = float(altitude_km)
    np.save(pos_path, positions)
    np.save(times_path, np.asarray(minutes, dtype=np.int32))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(expected, f, ensure_ascii=False, indent=2)
    print(
        f"[paper-40x40] Built position cache: {cache_dir} "
        f"shape={positions.shape} altitude_km={altitude_km:.3f}",
        flush=True,
    )
    return cache_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper 40x40 53deg constellation 3-zone multi-motif links in the 2D viewer."
    )
    parser.add_argument("--start-minute", type=int, default=0)
    parser.add_argument("--end-minute", type=int, default=10)
    parser.add_argument("--stride-minute", type=int, default=1)
    parser.add_argument("--author-motif-file", type=Path, default=DEFAULT_AUTHOR_MOTIF_FILE)
    parser.add_argument("--t0-sat-positions", type=Path, default=DEFAULT_T0_SAT_POSITIONS)
    parser.add_argument("--position-cache-dir", type=Path, default=None)
    parser.add_argument("--force-position-cache", action="store_true")
    parser.add_argument("--max-isl-km", type=float, default=5014.0)
    parser.add_argument("--no-range-filter", action="store_true")
    parser.add_argument("--no-plane-wrap", action="store_true")
    parser.add_argument("--topology-cache-dir", type=Path, default=None)
    parser.add_argument("--force-topology-cache", action="store_true")
    parser.add_argument("--no-topology-cache", action="store_true")
    parser.add_argument("--width", type=int, default=1700)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.stride_minute) <= 0:
        raise ValueError("--stride-minute must be positive")
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    minutes = list(range(int(args.start_minute), int(args.end_minute) + 1, int(args.stride_minute)))
    if not minutes:
        raise ValueError("empty minute range; check --start-minute/--end-minute/--stride-minute")

    position_cache_dir = args.position_cache_dir
    if position_cache_dir is None:
        position_cache_dir = default_position_cache_dir(
            int(args.start_minute), int(args.end_minute), int(args.stride_minute)
        )
    position_cache_dir = build_paper_position_cache(
        minutes=minutes,
        cache_dir=Path(position_cache_dir),
        t0_sat_positions=Path(args.t0_sat_positions),
        force=bool(args.force_position_cache),
    )

    motifs = read_level_wise_best_motif(Path(args.author_motif_file))
    max_isl_km = None if bool(args.no_range_filter) else float(args.max_isl_km)
    topology_cache_dir = None if bool(args.no_topology_cache) else args.topology_cache_dir
    if topology_cache_dir is None and not bool(args.no_topology_cache):
        topology_cache_dir = default_topology_cache_dir(
            start_minute=int(args.start_minute),
            end_minute=int(args.end_minute),
            stride_minute=int(args.stride_minute),
            author_motif_file=Path(args.author_motif_file),
            max_isl_km=max_isl_km,
            wrap_planes=not bool(args.no_plane_wrap),
        )

    series = load_or_build_paper_3zone_topology_cache(
        config=PAPER_VIEWER_CONFIG,
        steps=minutes,
        position_cache_dir=position_cache_dir,
        motifs=motifs,
        cache_dir=topology_cache_dir,
        force=bool(args.force_topology_cache),
        max_isl_km=max_isl_km,
        wrap_planes=not bool(args.no_plane_wrap),
        author_motif_file=Path(args.author_motif_file),
    )

    active_counts = series.edge_active_mask.sum(axis=1)
    print(
        f"[paper-40x40] config={PAPER_CONFIG_NAME} minutes={minutes[0]}..{minutes[-1]} "
        f"rows={len(minutes)} union_edges={series.edge_table.num_edges} "
        f"active_edges_min={int(active_counts.min())} active_edges_max={int(active_counts.max())} "
        f"range_filter={max_isl_km} author_motif={args.author_motif_file}",
        flush=True,
    )
    for idx, motif in enumerate(motifs):
        print(f"[paper-40x40] zone{idx}: {motif.label}", flush=True)

    if args.check_only:
        return 0

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = Paper3ZoneDynamicViewer(
        PAPER_VIEWER_CONFIG,
        steps=minutes,
        edge_table=series.edge_table,
        edge_active_mask=series.edge_active_mask,
        edge_zone_ids=series.edge_zone_ids,
        latitude_deg=series.latitude_deg,
        latitude_zone_ids=series.latitude_zone_ids,
        latitude_zones=series.latitude_zones,
        zone_motifs=series.motifs,
        group_data={},
        show_groups=False,
        show_grid_lines=False,
        window_title=(
            f"Paper constellation {PAPER_CONFIG_NAME} 3-zone dynamic links | "
            f"{minutes[0]}..{minutes[-1]} min"
        ),
        topology_edge_alpha=150,
    )

    screenshot = args.screenshot
    if screenshot is None and args.offscreen:
        screenshot = DEFAULT_SCREENSHOT
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
