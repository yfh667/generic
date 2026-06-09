from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PyQt5 import QtCore, QtWidgets


GENERIC_ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = Path(__file__).resolve().parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from src.config.viewer_config import G60_CONFIG
from full_link_delay_viewer import (
    build_full_option_edges,
    build_or_load_artifacts,
    choose_default_cache_dir,
    default_output_dir,
)
from grid_fake_delay_viewer_100s import GridFakeDelayViewer


PROJECT_ROOT = GENERIC_ROOT.parent
DEFAULT_EXISTING_DELAY_DIR = (
    PROJECT_ROOT
    / "data"
    / "postprocess"
    / "full_option_edge_delay"
    / "cache_0_1000_1s_t0_100_stride1"
)
DEFAULT_GROUP_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE_DIR = PROJECT_ROOT / "data" / "postprocess" / "full_option_edge_delay" / "group_data_cache"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the fixed-grid topology viewer with real propagation delays from satellite positions."
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--chunk-steps", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-incomplete-cache", action="store_true")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=760)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_GROUP_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--rev-group-base-id", type=int, default=None)
    return parser.parse_args(argv)


def build_real_delay_artifacts(args: argparse.Namespace):
    if args.out_dir is not None:
        existing = load_existing_delay_artifacts(args.out_dir)
        if existing is not None:
            return existing

    if (
        args.cache_dir is None
        and args.out_dir is None
        and int(args.start) == 0
        and int(args.end) == 100
        and int(args.stride) == 1
    ):
        existing = load_existing_delay_artifacts(DEFAULT_EXISTING_DELAY_DIR)
        if existing is not None:
            return existing

    cache_dir = args.cache_dir or choose_default_cache_dir()
    out_dir = args.out_dir or default_output_dir(cache_dir, args.start, args.end, args.stride)
    existing = load_existing_delay_artifacts(out_dir)
    if existing is not None and not args.force:
        return existing

    return build_or_load_artifacts(
        cache_dir=cache_dir,
        out_dir=out_dir,
        config=G60_CONFIG,
        start=args.start,
        end=args.end,
        stride=args.stride,
        force=args.force,
        chunk_steps=args.chunk_steps,
        allow_incomplete_cache=args.allow_incomplete_cache,
        options=(0, 1, 2, 4),
    )


def load_existing_delay_artifacts(out_dir: Path):
    out_dir = Path(out_dir)
    delay_path = out_dir / "edge_delay_ms.npy"
    time_indices_path = out_dir / "time_indices.npy"
    times_path = out_dir / "times_s.npy"
    meta_path = out_dir / "delay_meta.json"
    required = (delay_path, time_indices_path, times_path, meta_path)
    if not all(path.exists() for path in required):
        return None

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    edge_table = build_full_option_edges(G60_CONFIG, options=(0, 1, 2, 4))
    delay = np.load(delay_path, mmap_mode="r")
    if int(delay.shape[1]) != int(edge_table.num_edges):
        raise ValueError(f"{delay_path} has {delay.shape[1]} edges, expected {edge_table.num_edges}")

    return SimpleNamespace(
        cache_dir=Path(meta.get("signature", {}).get("cache_dir", "")),
        out_dir=out_dir,
        delay_path=delay_path,
        edges_csv_path=out_dir / "edges.csv",
        meta_path=meta_path,
        time_indices_path=time_indices_path,
        times_path=times_path,
        edge_table=edge_table,
        time_indices=np.load(time_indices_path, mmap_mode="r"),
        times_s=np.load(times_path, mmap_mode="r"),
        meta=meta,
    )


def group_cache_path(xml_file: Path, cache_dir: Path, start: int, end: int, stride: int) -> Path:
    stem = Path(xml_file).stem
    return Path(cache_dir) / f"{stem}_G60_t{int(start)}_{int(end)}_stride{int(stride)}.json"


def parse_xml_group_data_fast(xml_file: Path, start_step: int, end_step: int) -> dict[int, dict]:
    station_to_gid: dict[int, int] = {}
    for gid, info in G60_CONFIG.station_groups.items():
        for sid in info["stations"]:
            station_to_gid[int(sid)] = int(gid)

    groups_template = {int(gid): set() for gid in G60_CONFIG.station_groups}
    group_data: dict[int, dict] = {}
    context = ET.iterparse(str(xml_file), events=("end",))

    for _event, elem in context:
        if elem.tag != "time":
            continue
        step_attr = elem.get("step")
        if step_attr is None:
            elem.clear()
            continue
        try:
            step = int(step_attr)
        except ValueError:
            elem.clear()
            continue

        if step < int(start_step):
            elem.clear()
            continue
        if step > int(end_step):
            elem.clear()
            break

        groups = {gid: set() for gid in groups_template}
        all_mentioned: set[int] = set()
        stations_elem = elem.find("stations")
        if stations_elem is not None:
            for station_elem in stations_elem.findall("station"):
                sid_attr = station_elem.get("id")
                if sid_attr is None:
                    continue
                try:
                    sid = int(sid_attr)
                except ValueError:
                    continue
                gid = station_to_gid.get(sid)
                if gid is None:
                    continue

                sats = groups[gid]
                for sat_elem in station_elem:
                    if sat_elem.tag != "satellite":
                        continue
                    sat_id_attr = sat_elem.get("id")
                    if not sat_id_attr:
                        continue
                    try:
                        sat_id = int(sat_id_attr)
                    except ValueError:
                        try:
                            sat_id = int(float(sat_id_attr))
                        except ValueError:
                            continue
                    if 0 <= sat_id < int(G60_CONFIG.total_sats):
                        sats.add(sat_id)
                        all_mentioned.add(sat_id)

        group_data[int(step)] = {"groups": groups, "all_mentioned": all_mentioned}
        elem.clear()

    return group_data


def load_group_cache(path: Path, xml_file: Path, start: int, end: int, stride: int):
    if not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    source = payload.get("source", {})
    xml_stat = Path(xml_file).stat()
    expected = {
        "xml_file": str(Path(xml_file).resolve()),
        "xml_size": int(xml_stat.st_size),
        "xml_mtime_ns": int(xml_stat.st_mtime_ns),
        "start": int(start),
        "end": int(end),
        "stride": int(stride),
    }
    if source != expected:
        return None

    group_data: dict[int, dict] = {}
    for step_key, groups_obj in payload.get("groups_by_step", {}).items():
        groups = {int(gid): set(int(x) for x in nodes) for gid, nodes in groups_obj.items()}
        all_mentioned = set()
        for nodes in groups.values():
            all_mentioned.update(nodes)
        group_data[int(step_key)] = {"groups": groups, "all_mentioned": all_mentioned}
    return group_data


def save_group_cache(path: Path, xml_file: Path, start: int, end: int, stride: int, group_data: dict[int, dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    xml_stat = Path(xml_file).stat()
    payload = {
        "source": {
            "xml_file": str(Path(xml_file).resolve()),
            "xml_size": int(xml_stat.st_size),
            "xml_mtime_ns": int(xml_stat.st_mtime_ns),
            "start": int(start),
            "end": int(end),
            "stride": int(stride),
        },
        "groups_by_step": {
            str(step): {
                str(gid): sorted(int(x) for x in nodes)
                for gid, nodes in data.get("groups", {}).items()
            }
            for step, data in sorted(group_data.items())
        },
    }
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_or_build_group_data(args: argparse.Namespace, steps: list[int]) -> dict[int, dict]:
    if args.no_groups:
        return {}
    xml_file = Path(args.xml_file)
    if not xml_file.exists():
        print(f"[grid-real-delay] group xml not found: {xml_file}")
        return {}

    start = min(int(x) for x in steps)
    end = max(int(x) for x in steps)
    cache_path = group_cache_path(xml_file, args.group_cache_dir, start, end, args.stride)
    if not args.force_group_cache:
        cached = load_group_cache(cache_path, xml_file, start, end, args.stride)
        if cached is not None:
            print(f"[grid-real-delay] Reusing group cache: {cache_path}")
            return cached

    print(f"[grid-real-delay] Parsing region group data from {xml_file} for {start}..{end}")
    group_data = parse_xml_group_data_fast(xml_file, start, end)
    save_group_cache(cache_path, xml_file, start, end, args.stride, group_data)
    print(f"[grid-real-delay] Wrote group cache: {cache_path}")
    return group_data


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    artifacts = build_real_delay_artifacts(args)
    delay_ms = np.load(artifacts.delay_path, mmap_mode="r")
    steps = [int(x) for x in np.asarray(artifacts.time_indices)]
    group_data = load_or_build_group_data(args, steps)
    delay_min = float(artifacts.meta.get("delay_min_ms") or np.nanmin(delay_ms))
    delay_max = float(artifacts.meta.get("delay_max_ms") or np.nanmax(delay_ms))
    print(
        f"[grid-real-delay] steps={len(steps)}, edges_per_step={artifacts.edge_table.num_edges}, "
        f"delay_ms=({delay_min:.4f}, {delay_max:.4f}), "
        f"group_steps={len(group_data)}, cache={artifacts.cache_dir}"
    )

    if args.check_only:
        return 0

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = GridFakeDelayViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=artifacts.edge_table,
        delay_ms=delay_ms,
        delay_min_ms=delay_min,
        delay_max_ms=delay_max,
        window_title=f"Grid real full-option ISL propagation delay {steps[0]}..{steps[-1]}s",
        delay_value_label="delay_ms",
        group_data=group_data,
        show_groups=True,
        rev_group_base_id=args.rev_group_base_id,
    )
    viewer.resize(int(args.width), int(args.height))
    viewer.show()

    if args.screenshot is not None:
        screenshot_path = Path(args.screenshot)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        def save_screenshot_and_quit():
            app.processEvents()
            ok = viewer.grab().save(str(screenshot_path))
            print(f"[grid-real-delay] screenshot={screenshot_path} ok={ok}")
            app.quit()

        QtCore.QTimer.singleShot(800, save_screenshot_and_quit)

    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
