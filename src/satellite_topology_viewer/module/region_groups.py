from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path


def group_cache_path(
    *,
    xml_file: str | Path,
    cache_dir: str | Path,
    constellation_name: str,
    start: int,
    end: int,
    stride: int,
) -> Path:
    stem = Path(xml_file).stem
    return Path(cache_dir) / f"{stem}_{constellation_name}_t{int(start)}_{int(end)}_stride{int(stride)}.json"


def station_to_group_map(station_groups: dict[int, dict]) -> dict[int, int]:
    out: dict[int, int] = {}
    for gid, info in (station_groups or {}).items():
        for sid in info.get("stations", []) or []:
            out[int(sid)] = int(gid)
    return out


def parse_xml_group_data_fast(
    *,
    xml_file: str | Path,
    start_step: int,
    end_step: int,
    station_groups: dict[int, dict],
    total_sats: int,
) -> dict[int, dict]:
    station_to_gid = station_to_group_map(station_groups)
    groups_template = {int(gid): set() for gid in station_groups}
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
                    if 0 <= sat_id < int(total_sats):
                        sats.add(sat_id)
                        all_mentioned.add(sat_id)

        group_data[int(step)] = {"groups": groups, "all_mentioned": all_mentioned}
        elem.clear()

    return group_data


def load_group_cache(
    *,
    path: str | Path,
    xml_file: str | Path,
    start: int,
    end: int,
    stride: int,
):
    path = Path(path)
    xml_file = Path(xml_file)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    source = payload.get("source", {})
    xml_stat = xml_file.stat()
    expected = {
        "xml_file": str(xml_file.resolve()),
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


def save_group_cache(
    *,
    path: str | Path,
    xml_file: str | Path,
    start: int,
    end: int,
    stride: int,
    group_data: dict[int, dict],
) -> None:
    path = Path(path)
    xml_file = Path(xml_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    xml_stat = xml_file.stat()
    payload = {
        "source": {
            "xml_file": str(xml_file.resolve()),
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
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_or_build_group_data(
    *,
    xml_file: str | Path | None,
    group_cache_dir: str | Path,
    steps: list[int],
    station_groups: dict[int, dict],
    total_sats: int,
    constellation_name: str,
    stride: int = 1,
    enabled: bool = True,
    force: bool = False,
) -> dict[int, dict]:
    if not enabled or not station_groups or not xml_file:
        return {}

    xml_file = Path(xml_file)
    if not xml_file.exists():
        print(f"[sat-topology-viewer] group xml not found: {xml_file}", flush=True)
        return {}

    start = min(int(x) for x in steps)
    end = max(int(x) for x in steps)
    cache_path = group_cache_path(
        xml_file=xml_file,
        cache_dir=group_cache_dir,
        constellation_name=constellation_name,
        start=start,
        end=end,
        stride=stride,
    )
    if not force:
        cached = load_group_cache(path=cache_path, xml_file=xml_file, start=start, end=end, stride=stride)
        if cached is not None:
            print(f"[sat-topology-viewer] Reusing group cache: {cache_path}", flush=True)
            return cached

    print(f"[sat-topology-viewer] Parsing region group data from {xml_file} for {start}..{end}", flush=True)
    group_data = parse_xml_group_data_fast(
        xml_file=xml_file,
        start_step=start,
        end_step=end,
        station_groups=station_groups,
        total_sats=total_sats,
    )
    save_group_cache(path=cache_path, xml_file=xml_file, start=start, end=end, stride=stride, group_data=group_data)
    print(f"[sat-topology-viewer] Wrote group cache: {cache_path}", flush=True)
    return group_data
