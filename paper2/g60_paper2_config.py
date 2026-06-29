from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


GENERIC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
THIS_DIR = Path(__file__).resolve().parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_g60_voc_timeseries import GROUP_LABELS, build_station_specs
from src.config.viewer_config import ViewerConfig


DATA_ROOT = PROJECT_ROOT / "data_paper2" / "newdata"
DEFAULT_XML = DATA_ROOT / "xml" / "G60_baseraan_0.xml"
DEFAULT_GW_XML = DATA_ROOT / "xml" / "gw.xml"
DEFAULT_EPHEM_DIR = DATA_ROOT / "position" / "G60_baseraan0" / "satellite_pos"
DEFAULT_GW_EPHEM_DIR = DATA_ROOT / "position" / "gwposition" / "baseRaan_0" / "satellite_pos"
DEFAULT_POSITION_CACHE_ROOT = DATA_ROOT / "cache" / "position_cache" / "G60_baseraan0"
DEFAULT_GW_POSITION_CACHE_ROOT = DATA_ROOT / "cache" / "position_cache" / "GW_baseRaan_0"
DEFAULT_GROUP_CACHE_DIR = DATA_ROOT / "cache" / "paper2_g60_group_viewer" / "group_data"
DEFAULT_GW_GROUP_CACHE_DIR = DATA_ROOT / "cache" / "paper2_gw_group_viewer" / "group_data"
DEFAULT_MAPPING_DIR = DATA_ROOT / "cache" / "paper2_g60_group_viewer" / "station_mapping"
DEFAULT_GW_MAPPING_DIR = DATA_ROOT / "cache" / "paper2_gw_group_viewer" / "station_mapping"
DEFAULT_SCREENSHOT_DIR = DATA_ROOT / "cache" / "paper2_g60_group_viewer" / "screenshots"

GROUP_COLORS = [
    "#D81B60",  # China
    "#1E88E5",  # Europe
    "#FFC107",  # North America
    "#43A047",  # Africa
    "#8E24AA",  # South America
]


def build_paper2_constellation_config(*, name: str, p: int, n: int) -> ViewerConfig:
    specs = build_station_specs()
    station_groups: dict[int, dict] = {}
    group_order: list[str] = []
    for spec in specs:
        if spec.group not in group_order:
            group_order.append(spec.group)

    for group_id, group_name in enumerate(group_order):
        stations = [int(spec.xml_station_id) for spec in specs if spec.group == group_name]
        station_groups[group_id] = {
            "name": GROUP_LABELS.get(group_name, group_name),
            "key": group_name,
            "stations": stations,
        }

    return ViewerConfig(
        name=str(name),
        N=int(n),
        P=int(p),
        station_groups=station_groups,
        group_colors=GROUP_COLORS[: len(station_groups)],
    )


def build_paper2_g60_config() -> ViewerConfig:
    return build_paper2_constellation_config(name="G60_paper2", p=18, n=36)


def build_paper2_gw_config() -> ViewerConfig:
    return build_paper2_constellation_config(name="GW_paper2", p=18, n=48)


def write_station_group_mapping(*, out_dir: Path, config: ViewerConfig) -> None:
    specs = build_station_specs()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    xml_station_to_group = {}
    for gid, info in config.station_groups.items():
        for xml_station_id in info["stations"]:
            xml_station_to_group[int(xml_station_id)] = (int(gid), str(info["name"]), str(info.get("key", "")))

    for spec in specs:
        group_id, group_label, group_key = xml_station_to_group[int(spec.xml_station_id)]
        rows.append(
            {
                **asdict(spec),
                "group_id": group_id,
                "group_label": group_label,
                "group_key": group_key,
            }
        )

    with (out_dir / "paper2_g60_station_order.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with (out_dir / "paper2_g60_station_order.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "station_id",
                "xml_station_id",
                "group_id",
                "group",
                "group_label",
                "group_key",
                "group_index",
                "lat",
                "lon",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
