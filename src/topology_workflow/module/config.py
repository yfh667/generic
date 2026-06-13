from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.config.viewer_config import ViewerConfig


def load_workflow_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"workflow YAML root must be a mapping: {path}")
    return data


def optional_path(value: Any) -> Path | None:
    if value in (None, "", False):
        return None
    return Path(str(value))


def _station_groups_from_raw(raw: dict[str, Any]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for raw_gid, raw_info in (raw or {}).items():
        gid = int(raw_gid)
        info = dict(raw_info or {})
        out[gid] = {
            "name": str(info.get("name", f"Group {gid}")),
            "stations": [int(x) for x in (info.get("stations", []) or [])],
        }
    return out


def viewer_config_from_workflow(raw: dict[str, Any]) -> ViewerConfig:
    constellation = raw.get("constellation", {})
    if not isinstance(constellation, dict):
        raise ValueError("workflow.constellation must be a mapping")

    return ViewerConfig(
        name=str(constellation["name"]),
        P=int(constellation["p"]),
        N=int(constellation["n"]),
        station_groups=_station_groups_from_raw(constellation.get("station_groups", {})),
        group_colors=[str(x) for x in (constellation.get("group_colors", []) or [])],
    )


def time_axis_from_config(raw: dict[str, Any]) -> tuple[int, int, int]:
    time_raw = raw.get("time", {})
    if not isinstance(time_raw, dict):
        raise ValueError("workflow.time must be a mapping")
    start = int(time_raw.get("start", 0))
    end = int(time_raw.get("end", start))
    stride = int(time_raw.get("stride", 1))
    if end < start:
        raise ValueError("time.end must be >= time.start")
    if stride <= 0:
        raise ValueError("time.stride must be positive")
    return start, end, stride


def group_name(config: ViewerConfig, group_id: int) -> str:
    return str(config.station_groups.get(int(group_id), {}).get("name", f"Group {int(group_id)}"))
