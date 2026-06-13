from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_dict(path: str | Path) -> dict:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data
