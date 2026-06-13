from __future__ import annotations

from pathlib import Path

import yaml

from .support import MotifSupport, motif_support_from_dict


def load_yaml_dict(path: str | Path) -> dict:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_motif_support_yaml(path: str | Path) -> MotifSupport:
    """Load a support-style motif YAML file and return the motif object.

    The YAML may either be a bare motif mapping with ``w/h/support`` or a full
    example config whose motif lives under the top-level ``motif`` key.
    """

    return motif_support_from_dict(load_yaml_dict(path))
