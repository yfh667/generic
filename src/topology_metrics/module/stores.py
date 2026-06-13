from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .group_states import GroupState


@dataclass(frozen=True)
class MetricStoreLayout:
    """Conventional file names for a topology metric store."""

    root: Path

    @property
    def edges_csv(self) -> Path:
        return self.root / "edges.csv"

    @property
    def time_indices_npy(self) -> Path:
        return self.root / "time_indices.npy"

    @property
    def state_ids_npy(self) -> Path:
        return self.root / "state_ids.npy"

    @property
    def state_definitions_json(self) -> Path:
        return self.root / "state_definitions.json"

    @property
    def unique_state_values_npy(self) -> Path:
        return self.root / "unique_state_values.npy"

    @property
    def state_summary_csv(self) -> Path:
        return self.root / "state_summary.csv"

    @property
    def metric_values_npy(self) -> Path:
        return self.root / "metric_values.npy"

    @property
    def step_summary_csv(self) -> Path:
        return self.root / "step_summary.csv"

    @property
    def meta_json(self) -> Path:
        return self.root / "meta.json"


def write_state_definitions(path: str | Path, unique_states: Sequence[GroupState]) -> None:
    """Write unique source/target node-set states as JSON."""

    payload = [
        {
            "state_id": int(idx),
            "source_nodes": [int(x) for x in source_nodes],
            "target_nodes": [int(x) for x in target_nodes],
        }
        for idx, (source_nodes, target_nodes) in enumerate(unique_states)
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def expand_unique_state_values(
    *,
    unique_state_values: np.ndarray,
    state_ids: np.ndarray,
) -> np.ndarray:
    """Expand `(state, edge)` values to `(step, edge)` values."""

    state_ids = np.asarray(state_ids, dtype=np.int64)
    values = np.asarray(unique_state_values)
    if values.ndim != 2:
        raise ValueError("unique_state_values must be a 2D array")
    if state_ids.size and (np.min(state_ids) < 0 or np.max(state_ids) >= values.shape[0]):
        raise ValueError("state_ids contain an index outside unique_state_values")
    return values[state_ids, :]


def write_meta(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

