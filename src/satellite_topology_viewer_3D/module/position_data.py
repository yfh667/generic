from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.link_delay.module.position_cache import (
    PositionCacheStore,
    open_position_cache_for_interval,
)


@dataclass(frozen=True)
class PositionSeries:
    """A time-window view over a satellite position cache."""

    cache_dir: Path
    positions_km: np.ndarray
    cache_rows: np.ndarray
    steps: list[int]
    sat_ids: list[str]
    meta: dict

    @property
    def num_steps(self) -> int:
        return int(self.cache_rows.size)

    @property
    def num_sats(self) -> int:
        return int(self.positions_km.shape[1])

    def points_for_row(self, row: int) -> np.ndarray:
        row = int(max(0, min(int(row), self.num_steps - 1)))
        cache_row = int(self.cache_rows[row])
        return np.asarray(self.positions_km[cache_row, :, :], dtype=np.float32)


def load_position_series(
    *,
    start: int,
    end: int,
    stride: int = 1,
    cache_dir: str | Path | None = None,
    cache_root: str | Path | None = None,
    full_cache_dir: str | Path | None = None,
) -> PositionSeries:
    """Load a position time window without rebuilding the cache.

    Pass ``cache_dir`` for an explicit cache, or ``cache_root``/``full_cache_dir``
    when the caller wants the module to choose a cache that covers the interval.
    """

    if cache_root is not None or full_cache_dir is not None:
        store = open_position_cache_for_interval(
            int(start),
            int(end),
            stride=int(stride),
            cache_dir=cache_dir,
            cache_root=cache_root,
            full_cache_dir=full_cache_dir,
        )
    elif cache_dir is not None:
        store = PositionCacheStore(cache_dir)
    else:
        raise ValueError("pass cache_dir, cache_root, or full_cache_dir")

    rows = store.rows_for_interval(int(start), int(end), int(stride))
    steps = [int(x) for x in np.asarray(store.times_s[rows], dtype=np.int64)]
    return PositionSeries(
        cache_dir=Path(store.cache_dir),
        positions_km=store.positions_km,
        cache_rows=np.asarray(rows, dtype=np.int64),
        steps=steps,
        sat_ids=list(store.sat_ids),
        meta=dict(store.meta),
    )
