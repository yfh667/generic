from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable, build_full_option_edges
from src.link_delay.module.query import FullLinkDelayStore


@dataclass(frozen=True)
class EdgeDelayViewerData:
    store_dir: Path
    edge_table: EdgeTable
    delay_ms: np.ndarray
    steps: list[int]
    delay_min_ms: float
    delay_max_ms: float
    meta: dict


def load_store_meta(store_dir: str | Path) -> dict:
    path = Path(store_dir) / "delay_meta.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_edge_delay_data_for_viewer(
    *,
    store_dir: str | Path,
    config: ViewerConfig,
    start: int | None = None,
    end: int | None = None,
    stride: int = 1,
    options: Iterable[int] = (0, 1, 2, 4),
) -> EdgeDelayViewerData:
    store = FullLinkDelayStore(store_dir)
    if start is None:
        start = store.time_start
    if end is None:
        end = store.time_end

    rows = store.rows_for_interval(int(start), int(end), int(stride))
    steps = [int(x) for x in np.asarray(store.time_indices[rows], dtype=np.int64)]
    if int(stride) == 1 and rows.size:
        delay_ms = store.delay_ms_array[int(rows[0]) : int(rows[-1]) + 1, :]
    else:
        delay_ms = np.asarray(store.delay_ms_array[rows, :], dtype=np.float32)

    edge_table = build_full_option_edges(config, options=options)
    if int(delay_ms.shape[1]) != int(edge_table.num_edges):
        raise ValueError(
            f"delay store has {delay_ms.shape[1]} edges, but viewer config/options produce {edge_table.num_edges}"
        )

    meta = load_store_meta(store_dir)
    delay_min = meta.get("delay_min_ms")
    delay_max = meta.get("delay_max_ms")
    if delay_min is None:
        delay_min = float(np.nanmin(delay_ms))
    if delay_max is None:
        delay_max = float(np.nanmax(delay_ms))

    return EdgeDelayViewerData(
        store_dir=Path(store_dir),
        edge_table=edge_table,
        delay_ms=delay_ms,
        steps=steps,
        delay_min_ms=float(delay_min),
        delay_max_ms=float(delay_max),
        meta=meta,
    )
