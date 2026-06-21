from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable
from src.satellite_topology_viewer.module.edge_usage_viewer import EdgeUsageTopology2DViewer
from src.topology_workflow.module.edge_tables import INTRA_OPTION, add_intra_ring_records, make_edge_table_from_records


SYMBOLS = np.asarray(["", "A", "B", "C", "D"], dtype=object)
OPTION_TO_SYMBOL_ID = {
    0: 1,  # A
    1: 2,  # B
    4: 3,  # C
    2: 4,  # D
}
SYMBOL_COLORS = {
    "A": QtGui.QColor("#2563EB"),
    "B": QtGui.QColor("#7C3AED"),
    "C": QtGui.QColor("#059669"),
    "D": QtGui.QColor("#D97706"),
    "": QtGui.QColor("#CBD5E1"),
}


@dataclass(frozen=True)
class LinkSwitchViewerData:
    """Generic display payload for dynamic link-switch topology views.

    The viewer is intentionally data-only here: callers prepare the topology
    sequence and two edge-betweenness matrices outside the GUI, then pass the
    finished arrays into this payload. The two matrices are usually hop-based
    and delay-based edge betweenness, but their labels are caller-defined.
    """

    steps: np.ndarray
    edge_table: EdgeTable
    edge_active_mask: np.ndarray
    edge_building_mask: np.ndarray
    edge_betweenness_primary: np.ndarray
    edge_betweenness_secondary: np.ndarray
    primary_label: str = "hop"
    secondary_label: str = "delay"
    meta: dict | None = None

    def __post_init__(self) -> None:
        steps = np.asarray(self.steps, dtype=np.int64)
        active = np.asarray(self.edge_active_mask, dtype=bool)
        building = np.asarray(self.edge_building_mask, dtype=bool)
        primary = np.asarray(self.edge_betweenness_primary, dtype=np.float32)
        secondary = np.asarray(self.edge_betweenness_secondary, dtype=np.float32)

        if steps.ndim != 1 or steps.size == 0:
            raise ValueError("steps must be a non-empty 1-D array")
        expected = (int(steps.size), int(self.edge_table.num_edges))
        for name, value in (
            ("edge_active_mask", active),
            ("edge_building_mask", building),
            ("edge_betweenness_primary", primary),
            ("edge_betweenness_secondary", secondary),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} shape {value.shape} != expected {expected}")

        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "edge_active_mask", active)
        object.__setattr__(self, "edge_building_mask", building)
        object.__setattr__(self, "edge_betweenness_primary", primary)
        object.__setattr__(self, "edge_betweenness_secondary", secondary)
        object.__setattr__(self, "meta", dict(self.meta or {}))

    def values_for_mode(self, mode: str) -> np.ndarray:
        mode = str(mode)
        if mode in {"primary", str(self.primary_label)}:
            return self.edge_betweenness_primary
        if mode in {"secondary", str(self.secondary_label)}:
            return self.edge_betweenness_secondary
        if mode in {"any", "max"}:
            return np.maximum(self.edge_betweenness_primary, self.edge_betweenness_secondary)
        raise ValueError(
            f"mode must be primary/{self.primary_label}, secondary/{self.secondary_label}, any, or max; got {mode!r}"
        )


@dataclass(frozen=True)
class RightLinkStateStore:
    store_dir: Path
    meta: dict
    steps: np.ndarray
    topology_motif_id: np.ndarray
    active_right_edge_mask: np.ndarray
    edge_hop_betweenness: np.ndarray
    edge_delay_betweenness: np.ndarray
    right_neighbor: np.ndarray
    right_option: np.ndarray
    right_symbol_id: np.ndarray
    right_edge_idx: np.ndarray
    node_hop_betweenness: np.ndarray
    node_delay_betweenness: np.ndarray
    working_hop: np.ndarray
    working_delay: np.ndarray
    working_any: np.ndarray

    @property
    def n(self) -> int:
        return int(self.meta.get("constellation_n") or 36)

    @property
    def p(self) -> int:
        return int(self.meta.get("constellation_p") or 18)

    @property
    def num_nodes(self) -> int:
        return int(self.right_neighbor.shape[1])

    @property
    def primary_label(self) -> str:
        return str(self.meta.get("primary_label") or "hop")

    @property
    def secondary_label(self) -> str:
        return str(self.meta.get("secondary_label") or "delay")

    def slice_rows(self, row_indices: np.ndarray) -> "RightLinkStateStore":
        rows = np.asarray(row_indices, dtype=np.int64)
        meta = dict(self.meta)
        if rows.size:
            meta["start"] = int(self.steps[int(rows[0])])
            meta["end"] = int(self.steps[int(rows[-1])])
            if rows.size > 1:
                meta["stride"] = int(self.steps[int(rows[1])] - self.steps[int(rows[0])])
            meta["num_steps"] = int(rows.size)
        return RightLinkStateStore(
            store_dir=self.store_dir,
            meta=meta,
            steps=np.asarray(self.steps[rows], dtype=np.int64),
            topology_motif_id=np.asarray(self.topology_motif_id[rows], dtype=np.int16),
            active_right_edge_mask=np.asarray(self.active_right_edge_mask[rows, :], dtype=bool),
            edge_hop_betweenness=np.asarray(self.edge_hop_betweenness[rows, :], dtype=np.float32),
            edge_delay_betweenness=np.asarray(self.edge_delay_betweenness[rows, :], dtype=np.float32),
            right_neighbor=np.asarray(self.right_neighbor[rows, :], dtype=np.int32),
            right_option=np.asarray(self.right_option[rows, :], dtype=np.int16),
            right_symbol_id=np.asarray(self.right_symbol_id[rows, :], dtype=np.int8),
            right_edge_idx=np.asarray(self.right_edge_idx[rows, :], dtype=np.int32),
            node_hop_betweenness=np.asarray(self.node_hop_betweenness[rows, :], dtype=np.float32),
            node_delay_betweenness=np.asarray(self.node_delay_betweenness[rows, :], dtype=np.float32),
            working_hop=np.asarray(self.working_hop[rows, :], dtype=bool),
            working_delay=np.asarray(self.working_delay[rows, :], dtype=bool),
            working_any=np.asarray(self.working_any[rows, :], dtype=bool),
        )


def load_right_link_state_store(store_dir: str | Path) -> RightLinkStateStore:
    store_dir = Path(store_dir)
    with (store_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return RightLinkStateStore(
        store_dir=store_dir,
        meta=meta,
        steps=np.load(store_dir / "steps.npy"),
        topology_motif_id=np.load(store_dir / "topology_motif_id.npy"),
        active_right_edge_mask=np.load(store_dir / "active_right_edge_mask.npy", mmap_mode="r"),
        edge_hop_betweenness=np.load(store_dir / "right_edge_hop_betweenness.npy", mmap_mode="r"),
        edge_delay_betweenness=np.load(store_dir / "right_edge_delay_betweenness.npy", mmap_mode="r"),
        right_neighbor=np.load(store_dir / "right_neighbor.npy", mmap_mode="r"),
        right_option=np.load(store_dir / "right_option.npy", mmap_mode="r"),
        right_symbol_id=np.load(store_dir / "right_symbol_id.npy", mmap_mode="r"),
        right_edge_idx=np.load(store_dir / "right_edge_idx_by_node.npy", mmap_mode="r"),
        node_hop_betweenness=np.load(store_dir / "node_right_hop_betweenness.npy", mmap_mode="r"),
        node_delay_betweenness=np.load(store_dir / "node_right_delay_betweenness.npy", mmap_mode="r"),
        working_hop=np.load(store_dir / "node_right_working_hop.npy", mmap_mode="r"),
        working_delay=np.load(store_dir / "node_right_working_delay.npy", mmap_mode="r"),
        working_any=np.load(store_dir / "node_right_working_any.npy", mmap_mode="r"),
    )


def make_link_switch_viewer_data(
    *,
    steps: Sequence[int],
    edge_table: EdgeTable,
    edge_active_mask: np.ndarray,
    edge_betweenness_primary: np.ndarray,
    edge_betweenness_secondary: np.ndarray,
    edge_building_mask: np.ndarray | None = None,
    primary_label: str = "hop",
    secondary_label: str = "delay",
    meta: dict | None = None,
) -> LinkSwitchViewerData:
    """Build the generic viewer payload from prepared arrays."""

    steps_array = np.asarray(list(steps), dtype=np.int64)
    if edge_building_mask is None:
        edge_building_mask = np.zeros((int(steps_array.size), int(edge_table.num_edges)), dtype=bool)
    return LinkSwitchViewerData(
        steps=steps_array,
        edge_table=edge_table,
        edge_active_mask=np.asarray(edge_active_mask, dtype=bool),
        edge_building_mask=np.asarray(edge_building_mask, dtype=bool),
        edge_betweenness_primary=np.asarray(edge_betweenness_primary, dtype=np.float32),
        edge_betweenness_secondary=np.asarray(edge_betweenness_secondary, dtype=np.float32),
        primary_label=str(primary_label),
        secondary_label=str(secondary_label),
        meta=dict(meta or {}),
    )


def _right_owner_neighbor_for_edge(edge_table: EdgeTable, edge_idx: int) -> tuple[int, int] | None:
    if int(edge_table.option[int(edge_idx)]) == int(INTRA_OPTION):
        return None

    src = int(edge_table.src[int(edge_idx)])
    dst = int(edge_table.dst[int(edge_idx)])
    src_plane = int(edge_table.src_plane[int(edge_idx)])
    dst_plane = int(edge_table.dst_plane[int(edge_idx)])

    if src_plane < dst_plane:
        return src, dst
    if dst_plane < src_plane:
        return dst, src
    return None


def build_right_link_state_store_from_link_switch_data(
    data: LinkSwitchViewerData,
    *,
    config: ViewerConfig,
    store_dir: str | Path | None = None,
    topology_motif_id: Sequence[int] | None = None,
) -> tuple[RightLinkStateStore, np.ndarray]:
    """Derive per-node right-link state from generic edge masks and two metrics.

    The returned ``RightLinkStateStore`` is the format consumed by the right-side
    node inspector. The second return value maps right-link-store edge columns
    back to the input ``edge_table`` columns.
    """

    steps = np.asarray(data.steps, dtype=np.int64)
    num_steps = int(steps.size)
    num_nodes = int(config.total_sats)
    if topology_motif_id is None:
        motif_ids = np.zeros(num_steps, dtype=np.int16)
    else:
        motif_ids = np.asarray(list(topology_motif_id), dtype=np.int16)
        if int(motif_ids.size) != num_steps:
            raise ValueError(f"topology_motif_id length {motif_ids.size} != steps length {num_steps}")

    right_cols: list[int] = []
    owner_by_right: list[int] = []
    neighbor_by_right: list[int] = []
    option_by_right: list[int] = []
    symbol_by_right: list[int] = []
    for edge_idx in range(int(data.edge_table.num_edges)):
        owner_neighbor = _right_owner_neighbor_for_edge(data.edge_table, edge_idx)
        if owner_neighbor is None:
            continue
        owner, neighbor = owner_neighbor
        right_cols.append(int(edge_idx))
        owner_by_right.append(int(owner))
        neighbor_by_right.append(int(neighbor))
        option = int(data.edge_table.option[int(edge_idx)])
        option_by_right.append(option)
        symbol_by_right.append(int(OPTION_TO_SYMBOL_ID.get(option, 0)))

    right_to_viewer_edge_idx = np.asarray(right_cols, dtype=np.int32)
    active_right = np.asarray(data.edge_active_mask[:, right_to_viewer_edge_idx], dtype=bool)
    primary_right = np.asarray(data.edge_betweenness_primary[:, right_to_viewer_edge_idx], dtype=np.float32)
    secondary_right = np.asarray(data.edge_betweenness_secondary[:, right_to_viewer_edge_idx], dtype=np.float32)

    right_neighbor = np.full((num_steps, num_nodes), -1, dtype=np.int32)
    right_option = np.full((num_steps, num_nodes), -999, dtype=np.int16)
    right_symbol_id = np.zeros((num_steps, num_nodes), dtype=np.int8)
    right_edge_idx = np.full((num_steps, num_nodes), -1, dtype=np.int32)
    node_primary = np.zeros((num_steps, num_nodes), dtype=np.float32)
    node_secondary = np.zeros((num_steps, num_nodes), dtype=np.float32)
    working_primary = np.zeros((num_steps, num_nodes), dtype=bool)
    working_secondary = np.zeros((num_steps, num_nodes), dtype=bool)

    for right_idx, owner in enumerate(owner_by_right):
        rows = np.flatnonzero(active_right[:, right_idx])
        if rows.size == 0:
            continue
        right_neighbor[rows, int(owner)] = int(neighbor_by_right[right_idx])
        right_option[rows, int(owner)] = int(option_by_right[right_idx])
        right_symbol_id[rows, int(owner)] = int(symbol_by_right[right_idx])
        right_edge_idx[rows, int(owner)] = int(right_idx)
        node_primary[rows, int(owner)] = primary_right[rows, right_idx]
        node_secondary[rows, int(owner)] = secondary_right[rows, right_idx]
        working_primary[rows, int(owner)] = primary_right[rows, right_idx] > 0.0
        working_secondary[rows, int(owner)] = secondary_right[rows, right_idx] > 0.0

    meta = dict(data.meta or {})
    meta.update(
        {
            "constellation_name": str(config.name),
            "constellation_p": int(config.P),
            "constellation_n": int(config.N),
            "num_steps": num_steps,
            "start": int(steps[0]),
            "end": int(steps[-1]),
            "primary_label": str(data.primary_label),
            "secondary_label": str(data.secondary_label),
            "source": "link_switch_viewer_data",
        }
    )
    if num_steps > 1:
        meta["stride"] = int(steps[1] - steps[0])

    store = RightLinkStateStore(
        store_dir=Path("." if store_dir is None else store_dir),
        meta=meta,
        steps=steps,
        topology_motif_id=motif_ids,
        active_right_edge_mask=active_right,
        edge_hop_betweenness=primary_right,
        edge_delay_betweenness=secondary_right,
        right_neighbor=right_neighbor,
        right_option=right_option,
        right_symbol_id=right_symbol_id,
        right_edge_idx=right_edge_idx,
        node_hop_betweenness=node_primary,
        node_delay_betweenness=node_secondary,
        working_hop=working_primary,
        working_delay=working_secondary,
        working_any=working_primary | working_secondary,
    )
    return store, right_to_viewer_edge_idx


def row_indices_for_window(store: RightLinkStateStore, *, start: int, end: int, stride: int) -> np.ndarray:
    if int(stride) <= 0:
        raise ValueError("stride must be positive")
    steps = np.asarray(store.steps, dtype=np.int64)
    mask = (steps >= int(start)) & (steps <= int(end)) & (((steps - int(start)) % int(stride)) == 0)
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        raise ValueError(f"no stored rows found for start={start}, end={end}, stride={stride}")
    return rows


def _edge_key(src: int, dst: int) -> tuple[int, int]:
    src = int(src)
    dst = int(dst)
    return (src, dst) if src < dst else (dst, src)


def _edge_key_index(edge_table) -> dict[tuple[int, int], int]:
    return {
        _edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])): int(idx)
        for idx in range(edge_table.num_edges)
    }


def build_viewer_edge_payload(
    store: RightLinkStateStore,
    *,
    config: ViewerConfig,
    usage_mode: str,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    right_edges_csv = Path(store.store_dir) / "right_edges.csv"
    if not right_edges_csv.exists():
        raise FileNotFoundError(right_edges_csv)

    records: list[tuple[int, int, int, int, int]] = []
    right_idx_to_key: dict[int, tuple[int, int]] = {}
    with right_edges_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            right_idx = int(row["right_edge_idx"])
            owner = int(row["owner"])
            right = int(row["right"])
            records.append(
                (
                    int(row["owner_p"]),
                    int(row["owner_y"]),
                    int(row["right_p"]),
                    int(row["right_y"]),
                    int(row["option"]),
                )
            )
            right_idx_to_key[right_idx] = _edge_key(owner, right)

    add_intra_ring_records(records, p=int(config.P), n=int(config.N))
    edge_table = make_edge_table_from_records(p=int(config.P), n=int(config.N), records=records)
    viewer_key_to_idx = _edge_key_index(edge_table)

    right_count = int(store.active_right_edge_mask.shape[1])
    right_to_viewer = np.full(right_count, -1, dtype=np.int32)
    for right_idx, key in right_idx_to_key.items():
        if 0 <= int(right_idx) < right_count:
            right_to_viewer[int(right_idx)] = int(viewer_key_to_idx[key])

    active = np.zeros((len(store.steps), int(edge_table.num_edges)), dtype=bool)
    values = np.zeros((len(store.steps), int(edge_table.num_edges)), dtype=np.float32)
    intra_cols = np.asarray(np.asarray(edge_table.option) == -1).nonzero()[0]
    if intra_cols.size:
        active[:, intra_cols] = True

    for right_idx, viewer_idx in enumerate(right_to_viewer):
        if int(viewer_idx) < 0:
            continue
        active[:, int(viewer_idx)] = np.asarray(store.active_right_edge_mask[:, right_idx], dtype=bool)
        if usage_mode in {"hop", "primary", store.primary_label}:
            values[:, int(viewer_idx)] = np.asarray(store.edge_hop_betweenness[:, right_idx], dtype=np.float32)
        elif usage_mode in {"delay", "secondary", store.secondary_label}:
            values[:, int(viewer_idx)] = np.asarray(store.edge_delay_betweenness[:, right_idx], dtype=np.float32)
        elif usage_mode == "any":
            values[:, int(viewer_idx)] = np.maximum(
                np.asarray(store.edge_hop_betweenness[:, right_idx], dtype=np.float32),
                np.asarray(store.edge_delay_betweenness[:, right_idx], dtype=np.float32),
            )
        else:
            raise ValueError(
                "usage_mode must be "
                f"hop/primary/{store.primary_label}, delay/secondary/{store.secondary_label}, or any; "
                f"got {usage_mode!r}"
            )

    return edge_table, active, values, right_to_viewer


def nearest_row_for_step(steps: np.ndarray | list[int], step: int) -> int:
    values = np.asarray(steps, dtype=np.int64)
    if values.size == 0:
        raise ValueError("empty step axis")
    return int(np.argmin(np.abs(values - int(step))))


def node_xy(node: int, n: int) -> tuple[int, int]:
    return int(node // int(n)), int(node % int(n))


def contiguous_true_intervals(steps: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    steps = np.asarray(steps, dtype=np.int64)
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return []
    intervals: list[tuple[int, int]] = []
    start = int(rows[0])
    prev = int(rows[0])
    for row in rows[1:]:
        row = int(row)
        if row != prev + 1:
            intervals.append((int(steps[start]), int(steps[prev])))
            start = row
        prev = row
    intervals.append((int(steps[start]), int(steps[prev])))
    return intervals


class NodeRightLinkTimelineWidget(QtWidgets.QWidget):
    view_window_changed = QtCore.pyqtSignal(int, int)

    def __init__(self, store: RightLinkStateStore, *, working_mode: str):
        super().__init__()
        self.store = store
        self.working_mode = str(working_mode)
        self.selected_node: int | None = None
        self.current_row = 0
        self.view_start_step = int(self.store.steps[0])
        self.view_end_step = int(self.store.steps[-1])
        self._drag_pos: QtCore.QPoint | None = None
        self._drag_view_start = self.view_start_step
        self._drag_view_end = self.view_end_step
        self.hover_row: int | None = None
        self.hover_pos: QtCore.QPoint | None = None
        self._left_series_cache: dict[int, dict[str, np.ndarray]] = {}
        self._last_selected_node_for_intervals: int | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(520)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_selected_node(self, node: int | None) -> None:
        next_node = None if node is None else int(node)
        if self.selected_node == next_node:
            return
        self.selected_node = next_node
        self.update()

    def set_current_row(self, row: int) -> None:
        next_row = int(max(0, min(int(row), len(self.store.steps) - 1)))
        if self.current_row == next_row:
            return
        self.current_row = next_row
        self.update()

    def left_series_for_node(self, node: int) -> dict[str, np.ndarray]:
        node = int(node)
        cached = self._left_series_cache.get(node)
        if cached is not None:
            return cached

        num_rows = int(len(self.store.steps))
        owners = np.full(num_rows, -1, dtype=np.int32)
        edge_idx = np.full(num_rows, -1, dtype=np.int32)
        option = np.full(num_rows, -999, dtype=np.int16)
        symbol_id = np.zeros(num_rows, dtype=np.int8)
        primary = np.zeros(num_rows, dtype=np.float32)
        secondary = np.zeros(num_rows, dtype=np.float32)
        working_primary = np.zeros(num_rows, dtype=bool)
        working_secondary = np.zeros(num_rows, dtype=bool)

        matches = np.asarray(self.store.right_neighbor[:, :], dtype=np.int32) == int(node)
        has_left = np.any(matches, axis=1)
        hit_rows = np.flatnonzero(has_left)
        if hit_rows.size:
            hit_owners = np.argmax(matches[hit_rows, :], axis=1).astype(np.int32)
            owners[hit_rows] = hit_owners
            edge_idx[hit_rows] = np.asarray(self.store.right_edge_idx[hit_rows, hit_owners], dtype=np.int32)
            option[hit_rows] = np.asarray(self.store.right_option[hit_rows, hit_owners], dtype=np.int16)
            symbol_id[hit_rows] = np.asarray(self.store.right_symbol_id[hit_rows, hit_owners], dtype=np.int8)
            primary[hit_rows] = np.asarray(self.store.node_hop_betweenness[hit_rows, hit_owners], dtype=np.float32)
            secondary[hit_rows] = np.asarray(self.store.node_delay_betweenness[hit_rows, hit_owners], dtype=np.float32)
            working_primary[hit_rows] = np.asarray(self.store.working_hop[hit_rows, hit_owners], dtype=bool)
            working_secondary[hit_rows] = np.asarray(self.store.working_delay[hit_rows, hit_owners], dtype=bool)

        series = {
            "owner": owners,
            "edge_idx": edge_idx,
            "option": option,
            "symbol_id": symbol_id,
            "primary": primary,
            "secondary": secondary,
            "working_primary": working_primary,
            "working_secondary": working_secondary,
            "working_any": working_primary | working_secondary,
        }
        self._left_series_cache[node] = series
        return series

    def full_start_step(self) -> int:
        return int(self.store.steps[0])

    def full_end_step(self) -> int:
        return int(self.store.steps[-1])

    def min_view_span(self) -> int:
        if len(self.store.steps) <= 1:
            return 1
        diffs = np.diff(np.asarray(self.store.steps, dtype=np.int64))
        stride = int(np.median(diffs)) if diffs.size else 1
        return max(5 * stride, 60)

    def set_view_window(self, start_step: int, end_step: int) -> None:
        full_start = self.full_start_step()
        full_end = self.full_end_step()
        min_span = self.min_view_span()
        start_step = int(start_step)
        end_step = int(end_step)
        if end_step < start_step:
            start_step, end_step = end_step, start_step
        span = max(min_span, int(end_step - start_step))
        if span >= full_end - full_start:
            start_step, end_step = full_start, full_end
        else:
            if start_step < full_start:
                start_step = full_start
                end_step = start_step + span
            if end_step > full_end:
                end_step = full_end
                start_step = end_step - span
        changed = start_step != self.view_start_step or end_step != self.view_end_step
        self.view_start_step = int(start_step)
        self.view_end_step = int(end_step)
        if changed:
            self.view_window_changed.emit(self.view_start_step, self.view_end_step)
        self.update()

    def reset_zoom(self) -> None:
        self.set_view_window(self.full_start_step(), self.full_end_step())

    def zoom_by(self, factor: float, *, focus_step: int | None = None) -> None:
        factor = float(factor)
        if factor <= 0:
            return
        start = float(self.view_start_step)
        end = float(self.view_end_step)
        span = max(float(self.min_view_span()), end - start)
        focus = float(self.store.steps[self.current_row] if focus_step is None else focus_step)
        focus = max(start, min(end, focus))
        ratio = 0.5 if end <= start else (focus - start) / (end - start)
        next_span = max(float(self.min_view_span()), span * factor)
        next_start = focus - ratio * next_span
        next_end = next_start + next_span
        self.set_view_window(int(round(next_start)), int(round(next_end)))

    def focus_current(self, window_seconds: int) -> None:
        center = int(self.store.steps[self.current_row])
        half = max(self.min_view_span(), int(window_seconds)) // 2
        self.set_view_window(center - half, center + half)

    def _geometry(self) -> dict[str, float]:
        margin_l = 156.0
        margin_r = 26.0
        top = 18.0
        bottom = 46.0
        width = max(80.0, float(self.width()) - margin_l - margin_r)
        usable_h = max(430.0, float(self.height()) - top - bottom)
        track_h = max(20.0, min(32.0, usable_h * 0.052))
        gap = max(7.0, min(13.0, track_h * 0.38))
        curve_h = max(70.0, (usable_h - 7 * track_h - 8 * gap) / 2.0)
        return {
            "margin_l": margin_l,
            "margin_r": margin_r,
            "top": top,
            "width": width,
            "track_h": track_h,
            "gap": gap,
            "curve_h": curve_h,
        }

    def _time_rect(self) -> QtCore.QRectF:
        g = self._geometry()
        height = 7 * g["track_h"] + 8 * g["gap"] + 2 * g["curve_h"]
        return QtCore.QRectF(g["margin_l"], g["top"], g["width"], height)

    def _step_for_x(self, x: float, rect: QtCore.QRectF) -> int:
        if rect.width() <= 0:
            return int(self.store.steps[self.current_row])
        ratio = (float(x) - float(rect.left())) / float(rect.width())
        ratio = max(0.0, min(1.0, ratio))
        return int(round(self.view_start_step + ratio * (self.view_end_step - self.view_start_step)))

    def _x_for_step(self, step: float, rect: QtCore.QRectF) -> float:
        left = float(self.view_start_step)
        right = float(self.view_end_step)
        if right <= left:
            return float(rect.left())
        return float(rect.left() + (float(step) - left) / (right - left) * rect.width())

    def _x_for_row(self, row: int, rect: QtCore.QRectF) -> float:
        if len(self.store.steps) <= 1:
            return float(rect.left())
        step = float(self.store.steps[int(row)])
        return self._x_for_step(step, rect)

    def _row_for_step(self, step: int) -> int:
        values = np.asarray(self.store.steps, dtype=np.int64)
        if values.size == 0:
            return 0
        return int(np.argmin(np.abs(values - int(step))))

    def _row_for_pos(self, pos: QtCore.QPoint) -> int | None:
        if self.selected_node is None:
            return None
        rect = self._time_rect()
        if not rect.adjusted(-4, -8, 4, 8).contains(QtCore.QPointF(pos)):
            return None
        return self._row_for_step(self._step_for_x(float(pos.x()), rect))

    def _visible_row_bounds(self) -> tuple[int, int]:
        steps = np.asarray(self.store.steps, dtype=np.int64)
        if steps.size == 0:
            return 0, 0
        start = int(np.searchsorted(steps, int(self.view_start_step), side="left"))
        end = int(np.searchsorted(steps, int(self.view_end_step), side="right"))
        # Include one row on both sides so segments and curves crossing the
        # visible boundary are still clipped cleanly at the edge.
        start = max(0, start - 1)
        end = min(int(steps.size), max(start + 1, end + 1))
        return start, end

    def _x_for_row_exclusive(self, row: int, rect: QtCore.QRectF) -> float:
        steps = np.asarray(self.store.steps, dtype=np.int64)
        row = int(row)
        if row <= 0:
            return self._x_for_step(float(steps[0]), rect)
        if row >= int(steps.size):
            return self._x_for_step(float(steps[-1]), rect)
        return self._x_for_step(float(steps[row]), rect)

    def _downsample_rows_for_rect(self, start: int, end: int, rect: QtCore.QRectF) -> np.ndarray:
        count = max(0, int(end) - int(start))
        if count <= 0:
            return np.asarray([], dtype=np.int64)
        # A polyline cannot show more detail than the pixel width. Keep a small
        # oversampling factor so peaks remain visible without drawing 86k points
        # on every repaint.
        max_points = max(240, int(rect.width() * 2.0))
        stride = max(1, int(math.ceil(count / max_points)))
        rows = np.arange(int(start), int(end), stride, dtype=np.int64)
        last = int(end) - 1
        if rows.size == 0 or int(rows[-1]) != last:
            rows = np.append(rows, last)
        current = int(max(int(start), min(last, int(self.current_row))))
        if int(start) <= current <= last and not np.any(rows == current):
            rows = np.sort(np.append(rows, current)).astype(np.int64, copy=False)
        return rows

    def _hover_text_for_row(self, row: int) -> str:
        if self.selected_node is None:
            return ""
        row = int(max(0, min(int(row), len(self.store.steps) - 1)))
        node = int(self.selected_node)
        step = int(self.store.steps[row])
        motif_id = int(self.store.topology_motif_id[row])
        right = int(self.store.right_neighbor[row, node])
        symbol_id = int(self.store.right_symbol_id[row, node])
        symbol = str(SYMBOLS[symbol_id]) if 0 <= symbol_id < len(SYMBOLS) else ""
        right_text = "none" if right < 0 else f"{symbol or '?'}:{right}"
        left_series = self.left_series_for_node(node)
        left_owner = int(left_series["owner"][row])
        left_symbol_id = int(left_series["symbol_id"][row])
        left_symbol = str(SYMBOLS[left_symbol_id]) if 0 <= left_symbol_id < len(SYMBOLS) else ""
        left_text = "none" if left_owner < 0 else f"{left_symbol or '?'}:{left_owner}->node"
        primary_label = self.store.primary_label
        secondary_label = self.store.secondary_label
        right_primary = float(self.store.node_hop_betweenness[row, node])
        right_secondary = float(self.store.node_delay_betweenness[row, node])
        left_primary = float(left_series["primary"][row])
        left_secondary = float(left_series["secondary"][row])
        right_work_primary = int(bool(self.store.working_hop[row, node]))
        right_work_secondary = int(bool(self.store.working_delay[row, node]))
        left_work_primary = int(bool(left_series["working_primary"][row]))
        left_work_secondary = int(bool(left_series["working_secondary"][row]))
        return (
            f"step={step}s ({step / 3600.0:.2f}h)\n"
            f"node={node}, motif={motif_id}\n"
            f"right={right_text}: {primary_label}={right_primary:.3f}, {secondary_label}={right_secondary:.3f}, "
            f"work=({right_work_primary},{right_work_secondary})\n"
            f"left={left_text}: {primary_label}={left_primary:.3f}, {secondary_label}={left_secondary:.3f}, "
            f"work=({left_work_primary},{left_work_secondary})"
        )

    def _update_hover(self, pos: QtCore.QPoint, *, global_pos: QtCore.QPoint | None = None) -> None:
        row = self._row_for_pos(pos)
        if row is None:
            if self.hover_row is not None:
                self.hover_row = None
                self.hover_pos = None
                self.setToolTip("")
                self.update()
            return
        self.hover_row = int(row)
        self.hover_pos = QtCore.QPoint(pos)
        text = self._hover_text_for_row(int(row))
        self.setToolTip(text)
        if global_pos is not None:
            QtWidgets.QToolTip.showText(global_pos, text, self)
        self.update()

    def _paint_segments(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRectF,
        values: np.ndarray,
        *,
        color_for_value,
        text_for_value=None,
    ) -> None:
        if len(values) == 0:
            return
        painter.save()
        painter.setClipRect(rect.adjusted(-1, -1, 1, 1))
        start_bound, end_bound = self._visible_row_bounds()
        if end_bound <= start_bound:
            painter.restore()
            return
        values = np.asarray(values)
        start = int(start_bound)
        prev = values[0]
        if start > 0:
            prev = values[start]
        for row in range(start + 1, int(end_bound) + 1):
            end_segment = row == int(end_bound) or values[row] != prev
            if not end_segment:
                continue
            x0 = self._x_for_row(start, rect)
            x1 = self._x_for_row(max(start, row - 1), rect)
            if row < int(len(values)):
                x1 = self._x_for_row(row, rect)
            segment = QtCore.QRectF(x0, rect.top(), max(1.0, x1 - x0), rect.height())
            painter.fillRect(segment, color_for_value(prev))
            if text_for_value is not None and segment.width() >= 44:
                painter.setPen(QtGui.QColor("#111827"))
                painter.drawText(segment.adjusted(4, 0, -2, 0), QtCore.Qt.AlignVCenter, text_for_value(prev))
            start = row
            if row < int(len(values)):
                prev = values[row]
        painter.restore()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        factor = 0.78 if delta > 0 else 1.28
        focus_step = self._step_for_x(float(event.pos().x()), self._time_rect())
        self.zoom_by(factor, focus_step=focus_step)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_pos = QtCore.QPoint(event.pos())
            self._drag_view_start = int(self.view_start_step)
            self._drag_view_end = int(self.view_end_step)
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None:
            rect = self._time_rect()
            dx = float(event.pos().x() - self._drag_pos.x())
            span = int(self._drag_view_end - self._drag_view_start)
            shift = -int(round(dx / max(1.0, rect.width()) * span))
            self.set_view_window(self._drag_view_start + shift, self._drag_view_end + shift)
            event.accept()
            return
        self._update_hover(event.pos(), global_pos=event.globalPos())
        event.accept()
        return

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._drag_pos is not None:
            self._drag_pos = None
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.reset_zoom()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event):
        del event
        if self.hover_row is not None:
            self.hover_row = None
            self.hover_pos = None
            self.setToolTip("")
            self.update()

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        # This window is often kept open beside the topology canvas. During
        # ordinary window moves Windows/RDP requests many repaints, so avoid
        # expensive full-day antialiased drawing by default.
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QtGui.QColor("#F8FAFC"))

        g = self._geometry()
        margin_l = g["margin_l"]
        top = g["top"]
        width = g["width"]
        track_h = g["track_h"]
        gap = g["gap"]
        curve_h = g["curve_h"]

        font = QtGui.QFont("Arial")
        font.setPixelSize(14)
        painter.setFont(font)
        painter.setPen(QtGui.QColor("#334155"))

        if self.selected_node is None:
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Click a satellite node to inspect its left/right link usage.")
            painter.end()
            return

        node = int(self.selected_node)
        y_cursor = float(top)

        def next_track() -> QtCore.QRectF:
            nonlocal y_cursor
            rect = QtCore.QRectF(margin_l, y_cursor, width, track_h)
            y_cursor += track_h + gap
            return rect

        def next_curve() -> QtCore.QRectF:
            nonlocal y_cursor
            rect = QtCore.QRectF(margin_l, y_cursor, width, curve_h)
            y_cursor += curve_h + gap
            return rect

        motif_rect = next_track()
        right_rect = next_track()
        left_rect = next_track()
        right_primary_rect = next_track()
        right_secondary_rect = next_track()
        left_primary_rect = next_track()
        left_secondary_rect = next_track()
        right_curve_rect = next_curve()
        left_curve_rect = next_curve()

        label_font = QtGui.QFont("Arial")
        label_font.setPixelSize(15)
        label_font.setBold(True)
        painter.setFont(label_font)
        painter.drawText(12, int(motif_rect.center().y()) + 6, "motif")
        painter.drawText(12, int(right_rect.center().y()) + 6, "right link")
        painter.drawText(12, int(left_rect.center().y()) + 6, "left link")
        primary_label = self.store.primary_label
        secondary_label = self.store.secondary_label
        painter.drawText(12, int(right_primary_rect.center().y()) + 6, f"R {primary_label} work")
        painter.drawText(12, int(right_secondary_rect.center().y()) + 6, f"R {secondary_label} work")
        painter.drawText(12, int(left_primary_rect.center().y()) + 6, f"L {primary_label} work")
        painter.drawText(12, int(left_secondary_rect.center().y()) + 6, f"L {secondary_label} work")
        painter.drawText(12, int(right_curve_rect.center().y()) + 6, "R usage")
        painter.drawText(12, int(left_curve_rect.center().y()) + 6, "L usage")
        painter.setFont(font)

        motifs = np.asarray(self.store.topology_motif_id, dtype=np.int32)
        motif_colors = {
            int(self.store.meta.get("source_motif_id", 56)): QtGui.QColor("#CBD5E1"),
            int(self.store.meta.get("middle_motif_id", 61)): QtGui.QColor("#99F6E4"),
        }
        self._paint_segments(
            painter,
            motif_rect,
            motifs,
            color_for_value=lambda value: motif_colors.get(int(value), QtGui.QColor("#E5E7EB")),
            text_for_value=lambda value: str(int(value)),
        )

        right = np.asarray(self.store.right_neighbor[:, node], dtype=np.int32)
        right_symbol_ids = np.asarray(self.store.right_symbol_id[:, node], dtype=np.int8)
        right_packed = right.astype(np.int64) * 10 + right_symbol_ids.astype(np.int64)
        left_series = self.left_series_for_node(node)
        left_owner = np.asarray(left_series["owner"], dtype=np.int32)
        left_symbol_ids = np.asarray(left_series["symbol_id"], dtype=np.int8)
        left_packed = left_owner.astype(np.int64) * 10 + left_symbol_ids.astype(np.int64)

        def link_color(value):
            sid = int(value % 10)
            symbol = str(SYMBOLS[sid]) if 0 <= sid < len(SYMBOLS) else ""
            color = QtGui.QColor(SYMBOL_COLORS.get(symbol, QtGui.QColor("#94A3B8")))
            color.setAlpha(178)
            return color

        def right_text(value):
            sid = int(value % 10)
            neighbor = int(value // 10)
            symbol = str(SYMBOLS[sid]) if 0 <= sid < len(SYMBOLS) else ""
            if neighbor < 0:
                return "none"
            return f"{symbol}:{neighbor}"

        def left_text(value):
            sid = int(value % 10)
            owner = int(value // 10)
            symbol = str(SYMBOLS[sid]) if 0 <= sid < len(SYMBOLS) else ""
            if owner < 0:
                return "none"
            return f"{symbol}:{owner}->"

        self._paint_segments(painter, right_rect, right_packed, color_for_value=link_color, text_for_value=right_text)
        self._paint_segments(painter, left_rect, left_packed, color_for_value=link_color, text_for_value=left_text)

        right_work_hop = np.asarray(self.store.working_hop[:, node], dtype=bool)
        right_work_delay = np.asarray(self.store.working_delay[:, node], dtype=bool)
        left_work_hop = np.asarray(left_series["working_primary"], dtype=bool)
        left_work_delay = np.asarray(left_series["working_secondary"], dtype=bool)

        def paint_work_track(rect: QtCore.QRectF, values: np.ndarray, color: QtGui.QColor) -> None:
            painter.fillRect(rect, QtGui.QColor("#E5E7EB"))
            painter.save()
            painter.setClipRect(rect.adjusted(-1, -1, 1, 1))
            values = np.asarray(values, dtype=bool)
            start, end = self._visible_row_bounds()
            if end > start:
                local = values[start:end]
                padded = np.concatenate(
                    [
                        np.asarray([False], dtype=bool),
                        local,
                        np.asarray([False], dtype=bool),
                    ]
                )
                changes = np.flatnonzero(padded[1:] != padded[:-1])
                for begin_local, stop_local in zip(changes[0::2], changes[1::2]):
                    begin = int(start + begin_local)
                    stop = int(start + stop_local)
                    x0 = self._x_for_row(begin, rect)
                    x1 = self._x_for_row_exclusive(stop, rect)
                    painter.fillRect(QtCore.QRectF(x0, rect.top(), max(1.0, x1 - x0), rect.height()), color)
            painter.restore()

        paint_work_track(right_primary_rect, right_work_hop, QtGui.QColor("#2563EB"))
        paint_work_track(right_secondary_rect, right_work_delay, QtGui.QColor("#DC2626"))
        paint_work_track(left_primary_rect, left_work_hop, QtGui.QColor("#2563EB"))
        paint_work_track(left_secondary_rect, left_work_delay, QtGui.QColor("#DC2626"))

        right_primary = np.asarray(self.store.node_hop_betweenness[:, node], dtype=np.float64)
        right_secondary = np.asarray(self.store.node_delay_betweenness[:, node], dtype=np.float64)
        left_primary = np.asarray(left_series["primary"], dtype=np.float64)
        left_secondary = np.asarray(left_series["secondary"], dtype=np.float64)
        max_value = max(1.0, float(np.nanmax(np.r_[right_primary, right_secondary, left_primary, left_secondary])))

        def draw_curve_panel(rect: QtCore.QRectF, values_primary: np.ndarray, values_secondary: np.ndarray, title: str) -> None:
            painter.fillRect(rect, QtGui.QColor("#FFFFFF"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#E2E8F0"), 1.0))
            painter.drawRect(rect)

            def draw_curve(values: np.ndarray, color: str) -> None:
                painter.save()
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                painter.setClipRect(rect.adjusted(-1, -1, 1, 1))
                path = QtGui.QPainterPath()
                start, end = self._visible_row_bounds()
                rows = self._downsample_rows_for_rect(start, end, rect)
                for draw_idx, row in enumerate(rows.tolist()):
                    value = float(values[int(row)])
                    x = self._x_for_row(row, rect)
                    y = rect.bottom() - (max(0.0, float(value)) / max_value) * rect.height()
                    if draw_idx == 0:
                        path.moveTo(x, y)
                    else:
                        path.lineTo(x, y)
                painter.setPen(QtGui.QPen(QtGui.QColor(color), 1.6))
                painter.drawPath(path)
                painter.restore()

            draw_curve(values_primary, "#2563EB")
            draw_curve(values_secondary, "#DC2626")
            painter.setPen(QtGui.QColor("#334155"))
            painter.drawText(
                rect.adjusted(8, 4, -8, -4),
                QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft,
                f"{title}: {primary_label}=blue, {secondary_label}=red",
            )

        draw_curve_panel(right_curve_rect, right_primary, right_secondary, "right")
        draw_curve_panel(left_curve_rect, left_primary, left_secondary, "left")

        if self.hover_row is not None:
            hover_row = int(max(0, min(int(self.hover_row), len(self.store.steps) - 1)))
            hover_x = self._x_for_row(hover_row, self._time_rect())
            painter.setPen(QtGui.QPen(QtGui.QColor("#0F172A"), 1.2, QtCore.Qt.DashLine))
            painter.drawLine(
                QtCore.QPointF(hover_x, motif_rect.top() - 4),
                QtCore.QPointF(hover_x, left_curve_rect.bottom() + 4),
            )

            def draw_hover_dots(rect: QtCore.QRectF, primary_values: np.ndarray, secondary_values: np.ndarray) -> tuple[float, float]:
                primary_y = rect.bottom() - (max(0.0, float(primary_values[hover_row])) / max_value) * rect.height()
                secondary_y = rect.bottom() - (max(0.0, float(secondary_values[hover_row])) / max_value) * rect.height()
                painter.setPen(QtGui.QPen(QtGui.QColor("#FFFFFF"), 1.0))
                painter.setBrush(QtGui.QColor("#2563EB"))
                painter.drawEllipse(QtCore.QPointF(hover_x, primary_y), 4.0, 4.0)
                painter.setBrush(QtGui.QColor("#DC2626"))
                painter.drawEllipse(QtCore.QPointF(hover_x, secondary_y), 4.0, 4.0)
                return float(primary_y), float(secondary_y)

            right_py, right_sy = draw_hover_dots(right_curve_rect, right_primary, right_secondary)
            left_py, left_sy = draw_hover_dots(left_curve_rect, left_primary, left_secondary)
            hover_step = int(self.store.steps[hover_row])
            text = (
                f"t={hover_step}s ({hover_step / 3600.0:.2f}h)\n"
                f"R {primary_label}={float(right_primary[hover_row]):.3f}, "
                f"R {secondary_label}={float(right_secondary[hover_row]):.3f}\n"
                f"L {primary_label}={float(left_primary[hover_row]):.3f}, "
                f"L {secondary_label}={float(left_secondary[hover_row]):.3f}"
            )
            metrics = QtGui.QFontMetrics(font)
            text_rect = metrics.boundingRect(QtCore.QRect(0, 0, 360, 110), QtCore.Qt.AlignLeft, text)
            box_w = max(210.0, float(text_rect.width()) + 16.0)
            box_h = float(text_rect.height()) + 12.0
            box_x = hover_x + 10.0
            if box_x + box_w > right_curve_rect.right():
                box_x = hover_x - box_w - 10.0
            box_x = max(float(right_curve_rect.left()) + 4.0, box_x)
            dot_y = min(right_py, right_sy, left_py, left_sy)
            box_y = max(float(right_curve_rect.top()) + 8.0, dot_y - box_h - 10.0)
            label_rect = QtCore.QRectF(box_x, box_y, box_w, box_h)
            painter.setPen(QtGui.QPen(QtGui.QColor("#CBD5E1"), 1.0))
            painter.setBrush(QtGui.QColor(255, 255, 255, 238))
            painter.drawRoundedRect(label_rect, 5.0, 5.0)
            painter.setPen(QtGui.QColor("#111827"))
            painter.drawText(label_rect.adjusted(8, 6, -8, -6), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, text)

        current_x = self._x_for_row(self.current_row, self._time_rect())
        painter.setPen(QtGui.QPen(QtGui.QColor("#111827"), 1.6))
        painter.drawLine(
            QtCore.QPointF(current_x, motif_rect.top() - 4),
            QtCore.QPointF(current_x, left_curve_rect.bottom() + 4),
        )

        painter.setPen(QtGui.QColor("#334155"))
        visible_span = max(1, int(self.view_end_step - self.view_start_step))
        if visible_span <= 3600:
            tick_seconds = 600
        elif visible_span <= 4 * 3600:
            tick_seconds = 1800
        elif visible_span <= 12 * 3600:
            tick_seconds = 3600
        else:
            tick_seconds = 6 * 3600
        first_tick = int(math.ceil(self.view_start_step / tick_seconds) * tick_seconds)
        for step in range(first_tick, self.view_end_step + 1, tick_seconds):
            if step < self.full_start_step() or step > self.full_end_step():
                continue
            x = self._x_for_step(step, QtCore.QRectF(margin_l, top, width, 1))
            label = f"{step / 3600.0:.1f}h" if tick_seconds < 3600 else f"{int(step / 3600)}h"
            painter.drawText(int(x) - 18, int(self.height()) - 10, label)

        view_text = (
            f"visible {self.view_start_step}s..{self.view_end_step}s "
            f"({self.view_start_step / 3600.0:.2f}h..{self.view_end_step / 3600.0:.2f}h); "
            "wheel=zoom, drag=pan, double click=fit"
        )
        painter.drawText(int(margin_l), int(self.height()) - 32, view_text)

        painter.end()


class NodeRightLinkInspectorWindow(QtWidgets.QWidget):
    def __init__(self, store: RightLinkStateStore, *, working_mode: str, title: str):
        super().__init__()
        self.setWindowTitle(str(title))
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setStyleSheet(
            """
            QWidget {
                font-size: 16px;
            }
            QGroupBox {
                font-size: 17px;
                font-weight: 600;
            }
            QLabel#statusLabel {
                font-size: 16px;
                line-height: 1.35;
            }
            QPushButton {
                font-size: 16px;
                min-height: 34px;
                padding: 4px 10px;
            }
            QPlainTextEdit {
                font-family: Consolas, "Courier New", monospace;
                font-size: 16px;
            }
            QSplitter::handle:vertical {
                background: #CBD5E1;
                height: 10px;
                margin: 2px 0;
            }
            QSplitter::handle:vertical:hover {
                background: #94A3B8;
            }
            """
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        box = QtWidgets.QGroupBox("Node left/right-link working inspector")
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.setContentsMargins(8, 8, 8, 8)
        box_layout.setSpacing(6)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter.setHandleWidth(10)

        info_panel = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout(info_panel)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(8)

        self.node_status_label = QtWidgets.QLabel("Click one satellite node to inspect its 24h left/right-link state.")
        self.node_status_label.setObjectName("statusLabel")
        self.node_status_label.setWordWrap(True)
        info_layout.addWidget(self.node_status_label)

        tool_row = QtWidgets.QHBoxLayout()
        info_layout.addLayout(tool_row)
        self.fit_day_btn = QtWidgets.QPushButton("Fit day")
        self.zoom_in_btn = QtWidgets.QPushButton("Zoom +")
        self.zoom_out_btn = QtWidgets.QPushButton("Zoom -")
        self.focus_30m_btn = QtWidgets.QPushButton("Current +/-30m")
        self.focus_2h_btn = QtWidgets.QPushButton("Current +/-2h")
        for button in (
            self.fit_day_btn,
            self.zoom_in_btn,
            self.zoom_out_btn,
            self.focus_30m_btn,
            self.focus_2h_btn,
        ):
            tool_row.addWidget(button)
        self.fit_day_btn.clicked.connect(self.node_timeline_reset_requested)
        self.zoom_in_btn.clicked.connect(self.node_timeline_zoom_in_requested)
        self.zoom_out_btn.clicked.connect(self.node_timeline_zoom_out_requested)
        self.focus_30m_btn.clicked.connect(lambda: self.node_timeline.focus_current(3600))
        self.focus_2h_btn.clicked.connect(lambda: self.node_timeline.focus_current(4 * 3600))

        self.node_timeline = NodeRightLinkTimelineWidget(store, working_mode=working_mode)
        self.node_timeline.view_window_changed.connect(self.update_view_label)

        self.view_window_label = QtWidgets.QLabel("")
        self.view_window_label.setObjectName("statusLabel")
        info_layout.addWidget(self.view_window_label)

        self.node_intervals_text = QtWidgets.QPlainTextEdit()
        self.node_intervals_text.setReadOnly(True)
        self.node_intervals_text.setPlainText("Left/right working intervals will appear here after a node is selected.")

        self.splitter.addWidget(info_panel)
        self.splitter.addWidget(self.node_timeline)
        self.splitter.addWidget(self.node_intervals_text)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setCollapsible(2, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 5)
        self.splitter.setStretchFactor(2, 2)
        self.splitter.setSizes([165, 440, 230])
        box_layout.addWidget(self.splitter, stretch=1)

        layout.addWidget(box, stretch=1)

        help_text = QtWidgets.QLabel(
            "Left click a node on the topology canvas. "
            "This inspector is separate from the 2D viewer, so the original time slider and edge controls stay clean."
        )
        help_text.setWordWrap(True)
        help_text.setObjectName("statusLabel")
        layout.addWidget(help_text)
        self.resize(1080, 820)
        self.update_view_label(self.node_timeline.view_start_step, self.node_timeline.view_end_step)

    def node_timeline_reset_requested(self) -> None:
        self.node_timeline.reset_zoom()

    def node_timeline_zoom_in_requested(self) -> None:
        self.node_timeline.zoom_by(0.78)

    def node_timeline_zoom_out_requested(self) -> None:
        self.node_timeline.zoom_by(1.28)

    def update_view_label(self, start_step: int, end_step: int) -> None:
        self.view_window_label.setText(
            f"Visible range: {int(start_step)}s..{int(end_step)}s "
            f"({int(start_step) / 3600.0:.2f}h..{int(end_step) / 3600.0:.2f}h)"
        )


class NodeWorkingInspectorEdgeUsageViewer(EdgeUsageTopology2DViewer):
    def __init__(
        self,
        config: ViewerConfig,
        *,
        store: RightLinkStateStore,
        right_to_viewer_edge_idx: np.ndarray,
        working_mode: str = "any",
        inspector_mode: str = "window",
        **kwargs,
    ):
        self.node_state_store = store
        self.right_to_viewer_edge_idx = np.asarray(right_to_viewer_edge_idx, dtype=np.int32)
        self.working_mode = str(working_mode)
        self.inspector_mode = str(inspector_mode)
        if self.inspector_mode not in {"window", "right-panel"}:
            raise ValueError("inspector_mode must be 'window' or 'right-panel'")
        self.selected_node: int | None = None
        self._interval_cache: dict[tuple[int, str, str], list[tuple[int, int]]] = {}
        self._node_right_candidate_cache: dict[int, list[tuple[int, int, int]]] = {}
        self._last_intervals_text_node: int | None = None
        super().__init__(config, **kwargs)

    def _build_ui(self):
        super()._build_ui()
        if self.inspector_mode == "window":
            self.node_inspector_window = NodeRightLinkInspectorWindow(
                self.node_state_store,
                working_mode=self.working_mode,
                title="Node left/right-link working inspector",
            )
            self.node_status_label = self.node_inspector_window.node_status_label
            self.node_timeline = self.node_inspector_window.node_timeline
            self.node_intervals_text = self.node_inspector_window.node_intervals_text
            return

        root_layout = self.layout()
        if root_layout is not None:
            root_layout.removeWidget(self.main_splitter)

        self.outer_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.outer_splitter.setHandleWidth(10)
        self.outer_splitter.addWidget(self.main_splitter)

        self.node_side_panel = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(self.node_side_panel)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(8)

        box = QtWidgets.QGroupBox("Node left/right-link working inspector")
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.setContentsMargins(8, 8, 8, 8)
        box_layout.setSpacing(6)

        self.node_status_label = QtWidgets.QLabel("Click one satellite node to inspect its 24h left/right-link state.")
        self.node_status_label.setObjectName("statusLabel")
        self.node_status_label.setWordWrap(True)
        box_layout.addWidget(self.node_status_label)

        self.node_timeline = NodeRightLinkTimelineWidget(self.node_state_store, working_mode=self.working_mode)
        box_layout.addWidget(self.node_timeline)

        self.node_intervals_text = QtWidgets.QPlainTextEdit()
        self.node_intervals_text.setReadOnly(True)
        self.node_intervals_text.setPlainText("Left/right working intervals will appear here after a node is selected.")
        box_layout.addWidget(self.node_intervals_text)

        side_layout.addWidget(box)

        help_text = QtWidgets.QLabel(
            "Left click a node on the topology canvas. "
            "The left viewer keeps its original time slider and edge controls; "
            "this right panel only reports the selected node's left/right-link state."
        )
        help_text.setWordWrap(True)
        help_text.setObjectName("statusLabel")
        side_layout.addWidget(help_text)
        side_layout.addStretch(1)

        self.node_side_scroll = QtWidgets.QScrollArea()
        self.node_side_scroll.setWidgetResizable(True)
        self.node_side_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.node_side_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.node_side_scroll.setWidget(self.node_side_panel)
        self.node_side_scroll.setMinimumWidth(430)
        self.node_side_scroll.setMaximumWidth(620)
        self.outer_splitter.addWidget(self.node_side_scroll)
        self.outer_splitter.setStretchFactor(0, 5)
        self.outer_splitter.setStretchFactor(1, 0)
        self.outer_splitter.setCollapsible(0, False)
        self.outer_splitter.setCollapsible(1, True)
        self.outer_splitter.setSizes([1240, 460])

        if root_layout is not None:
            root_layout.addWidget(self.outer_splitter, stretch=1)

    def show_inspector_window(self):
        if self.inspector_mode != "window":
            return
        self.node_inspector_window.show()
        self.node_inspector_window.raise_()
        self.node_inspector_window.activateWindow()

    def _viewer_edge_for_node_row(self, node: int, row: int) -> int | None:
        state_edge_idx = int(self.node_state_store.right_edge_idx[int(row), int(node)])
        if state_edge_idx < 0 or state_edge_idx >= int(self.right_to_viewer_edge_idx.size):
            building = self._building_right_link_for_node_row(int(node), int(row))
            return None if building is None else int(building[0])
        viewer_idx = int(self.right_to_viewer_edge_idx[state_edge_idx])
        return None if viewer_idx < 0 else viewer_idx

    def _right_candidates_for_node(self, node: int) -> list[tuple[int, int, int]]:
        node = int(node)
        cached = self._node_right_candidate_cache.get(node)
        if cached is not None:
            return cached

        candidates: list[tuple[int, int, int]] = []
        for state_idx, viewer_idx in enumerate(np.asarray(self.right_to_viewer_edge_idx, dtype=np.int32)):
            viewer_idx = int(viewer_idx)
            if viewer_idx < 0:
                continue
            owner_neighbor = _right_owner_neighbor_for_edge(self.edge_table, viewer_idx)
            if owner_neighbor is None:
                continue
            owner, neighbor = owner_neighbor
            if int(owner) == node:
                candidates.append((int(state_idx), int(viewer_idx), int(neighbor)))
        self._node_right_candidate_cache[node] = candidates
        return candidates

    def _building_right_link_for_node_row(self, node: int, row: int) -> tuple[int, int, int, int] | None:
        row = int(row)
        building_row = 0 if int(self.edge_building_mask.shape[0]) == 1 else row
        for _state_idx, viewer_idx, neighbor in self._right_candidates_for_node(int(node)):
            if bool(self.edge_building_mask[building_row, int(viewer_idx)]):
                option = int(self.edge_table.option[int(viewer_idx)])
                symbol_id = int(OPTION_TO_SYMBOL_ID.get(option, 0))
                return int(viewer_idx), int(neighbor), option, symbol_id
        return None

    def pick_node(self, node: int):
        node = int(node)
        if not (0 <= node < int(self.config.total_sats)):
            return
        self.selected_node = node
        self.picked_nodes = [node]
        self.selected_edge_idx = self._viewer_edge_for_node_row(node, self.current_row)
        self.update_pick_markers()
        self.update_node_inspector()
        self.show_inspector_window()
        self.update_step(self.current_row)

    def clear_picked_nodes(self):
        self.selected_node = None
        self.picked_nodes = []
        self.selected_edge_idx = None
        self.pick_label.setText("Picked nodes: none")
        self.node_status_label.setText("Click one satellite node to inspect its 24h left/right-link state.")
        self.node_intervals_text.setPlainText("Left/right working intervals will appear here after a node is selected.")
        self._last_intervals_text_node = None
        self.node_timeline.set_selected_node(None)
        self.update_step(self.current_row)

    def update_step(self, row: int, *, sync_slider: bool = True):
        row = int(max(0, min(int(row), len(self.steps) - 1)))
        if self.selected_node is not None:
            self.selected_edge_idx = self._viewer_edge_for_node_row(self.selected_node, row)
        super().update_step(row, sync_slider=sync_slider)
        self.update_node_inspector()

    def _right_working_array(self, mode: str, node: int) -> np.ndarray:
        mode = str(mode)
        if mode in {"hop", "primary", self.node_state_store.primary_label}:
            return np.asarray(self.node_state_store.working_hop[:, node], dtype=bool)
        if mode in {"delay", "secondary", self.node_state_store.secondary_label}:
            return np.asarray(self.node_state_store.working_delay[:, node], dtype=bool)
        return np.asarray(self.node_state_store.working_any[:, node], dtype=bool)

    def _left_series_for_node(self, node: int) -> dict[str, np.ndarray]:
        return self.node_timeline.left_series_for_node(int(node))

    def _left_working_array(self, mode: str, node: int) -> np.ndarray:
        series = self._left_series_for_node(int(node))
        mode = str(mode)
        if mode in {"hop", "primary", self.node_state_store.primary_label}:
            return np.asarray(series["working_primary"], dtype=bool)
        if mode in {"delay", "secondary", self.node_state_store.secondary_label}:
            return np.asarray(series["working_secondary"], dtype=bool)
        return np.asarray(series["working_any"], dtype=bool)

    def _working_array(self, side: str, mode: str, node: int) -> np.ndarray:
        if str(side) == "left":
            return self._left_working_array(mode, int(node))
        return self._right_working_array(mode, int(node))

    def _intervals_for(self, side: str, node: int, mode: str) -> list[tuple[int, int]]:
        key = (int(node), str(side), str(mode))
        cached = self._interval_cache.get(key)
        if cached is not None:
            return cached
        intervals = contiguous_true_intervals(self.node_state_store.steps, self._working_array(side, mode, int(node)))
        self._interval_cache[key] = intervals
        return intervals

    def _format_intervals(self, node: int) -> str:
        lines = []
        modes = (self.node_state_store.primary_label, self.node_state_store.secondary_label, "any")
        for side in ("right", "left"):
            lines.append(f"[{side}]")
            for mode in modes:
                mask = self._working_array(side, mode, int(node))
                intervals = self._intervals_for(side, node, mode)
                lines.append(f"{mode}: working_steps={int(mask.sum())}/{len(mask)}, intervals={len(intervals)}")
                for start, end in intervals[:12]:
                    lines.append(f"  {start:>6}..{end:<6}  ({start / 3600.0:.2f}h..{end / 3600.0:.2f}h)")
                if len(intervals) > 12:
                    lines.append(f"  ... {len(intervals) - 12} more")
            lines.append("")
        return "\n".join(lines)

    def update_node_inspector(self):
        if not hasattr(self, "node_status_label"):
            return
        if self.selected_node is None:
            return

        node = int(self.selected_node)
        row = int(self.current_row)
        step = int(self.steps[row])
        n = int(self.config.N)
        p, y = node_xy(node, n)
        up_node = int(p * n + ((y + 1) % n))
        down_node = int(p * n + ((y - 1) % n))
        right = int(self.node_state_store.right_neighbor[row, node])
        right_text = "none"
        if right >= 0:
            rp, ry = node_xy(right, n)
            right_text = f"{right} ({rp}, {ry})"
        building = self._building_right_link_for_node_row(node, row)
        building_text = "none"
        if building is not None:
            _edge_idx, building_neighbor, building_option, building_symbol_id = building
            bp, by = node_xy(building_neighbor, n)
            building_symbol = str(SYMBOLS[building_symbol_id]) if 0 <= building_symbol_id < len(SYMBOLS) else ""
            building_text = (
                f"{building_neighbor} ({bp}, {by}), "
                f"symbol={building_symbol or 'none'}, option={building_option}"
            )
        symbol_id = int(self.node_state_store.right_symbol_id[row, node])
        symbol = str(SYMBOLS[symbol_id]) if 0 <= symbol_id < len(SYMBOLS) else ""
        option = int(self.node_state_store.right_option[row, node])
        left_series = self._left_series_for_node(node)
        left_owner = int(left_series["owner"][row])
        left_text = "none"
        if left_owner >= 0:
            lp, ly = node_xy(left_owner, n)
            left_text = f"{left_owner} ({lp}, {ly}) -> node"
        left_symbol_id = int(left_series["symbol_id"][row])
        left_symbol = str(SYMBOLS[left_symbol_id]) if 0 <= left_symbol_id < len(SYMBOLS) else ""
        left_option = int(left_series["option"][row])
        motif_id = int(self.node_state_store.topology_motif_id[row])
        primary_label = self.node_state_store.primary_label
        secondary_label = self.node_state_store.secondary_label
        right_primary_value = float(self.node_state_store.node_hop_betweenness[row, node])
        right_secondary_value = float(self.node_state_store.node_delay_betweenness[row, node])
        right_working_primary = bool(self.node_state_store.working_hop[row, node])
        right_working_secondary = bool(self.node_state_store.working_delay[row, node])
        right_working_any = bool(self.node_state_store.working_any[row, node])
        left_primary_value = float(left_series["primary"][row])
        left_secondary_value = float(left_series["secondary"][row])
        left_working_primary = bool(left_series["working_primary"][row])
        left_working_secondary = bool(left_series["working_secondary"][row])
        left_working_any = bool(left_series["working_any"][row])
        edge_idx = int(self.node_state_store.right_edge_idx[row, node])
        left_edge_idx = int(left_series["edge_idx"][row])

        self.pick_label.setText(
            f"Picked node: {node} ({p}, {y}); groups={self.node_group_text(node)}; "
            f"right edge idx={edge_idx if edge_idx >= 0 else 'none'}; "
            f"left edge idx={left_edge_idx if left_edge_idx >= 0 else 'none'}"
        )
        self.node_status_label.setText(
            f"step={step} row={row + 1}/{len(self.steps)} | motif={motif_id} | "
            f"node={node} ({p}, {y}) | intra_down={down_node}, intra_up={up_node} | "
            f"active_right={right_text}, symbol={symbol or 'none'}, "
            f"option={option if option > -900 else 'none'} | building_right={building_text} | "
            f"active_left={left_text}, symbol={left_symbol or 'none'}, "
            f"option={left_option if left_option > -900 else 'none'} | "
            f"right {primary_label}={right_primary_value:.3f}, "
            f"{secondary_label}={right_secondary_value:.3f}, "
            f"work=({int(right_working_primary)},{int(right_working_secondary)},{int(right_working_any)}) | "
            f"left {primary_label}={left_primary_value:.3f}, "
            f"{secondary_label}={left_secondary_value:.3f}, "
            f"work=({int(left_working_primary)},{int(left_working_secondary)},{int(left_working_any)})"
        )
        self.node_timeline.set_selected_node(node)
        self.node_timeline.set_current_row(row)
        if self._last_intervals_text_node != int(node):
            self.node_intervals_text.setPlainText(self._format_intervals(node))
            self._last_intervals_text_node = int(node)


class LinkSwitchTopologyViewer(NodeWorkingInspectorEdgeUsageViewer):
    """2D viewer for dynamic link switching with two edge-betweenness matrices.

    Parameters are deliberately plain arrays. The caller owns topology
    generation, LST scheduling, region constraints, and metric computation; this
    class only renders the already-prepared data and exposes the node inspector.
    """

    def __init__(
        self,
        config: ViewerConfig,
        *,
        data: LinkSwitchViewerData,
        value_mode: str = "secondary",
        working_mode: str = "any",
        inspector_mode: str = "window",
        topology_motif_id: Sequence[int] | None = None,
        group_data: dict | None = None,
        show_groups: bool = True,
        value_max: float | None = None,
        window_title: str | None = None,
        **viewer_kwargs,
    ):
        node_state_store, right_to_viewer_edge_idx = build_right_link_state_store_from_link_switch_data(
            data,
            config=config,
            topology_motif_id=topology_motif_id,
        )
        values = data.values_for_mode(value_mode)
        inferred_max = float(np.nanmax(values)) if values.size else 1.0
        label_by_mode = {
            "primary": data.primary_label,
            data.primary_label: data.primary_label,
            "secondary": data.secondary_label,
            data.secondary_label: data.secondary_label,
            "any": f"max({data.primary_label},{data.secondary_label})",
            "max": f"max({data.primary_label},{data.secondary_label})",
        }
        value_label = label_by_mode.get(str(value_mode), str(value_mode))
        defaults = {
            "edge_value_label": f"{value_label}_betweenness",
            "topology_edge_color": "#000000",
            "topology_edge_alpha": 165,
            "topology_edge_width": 0.018,
            "building_edge_color": "#1D4ED8",
            "building_edge_alpha": 230,
            "building_edge_width": 0.060,
            "value_width_min": 0.010,
            "value_width_max": 0.105,
            "value_alpha_min": 34,
            "value_alpha_max": 245,
            "hide_y_wrap_edges": True,
            "show_grid_lines": False,
        }
        defaults.update(viewer_kwargs)
        super().__init__(
            config,
            store=node_state_store,
            right_to_viewer_edge_idx=right_to_viewer_edge_idx,
            working_mode=str(working_mode),
            inspector_mode=str(inspector_mode),
            steps=[int(x) for x in data.steps],
            edge_table=data.edge_table,
            edge_usage_values=values,
            value_max=max(1.0, inferred_max) if value_max is None else float(value_max),
            edge_active_mask=data.edge_active_mask,
            edge_building_mask=data.edge_building_mask,
            window_title=window_title or f"{config.name} link-switch topology viewer",
            group_data=group_data or {},
            show_groups=show_groups,
            **defaults,
        )
