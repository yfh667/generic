from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from src.config.viewer_config import ViewerConfig
from src.satellite_topology_viewer.module.edge_usage_viewer import EdgeUsageTopology2DViewer
from src.topology_workflow.module.edge_tables import add_intra_ring_records, make_edge_table_from_records


SYMBOLS = np.asarray(["", "A", "B", "C", "D"], dtype=object)
SYMBOL_COLORS = {
    "A": QtGui.QColor("#2563EB"),
    "B": QtGui.QColor("#7C3AED"),
    "C": QtGui.QColor("#059669"),
    "D": QtGui.QColor("#D97706"),
    "": QtGui.QColor("#CBD5E1"),
}


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
        if usage_mode == "hop":
            values[:, int(viewer_idx)] = np.asarray(store.edge_hop_betweenness[:, right_idx], dtype=np.float32)
        elif usage_mode == "delay":
            values[:, int(viewer_idx)] = np.asarray(store.edge_delay_betweenness[:, right_idx], dtype=np.float32)
        elif usage_mode == "any":
            values[:, int(viewer_idx)] = np.maximum(
                np.asarray(store.edge_hop_betweenness[:, right_idx], dtype=np.float32),
                np.asarray(store.edge_delay_betweenness[:, right_idx], dtype=np.float32),
            )
        else:
            raise ValueError(f"usage_mode must be hop, delay, or any; got {usage_mode!r}")

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
        self.setMouseTracking(True)
        self.setMinimumHeight(300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_selected_node(self, node: int | None) -> None:
        self.selected_node = None if node is None else int(node)
        self.update()

    def set_current_row(self, row: int) -> None:
        self.current_row = int(max(0, min(int(row), len(self.store.steps) - 1)))
        self.update()

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
        margin_l = 124.0
        margin_r = 26.0
        top = 18.0
        bottom = 46.0
        width = max(80.0, float(self.width()) - margin_l - margin_r)
        usable_h = max(190.0, float(self.height()) - top - bottom)
        track_h = max(26.0, min(44.0, usable_h * 0.115))
        gap = max(11.0, min(18.0, track_h * 0.42))
        curve_h = max(78.0, usable_h - 3 * track_h - 3 * gap)
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
        height = 3 * g["track_h"] + 3 * g["gap"] + g["curve_h"]
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
        start = 0
        prev = values[0]
        for row in range(1, len(values) + 1):
            end_segment = row == len(values) or values[row] != prev
            if not end_segment:
                continue
            x0 = self._x_for_row(start, rect)
            x1 = self._x_for_row(max(start, row - 1), rect)
            if row < len(values):
                x1 = self._x_for_row(row, rect)
            segment = QtCore.QRectF(x0, rect.top(), max(1.0, x1 - x0), rect.height())
            painter.fillRect(segment, color_for_value(prev))
            if text_for_value is not None and segment.width() >= 44:
                painter.setPen(QtGui.QColor("#111827"))
                painter.drawText(segment.adjusted(4, 0, -2, 0), QtCore.Qt.AlignVCenter, text_for_value(prev))
            start = row
            if row < len(values):
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
        super().mouseMoveEvent(event)

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

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
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
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Click a satellite node to inspect its right-link usage.")
            painter.end()
            return

        node = int(self.selected_node)
        motif_rect = QtCore.QRectF(margin_l, top, width, track_h)
        right_rect = QtCore.QRectF(margin_l, top + track_h + gap, width, track_h)
        work_rect = QtCore.QRectF(margin_l, top + 2 * (track_h + gap), width, track_h)
        curve_rect = QtCore.QRectF(margin_l, top + 3 * (track_h + gap), width, curve_h)

        label_font = QtGui.QFont("Arial")
        label_font.setPixelSize(15)
        label_font.setBold(True)
        painter.setFont(label_font)
        painter.drawText(12, int(motif_rect.center().y()) + 6, "motif")
        painter.drawText(12, int(right_rect.center().y()) + 6, "right link")
        painter.drawText(12, int(work_rect.center().y()) + 6, "working")
        painter.drawText(12, int(curve_rect.center().y()) + 6, "usage")
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
        symbol_ids = np.asarray(self.store.right_symbol_id[:, node], dtype=np.int8)
        packed = right.astype(np.int64) * 10 + symbol_ids.astype(np.int64)

        def right_color(value):
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

        self._paint_segments(painter, right_rect, packed, color_for_value=right_color, text_for_value=right_text)

        work_any = np.asarray(self.store.working_any[:, node], dtype=bool)
        work_hop = np.asarray(self.store.working_hop[:, node], dtype=bool)
        work_delay = np.asarray(self.store.working_delay[:, node], dtype=bool)
        painter.fillRect(work_rect, QtGui.QColor("#E5E7EB"))
        painter.save()
        painter.setClipRect(work_rect.adjusted(-1, -1, 1, 1))
        for row, flag in enumerate(work_any):
            if not bool(flag):
                continue
            x0 = self._x_for_row(row, work_rect)
            x1 = self._x_for_row(min(row + 1, len(work_any) - 1), work_rect)
            if bool(work_hop[row]) and bool(work_delay[row]):
                color = QtGui.QColor("#C1121F")
            elif bool(work_delay[row]):
                color = QtGui.QColor("#EF4444")
            else:
                color = QtGui.QColor("#2563EB")
            painter.fillRect(QtCore.QRectF(x0, work_rect.top(), max(1.0, x1 - x0), work_rect.height()), color)
        painter.restore()

        hop = np.asarray(self.store.node_hop_betweenness[:, node], dtype=np.float64)
        delay = np.asarray(self.store.node_delay_betweenness[:, node], dtype=np.float64)
        max_value = max(1.0, float(np.nanmax(np.r_[hop, delay])))
        painter.fillRect(curve_rect, QtGui.QColor("#FFFFFF"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#E2E8F0"), 1.0))
        painter.drawRect(curve_rect)

        def draw_curve(values: np.ndarray, color: str) -> None:
            painter.save()
            painter.setClipRect(curve_rect.adjusted(-1, -1, 1, 1))
            path = QtGui.QPainterPath()
            for row, value in enumerate(values):
                x = self._x_for_row(row, curve_rect)
                y = curve_rect.bottom() - (max(0.0, float(value)) / max_value) * curve_rect.height()
                if row == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QtGui.QPen(QtGui.QColor(color), 1.6))
            painter.drawPath(path)
            painter.restore()

        draw_curve(hop, "#2563EB")
        draw_curve(delay, "#DC2626")

        current_x = self._x_for_row(self.current_row, self._time_rect())
        painter.setPen(QtGui.QPen(QtGui.QColor("#111827"), 1.6))
        painter.drawLine(
            QtCore.QPointF(current_x, motif_rect.top() - 4),
            QtCore.QPointF(current_x, curve_rect.bottom() + 4),
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

        box = QtWidgets.QGroupBox("Node right-link working inspector")
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.setContentsMargins(8, 8, 8, 8)
        box_layout.setSpacing(6)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter.setHandleWidth(10)

        info_panel = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout(info_panel)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(8)

        self.node_status_label = QtWidgets.QLabel("Click one satellite node to inspect its 24h right-link state.")
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
        self.node_intervals_text.setPlainText("Working intervals will appear here after a node is selected.")

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
        self._interval_cache: dict[tuple[int, str], list[tuple[int, int]]] = {}
        super().__init__(config, **kwargs)

    def _build_ui(self):
        super()._build_ui()
        if self.inspector_mode == "window":
            self.node_inspector_window = NodeRightLinkInspectorWindow(
                self.node_state_store,
                working_mode=self.working_mode,
                title="Node right-link working inspector",
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

        box = QtWidgets.QGroupBox("Node right-link working inspector")
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.setContentsMargins(8, 8, 8, 8)
        box_layout.setSpacing(6)

        self.node_status_label = QtWidgets.QLabel("Click one satellite node to inspect its 24h right-link state.")
        self.node_status_label.setObjectName("statusLabel")
        self.node_status_label.setWordWrap(True)
        box_layout.addWidget(self.node_status_label)

        self.node_timeline = NodeRightLinkTimelineWidget(self.node_state_store, working_mode=self.working_mode)
        box_layout.addWidget(self.node_timeline)

        self.node_intervals_text = QtWidgets.QPlainTextEdit()
        self.node_intervals_text.setReadOnly(True)
        self.node_intervals_text.setPlainText("Working intervals will appear here after a node is selected.")
        box_layout.addWidget(self.node_intervals_text)

        side_layout.addWidget(box)

        help_text = QtWidgets.QLabel(
            "Left click a node on the topology canvas. "
            "The left viewer keeps its original time slider and edge controls; "
            "this right panel only reports the selected node's right-link state."
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
            return None
        viewer_idx = int(self.right_to_viewer_edge_idx[state_edge_idx])
        return None if viewer_idx < 0 else viewer_idx

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
        self.node_status_label.setText("Click one satellite node to inspect its 24h right-link state.")
        self.node_intervals_text.setPlainText("Working intervals will appear here after a node is selected.")
        self.node_timeline.set_selected_node(None)
        self.update_step(self.current_row)

    def update_step(self, row: int, *, sync_slider: bool = True):
        row = int(max(0, min(int(row), len(self.steps) - 1)))
        if self.selected_node is not None:
            self.selected_edge_idx = self._viewer_edge_for_node_row(self.selected_node, row)
        super().update_step(row, sync_slider=sync_slider)
        self.update_node_inspector()

    def _working_array(self, mode: str, node: int) -> np.ndarray:
        if mode == "hop":
            return np.asarray(self.node_state_store.working_hop[:, node], dtype=bool)
        if mode == "delay":
            return np.asarray(self.node_state_store.working_delay[:, node], dtype=bool)
        return np.asarray(self.node_state_store.working_any[:, node], dtype=bool)

    def _intervals_for(self, node: int, mode: str) -> list[tuple[int, int]]:
        key = (int(node), str(mode))
        cached = self._interval_cache.get(key)
        if cached is not None:
            return cached
        intervals = contiguous_true_intervals(self.node_state_store.steps, self._working_array(mode, int(node)))
        self._interval_cache[key] = intervals
        return intervals

    def _format_intervals(self, node: int) -> str:
        lines = []
        for mode in ("hop", "delay", "any"):
            mask = self._working_array(mode, int(node))
            intervals = self._intervals_for(node, mode)
            lines.append(f"{mode}: working_steps={int(mask.sum())}/{len(mask)}, intervals={len(intervals)}")
            for start, end in intervals[:12]:
                lines.append(f"  {start:>6}..{end:<6}  ({start / 3600.0:.2f}h..{end / 3600.0:.2f}h)")
            if len(intervals) > 12:
                lines.append(f"  ... {len(intervals) - 12} more")
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
        symbol_id = int(self.node_state_store.right_symbol_id[row, node])
        symbol = str(SYMBOLS[symbol_id]) if 0 <= symbol_id < len(SYMBOLS) else ""
        option = int(self.node_state_store.right_option[row, node])
        motif_id = int(self.node_state_store.topology_motif_id[row])
        hop_value = float(self.node_state_store.node_hop_betweenness[row, node])
        delay_value = float(self.node_state_store.node_delay_betweenness[row, node])
        working_hop = bool(self.node_state_store.working_hop[row, node])
        working_delay = bool(self.node_state_store.working_delay[row, node])
        working_any = bool(self.node_state_store.working_any[row, node])
        edge_idx = int(self.node_state_store.right_edge_idx[row, node])

        self.pick_label.setText(
            f"Picked node: {node} ({p}, {y}); groups={self.node_group_text(node)}; "
            f"current right edge idx={edge_idx if edge_idx >= 0 else 'none'}"
        )
        self.node_status_label.setText(
            f"step={step} row={row + 1}/{len(self.steps)} | motif={motif_id} | "
            f"node={node} ({p}, {y}) | intra_down={down_node}, intra_up={up_node} | "
            f"right={right_text}, symbol={symbol or 'none'}, option={option if option > -900 else 'none'} | "
            f"hop_betweenness={hop_value:.3f}, delay_betweenness={delay_value:.3f} | "
            f"working hop={int(working_hop)}, delay={int(working_delay)}, any={int(working_any)}"
        )
        self.node_timeline.set_selected_node(node)
        self.node_timeline.set_current_row(row)
        self.node_intervals_text.setPlainText(self._format_intervals(node))
