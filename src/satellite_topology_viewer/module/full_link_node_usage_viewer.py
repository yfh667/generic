from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable
from src.satellite_topology_viewer.module.edge_usage_viewer import EdgeUsageTopology2DViewer
from src.topology_workflow.module.edge_tables import INTRA_OPTION


OPTION_ORDER = (0, 1, 2, 4)
OPTION_SYMBOL = {
    0: "A",
    1: "B",
    2: "D",
    4: "C",
}
OPTION_COLOR = {
    0: "#2563EB",
    1: "#7C3AED",
    2: "#D97706",
    4: "#059669",
}


@dataclass(frozen=True)
class FullLinkNodeUsageData:
    steps: np.ndarray
    edge_table: EdgeTable
    hop_usage: np.ndarray
    delay_usage: np.ndarray
    edge_active_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        steps = np.asarray(self.steps, dtype=np.int64)
        hop = np.asarray(self.hop_usage, dtype=np.float32)
        delay = np.asarray(self.delay_usage, dtype=np.float32)
        if steps.ndim != 1 or steps.size == 0:
            raise ValueError("steps must be a non-empty 1-D array")
        expected = (int(steps.size), int(self.edge_table.num_edges))
        if tuple(hop.shape) != expected:
            raise ValueError(f"hop_usage shape {hop.shape} != expected {expected}")
        if tuple(delay.shape) != expected:
            raise ValueError(f"delay_usage shape {delay.shape} != expected {expected}")
        if self.edge_active_mask is None:
            active = np.ones(expected, dtype=bool)
        else:
            active = np.asarray(self.edge_active_mask, dtype=bool)
            if tuple(active.shape) != expected:
                raise ValueError(f"edge_active_mask shape {active.shape} != expected {expected}")

        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "hop_usage", hop)
        object.__setattr__(self, "delay_usage", delay)
        object.__setattr__(self, "edge_active_mask", active)

    def values_for_mode(self, mode: str) -> np.ndarray:
        mode = str(mode)
        if mode in {"hop", "hops", "primary"}:
            return self.hop_usage
        if mode in {"delay", "secondary"}:
            return self.delay_usage
        if mode in {"any", "max"}:
            return np.maximum(self.hop_usage, self.delay_usage)
        raise ValueError(f"value mode must be hops, delay, or any; got {mode!r}")


def read_edge_table_csv(edges_csv: str | Path, *, total_nodes: int) -> EdgeTable:
    src: list[int] = []
    dst: list[int] = []
    option: list[int] = []
    src_plane: list[int] = []
    src_y: list[int] = []
    dst_plane: list[int] = []
    dst_y: list[int] = []
    with Path(edges_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            src.append(int(row["src_node"]))
            dst.append(int(row["dst_node"]))
            option.append(int(row["option"]))
            src_plane.append(int(row["src_plane"]))
            src_y.append(int(row["src_y"]))
            dst_plane.append(int(row["dst_plane"]))
            dst_y.append(int(row["dst_y"]))

    return EdgeTable(
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        option=np.asarray(option, dtype=np.int16),
        src_plane=np.asarray(src_plane, dtype=np.int16),
        src_y=np.asarray(src_y, dtype=np.int16),
        dst_plane=np.asarray(dst_plane, dtype=np.int16),
        dst_y=np.asarray(dst_y, dtype=np.int16),
        sat_ids=[str(idx + 1) for idx in range(int(total_nodes))],
    )


def _right_owner_neighbor_for_edge(edge_table: EdgeTable, edge_idx: int) -> tuple[int, int] | None:
    edge_idx = int(edge_idx)
    option = int(edge_table.option[edge_idx])
    if option == int(INTRA_OPTION) or option not in OPTION_ORDER:
        return None

    src = int(edge_table.src[edge_idx])
    dst = int(edge_table.dst[edge_idx])
    src_plane = int(edge_table.src_plane[edge_idx])
    dst_plane = int(edge_table.dst_plane[edge_idx])
    if src_plane < dst_plane:
        return src, dst
    if dst_plane < src_plane:
        return dst, src
    return None


def build_node_option_edge_index(
    edge_table: EdgeTable,
    *,
    total_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    edge_idx = np.full((int(total_nodes), len(OPTION_ORDER)), -1, dtype=np.int32)
    neighbor = np.full((int(total_nodes), len(OPTION_ORDER)), -1, dtype=np.int32)
    option_to_col = {int(option): idx for idx, option in enumerate(OPTION_ORDER)}
    for idx in range(int(edge_table.num_edges)):
        option = int(edge_table.option[idx])
        if option not in option_to_col:
            continue
        owner_neighbor = _right_owner_neighbor_for_edge(edge_table, idx)
        if owner_neighbor is None:
            continue
        owner, right = owner_neighbor
        col = option_to_col[option]
        edge_idx[int(owner), int(col)] = int(idx)
        neighbor[int(owner), int(col)] = int(right)
    return edge_idx, neighbor


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


class FullLinkNodeUsageTimelineWidget(QtWidgets.QWidget):
    def __init__(
        self,
        data: FullLinkNodeUsageData,
        *,
        option_edge_idx_by_node: np.ndarray,
        option_neighbor_by_node: np.ndarray,
        config: ViewerConfig,
    ):
        super().__init__()
        self.data = data
        self.option_edge_idx_by_node = np.asarray(option_edge_idx_by_node, dtype=np.int32)
        self.option_neighbor_by_node = np.asarray(option_neighbor_by_node, dtype=np.int32)
        self.config = config
        self.selected_node: int | None = None
        self.current_row = 0
        self.view_start_step = int(self.data.steps[0])
        self.view_end_step = int(self.data.steps[-1])
        self.hover_option_col: int | None = None
        self.hover_row: int | None = None
        self._drag_pos: QtCore.QPoint | None = None
        self._drag_view_start = self.view_start_step
        self._drag_view_end = self.view_end_step
        self.setMouseTracking(True)
        self.setMinimumHeight(520)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_selected_node(self, node: int | None) -> None:
        self.selected_node = None if node is None else int(node)
        self.update()

    def set_current_row(self, row: int) -> None:
        self.current_row = int(max(0, min(int(row), len(self.data.steps) - 1)))
        self.update()

    def min_view_span(self) -> int:
        if len(self.data.steps) <= 1:
            return 1
        diffs = np.diff(np.asarray(self.data.steps, dtype=np.int64))
        stride = int(np.median(diffs)) if diffs.size else 1
        return max(5 * stride, 60)

    def set_view_window(self, start_step: int, end_step: int) -> None:
        full_start = int(self.data.steps[0])
        full_end = int(self.data.steps[-1])
        span = max(self.min_view_span(), int(end_step) - int(start_step))
        if span >= full_end - full_start:
            self.view_start_step, self.view_end_step = full_start, full_end
        else:
            start_step = int(start_step)
            end_step = start_step + span
            if start_step < full_start:
                start_step = full_start
                end_step = start_step + span
            if end_step > full_end:
                end_step = full_end
                start_step = end_step - span
            self.view_start_step, self.view_end_step = int(start_step), int(end_step)
        self.update()

    def zoom_by(self, factor: float, *, focus_step: int | None = None) -> None:
        factor = float(factor)
        if factor <= 0:
            return
        start = float(self.view_start_step)
        end = float(self.view_end_step)
        span = max(float(self.min_view_span()), end - start)
        focus = float(self.data.steps[self.current_row] if focus_step is None else focus_step)
        focus = max(start, min(end, focus))
        ratio = 0.5 if end <= start else (focus - start) / (end - start)
        next_span = max(float(self.min_view_span()), span * factor)
        next_start = focus - ratio * next_span
        self.set_view_window(int(round(next_start)), int(round(next_start + next_span)))

    def _geometry(self) -> dict[str, float]:
        margin_l = 132.0
        margin_r = 24.0
        top = 18.0
        bottom = 46.0
        panel_gap = 13.0
        width = max(80.0, float(self.width()) - margin_l - margin_r)
        panel_h = max(104.0, (float(self.height()) - top - bottom - 3 * panel_gap) / 4.0)
        return {
            "margin_l": margin_l,
            "margin_r": margin_r,
            "top": top,
            "bottom": bottom,
            "panel_gap": panel_gap,
            "width": width,
            "panel_h": panel_h,
        }

    def _panel_rect(self, option_col: int) -> QtCore.QRectF:
        g = self._geometry()
        y = g["top"] + int(option_col) * (g["panel_h"] + g["panel_gap"])
        return QtCore.QRectF(g["margin_l"], y, g["width"], g["panel_h"])

    def _x_for_step(self, step: float, rect: QtCore.QRectF) -> float:
        left = float(self.view_start_step)
        right = float(self.view_end_step)
        if right <= left:
            return float(rect.left())
        return float(rect.left() + (float(step) - left) / (right - left) * rect.width())

    def _x_for_row(self, row: int, rect: QtCore.QRectF) -> float:
        return self._x_for_step(float(self.data.steps[int(row)]), rect)

    def _step_for_x(self, x: float, rect: QtCore.QRectF) -> int:
        if rect.width() <= 0:
            return int(self.data.steps[self.current_row])
        ratio = (float(x) - float(rect.left())) / float(rect.width())
        ratio = max(0.0, min(1.0, ratio))
        return int(round(self.view_start_step + ratio * (self.view_end_step - self.view_start_step)))

    def _row_for_step(self, step: int) -> int:
        values = np.asarray(self.data.steps, dtype=np.int64)
        return int(np.argmin(np.abs(values - int(step))))

    def _hover_target(self, pos: QtCore.QPoint) -> tuple[int, int] | None:
        if self.selected_node is None:
            return None
        for option_col in range(len(OPTION_ORDER)):
            rect = self._panel_rect(option_col)
            if rect.adjusted(-4, -4, 4, 4).contains(QtCore.QPointF(pos)):
                return option_col, self._row_for_step(self._step_for_x(float(pos.x()), rect))
        return None

    def _edge_idx(self, option_col: int) -> int:
        if self.selected_node is None:
            return -1
        return int(self.option_edge_idx_by_node[int(self.selected_node), int(option_col)])

    def _neighbor(self, option_col: int) -> int:
        if self.selected_node is None:
            return -1
        return int(self.option_neighbor_by_node[int(self.selected_node), int(option_col)])

    def _hover_text(self, option_col: int, row: int) -> str:
        edge_idx = self._edge_idx(option_col)
        option = OPTION_ORDER[int(option_col)]
        step = int(self.data.steps[int(row)])
        neighbor = self._neighbor(option_col)
        if edge_idx < 0:
            return f"step={step}s\noption={option} has no right edge for this node"
        hop = float(self.data.hop_usage[int(row), edge_idx])
        delay = float(self.data.delay_usage[int(row), edge_idx])
        return (
            f"step={step}s ({step / 3600.0:.2f}h)\n"
            f"option={option} ({OPTION_SYMBOL[option]}), right={neighbor}, edge_idx={edge_idx}\n"
            f"hops_usage={hop:.3f}, delay_usage={delay:.3f}\n"
            f"working hops={int(hop > 0.0)}, delay={int(delay > 0.0)}"
        )

    def _paint_bool_track(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRectF,
        values: np.ndarray,
        color: QtGui.QColor,
    ) -> None:
        painter.fillRect(rect, QtGui.QColor("#E5E7EB"))
        painter.save()
        painter.setClipRect(rect.adjusted(-1, -1, 1, 1))
        for row, flag in enumerate(np.asarray(values, dtype=bool)):
            if not bool(flag):
                continue
            x0 = self._x_for_row(row, rect)
            x1 = self._x_for_row(min(row + 1, len(values) - 1), rect)
            painter.fillRect(QtCore.QRectF(x0, rect.top(), max(1.0, x1 - x0), rect.height()), color)
        painter.restore()

    def _draw_curve(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRectF,
        values: np.ndarray,
        *,
        vmax: float,
        color: str,
    ) -> None:
        painter.save()
        painter.setClipRect(rect.adjusted(-1, -1, 1, 1))
        path = QtGui.QPainterPath()
        for row, raw_value in enumerate(np.asarray(values, dtype=np.float64)):
            x = self._x_for_row(row, rect)
            y = rect.bottom() - (max(0.0, float(raw_value)) / max(1.0, float(vmax))) * rect.height()
            if row == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QtGui.QPen(QtGui.QColor(color), 1.5))
        painter.drawPath(path)
        painter.restore()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        target = self._hover_target(event.pos())
        rect = self._panel_rect(0)
        focus_step = self._step_for_x(float(event.pos().x()), rect) if target is None else int(self.data.steps[target[1]])
        self.zoom_by(0.78 if delta > 0 else 1.28, focus_step=focus_step)
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
            rect = self._panel_rect(0)
            dx = float(event.pos().x() - self._drag_pos.x())
            span = int(self._drag_view_end - self._drag_view_start)
            shift = -int(round(dx / max(1.0, rect.width()) * span))
            self.set_view_window(self._drag_view_start + shift, self._drag_view_end + shift)
            event.accept()
            return

        target = self._hover_target(event.pos())
        if target is None:
            if self.hover_option_col is not None:
                self.hover_option_col = None
                self.hover_row = None
                self.setToolTip("")
                self.update()
            return
        option_col, row = target
        self.hover_option_col = int(option_col)
        self.hover_row = int(row)
        text = self._hover_text(option_col, row)
        self.setToolTip(text)
        QtWidgets.QToolTip.showText(event.globalPos(), text, self)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._drag_pos is not None:
            self._drag_pos = None
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.set_view_window(int(self.data.steps[0]), int(self.data.steps[-1]))
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#F8FAFC"))
        font = QtGui.QFont("Arial")
        font.setPixelSize(13)
        painter.setFont(font)

        if self.selected_node is None:
            painter.setPen(QtGui.QColor("#334155"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Click one satellite node to inspect full-link option usage.")
            painter.end()
            return

        label_font = QtGui.QFont("Arial")
        label_font.setPixelSize(13)
        label_font.setBold(True)
        node = int(self.selected_node)
        p, y = node_xy(node, int(self.config.N))

        for option_col, option in enumerate(OPTION_ORDER):
            panel = self._panel_rect(option_col)
            edge_idx = self._edge_idx(option_col)
            neighbor = self._neighbor(option_col)
            header_h = 20.0
            work_h = 11.0
            gap = 5.0
            header = QtCore.QRectF(panel.left(), panel.top(), panel.width(), header_h)
            hop_rect = QtCore.QRectF(panel.left(), panel.top() + header_h + gap, panel.width(), work_h)
            delay_rect = QtCore.QRectF(panel.left(), hop_rect.bottom() + gap, panel.width(), work_h)
            curve_rect = QtCore.QRectF(
                panel.left(),
                delay_rect.bottom() + gap,
                panel.width(),
                max(40.0, panel.bottom() - delay_rect.bottom() - gap),
            )

            color = QtGui.QColor(OPTION_COLOR[int(option)])
            painter.fillRect(header, QtGui.QColor("#EEF2F7"))
            painter.setPen(color)
            painter.setFont(label_font)
            if neighbor >= 0:
                np_, ny = node_xy(neighbor, int(self.config.N))
                title = f"option {option} ({OPTION_SYMBOL[option]})  node {node} ({p},{y}) -> {neighbor} ({np_},{ny})"
            else:
                title = f"option {option} ({OPTION_SYMBOL[option]})  node {node} ({p},{y}) -> none"
            painter.drawText(header.adjusted(6, 0, -4, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, title)
            painter.setFont(font)

            painter.setPen(QtGui.QColor("#475569"))
            painter.drawText(8, int(hop_rect.center().y()) + 4, "hops work")
            painter.drawText(8, int(delay_rect.center().y()) + 4, "delay work")
            painter.drawText(8, int(curve_rect.center().y()) + 4, "usage")

            if edge_idx < 0:
                painter.fillRect(hop_rect, QtGui.QColor("#E5E7EB"))
                painter.fillRect(delay_rect, QtGui.QColor("#E5E7EB"))
                painter.fillRect(curve_rect, QtGui.QColor("#FFFFFF"))
                painter.setPen(QtGui.QColor("#94A3B8"))
                painter.drawText(curve_rect, QtCore.Qt.AlignCenter, "no edge")
                continue

            hop = np.asarray(self.data.hop_usage[:, edge_idx], dtype=np.float64)
            delay = np.asarray(self.data.delay_usage[:, edge_idx], dtype=np.float64)
            self._paint_bool_track(painter, hop_rect, hop > 0.0, QtGui.QColor("#2563EB"))
            self._paint_bool_track(painter, delay_rect, delay > 0.0, QtGui.QColor("#DC2626"))
            painter.fillRect(curve_rect, QtGui.QColor("#FFFFFF"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#E2E8F0"), 1.0))
            painter.drawRect(curve_rect)
            vmax = max(1.0, float(np.nanmax(np.r_[hop, delay])))
            self._draw_curve(painter, curve_rect, hop, vmax=vmax, color="#2563EB")
            self._draw_curve(painter, curve_rect, delay, vmax=vmax, color="#DC2626")

            current_x = self._x_for_row(self.current_row, panel)
            painter.setPen(QtGui.QPen(QtGui.QColor("#111827"), 1.2))
            painter.drawLine(QtCore.QPointF(current_x, panel.top()), QtCore.QPointF(current_x, panel.bottom()))

            if self.hover_option_col == option_col and self.hover_row is not None:
                hover_row = int(max(0, min(int(self.hover_row), len(self.data.steps) - 1)))
                hover_x = self._x_for_row(hover_row, panel)
                painter.setPen(QtGui.QPen(QtGui.QColor("#0F172A"), 1.1, QtCore.Qt.DashLine))
                painter.drawLine(QtCore.QPointF(hover_x, panel.top()), QtCore.QPointF(hover_x, panel.bottom()))
                for values, dot_color in ((hop, "#2563EB"), (delay, "#DC2626")):
                    dot_y = curve_rect.bottom() - (max(0.0, float(values[hover_row])) / vmax) * curve_rect.height()
                    painter.setPen(QtGui.QPen(QtGui.QColor("#FFFFFF"), 1.0))
                    painter.setBrush(QtGui.QColor(dot_color))
                    painter.drawEllipse(QtCore.QPointF(hover_x, dot_y), 3.5, 3.5)

        painter.setPen(QtGui.QColor("#334155"))
        span = max(1, int(self.view_end_step - self.view_start_step))
        tick_seconds = 600 if span <= 3600 else 1800 if span <= 4 * 3600 else 3600 if span <= 12 * 3600 else 6 * 3600
        first_tick = int(math.ceil(self.view_start_step / tick_seconds) * tick_seconds)
        bottom_y = int(self.height()) - 11
        for step in range(first_tick, self.view_end_step + 1, tick_seconds):
            if step < int(self.data.steps[0]) or step > int(self.data.steps[-1]):
                continue
            x = self._x_for_step(step, self._panel_rect(0))
            label = f"{step / 3600.0:.1f}h" if tick_seconds < 3600 else f"{int(step / 3600)}h"
            painter.drawText(int(x) - 18, bottom_y, label)
        painter.drawText(
            132,
            int(self.height()) - 32,
            f"visible {self.view_start_step}s..{self.view_end_step}s; wheel=zoom, drag=pan, double click=fit",
        )
        painter.end()


class FullLinkNodeUsageInspectorWindow(QtWidgets.QWidget):
    def __init__(
        self,
        data: FullLinkNodeUsageData,
        *,
        option_edge_idx_by_node: np.ndarray,
        option_neighbor_by_node: np.ndarray,
        config: ViewerConfig,
    ):
        super().__init__()
        self.setWindowTitle("Full-link node option usage inspector")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.data = data
        self.option_edge_idx_by_node = option_edge_idx_by_node
        self.option_neighbor_by_node = option_neighbor_by_node
        self.config = config
        self.selected_node: int | None = None
        self.current_row = 0
        self.setStyleSheet(
            """
            QWidget { font-size: 15px; }
            QPushButton { font-size: 15px; min-height: 32px; padding: 4px 10px; }
            QLabel#statusLabel { font-size: 15px; line-height: 1.35; }
            QPlainTextEdit { font-family: Consolas, "Courier New", monospace; font-size: 14px; }
            """
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.status_label = QtWidgets.QLabel("Click one satellite node to inspect four full-link right-neighbor options.")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        tool_row = QtWidgets.QHBoxLayout()
        layout.addLayout(tool_row)
        self.fit_day_btn = QtWidgets.QPushButton("Fit day")
        self.zoom_in_btn = QtWidgets.QPushButton("Zoom +")
        self.zoom_out_btn = QtWidgets.QPushButton("Zoom -")
        for button in (self.fit_day_btn, self.zoom_in_btn, self.zoom_out_btn):
            tool_row.addWidget(button)
        tool_row.addStretch(1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.setHandleWidth(10)
        layout.addWidget(splitter, stretch=1)
        self.timeline = FullLinkNodeUsageTimelineWidget(
            data,
            option_edge_idx_by_node=option_edge_idx_by_node,
            option_neighbor_by_node=option_neighbor_by_node,
            config=config,
        )
        self.intervals_text = QtWidgets.QPlainTextEdit()
        self.intervals_text.setReadOnly(True)
        splitter.addWidget(self.timeline)
        splitter.addWidget(self.intervals_text)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([610, 170])

        self.fit_day_btn.clicked.connect(lambda: self.timeline.set_view_window(int(data.steps[0]), int(data.steps[-1])))
        self.zoom_in_btn.clicked.connect(lambda: self.timeline.zoom_by(0.78))
        self.zoom_out_btn.clicked.connect(lambda: self.timeline.zoom_by(1.28))
        self.resize(1220, 860)

    def update_for(self, node: int | None, row: int) -> None:
        self.selected_node = None if node is None else int(node)
        self.current_row = int(max(0, min(int(row), len(self.data.steps) - 1)))
        self.timeline.set_selected_node(self.selected_node)
        self.timeline.set_current_row(self.current_row)
        if self.selected_node is None:
            self.status_label.setText("Click one satellite node to inspect four full-link right-neighbor options.")
            self.intervals_text.setPlainText("")
            return

        node = int(self.selected_node)
        step = int(self.data.steps[self.current_row])
        p, y = node_xy(node, int(self.config.N))
        lines = [f"node={node} ({p},{y}), step={step}s ({step / 3600.0:.2f}h)"]
        interval_lines: list[str] = []
        for option_col, option in enumerate(OPTION_ORDER):
            edge_idx = int(self.option_edge_idx_by_node[node, option_col])
            neighbor = int(self.option_neighbor_by_node[node, option_col])
            if edge_idx < 0:
                lines.append(f"option {option} ({OPTION_SYMBOL[option]}): no right edge")
                interval_lines.append(f"option {option} ({OPTION_SYMBOL[option]}): no right edge")
                continue
            hop_now = float(self.data.hop_usage[self.current_row, edge_idx])
            delay_now = float(self.data.delay_usage[self.current_row, edge_idx])
            np_, ny = node_xy(neighbor, int(self.config.N))
            lines.append(
                f"option {option} ({OPTION_SYMBOL[option]}): right={neighbor} ({np_},{ny}), "
                f"edge_idx={edge_idx}, hops={hop_now:.3f}, delay={delay_now:.3f}"
            )
            interval_lines.append(f"option {option} ({OPTION_SYMBOL[option]}) edge_idx={edge_idx} right={neighbor}:")
            for label, values in (
                ("hops", self.data.hop_usage[:, edge_idx]),
                ("delay", self.data.delay_usage[:, edge_idx]),
            ):
                mask = np.asarray(values, dtype=np.float32) > 0.0
                intervals = contiguous_true_intervals(self.data.steps, mask)
                interval_lines.append(f"  {label}: working_steps={int(mask.sum())}/{len(mask)}, intervals={len(intervals)}")
                for start, end in intervals[:8]:
                    interval_lines.append(f"    {start:>6}..{end:<6} ({start / 3600.0:.2f}h..{end / 3600.0:.2f}h)")
                if len(intervals) > 8:
                    interval_lines.append(f"    ... {len(intervals) - 8} more")
        self.status_label.setText(" | ".join(lines))
        self.intervals_text.setPlainText("\n".join(interval_lines))


class FullLinkNodeUsageTopologyViewer(EdgeUsageTopology2DViewer):
    def __init__(
        self,
        config: ViewerConfig,
        *,
        data: FullLinkNodeUsageData,
        value_mode: str = "delay",
        group_data: dict | None = None,
        show_groups: bool = True,
        window_title: str | None = None,
        **viewer_kwargs,
    ):
        self.full_link_data = data
        self.value_mode = str(value_mode)
        self.option_edge_idx_by_node, self.option_neighbor_by_node = build_node_option_edge_index(
            data.edge_table,
            total_nodes=int(config.total_sats),
        )
        self.selected_node: int | None = None
        values = data.values_for_mode(value_mode)
        defaults = {
            "topology_edge_color": "#000000",
            "topology_edge_alpha": 135,
            "topology_edge_width": 0.010,
            "value_width_min": 0.008,
            "value_width_max": 0.095,
            "value_alpha_min": 30,
            "value_alpha_max": 245,
            "hide_y_wrap_edges": True,
            "show_grid_lines": False,
            "edge_value_label": f"{value_mode}_usage",
        }
        defaults.update(viewer_kwargs)
        super().__init__(
            config,
            steps=[int(x) for x in data.steps],
            edge_table=data.edge_table,
            edge_usage_values=values,
            value_max=max(1.0, float(np.nanmax(values)) if values.size else 1.0),
            edge_active_mask=data.edge_active_mask,
            window_title=window_title or f"{config.name} full-link node usage viewer",
            group_data=group_data or {},
            show_groups=show_groups,
            **defaults,
        )

    def _build_ui(self):
        super()._build_ui()
        self.full_link_inspector_window = FullLinkNodeUsageInspectorWindow(
            self.full_link_data,
            option_edge_idx_by_node=self.option_edge_idx_by_node,
            option_neighbor_by_node=self.option_neighbor_by_node,
            config=self.config,
        )

    def _best_edge_for_node_row(self, node: int, row: int) -> int | None:
        edge_indices = np.asarray(self.option_edge_idx_by_node[int(node), :], dtype=np.int32)
        values = self.full_link_data.values_for_mode(self.value_mode)
        best_idx = -1
        best_value = -1.0
        for edge_idx in edge_indices:
            edge_idx = int(edge_idx)
            if edge_idx < 0:
                continue
            value = float(values[int(row), edge_idx])
            if value > best_value:
                best_idx = edge_idx
                best_value = value
        return None if best_idx < 0 else int(best_idx)

    def show_inspector_window(self) -> None:
        self.full_link_inspector_window.show()
        self.full_link_inspector_window.raise_()
        self.full_link_inspector_window.activateWindow()

    def pick_node(self, node: int):
        node = int(node)
        if not (0 <= node < int(self.config.total_sats)):
            return
        self.selected_node = node
        self.picked_nodes = [node]
        self.selected_edge_idx = self._best_edge_for_node_row(node, self.current_row)
        self.update_pick_markers()
        self.update_full_link_inspector()
        self.show_inspector_window()
        self.update_step(self.current_row)

    def clear_picked_nodes(self):
        self.selected_node = None
        self.picked_nodes = []
        self.selected_edge_idx = None
        self.pick_label.setText("Picked nodes: none")
        self.full_link_inspector_window.update_for(None, self.current_row)
        self.update_step(self.current_row)

    def update_step(self, row: int, *, sync_slider: bool = True):
        row = int(max(0, min(int(row), len(self.steps) - 1)))
        if self.selected_node is not None:
            self.selected_edge_idx = self._best_edge_for_node_row(self.selected_node, row)
        super().update_step(row, sync_slider=sync_slider)
        self.update_full_link_inspector()

    def update_full_link_inspector(self):
        if self.selected_node is None:
            return
        node = int(self.selected_node)
        p, y = node_xy(node, int(self.config.N))
        option_texts = []
        for option_col, option in enumerate(OPTION_ORDER):
            neighbor = int(self.option_neighbor_by_node[node, option_col])
            edge_idx = int(self.option_edge_idx_by_node[node, option_col])
            if edge_idx < 0:
                option_texts.append(f"{option}:none")
            else:
                option_texts.append(f"{option}:{neighbor}(e{edge_idx})")
        self.pick_label.setText(
            f"Picked node: {node} ({p}, {y}); groups={self.node_group_text(node)}; "
            f"full-link right options: {', '.join(option_texts)}"
        )
        self.full_link_inspector_window.update_for(node, self.current_row)
