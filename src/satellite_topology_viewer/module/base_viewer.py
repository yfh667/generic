from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


GENERIC_ROOT = Path(__file__).resolve().parents[3]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import build_full_option_edges


COLOR_STOPS = (
    (0.00, (49, 54, 149)),
    (0.25, (69, 117, 180)),
    (0.50, (224, 243, 248)),
    (0.70, (254, 224, 144)),
    (0.85, (244, 109, 67)),
    (1.00, (165, 0, 38)),
)


def color_from_value(value: float, vmin: float, vmax: float, alpha: int) -> QtGui.QColor:
    if vmax <= vmin:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (float(value) - float(vmin)) / (float(vmax) - float(vmin))))

    rgb = COLOR_STOPS[-1][1]
    for idx in range(len(COLOR_STOPS) - 1):
        left_t, left_rgb = COLOR_STOPS[idx]
        right_t, right_rgb = COLOR_STOPS[idx + 1]
        if left_t <= t <= right_t:
            ratio = 0.0 if right_t == left_t else (t - left_t) / (right_t - left_t)
            rgb = tuple(
                int(round(left_rgb[channel] + ratio * (right_rgb[channel] - left_rgb[channel])))
                for channel in range(3)
            )
            break

    color = QtGui.QColor(*rgb)
    color.setAlpha(int(alpha))
    return color


def make_fake_edge_value(edge_table, step: int, edge_idx: int) -> float:
    option = int(edge_table.option[edge_idx])
    src_plane = int(edge_table.src_plane[edge_idx])
    src_y = int(edge_table.src_y[edge_idx])
    dst_y = int(edge_table.dst_y[edge_idx])

    base_by_option = {
        0: 4.5,
        1: 5.0,
        2: 9.0,
        4: 7.1,
    }
    phase = 0.055 * int(step) + 0.19 * src_y + 0.37 * src_plane + 0.11 * dst_y
    slow_wave = math.sin(phase)
    fast_wave = 0.35 * math.cos(0.13 * int(step) + 0.41 * src_plane)
    option_bias = 0.18 * (option in {1, 4}) - 0.12 * (option == 0)
    delay = base_by_option.get(option, 6.0) + 0.45 * slow_wave + fast_wave + option_bias
    return max(2.0, min(10.0, float(delay)))


def build_fake_edge_value_matrix(edge_table, steps: list[int]) -> np.ndarray:
    values = np.empty((len(steps), edge_table.num_edges), dtype=np.float32)
    for row, step in enumerate(steps):
        for edge_idx in range(edge_table.num_edges):
            values[row, edge_idx] = make_fake_edge_value(edge_table, step, edge_idx)
    return values


color_from_delay = color_from_value
make_fake_delay_ms = make_fake_edge_value
build_fake_delay_matrix = build_fake_edge_value_matrix


def distance_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def distance_point_to_polyline(px: float, py: float, points: list[tuple[float, float]]) -> float:
    best = math.inf
    for idx in range(len(points) - 1):
        ax, ay = points[idx]
        bx, by = points[idx + 1]
        best = min(best, distance_point_to_segment(px, py, ax, ay, bx, by))
    return best


class TopologyGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, owner: "SatelliteTopology2DViewer"):
        super().__init__()
        self.owner = owner
        self._panning = False
        self._left_press_pos: QtCore.QPoint | None = None
        self._left_dragging = False
        self._pan_start = QtCore.QPoint()
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setMouseTracking(True)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.owner.fit_on_next_resize:
            self.owner.fit_scene()
            self.owner.fit_on_next_resize = False

    def mouseMoveEvent(self, event):
        if self._left_press_pos is not None:
            delta_from_press = event.pos() - self._left_press_pos
            if self._left_dragging or delta_from_press.manhattanLength() >= 7:
                self._left_dragging = True
                delta = event.pos() - self._pan_start
                self._pan_start = QtCore.QPoint(event.pos())
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
                self.setCursor(QtCore.Qt.ClosedHandCursor)
                event.accept()
                return

        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = QtCore.QPoint(event.pos())
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        self.owner.handle_scene_hover(self.mapToScene(event.pos()))
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() in (QtCore.Qt.RightButton, QtCore.Qt.MiddleButton):
            self._panning = True
            self._pan_start = QtCore.QPoint(event.pos())
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton:
            self._left_press_pos = QtCore.QPoint(event.pos())
            self._pan_start = QtCore.QPoint(event.pos())
            self._left_dragging = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._left_press_pos is not None:
            if not self._left_dragging:
                self.owner.handle_scene_click(self.mapToScene(event.pos()))
            self._left_press_pos = None
            self._left_dragging = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
            return

        if event.button() in (QtCore.Qt.RightButton, QtCore.Qt.MiddleButton) and self._panning:
            self._panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        steps = delta / 120.0
        self.owner.zoom_view(1.18 ** steps)
        event.accept()


class SatelliteTopology2DViewer(QtWidgets.QWidget):
    def __init__(
        self,
        config: ViewerConfig,
        *,
        start: int = 0,
        end: int = 100,
        steps: list[int] | None = None,
        edge_table=None,
        edge_values=None,
        value_min: float | None = None,
        value_max: float | None = None,
        window_title: str = "Satellite topology 2D viewer",
        edge_value_label: str = "edge_value",
        group_data: dict | None = None,
        show_groups: bool = True,
    ):
        super().__init__()
        self.config = config
        self.steps = [int(x) for x in (steps if steps is not None else range(int(start), int(end) + 1))]
        self.full_steps = list(self.steps)
        self.visible_rows = list(range(len(self.steps)))
        self.visible_row_to_pos = {row: row for row in self.visible_rows}
        self.subrange_steps: list[int] | None = None
        self.edge_table = edge_table if edge_table is not None else build_full_option_edges(config, options=(0, 1, 2, 4))
        self.has_edge_values = edge_values is not None
        if self.has_edge_values:
            self.edge_values = np.asarray(edge_values, dtype=np.float32)
        else:
            self.edge_values = np.zeros((len(self.steps), int(self.edge_table.num_edges)), dtype=np.float32)
        if int(self.edge_values.shape[0]) != len(self.steps):
            raise ValueError(f"edge value rows {self.edge_values.shape[0]} != steps length {len(self.steps)}")
        if int(self.edge_values.shape[1]) != int(self.edge_table.num_edges):
            raise ValueError(f"edge value cols {self.edge_values.shape[1]} != edge count {self.edge_table.num_edges}")
        if self.has_edge_values:
            self.value_min = float(np.nanmin(self.edge_values)) if value_min is None else float(value_min)
            self.value_max = float(np.nanmax(self.edge_values)) if value_max is None else float(value_max)
        else:
            self.value_min = 0.0 if value_min is None else float(value_min)
            self.value_max = 0.0 if value_max is None else float(value_max)
        self.window_title = str(window_title)
        self.edge_value_label = str(edge_value_label)
        self.group_data = group_data or {}
        self.show_groups_default = bool(show_groups and self.group_data)

        self.edge_width = 0.018
        self.edge_alpha = 145
        self.node_radius = 0.14
        self.node_hit_radius = 0.34
        self.edge_hit_threshold = 0.13
        self.zoom_factor = 1.0
        self.min_zoom_factor = 0.35
        self.max_zoom_factor = 8.0
        self.fit_on_next_resize = True
        self.visible_options = {0: True, 1: True, 2: True, 4: True}
        self.current_row = 0
        self.selected_edge_idx: int | None = None
        self.preview_edge_idx: int | None = None
        self.picked_nodes: list[int] = []

        self.edge_items: list[QtWidgets.QGraphicsPathItem] = []
        self.node_items: list[QtWidgets.QGraphicsEllipseItem] = []
        self.node_base_pen = QtGui.QPen(QtGui.QColor(80, 80, 80), 0.0)
        self.edge_samples: list[list[tuple[float, float]]] = []
        self.node_to_edge: dict[tuple[int, int], int] = {}
        self.pick_marker_items: list[QtWidgets.QGraphicsEllipseItem] = []
        self.axis_label_items: list[QtWidgets.QGraphicsSimpleTextItem] = []

        self._build_ui()
        self._build_scene()
        self.update_step(0)

    def _build_ui(self):
        self.setWindowTitle(self.window_title)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self._apply_large_control_style()

        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = TopologyGraphicsView(self)
        self.view.setScene(self.scene)
        self.view.setMinimumHeight(160)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.main_splitter.setHandleWidth(10)
        self.main_splitter.addWidget(self.view)

        self.controls_panel = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(self.controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self.controls_scroll = QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.controls_scroll.setMinimumHeight(76)
        self.controls_scroll.setWidget(self.controls_panel)
        self.main_splitter.addWidget(self.controls_scroll)
        self.main_splitter.setStretchFactor(0, 8)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setSizes([720, 220])
        layout.addWidget(self.main_splitter, stretch=1)

        time_row = QtWidgets.QHBoxLayout()
        controls_layout.addLayout(time_row)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(self.visible_rows) - 1))
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.on_slider)
        self.slider.setMinimumHeight(28)
        time_row.addWidget(self.slider, stretch=1)

        range_layout = QtWidgets.QHBoxLayout()
        controls_layout.addLayout(range_layout)
        self.range_start_input = QtWidgets.QLineEdit()
        self.range_start_input.setPlaceholderText("区间起点")
        self.range_start_input.setMinimumWidth(150)
        self.range_end_input = QtWidgets.QLineEdit()
        self.range_end_input.setPlaceholderText("区间终点")
        self.range_end_input.setMinimumWidth(150)
        self.set_range_btn = QtWidgets.QPushButton("设定区间")
        self.exit_range_btn = QtWidgets.QPushButton("退出区间")
        self.set_range_btn.clicked.connect(self.set_subrange)
        self.exit_range_btn.clicked.connect(self.exit_subrange)
        range_layout.addWidget(self.range_start_input)
        range_layout.addWidget(self.range_end_input)
        range_layout.addWidget(self.set_range_btn)
        range_layout.addWidget(self.exit_range_btn)

        jump_layout = QtWidgets.QHBoxLayout()
        controls_layout.addLayout(jump_layout)
        self.jump_input = QtWidgets.QLineEdit()
        self.jump_input.setPlaceholderText("跳转到 step")
        self.jump_input.setMinimumWidth(190)
        self.jump_button = QtWidgets.QPushButton("跳转")
        self.jump_button.clicked.connect(self.on_jump)
        jump_layout.addWidget(self.jump_input)
        jump_layout.addWidget(self.jump_button)

        step_layout = QtWidgets.QHBoxLayout()
        controls_layout.addLayout(step_layout)
        self.prev_btn = QtWidgets.QPushButton("<")
        self.next_btn = QtWidgets.QPushButton(">")
        self.prev_btn.clicked.connect(self.step_prev)
        self.next_btn.clicked.connect(self.step_next)
        step_layout.addWidget(self.prev_btn)
        step_layout.addWidget(self.next_btn)

        self.step_label = QtWidgets.QLabel("")
        self.step_label.setObjectName("statusLabel")
        controls_layout.addWidget(self.step_label)
        self.coord_label = QtWidgets.QLabel("Cursor grid: none")
        self.coord_label.setObjectName("statusLabel")
        controls_layout.addWidget(self.coord_label)

        colorbar_row = QtWidgets.QHBoxLayout()
        controls_layout.addLayout(colorbar_row)
        colorbar_row.addStretch(1)
        self.colorbar_label = QtWidgets.QLabel()
        colorbar_row.addWidget(self.colorbar_label)
        colorbar_row.addStretch(1)

        group_layout = QtWidgets.QHBoxLayout()
        controls_layout.addLayout(group_layout)
        self.show_groups_checkbox = QtWidgets.QCheckBox("显示 region groups")
        self.show_groups_checkbox.setChecked(self.show_groups_default)
        self.show_groups_checkbox.stateChanged.connect(lambda _state: self.update_step(self.current_row))
        group_layout.addWidget(self.show_groups_checkbox)
        legend_parts = []
        for gid in sorted(self.config.station_groups):
            if int(gid) < len(self.config.group_colors):
                color = self.config.group_colors[int(gid)]
            else:
                color = "#777777"
            name = self.config.station_groups[int(gid)].get("name", f"Group {gid}")
            legend_parts.append(f'<span style="color:{color};">&#9679;</span> {name}')
        self.group_legend_label = QtWidgets.QLabel("  ".join(legend_parts))
        self.group_legend_label.setWordWrap(True)
        group_layout.addWidget(self.group_legend_label, stretch=1)

        control_row = QtWidgets.QHBoxLayout()
        controls_layout.addLayout(control_row)
        self.option_checks: dict[int, QtWidgets.QCheckBox] = {}
        for option in (0, 1, 2, 4):
            cb = QtWidgets.QCheckBox(f"option {option}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.on_option_changed)
            self.option_checks[option] = cb
            control_row.addWidget(cb)

        self.width_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.width_slider.setMinimum(8)
        self.width_slider.setMaximum(50)
        self.width_slider.setValue(int(self.edge_width * 1000))
        self.width_slider.valueChanged.connect(self.on_width_changed)
        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_slider.setMinimum(25)
        self.alpha_slider.setMaximum(190)
        self.alpha_slider.setValue(self.edge_alpha)
        self.alpha_slider.valueChanged.connect(self.on_alpha_changed)
        control_row.addWidget(QtWidgets.QLabel("width"))
        control_row.addWidget(self.width_slider)
        control_row.addWidget(QtWidgets.QLabel("alpha"))
        control_row.addWidget(self.alpha_slider)

        self.zoom_out_btn = QtWidgets.QPushButton("Zoom -")
        self.zoom_in_btn = QtWidgets.QPushButton("Zoom +")
        self.zoom_reset_btn = QtWidgets.QPushButton("Reset View")
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_view(1 / 1.25))
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_view(1.25))
        self.zoom_reset_btn.clicked.connect(self.reset_view)
        control_row.addWidget(self.zoom_out_btn)
        control_row.addWidget(self.zoom_in_btn)
        control_row.addWidget(self.zoom_reset_btn)

        info_row = QtWidgets.QHBoxLayout()
        controls_layout.addLayout(info_row)
        self.clear_btn = QtWidgets.QPushButton("Clear picked nodes")
        self.clear_btn.clicked.connect(self.clear_picked_nodes)
        self.pick_label = QtWidgets.QLabel("Picked nodes: none")
        self.pick_label.setObjectName("statusLabel")
        self.pick_label.setWordWrap(True)
        info_row.addWidget(self.clear_btn)
        info_row.addWidget(self.pick_label, stretch=1)

        self.hover_label = QtWidgets.QLabel("Nearest edge: none")
        self.selected_label = QtWidgets.QLabel("Selected edge: none")
        self.hover_label.setObjectName("statusLabel")
        self.selected_label.setObjectName("statusLabel")
        self.hover_label.setWordWrap(True)
        self.selected_label.setWordWrap(True)
        controls_layout.addWidget(self.hover_label)
        controls_layout.addWidget(self.selected_label)
        self.update_colorbar_labels()

    def _apply_large_control_style(self):
        base_font = QtGui.QFont()
        base_font.setPointSize(11)
        self.setFont(base_font)
        self.setStyleSheet(
            """
            QWidget {
                font-size: 14px;
            }
            QPushButton {
                font-size: 15px;
                min-height: 32px;
                padding: 4px 12px;
            }
            QLineEdit, QSpinBox {
                font-size: 15px;
                min-height: 30px;
                padding: 2px 6px;
            }
            QCheckBox {
                font-size: 14px;
                spacing: 6px;
            }
            QLabel#statusLabel {
                font-size: 15px;
                min-height: 24px;
            }
            QSplitter::handle:vertical {
                background: #C7D0DC;
                height: 10px;
                margin: 2px 0;
            }
            QSplitter::handle:vertical:hover {
                background: #8FA2BA;
            }
            QSlider::groove:horizontal {
                height: 8px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                margin: -6px 0;
            }
            """
        )

    def _refresh_visible_row_map(self):
        self.visible_row_to_pos = {int(row): pos for pos, row in enumerate(self.visible_rows)}

    def _set_visible_rows(self, rows: list[int], *, target_row: int | None = None):
        rows = [int(row) for row in rows if 0 <= int(row) < len(self.steps)]
        if not rows:
            return
        self.visible_rows = rows
        self._refresh_visible_row_map()
        self.slider.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(self.visible_rows) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.update_step(self.visible_rows[0] if target_row is None else target_row)

    def _current_visible_pos(self) -> int:
        return int(self.visible_row_to_pos.get(int(self.current_row), int(self.slider.value())))

    def on_slider(self, value: int):
        if not self.visible_rows:
            return
        pos = int(max(0, min(int(value), len(self.visible_rows) - 1)))
        self.update_step(self.visible_rows[pos], sync_slider=False)

    def step_prev(self):
        if not self.visible_rows:
            return
        pos = self._current_visible_pos()
        if pos > 0:
            self.update_step(self.visible_rows[pos - 1])

    def step_next(self):
        if not self.visible_rows:
            return
        pos = self._current_visible_pos()
        if pos < len(self.visible_rows) - 1:
            self.update_step(self.visible_rows[pos + 1])

    def on_jump(self):
        text = self.jump_input.text().strip()
        if not text:
            return
        try:
            step = int(text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "输入错误", "请输入有效的 step")
            return

        row = self.step_to_row(step)
        if row is None:
            QtWidgets.QMessageBox.warning(self, "跳转失败", f"未找到 step={step}")
            return
        if row not in self.visible_row_to_pos:
            self.exit_subrange()
        self.update_step(row)

    def set_subrange(self):
        try:
            start = int(self.range_start_input.text())
            end = int(self.range_end_input.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "输入错误", "请输入有效的区间起止 step")
            return
        if start > end:
            start, end = end, start

        rows = [idx for idx, step in enumerate(self.full_steps) if start <= int(step) <= end]
        if not rows:
            QtWidgets.QMessageBox.warning(self, "范围无效", "未找到该区间内的 step")
            return
        self.subrange_steps = [self.full_steps[row] for row in rows]
        self._set_visible_rows(rows)

    def exit_subrange(self):
        if self.subrange_steps is None and len(self.visible_rows) == len(self.full_steps):
            return
        self.subrange_steps = None
        self._set_visible_rows(list(range(len(self.full_steps))))

    def step_to_row(self, step: int) -> int | None:
        for idx, value in enumerate(self.full_steps):
            if int(value) == int(step):
                return idx
        return None

    def make_colorbar_pixmap(self, width: int, height: int) -> QtGui.QPixmap:
        pixmap = QtGui.QPixmap(int(width), int(height))
        pixmap.fill(QtGui.QColor(245, 245, 245))
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        font = QtGui.QFont("Arial", 10)
        painter.setFont(font)

        bar_x = 92
        bar_y = 26
        bar_w = max(40, int(width) - 184)
        bar_h = 18
        for x in range(bar_w):
            t = x / max(1, bar_w - 1)
            value = self.value_min + t * (self.value_max - self.value_min)
            painter.setPen(color_from_value(value, self.value_min, self.value_max, 255))
            painter.drawLine(bar_x + x, bar_y, bar_x + x, bar_y + bar_h)

        painter.setPen(QtGui.QColor(55, 55, 55))
        painter.drawRect(bar_x, bar_y, bar_w, bar_h)
        painter.drawText(int(width / 2) - 72, 16, f"{self.edge_value_label} color scale")
        painter.drawText(4, bar_y + 14, f"{self.value_min:.3f}")
        painter.drawText(bar_x + bar_w + 8, bar_y + 14, f"{self.value_max:.3f}")
        painter.end()
        return pixmap

    def update_colorbar_labels(self):
        if not hasattr(self, "colorbar_label"):
            return
        if not self.has_edge_values:
            self.colorbar_label.clear()
            self.colorbar_label.setVisible(False)
            return
        self.colorbar_label.setVisible(True)
        self.colorbar_label.setFixedSize(520, 52)
        self.colorbar_label.setPixmap(self.make_colorbar_pixmap(520, 52))

    def node_grid_pos(self, raw_node: int) -> tuple[float, float]:
        raw_node = int(raw_node)
        p, y = divmod(raw_node, int(self.config.N))
        return float(p), self.raw_y_to_scene_y(y)

    def grid_pos_to_node(self, p: int, y: int) -> int:
        raw_y = self.scene_y_to_raw_y(y)
        return int(p) * int(self.config.N) + int(raw_y)

    def raw_y_to_scene_y(self, raw_y: int | float) -> float:
        return float(int(self.config.N) - 1 - float(raw_y))

    def scene_y_to_raw_y(self, scene_y: int | float) -> int:
        return int(self.config.N) - 1 - int(scene_y)

    def scene_y_to_axis_y(self, scene_y: int | float) -> float:
        return float(int(self.config.N) - 1 - float(scene_y))

    def _build_scene(self):
        self.scene.clear()
        self.edge_items = []
        self.node_items = []
        self.edge_samples = []
        self.node_to_edge = {}
        self.pick_marker_items = []
        self.axis_label_items = []

        self.scene.setSceneRect(-2.3, -1.8, self.config.P + 3.0, self.config.N + 2.8)
        self._draw_grid()
        self._draw_axis_labels()
        self._draw_edges()
        self._draw_nodes()
        self._draw_pick_markers()
        self.fit_scene()

    def _draw_grid(self):
        grid_pen = QtGui.QPen(QtGui.QColor(215, 215, 215))
        grid_pen.setWidthF(0.0)
        for p in range(self.config.P):
            self.scene.addLine(p, -0.4, p, self.config.N - 0.6, grid_pen)
        for y in range(self.config.N):
            self.scene.addLine(-0.4, y, self.config.P - 0.6, y, grid_pen)

    def _make_axis_label(
        self,
        text: str,
        x: float,
        y: float,
        *,
        color: str = "#2F3A46",
        bold: bool = False,
        pixel_size: int = 8,
    ) -> QtWidgets.QGraphicsSimpleTextItem:
        item = QtWidgets.QGraphicsSimpleTextItem(str(text))
        font = QtGui.QFont("Arial")
        font.setPixelSize(int(pixel_size))
        font.setBold(bool(bold))
        item.setFont(font)
        item.setBrush(QtGui.QBrush(QtGui.QColor(color)))
        item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        item.setZValue(120)
        item.setPos(float(x), float(y))
        self.scene.addItem(item)
        self.axis_label_items.append(item)
        return item

    def _draw_axis_labels(self):
        axis_pen = QtGui.QPen(QtGui.QColor(92, 105, 120))
        axis_pen.setWidthF(0.0)
        self.scene.addLine(-0.4, -0.55, self.config.P - 0.6, -0.55, axis_pen)
        self.scene.addLine(-0.55, -0.4, -0.55, self.config.N - 0.6, axis_pen)
        self.scene.addLine(-0.4, self.config.N - 0.35, self.config.P - 0.6, self.config.N - 0.35, axis_pen)
        self.scene.addLine(self.config.P - 0.35, -0.4, self.config.P - 0.35, self.config.N - 0.6, axis_pen)

        self._make_axis_label("x", self.config.P + 0.48, self.config.N + 0.20, color="#1B4D89", bold=True, pixel_size=18)
        self._make_axis_label("y", -1.85, -0.45, color="#8A3FFC", bold=True, pixel_size=18)
        for p in range(int(self.config.P)):
            self._make_axis_label(str(p), p - 0.16, self.config.N + 0.20, color="#1B4D89", bold=True, pixel_size=15)
            self._make_axis_label(str(p), p - 0.16, -1.32, color="#1B4D89", pixel_size=15)

        right_x = float(self.config.P) + 0.12
        for raw_y in range(int(self.config.N)):
            if raw_y % 5 != 0 and raw_y != int(self.config.N) - 1:
                continue
            y_pos = self.raw_y_to_scene_y(raw_y) - 0.18
            self._make_axis_label(str(raw_y), -1.55, y_pos, color="#8A3FFC", bold=True, pixel_size=15)
            self._make_axis_label(str(raw_y), right_x, y_pos, color="#8A3FFC", bold=True, pixel_size=15)

    def _edge_path_and_samples(self, idx: int, row: int | None = None):
        x0, y0 = self.node_grid_pos(int(self.edge_table.src[idx]))
        x1, y1 = self.node_grid_pos(int(self.edge_table.dst[idx]))
        path = QtGui.QPainterPath()
        path.moveTo(x0, y0)

        if int(self.edge_table.option[idx]) == 2:
            ctrl_x = (x0 + x1) / 2.0
            ctrl_y = (y0 + y1) / 2.0 + 0.5 * abs(x1 - x0)
            path.quadTo(ctrl_x, ctrl_y, x1, y1)
            samples = []
            for t in np.linspace(0.0, 1.0, 17):
                qx = (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * ctrl_x + t * t * x1
                qy = (1 - t) * (1 - t) * y0 + 2 * (1 - t) * t * ctrl_y + t * t * y1
                samples.append((float(qx), float(qy)))
        else:
            path.lineTo(x1, y1)
            samples = [(x0, y0), (x1, y1)]

        return path, samples

    def _draw_edges(self):
        for idx in range(self.edge_table.num_edges):
            path, samples = self._edge_path_and_samples(idx)
            item = QtWidgets.QGraphicsPathItem(path)
            item.setZValue(3)
            self.scene.addItem(item)
            self.edge_items.append(item)
            self.edge_samples.append(samples)

            src = int(self.edge_table.src[idx])
            dst = int(self.edge_table.dst[idx])
            self.node_to_edge[(src, dst)] = idx
            self.node_to_edge[(dst, src)] = idx

    def _draw_nodes(self):
        for node in range(self.config.total_sats):
            p, y = self.node_grid_pos(node)
            item = QtWidgets.QGraphicsEllipseItem(
                float(p) - self.node_radius,
                float(y) - self.node_radius,
                2 * self.node_radius,
                2 * self.node_radius,
            )
            item.setPen(self.node_base_pen)
            item.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
            item.setZValue(20)
            self.scene.addItem(item)
            self.node_items.append(item)

    def _draw_pick_markers(self):
        colors = [QtGui.QColor("#1E88E5"), QtGui.QColor("#D81B60")]
        for color in colors:
            marker = QtWidgets.QGraphicsEllipseItem()
            pen = QtGui.QPen(color)
            pen.setWidthF(0.045)
            marker.setPen(pen)
            marker.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 0)))
            marker.setZValue(80)
            marker.setVisible(False)
            self.scene.addItem(marker)
            self.pick_marker_items.append(marker)

    def fit_scene(self):
        if hasattr(self, "view"):
            self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.zoom_factor = 1.0

    def reset_view(self):
        self.fit_scene()
        self.fit_on_next_resize = False

    def zoom_view(self, factor: float):
        if not hasattr(self, "view"):
            return
        factor = float(factor)
        if factor <= 0:
            return
        next_zoom = max(self.min_zoom_factor, min(self.max_zoom_factor, self.zoom_factor * factor))
        applied = next_zoom / self.zoom_factor
        if abs(applied - 1.0) < 1e-6:
            return
        self.zoom_factor = next_zoom
        self.view.scale(applied, applied)

    def on_option_changed(self):
        for option, cb in self.option_checks.items():
            self.visible_options[option] = cb.isChecked()
        self.update_step(self.current_row)

    def on_width_changed(self, value: int):
        self.edge_width = max(0.006, float(value) / 1000.0)
        self.update_step(self.current_row)

    def on_alpha_changed(self, value: int):
        self.edge_alpha = int(value)
        self.update_step(self.current_row)

    def update_step(self, row: int, *, sync_slider: bool = True):
        row = int(max(0, min(int(row), len(self.steps) - 1)))
        self.current_row = row
        slider_pos = self.visible_row_to_pos.get(row)
        if sync_slider and slider_pos is not None and self.slider.value() != slider_pos:
            self.slider.blockSignals(True)
            self.slider.setValue(slider_pos)
            self.slider.blockSignals(False)

        values = self.edge_values[row] if self.has_edge_values else None
        for idx, item in enumerate(self.edge_items):
            option = int(self.edge_table.option[idx])
            visible = bool(self.visible_options.get(option, False))
            item.setVisible(visible)
            if not visible:
                continue

            selected = idx == self.selected_edge_idx
            preview = idx == self.preview_edge_idx
            if selected:
                pen = QtGui.QPen(QtGui.QColor(10, 10, 10))
                pen.setWidthF(0.085)
                item.setZValue(50)
            elif preview:
                pen = QtGui.QPen(QtGui.QColor(35, 35, 35))
                pen.setWidthF(0.060)
                item.setZValue(40)
            elif not self.has_edge_values:
                color = QtGui.QColor(0, 0, 0)
                color.setAlpha(int(self.edge_alpha))
                pen = QtGui.QPen(color)
                pen.setWidthF(self.edge_width)
                item.setZValue(3)
            else:
                pen = QtGui.QPen(color_from_value(values[idx], self.value_min, self.value_max, self.edge_alpha))
                pen.setWidthF(self.edge_width)
                item.setZValue(3)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            item.setPen(pen)

        step = self.steps[row]
        pos = self.visible_row_to_pos.get(row, row)
        range_text = (
            f"range {self.visible_rows[0] + 1}-{self.visible_rows[-1] + 1}/{len(self.full_steps)}"
            if self.visible_rows
            else "range empty"
        )
        self.step_label.setText(
            f"step {step} | row {pos + 1}/{len(self.visible_rows)} | edges {self.edge_table.num_edges} | "
            f"{range_text}"
        )
        self.update_node_group_colors(row)
        self.update_pick_markers()
        self.update_selected_label()

    def update_node_group_colors(self, row: int):
        show = bool(self.group_data) and bool(self.show_groups_checkbox.isChecked())
        group_map = self.node_groups_for_row(row) if show else {}
        for node, item in enumerate(self.node_items):
            gids = group_map.get(int(node), [])
            if gids:
                gid = int(min(gids))
                color = self.group_color(gid)
                item.setBrush(QtGui.QBrush(color))
                item.setPen(QtGui.QPen(QtGui.QColor(35, 35, 35), 0.0))
            else:
                item.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
                item.setPen(self.node_base_pen)

    def group_color(self, gid: int) -> QtGui.QColor:
        if 0 <= int(gid) < len(self.config.group_colors):
            return QtGui.QColor(self.config.group_colors[int(gid)])
        return QtGui.QColor("#777777")

    def node_groups_for_row(self, row: int) -> dict[int, list[int]]:
        step = int(self.steps[int(row)])
        current = self.group_data.get(step, {}) if self.group_data else {}
        groups = current.get("groups", {}) if isinstance(current, dict) else {}
        node_to_groups: dict[int, list[int]] = {}
        for gid, nodes in groups.items():
            for node in nodes or []:
                node_i = int(node)
                if 0 <= node_i < int(self.config.total_sats):
                    node_to_groups.setdefault(node_i, []).append(int(gid))
        return node_to_groups

    def node_group_text(self, node: int, row: int | None = None) -> str:
        row = self.current_row if row is None else int(row)
        gids = self.node_groups_for_row(row).get(int(node), [])
        if not gids:
            return "none"
        parts = []
        for gid in sorted(set(int(x) for x in gids)):
            name = self.config.station_groups.get(gid, {}).get("name", f"Group {gid}")
            parts.append(f"{gid}:{name}")
        return ",".join(parts)

    def update_cursor_grid_label(self, x: float, y: float):
        p = int(round(float(x)))
        scene_y = int(round(float(y)))
        if not (0 <= p < int(self.config.P) and 0 <= scene_y < int(self.config.N)):
            self.coord_label.setText("Cursor grid: outside")
            return
        dist = math.hypot(float(x) - p, float(y) - scene_y)
        if dist > 0.55:
            axis_y = self.scene_y_to_axis_y(float(y))
            self.coord_label.setText(
                f"Cursor grid: display approx=({float(x):.2f}, {axis_y:.2f})"
            )
            return
        raw_node = self.grid_pos_to_node(p, scene_y)
        raw_p, raw_y = divmod(int(raw_node), int(self.config.N))
        self.coord_label.setText(
            f"Cursor grid: x={p}, y={raw_y}; node={raw_node}; "
            f"raw x={raw_p}, y={raw_y}; groups={self.node_group_text(raw_node)}"
        )

    def handle_scene_hover(self, pos: QtCore.QPointF):
        self.update_cursor_grid_label(float(pos.x()), float(pos.y()))
        edge_idx, dist = self.find_nearest_edge(float(pos.x()), float(pos.y()))
        if edge_idx is None:
            if self.preview_edge_idx is not None:
                self.preview_edge_idx = None
                self.hover_label.setText("Nearest edge: none")
                self.update_step(self.current_row)
            return

        if edge_idx != self.preview_edge_idx:
            self.preview_edge_idx = edge_idx
            self.hover_label.setText(self.describe_edge(edge_idx, prefix=f"Nearest edge ({dist:.3f})"))
            self.update_step(self.current_row)

    def handle_scene_click(self, pos: QtCore.QPointF):
        x = float(pos.x())
        y = float(pos.y())
        node = self.find_nearest_node(x, y)
        if node is not None:
            self.pick_node(node)
            return

        edge_idx, _ = self.find_nearest_edge(x, y)
        if edge_idx is not None:
            self.select_edge(edge_idx, via="direct")
            return

    def find_nearest_node(self, x: float, y: float) -> int | None:
        p = int(round(x))
        scene_y = int(round(y))
        if not (0 <= p < self.config.P and 0 <= scene_y < self.config.N):
            return None
        if math.hypot(x - p, y - scene_y) > self.node_hit_radius:
            return None
        return int(self.grid_pos_to_node(p, scene_y))

    def find_nearest_edge(self, x: float, y: float) -> tuple[int | None, float]:
        best_idx: int | None = None
        best_dist = math.inf
        for idx, samples in enumerate(self.edge_samples):
            option = int(self.edge_table.option[idx])
            if not self.visible_options.get(option, False):
                continue
            dist = distance_point_to_polyline(x, y, samples)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is None or best_dist > self.edge_hit_threshold:
            return None, best_dist
        return best_idx, best_dist

    def pick_node(self, node: int):
        if self.picked_nodes and self.picked_nodes[-1] == int(node):
            return
        if len(self.picked_nodes) >= 2:
            self.picked_nodes = []
        self.picked_nodes.append(int(node))
        self.update_pick_markers()

        if len(self.picked_nodes) == 1:
            p, y = divmod(int(node), self.config.N)
            self.pick_label.setText(
                f"Picked node A: node={node} ({p}, {y}); "
                f"groups={self.node_group_text(node)}."
            )
            return

        a, b = self.picked_nodes
        edge_idx = self.node_to_edge.get((int(a), int(b)))
        if edge_idx is None:
            pa, ya = divmod(int(a), self.config.N)
            pb, yb = divmod(int(b), self.config.N)
            self.pick_label.setText(
                f"No edge between node={a} ({pa}, {ya}) groups={self.node_group_text(a)} "
                f"and node={b} ({pb}, {yb}) groups={self.node_group_text(b)}."
            )
            self.selected_edge_idx = None
            self.update_step(self.current_row)
            return

        self.pick_label.setText(
            f"Picked node pair: raw={a} groups={self.node_group_text(a)} <-> "
            f"raw={b} groups={self.node_group_text(b)}."
        )
        self.select_edge(edge_idx, via="node pair")

    def clear_picked_nodes(self):
        self.picked_nodes = []
        self.selected_edge_idx = None
        self.pick_label.setText("Picked nodes: none")
        self.update_step(self.current_row)

    def update_pick_markers(self):
        radius = 0.28
        for idx, marker in enumerate(self.pick_marker_items):
            if idx >= len(self.picked_nodes):
                marker.setVisible(False)
                continue
            node = int(self.picked_nodes[idx])
            p, y = self.node_grid_pos(node)
            marker.setRect(float(p) - radius, float(y) - radius, 2 * radius, 2 * radius)
            marker.setVisible(True)

    def select_edge(self, edge_idx: int, *, via: str):
        self.selected_edge_idx = int(edge_idx)
        self.selected_label.setText(self.describe_edge(edge_idx, prefix=f"Selected edge ({via})"))
        self.update_step(self.current_row)

    def format_edge_value(self, edge_idx: int, row: int, value: float) -> str:
        return f"{self.edge_value_label}={float(value):.4f}"

    def edge_extra_description(self, edge_idx: int, row: int, value: float) -> str:
        return ""

    def describe_edge(self, edge_idx: int, *, prefix: str) -> str:
        edge_idx = int(edge_idx)
        src = int(self.edge_table.src[edge_idx])
        dst = int(self.edge_table.dst[edge_idx])
        value_segment = ""
        if self.has_edge_values:
            value = float(self.edge_values[self.current_row, edge_idx])
            value_text = self.format_edge_value(edge_idx, self.current_row, value)
            extra_text = self.edge_extra_description(edge_idx, self.current_row, value)
            extra_segment = f"; {extra_text}" if extra_text else ""
            value_segment = f"; {value_text}{extra_segment}"
        src_pos = self.node_grid_pos(src)
        dst_pos = self.node_grid_pos(dst)
        src_axis_y = int(round(self.scene_y_to_axis_y(src_pos[1])))
        dst_axis_y = int(round(self.scene_y_to_axis_y(dst_pos[1])))
        return (
            f"{prefix}: idx={edge_idx}; "
            f"raw {src} ({int(self.edge_table.src_plane[edge_idx])}, {int(self.edge_table.src_y[edge_idx])}) "
            f"grid=({int(src_pos[0])}, {src_axis_y}) -> "
            f"raw {dst} ({int(self.edge_table.dst_plane[edge_idx])}, {int(self.edge_table.dst_y[edge_idx])}) "
            f"grid=({int(dst_pos[0])}, {dst_axis_y}); "
            f"option={int(self.edge_table.option[edge_idx])}{value_segment}; "
            f"src_groups={self.node_group_text(src)}; dst_groups={self.node_group_text(dst)}; "
            f"step={self.steps[self.current_row]}"
        )

    def update_selected_label(self):
        if self.selected_edge_idx is None:
            self.selected_label.setText("Selected edge: none")
        else:
            self.selected_label.setText(self.describe_edge(self.selected_edge_idx, prefix="Selected edge"))

GridFakeDelayViewer = SatelliteTopology2DViewer
