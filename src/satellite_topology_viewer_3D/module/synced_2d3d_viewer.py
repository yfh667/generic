from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from src.config.viewer_config import ViewerConfig
from src.satellite_topology_viewer.module.edge_delay_data import EdgeDelayViewerData
from src.satellite_topology_viewer.module.edge_delay_viewer import EdgeDelayTopologyViewer

from .base_globe_viewer import SatelliteGlobe3DWidget
from .position_data import PositionSeries


class Synced2D3DTopologyWindow(QtWidgets.QWidget):
    """A shared-timeline container for the 2D topology viewer and a 3D globe."""

    def __init__(
        self,
        *,
        config: ViewerConfig,
        delay_data: EdgeDelayViewerData,
        position_series: PositionSeries,
        group_data: dict[int, dict] | None = None,
        show_groups: bool = True,
        show_3d_links: bool = True,
        show_3d_orbits: bool = True,
        link_stride: int = 1,
        timer_interval_ms: int = 180,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self.delay_data = delay_data
        self.position_series = position_series
        self.group_data = group_data or {}
        self.current_row = 0
        self.playing = False
        self.timer_interval_ms = max(1, int(timer_interval_ms))
        self._syncing = False

        if list(delay_data.steps) != list(position_series.steps):
            raise ValueError(
                "2D delay steps and 3D position steps must match exactly. "
                f"2D={delay_data.steps[:3]}..{delay_data.steps[-3:]}, "
                f"3D={position_series.steps[:3]}..{position_series.steps[-3:]}"
            )

        self.setWindowTitle(
            f"{config.name} synced 2D + 3D topology {delay_data.steps[0]}..{delay_data.steps[-1]}s"
        )
        self._build_ui(
            show_groups=show_groups,
            show_3d_links=show_3d_links,
            show_3d_orbits=show_3d_orbits,
            link_stride=link_stride,
        )
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.timer_interval_ms)
        self.set_playing(False)
        self.set_row(0)

    @property
    def steps(self) -> list[int]:
        return list(self.delay_data.steps)

    def _build_ui(self, *, show_groups: bool, show_3d_links: bool, show_3d_orbits: bool, link_stride: int) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #ffffff; color: #334155; font-size: 13px; }
            QPushButton {
                background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 5px;
                padding: 7px 12px; min-height: 28px;
            }
            QPushButton:hover { background: #eef2f7; }
            QLineEdit { border: 1px solid #cbd5e1; border-radius: 5px; padding: 6px 8px; }
            QLabel#timeLabel { font-weight: 700; color: #0f172a; }
            QSlider::groove:horizontal { height: 8px; border-radius: 4px; background: #dbe4ec; }
            QSlider::sub-page:horizontal { background: #2563eb; border-radius: 4px; }
            QSlider::handle:horizontal {
                background: #ffffff; border: 2px solid #2563eb; width: 18px;
                margin: -7px 0; border-radius: 9px;
            }
            """
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.viewer2d = EdgeDelayTopologyViewer(
            self.config,
            steps=self.delay_data.steps,
            edge_table=self.delay_data.edge_table,
            delay_ms=self.delay_data.delay_ms,
            delay_min_ms=self.delay_data.delay_min_ms,
            delay_max_ms=self.delay_data.delay_max_ms,
            window_title=f"{self.config.name} 2D topology",
            group_data=self.group_data,
            show_groups=show_groups,
        )
        self.viewer3d = SatelliteGlobe3DWidget(
            self.config,
            position_series=self.position_series,
            edge_table=self.delay_data.edge_table,
            edge_values=self.delay_data.delay_ms,
            value_min=self.delay_data.delay_min_ms,
            value_max=self.delay_data.delay_max_ms,
            group_data=self.group_data,
            show_groups=show_groups,
            show_links=show_3d_links,
            show_orbits=show_3d_orbits,
            link_stride=link_stride,
        )
        self.viewer2d.setMinimumWidth(620)
        self.viewer3d.setMinimumWidth(620)

        self._patch_2d_edge_selection_sync()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.viewer2d)
        splitter.addWidget(self.viewer3d)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([850, 850])
        root.addWidget(splitter, 1)

        timeline = QtWidgets.QFrame()
        timeline.setObjectName("timelinePanel")
        timeline_layout = QtWidgets.QVBoxLayout(timeline)
        timeline_layout.setContentsMargins(10, 8, 10, 8)
        timeline_layout.setSpacing(8)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(lambda: self.set_playing(not self.playing))
        self.prev_btn = QtWidgets.QPushButton("<")
        self.prev_btn.clicked.connect(lambda: self.set_row(self.current_row - 1))
        self.next_btn = QtWidgets.QPushButton(">")
        self.next_btn.clicked.connect(lambda: self.set_row(self.current_row + 1))
        self.time_label = QtWidgets.QLabel("")
        self.time_label.setObjectName("timeLabel")
        self.time_label.setMinimumWidth(260)
        self.jump_input = QtWidgets.QLineEdit()
        self.jump_input.setPlaceholderText(f"jump to time second, {self.steps[0]}-{self.steps[-1]}")
        self.jump_input.returnPressed.connect(self.jump_to_input_time)
        self.jump_btn = QtWidgets.QPushButton("Jump")
        self.jump_btn.clicked.connect(self.jump_to_input_time)

        controls.addWidget(self.play_btn)
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.next_btn)
        controls.addWidget(self.time_label)
        controls.addWidget(self.jump_input, 1)
        controls.addWidget(self.jump_btn)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, len(self.steps) - 1)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(max(1, len(self.steps) // 10))
        self.slider.setPageStep(max(1, len(self.steps) // 50))
        self.slider.valueChanged.connect(lambda value: self.set_row(int(value), source="main_slider"))

        axis = QtWidgets.QHBoxLayout()
        self.axis_min = QtWidgets.QLabel(f"{self.steps[0]}s")
        self.axis_mid = QtWidgets.QLabel("")
        self.axis_mid.setAlignment(QtCore.Qt.AlignCenter)
        self.axis_max = QtWidgets.QLabel(f"{self.steps[-1]}s")
        self.axis_max.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        axis.addWidget(self.axis_min)
        axis.addWidget(self.axis_mid, 1)
        axis.addWidget(self.axis_max)

        timeline_layout.addLayout(controls)
        timeline_layout.addWidget(self.slider)
        timeline_layout.addLayout(axis)
        root.addWidget(timeline)

        self.viewer2d.slider.valueChanged.connect(self._sync_from_2d_slider)

    def _patch_2d_edge_selection_sync(self) -> None:
        original_select_edge = self.viewer2d.select_edge
        original_clear_picked_nodes = self.viewer2d.clear_picked_nodes

        def select_edge_and_sync(edge_idx: int, *args, **kwargs):
            result = original_select_edge(edge_idx, *args, **kwargs)
            self.viewer3d.set_selected_edge(int(edge_idx))
            return result

        def clear_and_sync(*args, **kwargs):
            result = original_clear_picked_nodes(*args, **kwargs)
            self.viewer3d.set_selected_edge(None)
            return result

        self.viewer2d.select_edge = select_edge_and_sync
        self.viewer2d.clear_picked_nodes = clear_and_sync

    def _sync_from_2d_slider(self, _value: int) -> None:
        if self._syncing:
            return
        QtCore.QTimer.singleShot(0, lambda: self.set_row(self.viewer2d.current_row, source="viewer2d"))

    def set_playing(self, playing: bool) -> bool:
        self.playing = bool(playing)
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start(self.timer_interval_ms)
        else:
            self.timer.stop()
        return self.playing

    def _tick(self) -> None:
        if not self.playing:
            return
        self.set_row((self.current_row + 1) % len(self.steps), source="timer")

    def row_for_time(self, target_step: int | float) -> int:
        wanted = int(round(float(target_step)))
        pos = int(np.searchsorted(np.asarray(self.steps, dtype=np.int64), wanted))
        if pos <= 0:
            return 0
        if pos >= len(self.steps):
            return len(self.steps) - 1
        before = self.steps[pos - 1]
        after = self.steps[pos]
        return pos - 1 if abs(wanted - before) <= abs(after - wanted) else pos

    def jump_to_input_time(self) -> None:
        raw = self.jump_input.text().strip()
        if not raw:
            return
        try:
            row = self.row_for_time(float(raw))
        except ValueError:
            self.jump_input.selectAll()
            return
        self.set_playing(False)
        self.set_row(row)

    def set_row(self, row: int, *, source: str | None = None) -> None:
        if self._syncing:
            return
        row = int(max(0, min(int(row), len(self.steps) - 1)))
        self.current_row = row
        self._syncing = True
        try:
            self.slider.blockSignals(True)
            self.slider.setValue(row)
            self.slider.blockSignals(False)
            self.viewer2d.update_step(row)
            self.viewer3d.set_row(row)
            self.viewer3d.set_selected_edge(self.viewer2d.selected_edge_idx, render=False)
        finally:
            self._syncing = False
        step = int(self.steps[row])
        self.time_label.setText(f"step={step}s  row={row + 1}/{len(self.steps)}")
        self.axis_mid.setText(f"2D/3D synchronized | 3D=pyvista | source={source or 'api'}")

    def closeEvent(self, event) -> None:
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            self.viewer3d.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
