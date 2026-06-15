from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt5 import QtCore, QtWidgets


@dataclass(frozen=True)
class Topology2DPanel:
    title: str
    viewer: QtWidgets.QWidget
    stretch: int = 1


class UnifiedControlTopology2DViewer(QtWidgets.QWidget):
    """Use one original 2D control panel to drive multiple 2D canvases.

    The first panel is the master. Its original control panel is kept intact:
    timeline, jump, range filtering, option switches, width/alpha, zoom buttons,
    group toggle, and edge picking text all come from the original viewer. The
    drawing area is replaced by several original QGraphicsView canvases.

    This class is display-only. It does not transform topology, group, metric,
    or delay data; callers pass already-prepared viewer objects.
    """

    def __init__(
        self,
        *,
        title: str,
        panels: list[Topology2DPanel],
        shared_value_scale: bool = True,
        orientation: QtCore.Qt.Orientation = QtCore.Qt.Horizontal,
    ):
        super().__init__()
        if not panels:
            raise ValueError("UnifiedControlTopology2DViewer needs at least one panel")
        self.panels = list(panels)
        self.master = self.panels[0].viewer
        self.shared_value_scale = bool(shared_value_scale)
        self._syncing = False
        self.setWindowTitle(str(title))

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.main_splitter.setHandleWidth(10)
        layout.addWidget(self.main_splitter, stretch=1)

        self.canvas_splitter = QtWidgets.QSplitter(orientation)
        self.canvas_splitter.setHandleWidth(8)
        self.main_splitter.addWidget(self.canvas_splitter)
        for panel in self.panels:
            self._add_canvas_panel(panel)
            self._wire_panel(panel)
        self.canvas_splitter.setSizes([max(1, int(panel.stretch)) for panel in self.panels])

        controls_scroll = getattr(self.master, "controls_scroll", None)
        if controls_scroll is None:
            raise ValueError("master viewer does not expose controls_scroll")
        self.main_splitter.addWidget(controls_scroll)
        self.main_splitter.setStretchFactor(0, 8)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setSizes([720, 220])

        self._rewire_master_common_buttons()
        self._sync_from_master(int(getattr(self.master, "current_row", 0)), refresh_master=True)

    def _add_canvas_panel(self, panel: Topology2DPanel) -> None:
        frame = QtWidgets.QFrame()
        frame_layout = QtWidgets.QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(4)
        title = QtWidgets.QLabel(str(panel.title))
        title.setStyleSheet("font-size: 17px; font-weight: 700; color: #1f2933;")
        frame_layout.addWidget(title)

        view = getattr(panel.viewer, "view", None)
        if view is None:
            raise ValueError(f"panel {panel.title!r} viewer does not expose view")
        view.setMinimumHeight(220)
        frame_layout.addWidget(view, stretch=1)
        self.canvas_splitter.addWidget(frame)

    def _wire_panel(self, panel: Topology2DPanel) -> None:
        viewer = panel.viewer
        row_changed = getattr(viewer, "row_changed", None)
        if row_changed is not None:
            row_changed.connect(lambda row, source=viewer: self._on_row_changed(source, int(row)))
        visible_rows_changed = getattr(viewer, "visible_rows_changed", None)
        if visible_rows_changed is not None:
            visible_rows_changed.connect(
                lambda rows, target_row, source=viewer: self._on_visible_rows_changed(
                    source,
                    [int(x) for x in rows],
                    int(target_row),
                )
            )

    @staticmethod
    def _disconnect_button(button: QtWidgets.QAbstractButton) -> None:
        try:
            button.clicked.disconnect()
        except TypeError:
            pass

    def _rewire_master_common_buttons(self) -> None:
        zoom_out = getattr(self.master, "zoom_out_btn", None)
        zoom_in = getattr(self.master, "zoom_in_btn", None)
        zoom_reset = getattr(self.master, "zoom_reset_btn", None)
        clear_btn = getattr(self.master, "clear_btn", None)
        if zoom_out is not None:
            self._disconnect_button(zoom_out)
            zoom_out.clicked.connect(lambda: self.zoom_all(1 / 1.25))
        if zoom_in is not None:
            self._disconnect_button(zoom_in)
            zoom_in.clicked.connect(lambda: self.zoom_all(1.25))
        if zoom_reset is not None:
            self._disconnect_button(zoom_reset)
            zoom_reset.clicked.connect(self.reset_all_views)
        if clear_btn is not None:
            self._disconnect_button(clear_btn)
            clear_btn.clicked.connect(self.clear_all_picks)

    def _viewer_values_for_row(self, viewer: QtWidgets.QWidget, row: int) -> np.ndarray | None:
        if not bool(getattr(viewer, "has_edge_values", False)):
            return None
        if hasattr(viewer, "values_for_row"):
            values, _status = viewer.values_for_row(int(row))
            return np.asarray(values, dtype=np.float32)
        edge_values = getattr(viewer, "edge_values", None)
        if edge_values is None:
            return None
        values = np.asarray(edge_values)
        if values.ndim != 2 or values.shape[0] <= 0:
            return None
        value_row = 0 if values.shape[0] == 1 else int(row)
        value_row = max(0, min(value_row, values.shape[0] - 1))
        return np.asarray(values[value_row], dtype=np.float32)

    def _apply_shared_value_scale(self, row: int) -> None:
        if not self.shared_value_scale:
            return
        row_max = 1.0
        for panel in self.panels:
            values = self._viewer_values_for_row(panel.viewer, int(row))
            if values is None or values.size == 0:
                continue
            row_max = max(row_max, float(np.nanmax(values)))
        for panel in self.panels:
            if bool(getattr(panel.viewer, "has_edge_values", False)):
                setattr(panel.viewer, "value_max", float(row_max))

    def _set_suppressed(self, viewer: QtWidgets.QWidget, value: bool) -> None:
        if hasattr(viewer, "_suppress_multi_signals"):
            setattr(viewer, "_suppress_multi_signals", bool(value))

    def _copy_master_visual_state(self, target: QtWidgets.QWidget) -> None:
        if target is self.master:
            return
        for name in ("edge_width", "edge_alpha"):
            if hasattr(self.master, name) and hasattr(target, name):
                setattr(target, name, getattr(self.master, name))

        if hasattr(self.master, "visible_options") and hasattr(target, "visible_options"):
            master_options = getattr(self.master, "visible_options")
            target_options = getattr(target, "visible_options")
            for option in list(target_options):
                target_options[option] = bool(master_options.get(option, True))

        master_group_cb = getattr(self.master, "show_groups_checkbox", None)
        target_group_cb = getattr(target, "show_groups_checkbox", None)
        if master_group_cb is not None and target_group_cb is not None:
            target_group_cb.blockSignals(True)
            target_group_cb.setChecked(bool(master_group_cb.isChecked()))
            target_group_cb.blockSignals(False)

        master_option_checks = getattr(self.master, "option_checks", {})
        target_option_checks = getattr(target, "option_checks", {})
        for option, cb in target_option_checks.items():
            if option in master_option_checks:
                cb.blockSignals(True)
                cb.setChecked(bool(master_option_checks[option].isChecked()))
                cb.blockSignals(False)

    def _sync_from_master(self, row: int, *, refresh_master: bool) -> None:
        row = int(row)
        self._apply_shared_value_scale(row)
        for panel in self.panels:
            viewer = panel.viewer
            if viewer is self.master and not bool(refresh_master):
                continue
            self._copy_master_visual_state(viewer)
            update_step = getattr(viewer, "update_step", None)
            if update_step is None:
                continue
            self._set_suppressed(viewer, True)
            try:
                update_step(row)
            finally:
                self._set_suppressed(viewer, False)

    def _on_row_changed(self, source: QtWidgets.QWidget, row: int) -> None:
        if source is not self.master or self._syncing:
            return
        self._syncing = True
        try:
            self._sync_from_master(int(row), refresh_master=True)
        finally:
            self._syncing = False

    def _on_visible_rows_changed(self, source: QtWidgets.QWidget, rows: list[int], target_row: int) -> None:
        if source is not self.master or self._syncing:
            return
        self._syncing = True
        try:
            for panel in self.panels:
                viewer = panel.viewer
                if viewer is self.master:
                    continue
                set_rows = getattr(viewer, "_set_visible_rows", None)
                if set_rows is None:
                    continue
                self._set_suppressed(viewer, True)
                try:
                    set_rows(list(rows), target_row=int(target_row))
                finally:
                    self._set_suppressed(viewer, False)
            self._sync_from_master(int(target_row), refresh_master=False)
        finally:
            self._syncing = False

    def zoom_all(self, factor: float) -> None:
        for panel in self.panels:
            zoom_view = getattr(panel.viewer, "zoom_view", None)
            if zoom_view is not None:
                zoom_view(float(factor))

    def reset_all_views(self) -> None:
        for panel in self.panels:
            reset_view = getattr(panel.viewer, "reset_view", None)
            if reset_view is not None:
                reset_view()

    def clear_all_picks(self) -> None:
        for panel in self.panels:
            clear = getattr(panel.viewer, "clear_picked_nodes", None)
            if clear is not None:
                clear()
