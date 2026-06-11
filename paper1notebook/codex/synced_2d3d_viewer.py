from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


GENERIC_ROOT = Path(__file__).resolve().parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

import pyvista as pv
import vtk
from pyvistaqt import QtInteractor

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable
from src.link_delay.module.position_cache import PositionCacheStore
from src.satellite_topology_viewer.module.edge_delay_data import EdgeDelayViewerData
from src.satellite_topology_viewer.module.edge_delay_viewer import EdgeDelayTopologyViewer


EARTH_R_KM = 6371.0


@dataclass(frozen=True)
class PositionSeries:
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
    cache_dir: str | Path,
    start: int,
    end: int,
    stride: int = 1,
) -> PositionSeries:
    store = PositionCacheStore(cache_dir)
    rows = store.rows_for_interval(int(start), int(end), int(stride))
    steps = [int(x) for x in np.asarray(store.times_s[rows], dtype=np.int64)]
    return PositionSeries(
        cache_dir=Path(cache_dir),
        positions_km=store.positions_km,
        cache_rows=np.asarray(rows, dtype=np.int64),
        steps=steps,
        sat_ids=store.sat_ids,
        meta=store.meta,
    )


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    color = QtGui.QColor(str(value))
    if not color.isValid():
        color = QtGui.QColor("#3b82f6")
    return color.red(), color.green(), color.blue()


def _segments_to_polydata(points: np.ndarray, edges: Iterable[tuple[int, int]]):
    edges = list(edges)
    if not edges:
        return None

    seg_pts = np.empty((2 * len(edges), 3), dtype=np.float32)
    line_cells = np.empty((len(edges), 3), dtype=np.int64)
    for idx, (src, dst) in enumerate(edges):
        seg_pts[2 * idx] = points[int(src)]
        seg_pts[2 * idx + 1] = points[int(dst)]
        line_cells[idx] = [2, 2 * idx, 2 * idx + 1]

    mesh = pv.PolyData(seg_pts)
    mesh.lines = line_cells.ravel()
    return mesh


class SatelliteGlobe3DWidget(QtWidgets.QWidget):
    """PyVista-based 3D satellite widget driven by an external timeline."""

    def __init__(
        self,
        config: ViewerConfig,
        *,
        position_series: PositionSeries,
        edge_table: EdgeTable | None = None,
        edge_values: np.ndarray | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
        group_data: dict[int, dict] | None = None,
        show_groups: bool = True,
        show_links: bool = True,
        link_stride: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self.position_series = position_series
        self.edge_table = edge_table
        self.edge_values = None if edge_values is None else np.asarray(edge_values, dtype=np.float32)
        self.value_min = None if value_min is None else float(value_min)
        self.value_max = None if value_max is None else float(value_max)
        self.group_data = group_data or {}
        self.show_groups = bool(show_groups and self.group_data)
        self.show_links = bool(show_links and edge_table is not None)
        self.link_stride = max(1, int(link_stride))
        self.current_row = 0
        self.selected_sat_idxs: set[int] = set()
        self.selected_edge_idx: int | None = None
        self.path_by_time: dict[int, list[int]] = {}

        if self.position_series.num_sats != int(config.total_sats):
            raise ValueError(
                f"Position cache has {self.position_series.num_sats} satellites, "
                f"but config expects {config.total_sats}"
            )
        if self.edge_values is not None and self.edge_values.shape[0] != self.position_series.num_steps:
            raise ValueError(
                f"edge_values rows {self.edge_values.shape[0]} != position steps {self.position_series.num_steps}"
            )
        if self.edge_values is not None and edge_table is not None:
            if self.edge_values.shape[1] != edge_table.num_edges:
                raise ValueError(
                    f"edge_values cols {self.edge_values.shape[1]} != edge count {edge_table.num_edges}"
                )

        self._build_ui()
        self._build_scene()
        self.set_row(0, render=True)

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #ffffff; color: #475569; font-size: 13px; }
            QLabel#statusLabel { color: #334155; font-weight: 600; }
            """
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        status_row = QtWidgets.QHBoxLayout()
        status_row.setContentsMargins(8, 6, 8, 0)
        self.status_label = QtWidgets.QLabel("3D ready")
        self.status_label.setObjectName("statusLabel")
        self.pick_label = QtWidgets.QLabel("selected=none")
        self.pick_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.pick_label)

        self.plotter = QtInteractor(self, auto_update=False, multi_samples=0)
        layout.addLayout(status_row)
        layout.addWidget(self.plotter.interactor, 1)

    def _build_scene(self) -> None:
        self.plotter.set_background("#ffffff")
        earth = pv.Sphere(radius=EARTH_R_KM, theta_resolution=160, phi_resolution=160)
        self.plotter.add_mesh(
            earth,
            color="#eef2f5",
            smooth_shading=True,
            ambient=0.35,
            diffuse=0.65,
            specular=0.0,
            pickable=False,
        )
        self._add_earth_outline()

        initial_points = self.position_series.points_for_row(0)
        self.sat_centers = pv.PolyData(initial_points)
        self.sat_centers.point_data["colors"] = self._point_colors_for_row(0)
        self.plotter.add_mesh(
            self.sat_centers,
            name="sat_points",
            render_points_as_spheres=True,
            point_size=8,
            scalars="colors",
            rgb=True,
            ambient=0.35,
            opacity=1.0,
            pickable=True,
        )
        self.plotter.enable_point_picking(
            callback=self._on_pick_satellite,
            left_clicking=True,
            show_point=False,
            picker="point",
            tolerance=0.006,
            pickable_window=False,
            clear_on_no_selection=False,
            use_picker=False,
        )
        self.plotter.camera_position = [
            (18000, -15000, 11000),
            (0, 0, 0),
            (0, 0, 1),
        ]
        try:
            if hasattr(self.plotter, "iren") and self.plotter.iren is not None:
                self.plotter.iren.add_observer("EndInteractionEvent", self._on_camera_interaction_end)
        except Exception:
            pass

    def _add_earth_outline(self) -> None:
        earth_src = vtk.vtkEarthSource()
        earth_src.OutlineOn()
        if hasattr(earth_src, "SetOnRatio"):
            earth_src.SetOnRatio(1)
        if hasattr(earth_src, "SetRadius"):
            earth_src.SetRadius(EARTH_R_KM * 1.004)
        earth_src.Update()
        outline = pv.wrap(earth_src.GetOutput())
        if not hasattr(earth_src, "SetRadius"):
            outline.points = outline.points * (EARTH_R_KM * 1.004)
        self.plotter.add_mesh(
            outline.copy(),
            color="#ffffff",
            line_width=3.0,
            opacity=0.95,
            lighting=False,
            render_lines_as_tubes=True,
            pickable=False,
        )
        self.plotter.add_mesh(
            outline,
            color="#cbd5e1",
            line_width=1.2,
            opacity=1.0,
            lighting=False,
            render_lines_as_tubes=True,
            pickable=False,
        )

    def _point_colors_for_row(self, row: int) -> np.ndarray:
        colors = np.full((self.config.total_sats, 3), _hex_to_rgb("#3b82f6"), dtype=np.uint8)
        if not self.show_groups:
            return colors

        step = int(self.position_series.steps[int(row)])
        data = self.group_data.get(step, {})
        groups = data.get("groups", {}) if isinstance(data, dict) else {}
        for gid, nodes in groups.items():
            rgb = _hex_to_rgb(self.config.group_colors[int(gid)] if int(gid) < len(self.config.group_colors) else "#64748b")
            for node in nodes or []:
                node = int(node)
                if 0 <= node < self.config.total_sats:
                    colors[node] = rgb
        return colors

    def _status_text(self) -> str:
        step = int(self.position_series.steps[self.current_row])
        link_text = "links=off"
        if self.show_links and self.edge_table is not None:
            shown = len(self._link_edge_indices())
            link_text = f"links={shown}/{self.edge_table.num_edges}"
        return f"3D step={step}s row={self.current_row + 1}/{self.position_series.num_steps} sats={self.config.total_sats} {link_text}"

    def _link_edge_indices(self) -> np.ndarray:
        if not self.show_links or self.edge_table is None:
            return np.asarray([], dtype=np.int64)
        return np.arange(0, self.edge_table.num_edges, self.link_stride, dtype=np.int64)

    def _refresh_link_actor(self, points: np.ndarray, row: int) -> None:
        for actor_name in ("link_actor",):
            try:
                self.plotter.remove_actor(actor_name, render=False)
            except Exception:
                pass
        if not self.show_links or self.edge_table is None:
            return

        edge_indices = self._link_edge_indices()
        edges = [
            (int(self.edge_table.src[idx]), int(self.edge_table.dst[idx]))
            for idx in edge_indices
        ]
        mesh = _segments_to_polydata(points, edges)
        if mesh is None:
            return

        if self.edge_values is not None:
            values = np.asarray(self.edge_values[int(row), edge_indices], dtype=np.float32)
            mesh.cell_data["edge_value"] = values
            self.plotter.add_mesh(
                mesh,
                name="link_actor",
                scalars="edge_value",
                cmap="turbo",
                clim=(self.value_min, self.value_max) if self.value_min is not None and self.value_max is not None else None,
                line_width=1.0,
                opacity=0.36,
                lighting=False,
                render_lines_as_tubes=False,
                show_scalar_bar=False,
                pickable=False,
                reset_camera=False,
                render=False,
            )
        else:
            self.plotter.add_mesh(
                mesh,
                name="link_actor",
                color="#94a3b8",
                line_width=1.0,
                opacity=0.32,
                lighting=False,
                render_lines_as_tubes=False,
                pickable=False,
                reset_camera=False,
                render=False,
            )

    def _remove_actor_quietly(self, name: str) -> None:
        try:
            self.plotter.remove_actor(name, render=False)
        except Exception:
            pass

    def _refresh_selected_edge(self, points: np.ndarray) -> None:
        self._remove_actor_quietly("selected_edge_actor")
        if self.selected_edge_idx is None or self.edge_table is None:
            return
        idx = int(self.selected_edge_idx)
        if not (0 <= idx < self.edge_table.num_edges):
            return
        mesh = _segments_to_polydata(
            points,
            [(int(self.edge_table.src[idx]), int(self.edge_table.dst[idx]))],
        )
        if mesh is None:
            return
        self.plotter.add_mesh(
            mesh,
            name="selected_edge_actor",
            color="#f97316",
            line_width=6.0,
            opacity=1.0,
            lighting=False,
            render_lines_as_tubes=True,
            pickable=False,
            reset_camera=False,
            render=False,
        )

    def set_selected_edge(self, edge_idx: int | None, *, render: bool = True) -> None:
        self.selected_edge_idx = None if edge_idx is None else int(edge_idx)
        points = np.asarray(self.sat_centers.points, dtype=np.float32)
        self._refresh_selected_edge(points)
        if render:
            self.plotter.render()

    def set_paths(self, path_by_time: dict[int, list[int]] | None, *, render: bool = True) -> None:
        cleaned: dict[int, list[int]] = {}
        for key, value in (path_by_time or {}).items():
            if value is None:
                continue
            nodes = [int(x) for x in value]
            if len(nodes) >= 2:
                cleaned[int(key)] = nodes
        self.path_by_time = cleaned
        self._refresh_path_actor(np.asarray(self.sat_centers.points, dtype=np.float32))
        if render:
            self.plotter.render()

    def _refresh_path_actor(self, points: np.ndarray) -> None:
        for name in ("path_actor", "path_node_actor"):
            self._remove_actor_quietly(name)
        step = int(self.position_series.steps[self.current_row])
        nodes = self.path_by_time.get(step, self.path_by_time.get(self.current_row))
        if not nodes or len(nodes) < 2:
            return
        nodes = [int(x) for x in nodes if 0 <= int(x) < self.config.total_sats]
        if len(nodes) < 2:
            return
        mesh = _segments_to_polydata(points, zip(nodes[:-1], nodes[1:]))
        if mesh is not None:
            self.plotter.add_mesh(
                mesh,
                name="path_actor",
                color="#22c55e",
                line_width=7.0,
                opacity=1.0,
                lighting=False,
                render_lines_as_tubes=True,
                pickable=False,
                reset_camera=False,
                render=False,
            )
        node_mesh = pv.PolyData(points[nodes])
        self.plotter.add_mesh(
            node_mesh,
            name="path_node_actor",
            render_points_as_spheres=True,
            point_size=15,
            color="#fde047",
            ambient=0.35,
            pickable=False,
            reset_camera=False,
            render=False,
        )

    def set_row(self, row: int, *, render: bool = True) -> None:
        row = int(max(0, min(int(row), self.position_series.num_steps - 1)))
        self.current_row = row
        points = self.position_series.points_for_row(row)
        self.sat_centers.points = points
        self.sat_centers.point_data["colors"] = self._point_colors_for_row(row)
        self.sat_centers.Modified()
        self._refresh_link_actor(points, row)
        self._refresh_selected_edge(points)
        self._refresh_path_actor(points)
        self._refresh_selected_sat_labels()
        self.status_label.setText(self._status_text())
        if render:
            self.plotter.render()

    def _sat_display_name(self, sat_idx: int) -> str:
        if 0 <= int(sat_idx) < len(self.position_series.sat_ids):
            return f"SAT {self.position_series.sat_ids[int(sat_idx)]}"
        return f"node {int(sat_idx)}"

    def _on_pick_satellite(self, picked_point) -> None:
        if picked_point is None:
            return
        sat_idx = int(self.sat_centers.find_closest_point(picked_point))
        if sat_idx in self.selected_sat_idxs:
            self.selected_sat_idxs.remove(sat_idx)
        else:
            self.selected_sat_idxs.add(sat_idx)
        self._refresh_selected_sat_labels()
        if self.selected_sat_idxs:
            last = sorted(self.selected_sat_idxs)[-1]
            self.pick_label.setText(f"selected={self._sat_display_name(last)} labels={len(self.selected_sat_idxs)}")
        else:
            self.pick_label.setText("selected=none")
        self.plotter.render()

    def _is_sat_visible_from_camera(self, point: np.ndarray) -> bool:
        camera_pos = np.asarray(self.plotter.camera_position[0], dtype=float)
        target = np.asarray(point, dtype=float)
        ray = target - camera_pos
        ray_len2 = float(np.dot(ray, ray))
        if ray_len2 < 1e-12:
            return True
        t = np.clip(-float(np.dot(camera_pos, ray)) / ray_len2, 0.0, 1.0)
        closest = camera_pos + t * ray
        return np.linalg.norm(closest) > EARTH_R_KM * 1.001

    def _refresh_selected_sat_labels(self) -> None:
        self._remove_actor_quietly("sat_pick_labels")
        if not self.selected_sat_idxs:
            return

        camera_pos = np.asarray(self.plotter.camera_position[0], dtype=float)
        points = np.asarray(self.sat_centers.points, dtype=float)
        label_points = []
        label_texts = []
        for sat_idx in sorted(self.selected_sat_idxs):
            center = points[int(sat_idx)]
            if not self._is_sat_visible_from_camera(center):
                continue
            cam_vec = camera_pos - center
            cam_norm = np.linalg.norm(cam_vec)
            cam_dir = np.array([0.0, 0.0, 1.0]) if cam_norm < 1e-9 else cam_vec / cam_norm
            label_points.append(center + cam_dir * 220.0)
            label_texts.append(self._sat_display_name(int(sat_idx)))
        if not label_points:
            return
        self.plotter.add_point_labels(
            np.asarray(label_points, dtype=float),
            label_texts,
            name="sat_pick_labels",
            always_visible=True,
            show_points=False,
            text_color="#1d4ed8",
            shape_color="#eff6ff",
            shape_opacity=0.95,
            font_size=13,
            pickable=False,
            reset_camera=False,
            render=False,
        )

    def _on_camera_interaction_end(self, *_args) -> None:
        self._refresh_selected_sat_labels()
        self.plotter.render()

    def shutdown(self) -> None:
        for name in ("sat_pick_labels", "selected_edge_actor", "path_actor", "path_node_actor", "link_actor"):
            self._remove_actor_quietly(name)
        try:
            self.plotter.close()
        except Exception:
            pass
        try:
            self.plotter.deleteLater()
        except Exception:
            pass


class Synced2D3DTopologyWindow(QtWidgets.QWidget):
    """A shared-timeline container for the new 2D topology viewer and a 3D globe."""

    def __init__(
        self,
        *,
        config: ViewerConfig,
        delay_data: EdgeDelayViewerData,
        position_series: PositionSeries,
        group_data: dict[int, dict] | None = None,
        show_groups: bool = True,
        show_3d_links: bool = True,
        link_stride: int = 1,
        backend: str = "pyvista",
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
        self.backend = "pyvista"

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

    def _build_ui(self, *, show_groups: bool, show_3d_links: bool, link_stride: int) -> None:
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


def run_synced_2d3d_viewer(
    *,
    config: ViewerConfig,
    delay_data: EdgeDelayViewerData,
    position_series: PositionSeries,
    group_data: dict[int, dict] | None = None,
    width: int = 1600,
    height: int = 900,
    show_groups: bool = True,
    show_3d_links: bool = True,
    link_stride: int = 1,
    timer_interval_ms: int = 180,
    check_only: bool = False,
    offscreen: bool = False,
    screenshot: str | Path | None = None,
) -> int:
    if check_only:
        print(
            f"[synced-2d3d] check OK | steps={len(delay_data.steps)} "
            f"edges={delay_data.edge_table.num_edges} position_cache={position_series.cache_dir} "
            f"backend=pyvista",
            flush=True,
        )
        return 0

    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    window = Synced2D3DTopologyWindow(
        config=config,
        delay_data=delay_data,
        position_series=position_series,
        group_data=group_data,
        show_groups=show_groups,
        show_3d_links=show_3d_links,
        link_stride=link_stride,
        timer_interval_ms=timer_interval_ms,
    )
    window.resize(int(width), int(height))
    window.show()

    def settle_initial_layout():
        try:
            window.viewer2d.fit_scene()
        except Exception:
            pass
        try:
            window.viewer3d.set_row(window.current_row)
        except Exception:
            pass

    QtCore.QTimer.singleShot(120, settle_initial_layout)

    if screenshot is not None:
        screenshot_path = Path(screenshot)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        def save_screenshot_and_quit():
            app.processEvents()
            ok = window.grab().save(str(screenshot_path))
            print(f"[synced-2d3d] screenshot={screenshot_path} ok={ok}", flush=True)
            app.quit()

        QtCore.QTimer.singleShot(2200, save_screenshot_and_quit)

    return int(app.exec_())
