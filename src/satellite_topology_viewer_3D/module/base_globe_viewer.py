from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk
from PyQt5 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable

from .geometry import EARTH_R_KM, build_orbit_edges, hex_to_rgb, segments_to_polydata
from .position_data import PositionSeries


class SatelliteGlobe3DWidget(QtWidgets.QWidget):
    """PyVista-based 3D satellite widget driven by external data."""

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
        show_orbits: bool = True,
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
        self.show_orbits = bool(show_orbits)
        self.link_stride = max(1, int(link_stride))
        self.orbit_edges = build_orbit_edges(config)
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
        self.status_label.setToolTip(str(self.position_series.cache_dir))
        self.pick_label = QtWidgets.QLabel("selected=none")
        self.pick_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.links_button = QtWidgets.QPushButton()
        self.links_button.setCheckable(True)
        self.links_button.setChecked(self.show_links)
        self.links_button.clicked.connect(lambda checked: self.set_links_visible(bool(checked)))

        self.orbits_button = QtWidgets.QPushButton()
        self.orbits_button.setCheckable(True)
        self.orbits_button.setChecked(self.show_orbits)
        self.orbits_button.clicked.connect(lambda checked: self.set_orbits_visible(bool(checked)))

        self._sync_layer_button_texts()
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.links_button)
        status_row.addWidget(self.orbits_button)
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
        colors = np.full((self.config.total_sats, 3), hex_to_rgb("#3b82f6"), dtype=np.uint8)
        if not self.show_groups:
            return colors

        step = int(self.position_series.steps[int(row)])
        data = self.group_data.get(step, {})
        groups = data.get("groups", {}) if isinstance(data, dict) else {}
        for gid, nodes in groups.items():
            if int(gid) < len(self.config.group_colors):
                rgb = hex_to_rgb(self.config.group_colors[int(gid)])
            else:
                rgb = hex_to_rgb("#64748b")
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
        orbit_text = f"orbits={len(self.orbit_edges)}" if self.show_orbits else "orbits=off"
        cache_name = self.position_series.cache_dir.name
        return (
            f"3D real-cache step={step}s row={self.current_row + 1}/{self.position_series.num_steps} "
            f"sats={self.config.total_sats} {link_text} {orbit_text} cache={cache_name}"
        )

    def _sync_layer_button_texts(self) -> None:
        if hasattr(self, "links_button"):
            self.links_button.setText("Links on" if self.show_links else "Links off")
        if hasattr(self, "orbits_button"):
            self.orbits_button.setText("Orbits on" if self.show_orbits else "Orbits off")

    def set_links_visible(self, visible: bool, *, render: bool = True) -> None:
        self.show_links = bool(visible and self.edge_table is not None)
        self._sync_layer_button_texts()
        points = np.asarray(self.sat_centers.points, dtype=np.float32)
        self._refresh_link_actor(points, self.current_row)
        self.status_label.setText(self._status_text())
        if render:
            self.plotter.render()

    def set_orbits_visible(self, visible: bool, *, render: bool = True) -> None:
        self.show_orbits = bool(visible)
        self._sync_layer_button_texts()
        points = np.asarray(self.sat_centers.points, dtype=np.float32)
        self._refresh_orbit_actor(points)
        self.status_label.setText(self._status_text())
        if render:
            self.plotter.render()

    def _link_edge_indices(self) -> np.ndarray:
        if not self.show_links or self.edge_table is None:
            return np.asarray([], dtype=np.int64)
        return np.arange(0, self.edge_table.num_edges, self.link_stride, dtype=np.int64)

    def _refresh_link_actor(self, points: np.ndarray, row: int) -> None:
        self._remove_actor_quietly("link_actor")
        if not self.show_links or self.edge_table is None:
            return

        edge_indices = self._link_edge_indices()
        edges = [
            (int(self.edge_table.src[idx]), int(self.edge_table.dst[idx]))
            for idx in edge_indices
        ]
        mesh = segments_to_polydata(points, edges)
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
                clim=(self.value_min, self.value_max)
                if self.value_min is not None and self.value_max is not None
                else None,
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

    def _refresh_orbit_actor(self, points: np.ndarray) -> None:
        self._remove_actor_quietly("orbit_actor")
        if not self.show_orbits:
            return
        mesh = segments_to_polydata(points, self.orbit_edges)
        if mesh is None:
            return
        self.plotter.add_mesh(
            mesh,
            name="orbit_actor",
            color="#111827",
            line_width=1.8,
            opacity=0.62,
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
        mesh = segments_to_polydata(
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
        mesh = segments_to_polydata(points, zip(nodes[:-1], nodes[1:]))
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
        self._refresh_orbit_actor(points)
        self._refresh_selected_edge(points)
        self._refresh_path_actor(points)
        self._refresh_selected_sat_labels()
        self.status_label.setText(self._status_text())
        if render:
            self.plotter.render()

    def _sat_display_name(self, sat_idx: int) -> str:
        node = int(sat_idx)
        if 0 <= node < int(self.config.total_sats):
            p, y = divmod(node, int(self.config.N))
            return f"node={node} (x={p}, y={y})"
        return f"node={node}"

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
        for name in (
            "sat_pick_labels",
            "selected_edge_actor",
            "path_actor",
            "path_node_actor",
            "link_actor",
            "orbit_actor",
        ):
            self._remove_actor_quietly(name)
        try:
            self.plotter.close()
        except Exception:
            pass
        try:
            self.plotter.deleteLater()
        except Exception:
            pass
