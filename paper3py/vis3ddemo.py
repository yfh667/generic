# demo_globe_pyvista.py
# pip install pyvista pyvistaqt pyqt5 vtk numpy

import numpy as np
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel
)
from shapely.ops import unary_union
import geopandas as gpd
import numpy as np
import pyvista as pv

EARTH_R_KM = 6371.0
SAT_ALT_KM = 550.0
INC_DEG = 53.0

# 建议放一张“浅色、无标注”的地球纹理；
# 没有就走纯色球 + 大陆轮廓，整体观感也会接近 Cesium 那套白底风格
EARTH_TEXTURE_PATH = None

# ===== Cesium / Positron 风格配色 =====
BG_COLOR = "#ffffff"       # 纯白背景
EARTH_COLOR = "#eef2f5"    # 极浅灰白地球
COAST_COLOR = "#c9d1d9"    # 大陆轮廓线
ORBIT_COLOR = "#d7dee5"    # 轨道参考线
SAT_COLOR = "#3b82f6"   # 蓝色
     # 卫星点
LINK_COLOR = "#b8c3ce"     # 常规链路
PATH_COLOR = "#4f8fc4"     # 高亮路径
STATION_COLOR = "#7c8794"  # 地面站

UI_BG = "#ffffff"
UI_PANEL = "#f8fafc"
UI_BORDER = "#d9e1e8"
TEXT_COLOR = "#5f6b78"

# pip install geopandas pyogrio shapely
import geopandas as gpd

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def ll_to_xyz(lat_deg, lon_deg, r):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z], dtype=float)


class GlobeSatDemo(QWidget):
    def __init__(self, parent=None, P=18, N=36):
        super().__init__(parent)
        self.setWindowTitle("3D Globe + Satellites Demo (PyVista)")
        self.resize(1400, 900)

        self.P = P
        self.N = N
        self.total = P * N
        self.step = 0
        self.playing = True

        self._build_ui()
        self._build_scene()
        self._build_static_layers()
        self._update_frame(0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(50)

    def sid(self, p, s):
        return p * self.N + s

    def _build_ui(self):
        self.setStyleSheet(f"""
        QWidget {{
            background: {UI_BG};
            color: {TEXT_COLOR};
            font-size: 13px;
        }}
        QPushButton {{
            background: {UI_PANEL};
            border: 1px solid {UI_BORDER};
            border-radius: 5px;
            padding: 6px 12px;
            min-height: 28px;
        }}
        QPushButton:hover {{
            background: #f1f5f9;
        }}
        QLabel {{
            color: {TEXT_COLOR};
        }}
        QSlider::groove:horizontal {{
            border: 0;
            height: 4px;
            background: #dce3ea;
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: #8c98a5;
            border: 0;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        bar = QHBoxLayout()
        bar.setSpacing(8)

        self.btn = QPushButton("Pause")
        self.btn.clicked.connect(self._toggle_play)

        self.lbl = QLabel("step=0")
        self.lbl.setFixedWidth(90)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 5000)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider)

        bar.addWidget(self.btn)
        bar.addWidget(self.lbl)
        bar.addWidget(self.slider)

        self.plotter = QtInteractor(self)

        root.addLayout(bar)
        root.addWidget(self.plotter.interactor)

    def _build_scene(self):
        # 白底，和 Cesium 示例一致
        self.plotter.set_background(BG_COLOR)
        self.plotter.enable_anti_aliasing()

        earth = pv.Sphere(radius=EARTH_R_KM, theta_resolution=220, phi_resolution=220)

        if EARTH_TEXTURE_PATH:
            tex = pv.read_texture(EARTH_TEXTURE_PATH)
            self.plotter.add_mesh(
                earth,
                texture=tex,
                smooth_shading=True,
                ambient=0.28,
                diffuse=0.72,
                specular=0.0,
            )
        else:
            # 没有浅色纹理时，用浅灰白球体 + 大陆轮廓，整体气质接近 Cesium Positron
            self.plotter.add_mesh(
                earth,
                color=EARTH_COLOR,
                smooth_shading=True,
                ambient=0.35,
                diffuse=0.65,
                specular=0.0,
            )
            #self._add_continent_outlines()

            self.add_country_outlines_from_polygons(r"D:\paper3\data\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp")

        # 卫星点云（动态）
        self.sat_poly = pv.PolyData(np.zeros((self.total, 3), dtype=float))
        self.sat_actor = self.plotter.add_mesh(
            self.sat_poly,
            render_points_as_spheres=True,
            point_size=8,
            color=SAT_COLOR,
            ambient=0.5,
        )

        self.link_actor = None
        self.path_actor = None

        # 改成更接近 Cesium 的明亮背景下观察角度
        self.plotter.camera_position = [
            (18000, -15000, 11000),
            (0, 0, 0),
            (0, 0, 1),
        ]

    # def _add_continent_outlines(self):
    #     """
    #     用 VTK 自带的地球大陆轮廓，模拟 Cesium 里的浅色无标注底图。
    #     这样即使没有纹理，也不会只剩一个纯色球。
    #     """
    #     earth_src = vtk.vtkEarthSource()
    #     earth_src.OutlineOn()
    #
    #     if hasattr(earth_src, "SetOnRatio"):
    #         earth_src.SetOnRatio(1)
    #
    #     if hasattr(earth_src, "SetRadius"):
    #         earth_src.SetRadius(EARTH_R_KM * 1.0015)
    #
    #     earth_src.Update()
    #     coast = pv.wrap(earth_src.GetOutput())
    #
    #     # 老版本 vtkEarthSource 可能没有 SetRadius
    #     if not hasattr(earth_src, "SetRadius"):
    #         coast.points = coast.points * (EARTH_R_KM * 1.0015)
    #
    #     self.plotter.add_mesh(
    #         coast,
    #         color=COAST_COLOR,
    #         line_width=1.0,
    #         opacity=0.95,
    #         lighting=False,
    #     )
    def _iter_lines(self, geom):
        if geom is None or geom.is_empty:
            return

        gt = geom.geom_type
        if gt == "LineString":
            yield geom
        elif gt == "MultiLineString":
            for g in geom.geoms:
                yield from self._iter_lines(g)
        elif gt == "GeometryCollection":
            for g in geom.geoms:
                yield from self._iter_lines(g)

    def add_country_outlines_from_polygons(self, shp_path):
        import geopandas as gpd
        from shapely.ops import unary_union

        gdf = gpd.read_file(shp_path)

        # 关键：合并“边界线”，不是合并“国家面”
        boundary = unary_union(gdf.geometry.boundary)

        r = EARTH_R_KM * 1.004
        segments = []

        for line in self._iter_lines(boundary):
            xyz = np.array(
                [ll_to_xyz(lat, lon, r) for lon, lat in line.coords],
                dtype=float
            )
            if len(xyz) >= 2:
                segments.append(xyz)

        if not segments:
            return

        mesh = self._segments_to_polydata(segments)

        self.plotter.add_mesh(
            mesh.copy(),
            color="#ffffff",
            line_width=3.0,
            opacity=0.95,
            lighting=False,
            render_lines_as_tubes=True,
        )
        self.plotter.add_mesh(
            mesh,
            color="#d7dde4",
            line_width=1.2,
            opacity=1.0,
            lighting=False,
            render_lines_as_tubes=True,
        )

    def _segments_to_polydata(self, segments):
        pts = np.vstack(segments)
        cells = []
        offset = 0

        for seg in segments:
            cells.extend([len(seg), *range(offset, offset + len(seg))])
            offset += len(seg)

        mesh = pv.PolyData(pts)
        mesh.lines = np.array(cells, dtype=np.int64)
        return mesh

    def add_country_borders(self, shp_path):
        gdf = gpd.read_file(shp_path)
        r = EARTH_R_KM * 1.004
        segments = []

        for geom in gdf.geometry:
            if geom is None:
                continue
            geoms = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
            for line in geoms:
                xyz = np.array(
                    [ll_to_xyz(lat, lon, r) for lon, lat in line.coords],
                    dtype=float
                )
                if len(xyz) >= 2:
                    segments.append(xyz)

        border_mesh = self._segments_to_polydata(segments)


        self.plotter.add_mesh(
            border_mesh.copy(),
            color="#ffffff",
            line_width=3.0,
            opacity=0.9,
            lighting=False,
            render_lines_as_tubes=True,
        )
        self.plotter.add_mesh(
            border_mesh,
            color="#6f7c89",
            line_width=1.2,
            opacity=1.0,
            lighting=False,
            render_lines_as_tubes=True,
        )

    def _add_continent_outlines(self):
        earth_src = vtk.vtkEarthSource()
        earth_src.OutlineOn()
        earth_src.SetOnRatio(1)

        outline_r = EARTH_R_KM * 1.005
        if hasattr(earth_src, "SetRadius"):
            earth_src.SetRadius(outline_r)

        earth_src.Update()
        coast = pv.wrap(earth_src.GetOutput())

        if not hasattr(earth_src, "SetRadius"):
            coast.points = coast.points * outline_r

        # 白色底描边，制造“发亮边”
        self.plotter.add_mesh(
            coast.copy(),
            color="#ffffff",
            line_width=3.6,
            opacity=0.95,
            lighting=False,
            render_lines_as_tubes=True,
        )

        # 主轮廓线
        self.plotter.add_mesh(
            coast,
            color="#7f8a96",
            line_width=1.6,
            opacity=1.0,
            lighting=False,
            render_lines_as_tubes=True,
        )

    def _build_static_layers(self):
        # 示例地面站：配色压暗，避免在白底上太炸
        stations_ll = [
            (-15.7939, -47.8828),   # Brazil
            (12.1140, -86.2362),    # Nicaragua
            (30.0444, 31.2357),     # Egypt
            (39.9042, 116.4074),    # Beijing
            (-33.8688, 151.2093),   # Sydney
        ]
        station_xyz = np.array([ll_to_xyz(lat, lon, EARTH_R_KM) for lat, lon in stations_ll], dtype=float)
        st_poly = pv.PolyData(station_xyz)
        self.plotter.add_mesh(
            st_poly,
            render_points_as_spheres=True,
            point_size=11,
            color=STATION_COLOR,
            ambient=0.25,
        )

        # 轨道平面参考线（静态）
        inc = np.deg2rad(INC_DEG)
        u = np.linspace(0, 2 * np.pi, 300)
        r = EARTH_R_KM + SAT_ALT_KM
        base = np.vstack([r * np.cos(u), r * np.sin(u), np.zeros_like(u)]).T

        for p in range(self.P):
            raan = 2 * np.pi * p / self.P
            pts = (rot_z(raan) @ (rot_x(inc) @ base.T)).T
            line = pv.Spline(pts, 600)
            self.plotter.add_mesh(
                line,
                color=ORBIT_COLOR,
                line_width=1,
                opacity=0.60,
                lighting=False,
            )

    def _sat_positions(self, step):
        # 简单轨道动力学演示：按均匀角速度推进
        t = step * 0.015
        w = 0.9
        inc = np.deg2rad(INC_DEG)
        r = EARTH_R_KM + SAT_ALT_KM

        pts = np.zeros((self.total, 3), dtype=float)
        for p in range(self.P):
            raan = 2 * np.pi * p / self.P
            R = rot_z(raan) @ rot_x(inc)
            for s in range(self.N):
                u = 2 * np.pi * s / self.N + w * t
                vec_orb = np.array([r * np.cos(u), r * np.sin(u), 0.0], dtype=float)
                pts[self.sid(p, s)] = R @ vec_orb

        # 地球坐标系有一点自转视觉
        earth_spin = -0.08 * t
        pts = (rot_z(earth_spin) @ pts.T).T
        return pts

    @staticmethod
    def _edge_mesh(points, edges):
        if not edges:
            return None

        seg_pts = np.empty((2 * len(edges), 3), dtype=float)
        line_cells = np.empty((len(edges), 3), dtype=np.int64)
        for k, (i, j) in enumerate(edges):
            seg_pts[2 * k] = points[i]
            seg_pts[2 * k + 1] = points[j]
            line_cells[k] = [2, 2 * k, 2 * k + 1]

        mesh = pv.PolyData(seg_pts)
        mesh.lines = line_cells.ravel()
        return mesh

    def _build_edges(self):
        edges = []

        # 同轨
        for p in range(self.P):
            for s in range(self.N):
                i = self.sid(p, s)
                j = self.sid(p, (s + 1) % self.N)
                edges.append((i, j))

        # 跨轨（稀疏一些，避免太乱）
        for p in range(self.P):
            pn = (p + 1) % self.P
            for s in range(0, self.N, 3):
                i = self.sid(p, s)
                j = self.sid(pn, s)
                edges.append((i, j))

        return edges

    def _build_path_nodes(self, step):
        # 演示路径：随时间平移
        p0 = (step // 30) % self.P
        s = 2
        nodes = []
        for k in range(12):
            pp = (p0 + k) % self.P
            ss = (s + k) % self.N
            nodes.append(self.sid(pp, ss))
        return nodes

    def _update_frame(self, step):
        self.step = step
        pts = self._sat_positions(step)

        self.sat_poly.points = pts
        self.sat_poly.Modified()

        # 常规链路
        edges = self._build_edges()
        edge_mesh = self._edge_mesh(pts, edges)
        if self.link_actor is not None:
            self.plotter.remove_actor(self.link_actor)
        self.link_actor = self.plotter.add_mesh(
            edge_mesh,
            color=LINK_COLOR,
            line_width=1,
            opacity=0.42,
            lighting=False,
        )

        # 高亮路径
        path_nodes = self._build_path_nodes(step)
        path_edges = [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)]
        path_mesh = self._edge_mesh(pts, path_edges)
        if self.path_actor is not None:
            self.plotter.remove_actor(self.path_actor)
        self.path_actor = self.plotter.add_mesh(
            path_mesh,
            color=PATH_COLOR,
            line_width=3.2,
            opacity=1.0,
            lighting=False,
        )

        self.lbl.setText(f"step={step}")
        self.plotter.render()

    def _tick(self):
        if not self.playing:
            return

        nxt = (self.step + 1) % (self.slider.maximum() + 1)
        self.slider.blockSignals(True)
        self.slider.setValue(nxt)
        self.slider.blockSignals(False)
        self._update_frame(nxt)

    def _on_slider(self, v):
        self._update_frame(int(v))

    def _toggle_play(self):
        self.playing = not self.playing
        self.btn.setText("Pause" if self.playing else "Play")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = GlobeSatDemo(P=18, N=36)  # 可以先改成 P=12, N=24 看性能
    w.show()
    sys.exit(app.exec_())