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

EARTH_R_KM = 6371.0
SAT_ALT_KM = 550.0
INC_DEG = 53.0

# 可选：如果你后面准备了“浅色无标注”的等经纬度纹理，直接填这里
EARTH_TEXTURE_PATH = None
# 例如：
# EARTH_TEXTURE_PATH = r"D:\paper3\data\textures\earth_light_nolabels.jpg"

# ===== 按 Cesium 这份配色来 =====
BG_COLOR = "#ffffff"             # scene.backgroundColor = WHITE
EARTH_BASE_COLOR = "#ffffff"     # globe.baseColor = WHITE

# 用很淡的灰模拟 CartoDB light_nolabels 的地表层次
LAND_FILL_COLOR = "#eef2f5"
COAST_COLOR = "#cfd6de"

# 其余元素也压成更接近 Cesium 的冷灰蓝
SAT_COLOR = "#6f7b87"
LINK_COLOR = "#bcc8d4"
PATH_COLOR = "#4f8fc4"
ORBIT_COLOR = "#d3dae2"
STATION_COLOR = "#7e8996"


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
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.btn = QPushButton("Pause")
        self.btn.clicked.connect(self._toggle_play)
        self.lbl = QLabel("step=0")
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

    def _earth_source(self, outline=False, radius_scale=1.001):
        """
        用 vtkEarthSource 生成一层非常轻的陆地/海岸线，
        去模拟 Cesium 里的 light_nolabels 地表细节。
        """
        src = vtk.vtkEarthSource()

        if outline:
            src.OutlineOn()
        else:
            src.OutlineOff()

        if hasattr(src, "SetOnRatio"):
            src.SetOnRatio(1)

        radius = EARTH_R_KM * radius_scale
        if hasattr(src, "SetRadius"):
            src.SetRadius(radius)

        src.Update()
        mesh = pv.wrap(src.GetOutput())

        # 兼容少数旧版 VTK：没有 SetRadius 时，手动缩放
        if not hasattr(src, "SetRadius"):
            mesh.points = mesh.points * radius

        return mesh

    def _add_cesium_surface_details(self):
        # 陆地填充：极淡灰
        land = self._earth_source(outline=False, radius_scale=1.0008)
        self.plotter.add_mesh(
            land,
            color=LAND_FILL_COLOR,
            opacity=1.0,
            lighting=False,
        )

        # 海岸线：略深一层灰
        coast = self._earth_source(outline=True, radius_scale=1.0015)
        self.plotter.add_mesh(
            coast,
            color=COAST_COLOR,
            line_width=1.0,
            opacity=1.0,
            lighting=False,
        )

    def _build_scene(self):
        # 对齐 Cesium：纯白背景
        self.plotter.set_background(BG_COLOR)
        self.plotter.enable_anti_aliasing()

        # 去掉 show_axes()，Cesium 里没有这套彩色坐标轴
        earth = pv.Sphere(radius=EARTH_R_KM, theta_resolution=220, phi_resolution=220)

        if EARTH_TEXTURE_PATH:
            tex = pv.read_texture(EARTH_TEXTURE_PATH)
            self.plotter.add_mesh(
                earth,
                texture=tex,
                smooth_shading=True,
                ambient=0.18,
                diffuse=0.82,
                specular=0.0,
            )
        else:
            # base sphere 必须回到纯白，对齐 globe.baseColor = WHITE
            self.plotter.add_mesh(
                earth,
                color=EARTH_BASE_COLOR,
                smooth_shading=True,
                ambient=0.16,
                diffuse=0.84,
                specular=0.0,
            )
            # 地表可见性不靠 baseColor，而靠这层浅灰陆地细节
            self._add_cesium_surface_details()

        # 不加蓝色大气层：Cesium 这份配置里 skyAtmosphere 是关掉的
        # 不加 skyBox：背景就是纯白

        # 卫星点云（动态）
        self.sat_poly = pv.PolyData(np.zeros((self.total, 3), dtype=float))
        self.sat_actor = self.plotter.add_mesh(
            self.sat_poly,
            render_points_as_spheres=True,
            point_size=8,
            color=SAT_COLOR,
            ambient=0.25,
        )

        self.link_actor = None
        self.path_actor = None

        # 相机初始视角也尽量贴近你那份 Cesium：非洲方向
        cam = ll_to_xyz(10.32, 19.57, EARTH_R_KM + 20000.0)  # lat=10.32, lon=19.57, h=20000km
        self.plotter.camera_position = [
            tuple(cam),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]

    def _build_static_layers(self):
        # 几个示例地面站
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

        # 让地球坐标系有一点自转视觉
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
            opacity=0.38,
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
    w = GlobeSatDemo(P=18, N=36)  # 你可以先改成 P=12,N=24 看性能
    w.show()
    sys.exit(app.exec_())