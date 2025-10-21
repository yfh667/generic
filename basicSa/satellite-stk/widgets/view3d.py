from vispy import scene
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton  # 添加QPushButton
from PyQt5.QtCore import Qt, pyqtSignal

import numpy as np
class Satellite3DView(QWidget):
    play_btn_click = pyqtSignal()  # 添加缺失的信号
    satellite_positions_3d_signal = pyqtSignal(dict)  # 信号传递一个列表，包含卫星的2D坐标

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.markers = {}
        self.current_time = 0.0

        # 初始化3D视图
        self._init_3d_canvas()

        # 初始化控制按钮
        self.play_btn = QPushButton('Run')  # 现在可以正确引用
        self.play_btn.clicked.connect(self.toggle_animation)

        # 布局设置
        layout = QVBoxLayout()
        layout.addWidget(self.canvas.native)
        layout.addWidget(self.play_btn)  # 将按钮添加到布局
        self.setLayout(layout)

        # 动画定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_positions)

    def _init_3d_canvas(self):
        """初始化3D画布和地球"""
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='white',parent=self,vsync=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'arcball'
         # 创建地球
        self.sphere = scene.visuals.Sphere(
            radius=6378000,
            method='latitude',
            parent=self.view.scene,
            color=(0.7, 0.7, 0.7),
            edge_color='black'
        )
        self.view.camera.set_range(x=[-7e6, 7e6], y=[-7e6, 7e6], z=[-7e6, 7e6])


    def toggle_animation(self):
        self.play_btn_click.emit()
        # if self.timer.isActive():
        #     self.timer.stop()
        #     self.play_btn.setText('Run')
        # else:
        #     self.timer.basicSa(50)  # 50ms更新间隔
        #     self.play_btn.setText('Pause')

    def update_positions(self, current_time=None):
        """更新卫星的位置"""
        if current_time is not None:
            self.current_time = current_time  # 使用传入的时间

        # 初始化字典存储卫星3D位置
        satellite_positions_3d = {}

        for sat in self.manager.satellites.values():
            # 找到当前时间对应的位置
            idx = next(
                (i for i, p in enumerate(sat.trajectory)
                 if p.time >= self.current_time),
                0
            )
            point = sat.trajectory[idx]

            # 更新3D视图中的标记
            self.update_marker(sat.id, point.x, point.y, point.z)

            # here we need add more point
            point_next = sat.trajectory[idx+1]
            # 将位置存入字典（键：sat.id，值：point对象）

            satellite_positions_3d[sat.id] = [point,point_next]


        self.canvas.update()
        self.satellite_positions_3d_signal.emit(satellite_positions_3d)

    # 在Satellite3DView类中修改update_marker方法
    def update_marker(self, sat_id, x, y, z):
        color = (0.0, 0.0, 1.0, 1.0)  # RGBA格式，蓝色不透明

        if sat_id not in self.markers:
            # 创建新标记
            marker = scene.visuals.Markers()
            marker.set_data(
                pos=np.array([[x, y, z]]),
                edge_color=color,
                face_color=color,
                size=12,
                symbol='o',
                edge_width=1.5,
            )
            self.markers[sat_id] = marker
            self.view.add(marker)
        else:
            # 更新现有标记
            marker = self.markers[sat_id]

            # 需要同时更新位置和颜色
            marker.set_data(
                pos=np.array([[x, y, z]]),
                edge_color=color,  # 必须重新设置
                face_color=color,  # 必须重新设置
                size=12,
                symbol='o',
                edge_width=1.5
            )

            # 强制更新图形管线
            marker.update()


