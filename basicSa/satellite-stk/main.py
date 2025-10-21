import sys
import numpy as np
from vispy import scene
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from widgets.view3d import Satellite3DView
from widgets.control import ControlPanel
from core.satellite import SatelliteManager
from widgets.plot2d import Satellite2DView
from widgets.time_control import TimeControl
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QDockWidget, QApplication, QWidget, QVBoxLayout
)
from widgets.view3d import Satellite3DView
from widgets.control import ControlPanel
from core.satellite import SatelliteManager
from PyQt5.QtCore import Qt, QTimer, QSettings, QEvent, QRect,QObject
from widgets.plot2d import Satellite2DView
from widgets.time_control import TimeControl
from weakref import ref
from PyQt5.QtCore import pyqtSlot, QObject, QEvent
from weakref import ref


class CustomSubWindow(QMdiSubWindow):
    def __init__(self, widget, title, parent=None):
        super().__init__(parent)
        self.setWidget(widget)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.manager = SatelliteManager()
        self.timer = QTimer()
        self.timer.setInterval(50)

        # 创建MDI区域
        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)
        self._create_initial_windows()
        self._init_ui()
        self._connect_signals()


    def _init_ui(self):
        self.setWindowTitle("STK-like Satellite Viewer (MDI)")
        self.resize(1600, 900)

        # 创建菜单栏
        bar = self.menuBar()
        window_menu = bar.addMenu("窗口")
        window_menu.addAction("新建3D视图", self.create_3d_window)
        window_menu.addAction("平铺排列", self.mdi.tileSubWindows)
        window_menu.addAction("层叠排列", self.mdi.cascadeSubWindows)

    def _create_initial_windows(self):
        # 创建初始窗口
        self.view3d = Satellite3DView(self.manager)
        self.control_panel = ControlPanel(self.manager)
        self.plot2d = Satellite2DView(self.manager)
        self.time_control = TimeControl()

        windows = [
            (self.view3d, "3D视图"),
            (self.control_panel, "控制面板"),
            (self.plot2d, "2D卫星网格"),
            (self.time_control, "时间控制")
        ]

        for widget, title in windows:
            sub = CustomSubWindow(widget, title)
            self.mdi.addSubWindow(sub)
            sub.show()

    def create_3d_window(self):
        """创建新的3D视图窗口"""
        view = Satellite3DView(self.manager)
        sub = CustomSubWindow(view, "3D视图")
        self.mdi.addSubWindow(sub)
        sub.show()

    def _connect_signals(self):
        # 连接控制面板信号
        self.control_panel.orbit_params_changed.connect(self.plot2d.update_layout)
        self.control_panel.trajectory_loaded.connect(self._update_time_control)

        # 连接时间控制信号
        self.time_control.time_changed.connect(self._on_time_changed)

        # 连接3D视图信号
        self.view3d.satellite_positions_3d_signal.connect(self.plot2d.update_satellite_positions)

    def _update_time_control(self):
        """数据加载后更新时间控件"""
        self.time_control.set_time_range(self.manager.max_time)

    def _on_time_changed(self, normalized_time):
        """处理时间变化"""
        if self.timer.isActive():
            self.timer.stop()

        current_time = normalized_time * self.manager.max_time
        self.manager.current_time = current_time
        self.view3d.update_positions(current_time)

    def toggle_animation(self):
        """处理动画启动/停止"""
        self.plot2d.set_animation_state(self.timer.isActive())

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
