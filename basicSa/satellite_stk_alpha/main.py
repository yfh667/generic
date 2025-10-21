from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QApplication
)
from widgets.control import ControlPanel
from widgets.station import StationData

# from core.basicSa import SatelliteManager
#
#
#
# from core.basicSa import StationManager

import basicSa.simulator.nodemanager as nodemanager

from PyQt5.QtCore import Qt, QTimer

from widgets.time_control import TimeControl

from widgets.routegraph import Route2DView
from widgets.postview import PostControlPanel

from core.satellitedc import SatelliteSimulator

from core.justrun import justrun
from postcore.readfile import TimeDatabase  # 新增导入
##ATTENTION
from core.linkengine2 import LinkEngine



class CustomSubWindow(QMdiSubWindow):
    def __init__(self, widget, title, parent=None):
        super().__init__(parent)
        self.setWidget(widget)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.manager = nodemanager.SatelliteManager()

        self.TimeDatabase = TimeDatabase()


        self.stationmanager = nodemanager.StationManager()


        self.sim = SatelliteSimulator(  self.manager )  # 创建数据中心


        self.LinkEngine = LinkEngine( self.sim)  # 创建数据中心
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

        window_menu.addAction("平铺排列", self.mdi.tileSubWindows)
        window_menu.addAction("层叠排列", self.mdi.cascadeSubWindows)




    def _create_initial_windows(self):
        # 创建初始窗口

     #   self.view3d = CesiumSatellite3DView(self.sim)


        self.control_panel = ControlPanel(self.manager)

        self.control_panel2 = PostControlPanel(self.TimeDatabase)


        self.station = StationData(self.stationmanager)



## attention we negelect the linkgraph
        #self.plot2d = Satellite2DView(self.LinkEngine)

        self.routing =Route2DView(self.LinkEngine,self.stationmanager)


        self.simulate =justrun( self.sim,self.manager ,self.stationmanager)

        self.time_control = TimeControl()

        windows = [

            (self.control_panel, "卫星数据导入"),
            (self.station, "地面站"),
           # (self.plot2d, "卫星建链网格"),
            (self.time_control, "时间控制"),
           (self.routing, "路由规划"),
            (self.control_panel2, "后期数据分析"),
            (self.simulate, "仿真"),

        ]

        for widget, title in windows:
            sub = CustomSubWindow(widget, title)
            self.mdi.addSubWindow(sub)
            sub.show()



    def _connect_signals(self):
        # 连接控制面板信号

        self.control_panel.orbit_params_changed.connect(self.LinkEngine.set_parameters)

        self.control_panel.trajectory_loaded.connect(self._update_time_control)
        # 先连接计算模块


        self.sim.positions_updated.connect(      self.LinkEngine.on_positions_updated)

        # 后连接可视化模块
       # self.sim.positions_updated.connect(self.view3d.on_positions_updated)
        self.LinkEngine.modify_nodes.connect(self.time_control.on_positions_updated)
        # 连接时间控制信号
        self.time_control.time_changed.connect(self._on_time_changed)




        self.station.station_added.connect(    self.sim._update_stations )


    def _update_time_control(self):
        """数据加载后更新时间控件"""
        self.time_control.set_time_range(self.manager.max_time)

    def _on_time_changed(self, normalized_time):
        """处理时间变化"""
        if self.timer.isActive():
            self.timer.stop()



        self.manager.current_time = normalized_time

        self.sim.current_time =normalized_time



    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
