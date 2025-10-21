from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout,
    QPushButton, QLineEdit, QTextEdit, QLabel
)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from core.satellite import SatelliteManager
import basicSa.satellite_stk_alpha.core.satellite as satellite
import basicSa.utilis.Node as Node
import basicSa.utilis.readdata as readdata

from core.satellitedc import SatelliteSimulator
import os


class justrun(QWidget):


    def __init__(self, sim:SatelliteSimulator,satellites: satellite.SatelliteManager,stations: satellite.StationManager  ,  parent=None):  # 添加manager参数
        super().__init__(parent)
        self.sim = sim  # 初始化manager
        self.satellites = satellites  # 初始化manager
        self.stations = stations  # 初始化manager
        self._init_ui()

        self._position_cache = {}
    def _init_ui(self):
        layout = QVBoxLayout()

        # 文件加载按钮
        self.load_btn = QPushButton('begain simulator')
        self.load_btn.clicked.connect(self.run)
        layout.addWidget(self.load_btn)  # 添加到布局
     # Set layout for the widget
        self.setLayout(layout)



    def run(self):
        first_satellite = next(iter(self.satellites.satellites.values()))  # Get the first basicSa object
        first_trajectory = first_satellite.trajectory
        max_time  = len(first_trajectory)



        for i in range(100):
            self.sim.current_time = i

        print("1")



        # self._position_cache.clear()

        # for time in range(1):
        #     for sat in self.satellites.satellites.values():
        #         idx = next((i for i, p in enumerate(sat.trajectory)
        #                   if p.time >= time), 0)
        #         self._position_cache[sat.id] = (
        #             sat.trajectory[idx],  # 当前时刻
        #             sat.trajectory[idx+1] # 下一时刻（用于计算）
        #         )

