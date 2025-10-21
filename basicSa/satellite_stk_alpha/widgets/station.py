from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout,
    QPushButton, QLineEdit, QTextEdit, QLabel
)
from PyQt5 import QtGui
from PyQt5.QtGui import QTextCursor

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from core.satellite import  StationManager
import basicSa.utilis.Node as Node
import basicSa.utilis.readdata as readdata
from config.settings import SatelliteConfig
import os
import basicSa.utilis.Node as Node
import math
from config.settings import SatelliteConfig
import basicSa.fileread.readstation as readstation
class StationData(QWidget):
    station_added = pyqtSignal(Node.Node)  # (sta_id, x, y, z)

    def __init__(self, manager: StationManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # 地面站输入表单
        station_form = QFormLayout()
        self.sta_x = QLineEdit()
        self.sta_y = QLineEdit()
        self.sta_z = QLineEdit()
        self.sta_angle = QLineEdit()
        station_form.addRow("X坐标 (米):", self.sta_x)
        station_form.addRow("Y坐标 (米):", self.sta_y)
        station_form.addRow("Z坐标 (米):", self.sta_z)
        station_form.addRow("最小仰角 (度):", self.sta_angle)

        self.add_station_btn = QPushButton('添加地面站')
        self.add_station_btn.clicked.connect(self.add_ground_station_single)

        # 日志显示区域
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("""
            QTextEdit {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                padding: 5px;
                font-family: Consolas;
            }
        """)





        layout.addLayout(station_form)
        layout.addWidget(self.add_station_btn)
        layout.addWidget(QLabel("地面站数据:"))
        layout.addWidget(self.log)

        self.setLayout(layout)

        # 文件加载按钮
        self.load_btn = QPushButton('Load station')
        self.load_btn.clicked.connect(self.load_station)
        layout.addWidget(self.load_btn)  # 添加到布局

    def load_station(self):
        """加载文件夹下所有txt文件"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Trajectory Folder", ""
        )
        if not dir_path:
            return

        self.StationManager=readstation.readstation_path(dir_path,SatelliteConfig.stationangle)


        stations =  self.StationManager.stations

        for i in range(len(stations)):
            self.station_added.emit(stations[i].trajectory[0])


        # for station in stations:



        # for station in StationManager:
        #
        # for station in stationNodes:
        #       self.station_added.emit(station)

        # # 获取所有txt文件并按数字顺序排序
        # files = []
        # for filename in os.listdir(dir_path):
        #     if filename.endswith(".txt"):
        #         try:
        #             # 提取文件名中的数字部分（假设文件名格式为数字.txt）
        #             num = int(os.path.splitext(filename)[0])
        #             files.append((num, filename))
        #         except ValueError:
        #             continue  # 跳过不符合命名规范的文件
        #
        # # 按数字顺序排序
        # files.sort(key=lambda x: x[0])
        #
        # # 按顺序处理文件
        # for num, filename in files:
        #     file_path = os.path.join(dir_path, filename)
        #     with open(file_path, "r") as file:
        #         lines = file.readlines()
        #
        #         for line in lines:
        #             try:
        #                 data = line.split()
        #                 x = float(data[0])
        #                 y = float(data[1])
        #                 z = float(data[2])
        #                 angle = math.radians(SatelliteConfig.stationangle)
        #
        #                 sta_id = self.manager.add_ground_station(x, y, z, angle)
        #
        #                 self.station_added.emit(Node.Node(x=x, y=y, z=z, angle=angle, nodeid=sta_id))
        #
        #             except (ValueError, IndexError) as e:
        #                 print(f"Error parsing line in {filename}: {e}")
        #                 continue

    def add_ground_station_single(self):  # 修正缩进，作为类方法
        """处理地面站添加"""
        try:
            x = float(self.sta_x.text())
            y = float(self.sta_y.text())
            z = float(self.sta_z.text())
            angle = float(self.sta_angle.text())

            angle = math.radians(angle)
            # if not self.manager.validate_coordinates(x, y, z):
            #     raise ValueError("坐标不在合理地球表面范围内")

            sta_id = self.manager.add_ground_station(x, y, z,angle)


            self.station_added.emit(Node.Node(x=x, y=y, z=z, angle=angle,nodeid=sta_id))




            # 记录成功日志
            log_msg = f"地面站 {sta_id}: ({x:.2f}, {y:.2f}, {z:.2f})\n"
            self._append_log(log_msg, "success")

            QMessageBox.information(
                self,
                "成功",
                f"地面站 {sta_id} 已添加\n坐标: ({x:.2f}, {y:.2f}, {z:.2f})"
            )

            # 清空输入框
            self.sta_x.clear()
            self.sta_y.clear()
            self.sta_z.clear()

        except ValueError as e:
            QMessageBox.critical(self, "输入错误", str(e))




    def _append_log(self, message: str, msg_type: str = "info"):
        """追加日志到显示区域"""
        color_map = {
            "success": "#28a745",
            "error": "#dc3545",
            "info": "#17a2b8"
        }

        self.log.moveCursor(QtGui.QTextCursor.End)
        self.log.setTextColor(QtGui.QColor(color_map.get(msg_type, "#000000")))
        self.log.insertPlainText(message + "\n")

        # 自动滚动到底部
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
