from copy import deepcopy

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout,
    QPushButton, QLineEdit, QTextEdit, QLabel, QHBoxLayout, QSlider, QComboBox
)

from PyQt5.QtWidgets import QFileDialog, QMessageBox

import basicSa.Dataprocessing.readns3file as readns3file

# import basicSa.Dataprocessing.readpyfile as readpyfile

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import numpy as np
import json
import websockets
import asyncio
from threading import Thread
import  os
class PostControlPanel(QWidget):

    trajectory_loaded = pyqtSignal()  # 添加缺失的信号


    def __init__(self, database: readns3file.TimeDatabase, parent=None):  # 添加manager参数
        super().__init__(parent)
        self.database = database  # 初始化manager

        self.is_playing = False
        self.timer = None  # Initialize the timer (this will need to be set up later for real-time updates)
        self._init_timer()
        self.totaltime = 100
        self.clients = set()  # 添加此行初始化
        self.flag = 1
        self.flag_route = 1
        self.current_paths = {}  # 存储当前路径数据
        self.path_mapping = []   # 存储路径索引映射

        self.fixed_paths =[[0,1],[2,3],[4,5],[6,7]]
        self.path_selected= [4,5]

        self.stationnum = 0
        self._init_ui()

        self.satellitenum = 0

        self.sensor_visibility = True  # Initial state: sensors are visible

        self.visible_satellites = []  # Store satellites the user wants to see
    def _init_ui(self):
        layout = QVBoxLayout()
        self.currenttime  = 0
        self.load_btn = QPushButton('gaibian ')
        self.load_btn.clicked.connect(self.route2)
        layout.addWidget(self.load_btn)  # Add the first load button to the layout

        # Add batch load button (second button)
        self.load_btn2 = QPushButton('Load ALL Trajectory')
        self.load_btn2.clicked.connect(self.load_trajectory_filenew)
        layout.addWidget(self.load_btn2)  # Correctly add the second button to the layout

        # 输入表单
        self.load_btn3 = QPushButton('luyou ')
        self.load_btn3.clicked.connect(self.route3)
        layout.addWidget(self.load_btn3)  # Add the first load button to the layout

        self.setLayout(layout)

        # 控制行布局
        control_layout = QHBoxLayout()

        # 播放/暂停按钮
        self.play_btn = QPushButton()
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_btn)

        # 时间标签
        self.time_label = QLabel("Time: 0.0s / 0.0s")
        control_layout.addWidget(self.time_label)

        layout.addLayout(control_layout)
        self.server_thread_loop = None  # 新增属性
        # 进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.valueChanged.connect(self.update_time_label)
        self.slider.sliderPressed.connect(self.pause)  # 手动拖动时暂停
        layout.addWidget(self.slider)

        # 添加时间跳转输入框和按钮
        jump_layout = QHBoxLayout()
        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText("输入时间（整数）")
        self.jump_btn = QPushButton("跳转")
        self.jump_btn.clicked.connect(self.jump_to_time)
        jump_layout.addWidget(self.time_input)
        jump_layout.addWidget(self.jump_btn)
        layout.addLayout(jump_layout)

        self.start_websocket_server()

        # 添加路径选择控件
        path_control_layout = QHBoxLayout()



        # 刷新按钮
        self.refresh_btn = QPushButton("刷新路径")
        self.refresh_btn.clicked.connect(lambda: self.show_path(self.currenttime))
        path_control_layout.addWidget(self.refresh_btn)

        layout.addLayout(path_control_layout)

        # 添加路径显示框
        self.path_display = QTextEdit()
        self.path_display.setReadOnly(True)
        self.path_display.setMinimumHeight(150)  # 设置最小高度
        layout.addWidget(QLabel("当前路径列表:"))
        layout.addWidget(self.path_display)

        layout.addSpacing(20)  # 添加间距

        # 路径选择下拉框
        self.path_combobox = QComboBox()
        self.path_combobox.currentIndexChanged.connect(self.on_path_selected)
        path_control_layout.addWidget(QLabel("选择路径:"))
        path_control_layout.addWidget(self.path_combobox)

        self.path_combobox.addItems([
            "路径 0→1",
            "路径 2→3",
            "路径 4→5"
        ])
        self.path_combobox.currentIndexChanged.connect(self.on_fixed_path_selected)


        ## Sensor visibility toggle button

        self.toggle_sensors_btn = QPushButton("Hide Sensors")  # Initial text
        self.toggle_sensors_btn.clicked.connect(self.toggle_sensor_visibility)
        layout.addWidget(self.toggle_sensors_btn)

        # 添加卫星ID输入框和刷新按钮
        self.satellite_ids_input = QLineEdit()
        self.satellite_ids_input.setPlaceholderText("输入卫星ID，用逗号分隔 (例如: 1,2,3)")
        layout.addWidget(self.satellite_ids_input)

        self.update_sensors_btn = QPushButton("刷新传感器显示")
        self.update_sensors_btn.clicked.connect(self.update_visible_satellites)
        layout.addWidget(self.update_sensors_btn)



    def on_fixed_path_selected(self, index):
        """处理固定路径选择"""
        if 0 <= index < len(self.fixed_paths):
            self.path_selected = self.fixed_paths[index]
         #   self.show_selected_fixed_path()
        else:
            self.selected_fixed_path = None
    def route2(self):
        # Toggle flag between 0 and 1
        self.flag = 1 - self.flag  # If flag is 0, it becomes 1. If flag is 1, it becomes 0
        print(f"flag = {self.flag}")


        self.on_positions_updated(  self.currenttime)


    def route3(self):
        # Toggle flag between 0 and 1
        self.flag_route = 1 - self.flag_route  # If flag is 0, it becomes 1. If flag is 1, it becomes 0

    def update_visible_satellites(self):
        """Updates the list of visible satellites based on user input."""
        sat_ids_text = self.satellite_ids_input.text()
        try:
            input = [int(sat_id.strip()) for sat_id in sat_ids_text.split(',') if sat_id.strip()]
            if len(self.visible_satellites)>0:
                self.visible_satellites.extend(input)
            else:
                self.visible_satellites = input
            self.satellite_ids_input.clear()
      #      print("1")
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的卫星ID，用逗号分隔")
            return
        # Send the updated visible satellites to Cesium

        self.sendtocesium(self.database.snapshots[self.currenttime].nodes)


    def update_time_label(self):
        """Update the time label when the slider value changes"""
        current_value = self.slider.value()
        current_time = int(round((current_value / 1000.0) * self.totaltime))  # 四舍五入取整

        # 保证时间在有效范围内
        current_time = max(0, min(current_time, self.totaltime))
        self.currenttime = current_time
        self.time_label.setText(f"Time: {current_time}s / {self.totaltime}s")
        self.handle_playback_action(current_time)
        print("1")


    def load_trajectory_filenew(self):
        """加载文件夹下所有txt文件"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Trajectory Files", "", "Text Files (*.xml)"
        )
        if not file_paths:
            return


        db = readns3file.readxml(file_paths[0])  # 获取地面站数据

        self.database = db

        self.totaltime = len(db.snapshots)


        self.stationnum = db.stationnum
        self.satellitenum = db.satellitenum


    def _init_timer(self):
        self.timer = QTimer()
        self.timer.setInterval(50)  # 20fps
        self.timer.timeout.connect(self.advance_playback)
    def toggle_play(self, checked):
            """切换播放状态"""
            self.is_playing = checked
            self.play_btn.setText("⏸ Pause" if checked else "▶ Play")
            if checked:
                if self.slider.value() >= 1000:
                    self.slider.setValue(0)
                self.timer.start()
            else:
                self.timer.stop()
    def pause(self):
        """暂停播放"""
        if self.is_playing:
            self.toggle_play(False)


    def on_positions_updated(self, current_time ):
        snapshot = self.database.snapshots[current_time]
        self.current_paths = snapshot.active_paths

        # 清空旧数据
     #   self.path_combobox.clear()

        self.path_mapping = []

        if not self.current_paths:
            self.path_display.setText("当前时间点无有效路径")
            self.sendtocesium(snapshot.nodes)

            return

        # 生成路径选项
        index = 0
        for dest, path_list in self.current_paths.items():
            for path_idx, path_info in enumerate(path_list):
                path_str = f"{path_info['path'][0]}→{dest} (路径{path_idx + 1})"

                self.path_mapping.append([ path_info['path'][0],dest])
                if dest == self.path_selected[1] and path_info['path'][0] == self.path_selected[0]:
                    paths_select = path_info['path']
                index += 1

        if self.flag_route:
            # route
            if self.path_selected in self.path_mapping:
                self.highlight_selected_path(paths_select)
            else:

                snapshot = self.database.snapshots[self.currenttime]
                Nodes = deepcopy(snapshot.nodes)

                # 重置所有链路
                for node in Nodes:
                    node.linked_array = [-1] * 5

                self.sendtocesium(Nodes)
        else:
            snapshot = self.database.snapshots[current_time]
            self.sendtocesium(snapshot.nodes)

    def on_path_selected(self, index):
        """处理路径选择事件"""
        if not self.path_mapping or index < 0:
            return

        # 获取选中的路径数据
        src, dest = self.path_mapping[index]
        path_info = self.current_paths[dest][0]
        path = path_info['path']

        # 高亮显示选中路径
        self.highlight_selected_path(path)

        # 更新文本显示
        self.show_single_path(path_info)


    def highlight_selected_path(self, path):
        """更新Cesium显示指定路径"""
        snapshot = self.database.snapshots[self.currenttime]
        Nodes = deepcopy(snapshot.nodes)

        # 重置所有链路
        for node in Nodes:
            node.linked_array = [-1] * 5


        src_id =path[0]
        dest_id = path[-1]
        Nodes[src_id].linked_array[0] = path[1]
        Nodes[dest_id].linked_array[0] = path[-2]

        for i in range(1, len(path) - 2):
            Nodes[path[i]].linked_array[1] = path[i + 1]



        self.sendtocesium(Nodes)

    def toggle_sensor_visibility(self):
        """Toggles the visibility of the basicSa conic sensors."""
        self.sensor_visibility = not self.sensor_visibility
        if self.sensor_visibility:
            self.toggle_sensors_btn.setText("Hide Sensors")
        else:
            self.visible_satellites = []
            self.toggle_sensors_btn.setText("Show Sensors")
        # Send a message to Cesium to update the sensor visibility.
        self.sendtocesium(self.database.snapshots[self.currenttime].nodes)  # 调用sendtocesium


    def show_single_path(self, path_info):
        """在文本框中高亮显示单个路径"""
        self.path_display.clear()
        path_str = " → ".join(map(str, path_info['path']))
        self.path_display.append(f"当前选择路径:")
        self.path_display.append(f"  起点: {path_info['path'][0]}")
        self.path_display.append(f"  终点: {path_info['path'][-1]}")
        self.path_display.append(f"  完整路径: {path_str}")
        self.path_display.append(f"  跳数: {len(path_info['path']) - 1}")

    def show_path(self, current_time):
        """显示并更新路径选择"""
        snapshot = self.database.snapshots[current_time]
        self.current_paths = snapshot.active_paths

        # 清空旧数据
        self.path_combobox.clear()


        self.path_mapping = []

        if not self.current_paths:
            self.path_display.setText("当前时间点无有效路径")
            return
        paths_select = []
        # 生成路径选项
        index = 0
        for dest, path_list in self.current_paths.items():
            for path_idx, path_info in enumerate(path_list):
                path_str = f"{path_info['path'][0]}→{dest} (路径{path_idx + 1})"
                self.path_combobox.addItem(path_str)
                self.path_mapping.append((dest, path_info['path'][0]))
                if dest == self.path_selected[0] and path_info['path'][0] == self.path_selected[1]:
                    paths_select = path_info['path']
                index += 1



        if self.flag_route :
            #route
            if self.path_selected in self.path_mapping:
                self.highlight_selected_path(paths_select)
            else:

                snapshot = self.database.snapshots[self.currenttime]
                Nodes = deepcopy(snapshot.nodes)

                # 重置所有链路
                for node in Nodes:
                    node.linked_array = [-1] * 5

                self.sendtocesium(Nodes)
        else:
            snapshot = self.database.snapshots[current_time]
            self.sendtocesium(snapshot.nodes)



    def sendtocesium(self,Nodes):
        stationmun = self.stationnum

        gnd = Nodes[0:stationmun]
        positions = Nodes[stationmun:]


        sat_data = []

        gnd_data = []
        gndnum = len(gnd)

        for j in range(len(gnd)):
            node = gnd[j]
            id = j
            gnd_data.append({
                "id": f"gnd{id}",
                "position": [node.x, node.y, node.z],  # 使用当前节点的坐标
                "link_array": node.linked_array if hasattr(node, 'linked_array') else []
            })

        length = len(positions)

        for i in range(length):
            node = positions[i]
            satid = i

            sat_data.append({
                "id": f"sat{satid}",  # ID从0开始连续编号
                "position": [node.x, node.y, node.z],
                "link_array": node.linked_array if hasattr(node, 'linked_array') else []
            })

        # 构建符合Cesium前端格式的数据
        combined_data = {
            "stations": gnd_data,  # 空地面站列表
            "sats": sat_data,
            "visibleSatellites": self.visible_satellites,  # List of visible basicSa IDs

            "sensorVisibility": self.sensor_visibility  # 添加覆盖角显示状态

        }

        if self.server_thread_loop is None:
            return

        async def _broadcast():
            for client in self.clients.copy():
                try:
                    if not client.closed:
                        await client.send(json.dumps(combined_data))
                except:
                    self.clients.discard(client)

        asyncio.run_coroutine_threadsafe(
            _broadcast(),
            self.server_thread_loop  # 使用保存的循环
        )


    def handle_playback_action(self,current_time):
        """播放/滑动时的响应函数"""

        self.on_positions_updated( current_time)


    def jump_to_time(self):
        """跳转到指定时间"""
        try:
            target_time = int(self.time_input.text())
            if 0 <= target_time <= self.totaltime:
                # 计算对应的滑动条值
                slider_value = int((target_time / self.totaltime) * 1000)
                self.slider.setValue(slider_value)
            else:
                QMessageBox.warning(self, "错误", f"时间必须在0到{self.totaltime}之间")
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的整数时间")

    def advance_playback(self):
        """推进播放进度（按整数步进）"""
        current_time = self.get_current_time()
        new_time = current_time + 1  # 每次前进1秒
        if new_time >= self.totaltime:
            new_time = self.totaltime
            self.toggle_play(False)
        self.slider.setValue(int((new_time / self.totaltime) * 1000))

    def get_current_time(self):
        """获取当前整数时间"""
        return int(round((self.slider.value() / 1000.0) * self.totaltime))
    def start_websocket_server(self):
        def run_server():
            self.server_thread_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.server_thread_loop)

            async def server_main():
                # 使用与示例代码相同的参数结构
                async with websockets.serve(
                    self._websocket_handler,  # 直接传递处理方法
                    "0.0.0.0", 8766,
                    ping_interval=30
                ):
                    print("WebSocket服务器已启动")
                    await asyncio.Future()

            try:
                self.server_thread_loop.run_until_complete(server_main())
            finally:
                self.server_thread_loop.close()
                self.server_thread_loop = None  # 清理

        Thread(target=run_server, daemon=True).start()

    async def _websocket_handler(self, websocket, path):
        """保持两个参数，无需默认值"""
        self.clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        finally:
            self.clients.remove(websocket)

    def run_websocket_client(self):
        """运行WebSocket客户端"""
        asyncio.run(self.websocket_client())

    async def websocket_client(self):
        """WebSocket客户端主循环"""
        async with websockets.connect("ws://localhost:8766") as websocket:
            self.websocket = websocket
            while True:
                await asyncio.sleep(0.1)  # 维持连接


