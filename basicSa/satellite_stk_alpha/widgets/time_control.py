from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSlider, QLabel, QPushButton, QHBoxLayout, QLineEdit
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from vispy import scene
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton  # 添加QPushButton
from PyQt5.QtCore import Qt, pyqtSignal
from copy import deepcopy
import numpy as np
from vispy import scene
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import numpy as np
import json
import websockets
import asyncio
from threading import Thread
import basicSa.simulator.websocket as websocket
import time
import logging
import basicSa.simulator.sendtocesium as sendtocesium2
import basicSa.simulator.cesiumswitch.heightlight as heightlight
class TimeControl(QWidget):
    time_changed = pyqtSignal(float)  # 发送归一化的时间值[0.0-1.0]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_time = 0.0
        self.is_playing = False
        self._init_ui()
        self._init_timer()
        self._set_style()
        self.flag =1
        self.current_time = 0.0
        self._is_programmatic_update = False  # 新增标志位
        self._init_connections()


        self.stationnum = 8


        self.clients = set()  # 添加此行初始化
        # self.current_time = 0.0
        self.ws_thread = None
     #   self.server_thread_loop = None  # 新增属性
        self.flag_route = 1
        # 启动WebSocket服务器
    #    self.start_websocket_server()

        # 替换原来的WebSocket实现
        self.ws_manager = websocket.WebSocketManager()
        self.ws_manager.start_server()

    def _send_test_data(self):
        """测试数据发送"""
        test_data = {
            "type": "test",
            "timestamp": time.time(),
            "message": "这是一条测试消息"
        }
        self.ws_manager.broadcast(test_data)
        logging.info("发送测试数据")

    def _init_ui(self):
        # 主布局
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 控制行布局
        control_layout = QHBoxLayout()

        # 播放/暂停按钮
        self.play_btn = QPushButton()
        self.play_btn.setCheckable(True)

        control_layout.addWidget(self.play_btn)

        # 时间标签
        self.time_label = QLabel("Time: 0.0s / 0.0s")
        control_layout.addWidget(self.time_label)

        # 输入表单
        self.load_btn3 = QPushButton('luyou ')
        self.load_btn3.clicked.connect(self.route3)
        layout.addWidget(self.load_btn3)  # Add the first load button to the layout


        layout.addLayout(control_layout)

        # 进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)

        layout.addWidget(self.slider)


     # 跳转到某个时间的控件
        jump_layout = QHBoxLayout()

        # 跳转时间输入框
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("Enter time (s)")
        jump_layout.addWidget(self.jump_input)

        # 跳转按钮
        self.jump_btn = QPushButton("Jump to Time")
        self.jump_btn.clicked.connect(self.jump_to_time)
        jump_layout.addWidget(self.jump_btn)

        layout.addLayout(jump_layout)
    def _init_timer(self):
        self.timer = QTimer()
        self.timer.setInterval(50)  # 20fps


    def _init_connections(self):
        """初始化信号连接"""
        # 手动拖动事件
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderPressed.connect(self._on_slider_pressed)

        # 跳转按钮
        self.jump_btn.clicked.connect(self.jump_to_time)

        # 自动播放更新
        self.timer.timeout.connect(self._advance_playback)

    def _on_slider_pressed(self):
        self.toggle_play(False)  # 开始拖动时自动暂停
    def _set_style(self):
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #eee;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QPushButton {
                min-width: 60px;
                padding: 5px;
                border-radius: 4px;
                background: #2ecc71;
                color: white;
            }
            QPushButton:checked {
                background: #e74c3c;
            }
        """)
        self.play_btn.setText("▶ Play")


    def route3(self):
        # Toggle flag between 0 and 1
        self.flag_route = 1 - self.flag_route  # If flag is 0, it becomes 1. If flag is 1, it becomes 0




    def jump_to_time(self):
        """场景1：跳转到指定时间"""
        try:
            target_time = float(self.jump_input.text())
            if 0 <= target_time <= self.max_time:
                self._set_current_time(target_time, is_user_action=False)
            else:
                self._show_input_error()
        except ValueError:
            self._show_input_error()

    def set_time_range(self, max_time):
        """设置时间范围（秒）"""
        self.max_time = max_time
        self.slider.setValue(0)
       # self.update_time_label()
        self.setEnabled(max_time > 0)

    def _on_slider_changed(self, value):
        """统一处理所有滑块变化"""
        if self._is_programmatic_update:
            return

        new_time = value / 1000 * self.max_time
        self._handle_time_change(new_time, is_user_action=True)

    def _handle_time_change(self, new_time, is_user_action):
        """处理时间变化事件"""
        self.current_time = max(0, min(new_time, self.max_time))
        self._update_display()

        # 用户操作立即发射信号，程序操作由_set_current_time处理
        if is_user_action:
            self.time_changed.emit(self.current_time)

            # 如果正在播放，保持同步
            if self.is_playing:
                self._sync_playback_position()

    def _sync_playback_position(self):
        """同步播放位置"""
        self.timer.stop()
        self.timer.start()  # 重置定时器以保持播放节奏

    def _update_display(self):
        """更新界面显示"""
        self.time_label.setText(
            f"Time: {self.current_time:.1f}s / {self.max_time:.1f}s"
        )


    def toggle_play(self, checked=None):
        """改进的播放控制"""
        if checked is None:
            checked = not self.is_playing

        self.is_playing = checked
        self.play_btn.setText("⏸ Pause" if checked else "▶ Play")

        if checked and self.current_time >= self.max_time:
            self._set_current_time(0.0, is_user_action=False)

        self.timer.start() if checked else self.timer.stop()

    def _advance_playback(self):
        if self.max_time == 0:
            return

        # 基于实际时间计算步长（假设定时器50ms间隔）
        step = (self.max_time * 50) / 1000  # 每秒推进max_time的5%
        new_time = min(self.current_time + step, self.max_time)
        self._set_current_time(new_time, is_user_action=False)

    def _set_current_time(self, new_time, is_user_action):
        """安全设置时间"""
        self._is_programmatic_update = True
        try:
            self.current_time = max(0, min(new_time, self.max_time))
            self.slider.setValue(int(self.current_time / self.max_time * 1000))
            self._update_display()

            # 根据操作类型决定是否立即发射信号
            if not is_user_action:
                self.time_changed.emit(self.current_time)
        finally:
            self._is_programmatic_update = False



    def pause(self):
        """暂停播放"""
        if not self.is_playing:
            current = self.slider.value() / 1000 * self.max_time
            self.current_time = current
            self.toggle_play(False)



    def sendtocesium(self,Nodes):


       sendtocesium2.CesiumSender.send_to_cesium(Nodes, self.stationnum, self.ws_manager)

    # stationmun = self.stationnum
    #
    # gnd = Nodes[0:stationmun]
    # positions = Nodes[stationmun:]
    #
    #
    # sat_data = []
    #
    # gnd_data = []
    # gndnum = len(gnd)
    #
    # for j in range(len(gnd)):
    #     node = gnd[j]
    #     id = j
    #     gnd_data.append({
    #         "id": f"gnd{id}",
    #         "position": [node.x, node.y, node.z],  # 使用当前节点的坐标
    #         "link_array": node.linked_array if hasattr(node, 'linked_array') else []
    #     })
    #
    # length = len(positions)
    #
    # for i in range(length):
    #     node = positions[i]
    #     satid = i
    #
    #     sat_data.append({
    #         "id": f"sat{satid}",  # ID从0开始连续编号
    #         "position": [node.x, node.y, node.z],
    #         "link_array": node.linked_array if hasattr(node, 'linked_array') else []
    #     })
    #
    # # 构建符合Cesium前端格式的数据
    # combined_data = {
    #     "stations": gnd_data,  # 空地面站列表
    #     "sats": sat_data
    # }
    #
    #
    # self.ws_manager.broadcast(combined_data)




        # if self.server_thread_loop is None:
        #     return
        #
        # async def _broadcast():
        #     for client in self.clients.copy():
        #         try:
        #             if not client.closed:
        #                 await client.send(json.dumps(combined_data))
        #         except:
        #             self.clients.discard(client)
        #
        # asyncio.run_coroutine_threadsafe(
        #     _broadcast(),
        #     self.server_thread_loop  # 使用保存的循环
        # )

    def highlight_selected_path(self, taget,RAWnodes):
        """更新Cesium显示指定路径"""
        #snapshot = self.database.snapshots[self.currenttime]
        # Nodes = deepcopy(RAWnodes)
        #
        # srcid = taget[0]
        # destid = taget[1]
        # if Nodes[srcid].linked_array[0]==-1:
        #     # 重置所有链路
        #     for node in Nodes:
        #         node.linked_array = [-1] * 5
        #
        #
        # else:
        #     path = []
        #     path.append(srcid)
        #     nexthop = Nodes[srcid].linked_array[0]
        #     flag=1
        #
        #     path.append(nexthop)
        #
        #     while(flag):
        #
        #             for i in range(5,15):
        #                 if Nodes[nexthop].get_value(i)==[-1,-1]:
        #                     flag=0
        #                     break
        #                 elif Nodes[nexthop].get_value(i)[0]==destid:
        #
        #                     port = Nodes[nexthop].get_value(i)[1]
        #                   #  destid =  Nodes[nexthop].get_value(i)[0]
        #                     if port == 0:
        #                         path.append(destid)
        #                         flag = 0
        #                         break
        #                     else:
        #                         nexthop = Nodes[nexthop].linked_array[port]
        #                         path.append(nexthop)
        #                     break
        #
        #
        #     # 重置所有链路
        #     for node in Nodes:
        #         node.linked_array = [-1] * 5
        #
        #     #
        #
        #     src_id = path[0]
        #     dest_id = path[-1]
        #     Nodes[src_id].linked_array[0] = path[1]
        #     Nodes[dest_id].linked_array[0] = path[-2]
        #
        #     for i in range(1, len(path) - 2):
        #         Nodes[path[i]].linked_array[1] = path[i + 1]

        Nodes = heightlight.highlight_selected_path(taget,RAWnodes)

        self.sendtocesium(Nodes)



    #
    def on_positions_updated(self,Nodes):
        if self.flag_route:
            # route
         #   if self.path_selected in self.path_mapping:
                self.highlight_selected_path([0,1],Nodes)



        else:

            self.sendtocesium(Nodes)



    def toggle_animation(self):
        self.play_btn_click.emit()

    # def send_data(self):
    #     # 主线程中安全调用
    #     data = {"time": time.time()}
    #     self.ws_manager.broadcast(data)

    def closeEvent(self, event):
        self.ws_manager.cleanup()  # 窗口关闭时清理
        event.accept()


    def start_websocket_server(self):
        def run_server():
            self.server_thread_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.server_thread_loop)

            async def server_main():
                # 使用与示例代码相同的参数结构
                async with websockets.serve(
                    self._websocket_handler,  # 直接传递处理方法
                    "0.0.0.0", 8765,
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
        async with websockets.connect("ws://localhost:8765") as websocket:
            self.websocket = websocket
            while True:
                await asyncio.sleep(0.1)  # 维持连接
    def is_connected(self):
        return hasattr(self, 'websocket') and not self.websocket.closed
