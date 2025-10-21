from vispy import scene
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton  # 添加QPushButton
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import numpy as np
from vispy import scene
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import numpy as np
import json
import websockets
import asyncio
from threading import Thread

class PostCesiumDView(QWidget):
    play_btn_click = pyqtSignal()  # 添加缺失的信号
    satellite_positions_3d_signal = pyqtSignal(dict)  # 信号传递一个列表，包含卫星的2D坐标

    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.markers = {}
        self.clients = set()  # 添加此行初始化



        self.current_time = 0.0
        self.ws_thread = None
        self.server_thread_loop = None  # 新增属性

        # 启动WebSocket服务器
        self.start_websocket_server()

    def is_connected(self):
        return hasattr(self, 'websocket') and not self.websocket.closed

    def print_connection_status(self):
        print("Connection status:",
              "Exists" if hasattr(self, 'websocket') else "No connection",
              "Open" if self.is_connected() else "Closed")

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



    def on_positions_updated(self, positions,gnd):
        """将位置数据转换为Cesium格式并通过WebSocket发送（无地面站版本）"""
        sat_data = []

        gnd_data = []
        gndnum = len(gnd)
        for node in gnd:  # 变量名错误修复
            id = node.nodeid
            gnd_data.append({
                "id": f"gnd{id}",
                "position": [node.x, node.y, node.z],  # 使用当前节点的坐标
                "link_array": node.linked_array if hasattr(node, 'linked_array') else []
            })

        length = len(positions)

        for i in range(length):
            node = positions[i][0]
            satid = node.nodeid - 1

            sat_data.append({
                "id": f"sat{satid}",  # ID从0开始连续编号
                "position": [node.x, node.y, node.z],
                "link_array": node.linked_array if hasattr(node, 'linked_array') else []
            })

        # 构建符合Cesium前端格式的数据
        combined_data = {
            "stations": gnd_data,  # 空地面站列表
            "sats": sat_data
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

    def toggle_animation(self):
        self.play_btn_click.emit()

