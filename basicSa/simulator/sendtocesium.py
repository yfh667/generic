# cesium_sender.py
import json
from typing import List, Any


class CesiumSender:
    @staticmethod
    def send_to_cesium(nodes: List[Any], station_num: int, ws_manager):
        """
        发送数据到Cesium

        参数:
            nodes: 节点列表
            station_num: 地面站数量
            ws_manager: WebSocket管理器实例
        """
        gnd = nodes[:station_num]
        positions = nodes[station_num:]

        # 准备地面站数据
        gnd_data = [
            {
                "id": f"gnd{j}",
                "position": [node.x, node.y, node.z],
                "link_array": getattr(node, 'linked_array', [])
            }
            for j, node in enumerate(gnd)
        ]

        # 准备卫星数据
        sat_data = [
            {
                "id": f"sat{i}",
                "position": [node.x, node.y, node.z],
                "link_array": getattr(node, 'linked_array', [])
            }
            for i, node in enumerate(positions)
        ]

        # 构建完整数据包
        combined_data = {
            "stations": gnd_data,
            "sats": sat_data
        }

        # 通过WebSocket发送
        ws_manager.broadcast(combined_data)
