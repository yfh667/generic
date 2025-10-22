from dataclasses import dataclass
import numpy as np
from typing import List, Dict
import basicSa.utilis.Node as Node

class Satellite:
    def __init__(self, sat_id: int, trajectory: List[Node]):
        # 确保这里接收的是 sat_id
        self.sat_id = sat_id
        # 将轨迹存储为时间->节点的映射
        self.trajectory = {node.time: node for node in trajectory}

    def __getitem__(self, time: float) -> Node:
        """使得可以通过时间直接访问节点"""
        return self.trajectory.get(time, None)
    def __len__(self):
        """返回轨迹的长度，即节点数量"""
        return len(self.trajectory)


# 确保SatelliteManager中的 add_trajectory 正常工作
class SatelliteManager:
    def __init__(self):
        self.satellites = {}  # {sat_id: Satellite}
        self.max_time = 0.0
        self.current_time = 0.0  # 新增当前时间属性

    def get_sorted_ids(self):
        """获取已加载的排序后的卫星ID列表"""
        return sorted(self.satellites.keys())

    def step_simulation(self, step=0.1):
        """更新模拟时间"""
        self.current_time += step
        if self.current_time > self.max_time:
            self.current_time = 0.0

    def add_trajectory(self, sat_id: int, trajectory: List[Node]):
        """添加卫星轨迹并更新最大时间"""
        if sat_id not in self.satellites:
            # 在这里创建Satellite实例时，确保传入了正确的参数
            self.satellites[sat_id] = Satellite(sat_id=sat_id, trajectory=trajectory)
            self.max_time = max(self.max_time, max(p.time for p in trajectory))

    def find_satellite_at_time(self, target_time=100):
        """直接查找所有在特定时间（如100）有数据的卫星"""
        result = []
        for sat_id, satellite in self.satellites.items():
            node = satellite[target_time]  # 使用__getitem__访问
            if node is not None:
                result.append((sat_id, node))
        return result




# @dataclass
# class Satellite:
#     id: int
#     trajectory: List[Node.Node]
#     current_index: int = 0
#
# class SatelliteManager:
#     def __init__(self):
#         self.satellites = {}  # {sat_id: Satellite}
#         self.max_time = 0.0
#         self.current_time = 0.0  # 新增当前时间属性
#
#
#         def get_sorted_ids(self):
#             """获取已加载的排序后的卫星ID列表"""
#             return sorted(self.satellites.keys())
#     def step_simulation(self, step=0.1):
#         """更新模拟时间"""
#         self.current_time += step
#         if self.current_time > self.max_time:
#             self.current_time = 0.0
#     def add_trajectory(self, sat_id: int, trajectory: List[Node.Node]):
#         if sat_id not in self.satellites:
#             self.satellites[sat_id] = Satellite(id=sat_id, trajectory=trajectory)
#             self.max_time = max(self.max_time, max(p.time for p in trajectory))


class StationManager:
    def __init__(self):
        self.stations: Dict[int, Station] = {}  # {sta_id: Station}
        self.next_sta_id = 0  # 自增ID生成器

    def add_ground_station(self, x: float, y: float, z: float,angle:float) -> int:
        """添加新地面站返回分配ID"""
        sta_id = self.next_sta_id
        trajectory = [Node.Node(time=0.0, x=x, y=y, z=z,angle=angle,nodeid=sta_id)]  # 固定位置轨迹
        self.stations[sta_id] = Station(sta_id, trajectory)
        self.next_sta_id += 1
        return sta_id

    def get_station_positions(self) -> Dict[int, tuple]:
        """获取所有地面站最新坐标"""
        return {
            sta_id: (sta.trajectory[-1].x, sta.trajectory[-1].y, sta.trajectory[-1].z)
            for sta_id, sta in self.stations.items()
        }

    def validate_coordinates(self, x: float, y: float, z: float) -> bool:
        """坐标有效性验证"""
        radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return 6370000 - 100000 < radius < 6370000 + 100000  # 允许±100公里误差


class Station:
    def __init__(self, sta_id: int, trajectory: list[Node]):
        self.id = sta_id
        self.trajectory = trajectory  # 固定位置时只有一个节点

    @property
    def current_position(self) -> Node:
        return self.trajectory[-1]
