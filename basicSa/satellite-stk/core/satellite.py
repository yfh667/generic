from dataclasses import dataclass
import numpy as np
from typing import List, Dict
import basicSa.utilis.Node as Node

# @dataclass
# class TrajectoryPoint:
#     time: float
#     x: float
#     y: float
#     z: float


@dataclass
class Satellite:
    id: int
    trajectory: List[Node.Node]
    current_index: int = 0





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
    def add_trajectory(self, sat_id: int, trajectory: List[Node.Node]):
        if sat_id not in self.satellites:
            self.satellites[sat_id] = Satellite(id=sat_id, trajectory=trajectory)
            self.max_time = max(self.max_time, max(p.time for p in trajectory))
