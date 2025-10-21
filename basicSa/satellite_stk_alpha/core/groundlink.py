# core/link_engine.py
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from collections import defaultdict
from config.settings import SatelliteConfig
import basicSa.calculate_angular.calculate_angular as calculate_angular





class LinkEngine(QObject):
  #  topology_updated = pyqtSignal(dict)  # 新增拓扑更新信号

    def __init__(self, data_hub):
        super().__init__()
        self.data_hub = data_hub
        self.current_params = (0, 0)  # (P, N)
       # self.vector_cls = SatelliteVector  # 可替换计算实现

        # 连接数据更新信号
        self.data_hub.positions_updated.connect(self.on_positions_updated)

    def set_parameters(self, P, N):
        """更新轨道参数"""
        self.current_params = (P, N)




    def on_positions_updated(self, positions):
        """位置更新时的处理"""
        P, N = self.current_params
        if P == 0 or N == 0:
            return

        # 使用向量化计算
        sv = SatelliteVector(P, N)
        sv.load_from_3dview(positions)

        # 批量计算
        angular_vel = sv.calculate_angular_velocity()
        los_mask = angular_vel < SatelliteConfig.ANGULAR_VELOCITY

        # 生成邻接表
        adj_list = defaultdict(list)
        for i in range(len(los_mask)):
            if los_mask[i] and sv._get_right_neighbor(i) is not None:
                orbit = i // N
                plane = i % N
                adj_list[(orbit, plane)].append((orbit + 1, plane))

        # 发射拓扑数据（附加时间戳）
        self.topology_updated.emit({
            "adjacency": adj_list,
            "params": self.current_params,
            "timestamp": self.data_hub.current_time
        })
