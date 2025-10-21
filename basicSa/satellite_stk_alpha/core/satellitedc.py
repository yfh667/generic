from PyQt5.QtCore import QObject, pyqtSignal
from basicSa.los import  Sat2Gnd

class SatelliteSimulator(QObject):  # 继承自QObject
    positions_updated = pyqtSignal(float,dict, list)  # 保持字典参数类型
    def __init__(self, manager, ):
        super().__init__()  # 初始化QObject

        self.manager = manager

        self._current_time = 0.0

        self._position_cache = {}

        self._ground_cache = []

    @property
    def current_time(self):
        return self._current_time

    @current_time.setter
    def current_time(self, value):
        self._current_time = value


        self._update_positions()

        self._notify_subscribers()

    def _update_positions(self):
        """统一更新所有卫星位置"""
        self._position_cache.clear()
        for sat in self.manager.satellites.values():
            idx = next((i for i, p in enumerate(sat.trajectory)
                      if p.time >= self.current_time), 0)
            self._position_cache[sat.id] = (
                sat.trajectory[idx],  # 当前时刻
                sat.trajectory[idx+1] # 下一时刻（用于计算）
            )

    def _update_stations(self,node):
       # pass
        self._ground_cache.append(node)


    def _notify_subscribers(self):
        """通过信号通知订阅者"""
        self.positions_updated.emit(   self._current_time,self._position_cache, self._ground_cache)  # 发射信号
