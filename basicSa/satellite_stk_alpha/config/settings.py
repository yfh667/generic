# config/settings.py
from datetime import datetime, timezone


class SatelliteConfig:
    # 轨道参数配置
    RAAN_INCREMENT = 10.2 # RAAN递增量 (度)
  #  DEFAULT_ORBIT_ALTITUDE = 550  # 默认轨道高度 (公里)
    ANGULAR_VELOCITY = 0.2  #相对角速度约束
    # 可视化配置
    stationangle = 20
    DEFAULT_COLOR_SCHEME = 'viridis'
    basetime = datetime(2012, 2, 24, 18, 0, 0)
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("Current Satellite Configuration:")
        print(f"RAAN_INCREMENT: {cls.RAAN_INCREMENT}°")
     #   print(f"ORBIT_ALTITUDE: {cls.DEFAULT_ORBIT_ALTITUDE}km")
