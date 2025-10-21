# @dataclass



class TimeSnapshot:
    """单个时间点的网络快照（完全按照需求定义）"""
    def __init__(self, timestamp_str, station_snapshot, satellite_dsnapshot):
        self.timestamp = timestamp_str  # ISO 8601时间字符串
      #  self.nodes = nodes_dict         # 节点ID到节点对象的映射

        self.stations = station_snapshot
        self.satellites = satellite_dsnapshot
       # self.active_paths = active_paths_dict  # 路径字典

class TimeDatabase:
    """时间序列数据库"""
    def __init__(self):
        self.snapshots = []  # 存储TimeSnapshot对象的列表
        self.stationnum = 0
        self.satellitenum = 0
    # Implement __getitem__ to allow indexing
    def __getitem__(self, index):
        return self.snapshots[index]
