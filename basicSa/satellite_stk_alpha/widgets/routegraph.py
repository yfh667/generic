# widgets/plot2d.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
from core.render_2d import NetworkRenderer
from contrib.route import SatelliteRouter


from basicSa.los import  Sat2Gnd
class Route2DView(QWidget):
    def __init__(self, link_engine,stationData, parent=None):
        super().__init__(parent)
        self.link_engine = link_engine
        self.StationData = stationData

        self.router = None  # 新增路由计算器

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        # 创建绘图区域
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)

        # 初始化渲染器
        self.renderer = NetworkRenderer(self.plot)

    def _connect_signals(self):
        # 连接拓扑更新信号
        self.link_engine.topology_updated.connect(self._handle_topology_update)

    def _handle_topology_update(self, data):
        """处理拓扑更新"""



        stationmaneger  = self.StationData.stations
       # node1 = stationmaneger[0].current_position

        P = data["params"][0]
        N = data["params"][1]
        base_adj = data["adjacency"]



        satellites = []
        for i in range(len(data["satellites"])):
            satellites.append(data["satellites"][i][0])



        # if route_adj:
        # # 更新渲染器
        #     self.renderer.update_topology(
        #         P=P,
        #         N=N,
        #         adjacency=route_adj
        #     )
