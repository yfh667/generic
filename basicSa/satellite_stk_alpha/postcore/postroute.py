# widgets/plot2d.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
from core.render_2d import NetworkRenderer
from contrib.route import SatelliteRouter


from basicSa.los import  Sat2Gnd
class PostRouteDView(QWidget):
    def __init__(self, link_engine,stationData, parent=None):
        super().__init__(parent)
    #    self.link_engine = link_engine
     #   self.StationData = stationData

        self.router = None  # 新增路由计算器

        self._init_ui()
    #    self._connect_signals()

    def _init_ui(self):
        # 创建绘图区域
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)

        # 初始化渲染器
        self.renderer = NetworkRenderer(self.plot)

    # def _connect_signals(self):
    #     # 连接拓扑更新信号
    #     self.link_engine.topology_updated.connect(self._handle_topology_update)



    def _handle_topology_update(self, data):
        """处理拓扑更新"""

        P = 10
        N = 10

        base_adj = data["adjacency"]

      #  visibility = Sat2Gnd.Ground_Sat(node1, node2)

        # 初始化路由器
        self.router = SatelliteRouter(P, N)


        self.router.build_topology(base_adj)

        # 计算示例路由（0号节点到31号节点）


#这里的路由，是sat-sat的路由
      #  basicSa = data["basicSa"]
      #  end = data["end"]


        route_adj =[]


        # if basicSa!=-1 and end!=-1:
        #     path = self.router.find_path(basicSa, end)
        #     route_adj = self.router.generate_route_adj(path)
        #     next we need write  the route table
        #



        if route_adj:
        # 更新渲染器
            self.renderer.update_topology(
                P=P,
                N=N,
                adjacency=route_adj
            )

