# widgets/plot2d.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
from core.render_2d import NetworkRenderer


class Satellite2DView(QWidget):
    def __init__(self, link_engine, parent=None):
        super().__init__(parent)
        self.link_engine = link_engine
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
      #  pass we negel
        self.link_engine.topology_updated.connect(self._handle_topology_update)

    def _handle_topology_update(self, data):


        adjacency = data['adjacency']
        N = data["params"][1]
        P = data["params"][0]
        # for i in range(P):
        #     a
        for i in range(P-1):
            if adjacency[(i,0)] :
                adjacency[(i, N)].append((i+1, N))



        """处理拓扑更新"""
        self.renderer.update_topology(
            N=data["params"][1],
            P=data["params"][0],
            adjacency=data["adjacency"]
        )
