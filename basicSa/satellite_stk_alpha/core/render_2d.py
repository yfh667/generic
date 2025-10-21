# core/renderer/network_renderer.py
import pyqtgraph as pg
from PyQt5.QtGui import QFont, QColor

class NetworkRenderer:
    def __init__(self, plot_widget):
        self.plot = plot_widget
        self._init_components()
        self._init_styles()
        self._setup_plot()

    def _init_components(self):
        """初始化图形组件"""
        self.scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.scatter)
        self.lines = []
        self.labels = []

    def _init_styles(self):
        """初始化样式配置"""
        self.node_style = {
            'size': 18,
            'pen': pg.mkPen(QColor(30, 144, 255), width=2),
            'brush': pg.mkBrush(QColor(135, 206, 250))
        }
        self.line_style = {
            'color': QColor(231, 76, 60),
            'width': 1.5
        }
        self.label_font = QFont("SimHei", 8)
        self.label_color = '#2c3e80'

    def _setup_plot(self):
        """配置绘图区域"""
        self.plot.setAspectLocked(True)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.setBackground('w')

    def update_topology(self, N, P, adjacency):
        """更新网络拓扑"""
        self._clear()
        pos = self._generate_positions(N, P)
        self._draw_nodes(pos)
        self._draw_connections(pos, adjacency)
        self._draw_labels(pos, N)
        self._set_view_range(P, N)

    def _generate_positions(self, N, P):
        """生成网格位置"""
        return {(i, j): (i, -j) for i in range(P) for j in range(N + 1)}

    def _draw_nodes(self, positions):
        """绘制节点"""
        points = [{'pos': pos, **self.node_style} for pos in positions.values()]
        self.scatter.setData(points)

    def _draw_connections(self, positions, adjacency):
        """绘制连接线"""
        for node, neighbors in adjacency.items():
            if node not in positions:
                continue
            x1, y1 = positions[node]
            for neighbor in neighbors:
                if neighbor in positions:
                    x2, y2 = positions[neighbor]
                    line = pg.PlotCurveItem(
                        x=[x1, x2], y=[y1, y2],
                        pen=pg.mkPen(**self.line_style)
                    )
                    self.plot.addItem(line)
                    self.lines.append(line)

    def _draw_labels(self, positions, N):
        """绘制标签"""
        for (i, j), (x, y) in positions.items():
            text = f"({i},{j})" if j < N else f"({i},0_COPY)"
            label = pg.TextItem(
                html=f'<span style="font-family: SimHei; font-size: 8pt; color: {self.label_color};">{text}</span>',
                anchor=(0.5, 1)
            )
            label.setPos(x, y - 0.1)
            self.plot.addItem(label)
            self.labels.append(label)

    def _set_view_range(self, P, N):
        """设置视图范围"""
        self.plot.setXRange(-1, P)
        self.plot.setYRange(-N-1, 1)

    def _clear(self):
        """清除图形元素"""
        self.scatter.clear()
        for item in self.lines + self.labels:
            self.plot.removeItem(item)
        self.lines.clear()
        self.labels.clear()
