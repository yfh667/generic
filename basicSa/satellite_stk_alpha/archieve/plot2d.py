
# widgets/plot2d.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg




class Satellite2DView(QWidget):
    def __init__(self, link_engine, parent=None):
        super().__init__(parent)


        self.link_engine = link_engine
        self.link_engine.topology_updated.connect(self.update_topology_sim)

        self.current_params = (0, 0)
        self.active_sats = set()
        self.is_animating = False


        # 存储图形对象
        self.nodes = {}
        self.lines = []
        self.labels = []
        self.scatter = None


        self.base_font = QFont("SimHei", 8)  # 初始化基础字体
        self.label_style = {'color': '#2c3e80', 'font-size': '8pt'}  # 使用CSS样式

        self._init_ui()
        self._init_view_scaling()


    def _init_view_scaling(self):
        """初始化视图缩放处理"""
        # 连接视图范围变化信号
        self.plot.sigRangeChanged.connect(self._update_font_sizes)
        # 记录初始视图范围
        self.original_view = self.plot.viewRect()

    def _update_font_sizes(self):
        """修正字体更新逻辑"""
        current_view = self.plot.viewRect()
        width_ratio = current_view.width() / self.original_view.width()

        # 计算新字号
        new_size = min(24, max(6, int(self.base_font.pointSize() / width_ratio)))

        # 更新所有标签
        for label in self.labels:
            # 通过HTML更新字体大小
            text = label.textItem.toPlainText()
            label.setHtml(
                f'<span style="font-family: SimHei; font-size: {new_size}pt; '
                f'color: {self.label_style["color"]};">{text}</span>'
            )
    def _init_ui(self):
        # 配置PyQtGraph
        pg.setConfigOptions(antialias=True, foreground='k', background='w')

        # 创建主布局
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 创建绘图部件
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=False)
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.enableAutoRange()

        #隐藏坐标周
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')


        # 配置中文字体
        font = QFont()
        font.setFamily("SimHei")
        font.setPointSize(10)
        self.plot.getAxis("left").setStyle(tickFont=font)
        self.plot.getAxis("bottom").setStyle(tickFont=font)

        layout.addWidget(self.plot)

        # 初始化散点图
        self._init_scatter()

    def _init_scatter(self):
        """初始化散点图对象"""
        if self.scatter is None:
            self.scatter = pg.ScatterPlotItem(
                size=18,
                pen=pg.mkPen(QColor(30, 144, 255), width=2),
                brush=pg.mkBrush(QColor(135, 206, 250)),
                symbol='d',
                hoverable=True
            )
            self.plot.addItem(self.scatter)
        else:
            self.scatter.clear()

    def update_topology(self, N, P, adjacency):
        """更新网络拓扑"""
     #   self.current_params = (N, P)
        self._clear_drawing()
        pos = self._generate_positions(N, P)
        self._draw_nodes(pos)
        self._draw_connections(pos, adjacency)
        self._draw_labels(pos, N)
        self._set_view_range(N, P)

    def _generate_positions(self, N, P):
        """生成节点位置"""
        pos = {}

        for i in range(P):
            for j in range(N + 1):

                pos[(i, j)] = (i, -j)
        return pos

    def _draw_nodes(self, pos):
        """绘制节点"""
        points = []
        for (x, y) in pos.values():
            points.append({'pos': (x, y), 'data': 1})
        self.scatter.setData(points)

    def _draw_connections(self, pos, adjacency):
        """绘制连接线"""
        for node, neighbors in adjacency.items():
            if node not in pos:
                continue
            x1, y1 = pos[node]
            for neighbor in neighbors:
                if neighbor in pos:
                    x2, y2 = pos[neighbor]
                    line = pg.PlotCurveItem(
                        x=[x1, x2], y=[y1, y2],
                        pen=pg.mkPen(QColor(231, 76, 60), width=1.5)
                    )
                    self.plot.addItem(line)
                    self.lines.append(line)

    def _draw_labels(self, pos, N):
        """修正后的标签绘制方法"""
        for (i, j), (x, y) in pos.items():
            text = f"({i},{j})" if j < N else f"({i},0_COPY)"

            # 使用HTML样式设置字体
            label = pg.TextItem(
                html=f'<span style="font-family: SimHei; font-size: {self.base_font.pointSize()}pt; color: {self.label_style["color"]};">{text}</span>',
                anchor=(0.5, 1)
            )
            label.setPos(x, y - 0.1)
            self.plot.addItem(label)
            self.labels.append(label)

    def _set_view_range(self, N, P):
        """设置视图范围"""
        self.plot.setXRange(-1, P)
        self.plot.setYRange(-N - 1, 1)

    def _clear_drawing(self):
        """清除图形元素"""
        # 清除线条
        for line in self.lines:
            self.plot.removeItem(line)
        self.lines.clear()

        # 清除标签
        for label in self.labels:
            self.plot.removeItem(label)
        self.labels.clear()

        # 清空散点数据（使用兼容方式）
        if self.scatter is not None:
            self.scatter.setData([])  # 替换clear()方法
        else:
            self._init_scatter()  # 确保scatter存在

    def highlight_nodes(self, active_nodes):
        """高亮指定节点"""
        colors = []
        points = self.scatter.data

        for p in points:
            pos = (round(p['pos'][0]), round(-p['pos'][1]))
            if pos in active_nodes:
                colors.append((231, 76, 60, 255))  # 红色
            else:
                colors.append((52, 152, 219, 255))  # 蓝色

        self.scatter.setBrush(colors)

    def update_topology_sim(self, data):
        """纯显示更新方法"""
        adj_list = data["adjacency"]
        P, N = data["params"]

        self.update_topology(N, P, adj_list)







