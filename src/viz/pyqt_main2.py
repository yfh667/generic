# 下面是非阻塞版本，简单来说，可以画出图后，继续在控制台操作，这样可以省去每次读取xml文件的时间


from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg
import draw.basic_functio.basic_motif_option as basic_motif_option
# # ====== 常量（与原始一致） ======
# N = 48   # 每轨道卫星数
# # N = 36   # 每轨道卫星数
# # P = 18   # 轨道平面数
# P = 18   # 轨道平面数
#
#
# TOTAL_SATS = N * P
#
# STATION_GROUPS = {
#     0: {"name": "Group 0", "stations": list(range(0, 4))},
#     1: {"name": "Group 1", "stations": list(range(4, 9))},
#     2: {"name": "Group 2", "stations": [9]},
#     3: {"name": "Group 3", "stations": [10]},
#     4: {"name": "Group 4", "stations": list(range(11, 15))},
#     5: {"name": "Group 5", "stations": list(range(15, 17))},
#     6: {"name": "Group 6", "stations": list(range(17, 20))},
# }
#
# GROUP_COLORS = [
#     '#FF0000',  # 红 (Group 0)
#     '#00FF00',  # 绿 (Group 1)
#     '#0000FF',  # 蓝 (Group 2)
#     '#FFA500',  # 橙 (Group 3)
#     '#800080',  # 紫 (Group 4)
#     '#00FFFF',  # 青 (Group 5)
#     '#FFFF00',  # 黄 (Group 6)
# ]

# def getoption(x1,y1,x2,y2,N):
#
#     if x2-x1==1 and y2-y1==0:
#         return 0
#     elif x2-x1==1 and  y2==(y1 - 1 + N) % N :
#         return 1
#     elif x2-x1==2 and y2==y1:
#         return 2
#     elif x2-x1==1 and y2==(y1 + 1) % N:
#         return 4
#     elif x2-x1==2 and y2==(y1 - 1 + N) % N:
#         return 5

def getoption(x1, y1, x2, y2, N):
    # 先保证 x1 <= x2，把边规范成“从左到右”
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1

    dx = x2 - x1

    if dx == 1 and y2 == y1:
        return 0
    elif dx == 1 and y2 == (y1 - 1 + N) % N:
        return 1
    elif dx == 2 and y2 == y1:
        return 2
    elif dx == 1 and y2 == (y1 + 1) % N:
        return 4
    elif dx == 2 and y2 == (y1 - 1 + N) % N:
        return 5
    else:
        # 不符合你定义的五类，可以打印出来检查
        # print("Unknown edge type:", x1, y1, "->", x2, y2)
        return None

# ========== UI 主类 ==========
class SatelliteViewer(QtWidgets.QWidget):
    def __init__(self, group_data,config):
        pg.setConfigOption('background', 'w')
        super().__init__()
        # 允许空态启动
        self.config = config
        self.N = config.N
        self.P = config.P
        self.total_sats = config.total_sats
        self.station_groups = config.station_groups
        self.group_colors = config.group_colors


        self.group_data = group_data or {}
        self.steps = sorted(self.group_data.keys())
        self.full_steps = self.steps
        self.subrange_steps = None
        self.envelopes = {}
        self._edge_lines = []
        self._edges_line_items = []
        self.envelopesflag = 0  # 默认不显示包络
        self.envelope_regions = getattr(self, "envelope_regions", None)  # 若外部后来设置

        self._init_ui_core()
        self.register_envelope(4, color="deeppink")
        self.register_envelope(0, color="orange")
        #route path
        self.path_by_step = {}  # { step: [node_ids...] }
        self._init_src_dst_markers()
        self._route_items = []  # 当前路由的 QGraphicsPathItem 列表
        self.route_color = '#1E88E5'  # 蓝色
        self.route_width = 2.0
        self.route_offset = 0.08  # 路由整体向法线方向侧移（坐标单位），避免和原链路重叠
        self.route_curve_k = 0.50  # 跨多轨时的弧度系数（和你 draw_curved_edge 一致）

        # 有数据则画首帧
        if self.steps:
            self._apply_steps_and_draw()

    # ---------- 槽：后台解析完成 ----------
    @QtCore.pyqtSlot(dict)
    def on_group_ready(self, gd):
        self.group_data = gd or {}
        self.steps = sorted(self.group_data.keys())
        self.full_steps = self.steps
        self._apply_steps_and_draw()

    # ---------- 槽：某一步的边集合准备好 ----------
    @QtCore.pyqtSlot(int, dict)
    def on_edges_ready(self, step, edges):
        if not hasattr(self, "edges_by_step"):
            self.edges_by_step = {}
        self.edges_by_step[step] = edges
        if self.steps and self.slider.value() == step:
            self.draw_edges(step, edges)


##-----------------route path
    # ===== 路由绘制（按链路规则 + 轻微偏移） =====
    ##-----------------route path
    # ===== 路由绘制（按链路规则 + 轻微偏移） =====
    def _geom_offset(self, x0, y0, x1, y1, offset):
        """按 chord 法向整体侧移直线段的两个端点"""
        dx, dy = (x1 - x0), (y1 - y0)
        L = (dx * dx + dy * dy) ** 0.5
        if L == 0:
            return x0, y0, x1, y1, 0.0, 0.0
        nx, ny = (-dy / L, dx / L)  # 单位法向
        ox, oy = (nx * offset, ny * offset)
        return x0 + ox, y0 + oy, x1 + ox, y1 + oy, ox, oy

    def draw_straight_edge_route(self, x0, y0, x1, y1):
        """
        路由直线：与 draw_straight_edge 规则一致，但整体法向偏移一点、颜色蓝、置顶。
        """
        x0o, y0o, x1o, y1o, _, _ = self._geom_offset(x0, y0, x1, y1, self.route_offset)
        path = QPainterPath()
        path.moveTo(x0o, y0o)
        path.lineTo(x1o, y1o)
        item = QGraphicsPathItem(path)
        pen = pg.mkPen(self.route_color, width=self.route_width, style=QtCore.Qt.SolidLine)
        item.setPen(pen)
        item.setZValue(140)  # 高于普通链路(3)
        self.plot_widget.addItem(item)
        self._route_items.append(item)

    def draw_curved_edge_route(self, x0, y0, x1, y1, curve=None):
        """
        路由曲线：与 draw_curved_edge 完全同款控制点规则，但整体法向偏移一点、颜色蓝、置顶。
        """
        # *** 修改点 ***
        # 优先使用传入的 curve，否则使用 __init__ 中定义的 route_curve_k
        k = curve if curve is not None else self.route_curve_k

        # 1. 先按你的规则计算控制点
        ctrl_x = (x0 + x1) / 2.0
        ctrl_y = (y0 + y1) / 2.0 + k * abs(x1 - x0)  # 使用 k

        # 2. 然后对整个几何（两端点+控制点）做相同的整体法向偏移
        x0o, y0o, x1o, y1o, ox, oy = self._geom_offset(x0, y0, x1, y1, self.route_offset)
        path = QPainterPath()
        path.moveTo(x0o, y0o)
        path.quadTo(ctrl_x + ox, ctrl_y + oy, x1o, y1o)

        item = QGraphicsPathItem(path)
        pen = pg.mkPen(self.route_color, width=self.route_width, style=QtCore.Qt.SolidLine)
        item.setPen(pen)
        item.setZValue(140)
        self.plot_widget.addItem(item)
        self._route_items.append(item)

    def _clear_route(self):
        for it in self._route_items:
            self.plot_widget.removeItem(it)
        self._route_items = []

    def draw_route_path(self, step: int):
        """
        用“画链路”的判定逻辑来画整条 path（同/邻轨=直线；跨多轨=曲线），
        但把每一段统一整体法向偏移一点（route_offset），并使用蓝色画笔。
        """
        self._clear_route()
        path_nodes = self.path_by_step.get(int(step))
        if not path_nodes or len(path_nodes) < 2:
            return

        # 可选：若你想只在当步存在的边上画（严格依赖 edges_by_step），取消注释下面两行：
        # edges = getattr(self, "edges_by_step", {}).get(step, {})
        # def has_edge(a,b): return b in edges.get(a, set())

        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            a = int(u);
            b = int(v)
            x0, y0 = self._all_cols[a], self._all_rows[a]
            x1, y1 = self._all_cols[b], self._all_rows[b]

            # 与 draw_edges 的判定保持一致
            is_curved = abs(self._all_cols[a] - self._all_cols[b]) > 1

            if is_curved:
                # *** 修改点 ***
                # 不再硬编码 curve=0.5，让它自动使用 self.route_curve_k (在 __init__ 中定义为 0.5)
                self.draw_curved_edge_route(x0, y0, x1, y1)
            else:
                self.draw_straight_edge_route(x0, y0, x1, y1)

    def _init_src_dst_markers(self):
        # 画两个空心圆，z 值高于散点/连线
        pen_src = pg.mkPen(color='#1E88E5', width=2)  # 蓝色
        pen_dst = pg.mkPen(color='#1E88E5', width=2, style=QtCore.Qt.DotLine)

        self.src_marker = QtWidgets.QGraphicsEllipseItem()
        self.src_marker.setPen(pen_src);
        self.src_marker.setVisible(False);
        self.src_marker.setZValue(200)
        self.plot_widget.addItem(self.src_marker)

        self.dst_marker = QtWidgets.QGraphicsEllipseItem()
        self.dst_marker.setPen(pen_dst);
        self.dst_marker.setVisible(False);
        self.dst_marker.setZValue(200)
        self.plot_widget.addItem(self.dst_marker)

    def _node_xy(self, node_id: int):
        """把节点 id 转成当前画布坐标（列=plane=x，行=in-plane=y）"""
        x = int(node_id) // self.N if hasattr(self, 'N') else (int(node_id) // N)
        y = int(node_id) % (self.N if hasattr(self, 'N') else N)
        return float(x), float(y)

    def update_src_dst_marker(self, step: int, *, radius=0.48):
        """
        半径单位用“坐标单位”（与你的网格一致）。
        - source: 实线蓝圈
        - destination: 点线蓝圈
        """
        path = self.path_by_step.get(int(step))
        if not path or len(path) < 1:
            self.src_marker.setVisible(False)
            self.dst_marker.setVisible(False)
            return

        src = int(path[0])
        dst = int(path[-1])

        # 放置 src
        sx, sy = self._node_xy(src)
        self.src_marker.setRect(sx - radius, sy - radius, 2 * radius, 2 * radius)
        self.src_marker.setVisible(True)

        # 放置 dst
        dx, dy = self._node_xy(dst)
        self.dst_marker.setRect(dx - radius, dy - radius, 2 * radius, 2 * radius)
        self.dst_marker.setVisible(True)

    def set_paths(self, path_by_step: dict):
        """
        传入 {step: [node_ids...] }。例如：
          { 1: [355,356,...,44], 2: [...], ... }
        """
        self.path_by_step = {int(k): list(map(int, v)) for k, v in (path_by_step or {}).items()}
        # 如果当前 slider 停在某个 step，立即刷新一次蓝圈
        if self.steps:
            self.update_src_dst_marker(self.slider.value())










    ##---------------
    def register_envelope(self, group_id, color="deeppink", need_count=1):
        """确保某个分组有 need_count 个 RectangleEnvelope（支持多个矩形）。"""
        lst = self.envelopes.get(group_id, [])
        while len(lst) < need_count:
            env = RectangleEnvelope(color=color)
            lst.append(env)
            self.plot_widget.addItem(env.rect_item)
        self.envelopes[group_id] = lst
        return lst

    def _hide_all_envelopes(self):
        for lst in self.envelopes.values():
            for env in lst:
                env.hide()

    def show_envelopes_static(self, rects_by_group, expand=0.35, colors=None, persist=True):
        """
        静态一次性显示包络矩形（不随 step 变化）。
        rects_by_group: {gid: [(xmin,xmax,ymin,ymax), ...], ...}
        expand: 外扩量（坐标单位，避免与点重合）
        colors: 可选 {gid: "color"}
        persist: True 时，后续刷新不会覆盖/隐藏这些矩形
        """
        # 标记为“静态持久”
        self._static_envelopes = bool(persist)

        if not rects_by_group:
            self._hide_all_envelopes()
            return

        # 不清空对象，只先全部隐藏，再逐个设置（避免频繁新建/销毁 Item）
        self._hide_all_envelopes()

        for gid, rect_list in rects_by_group.items():
            if not rect_list:
                continue

            # 颜色策略
            if colors and gid in colors:
                c = colors[gid]
            elif gid == 4:
                c = "deeppink"
            elif gid == 0:
                c = "orange"
            else:
                c = "deeppink"

            env_list = self.register_envelope(gid, color=c, need_count=len(rect_list))

            for idx, rect in enumerate(rect_list):
                if rect is None:
                    env_list[idx].hide()
                    continue
                xmin, xmax, ymin, ymax = rect
                if xmin > xmax: xmin, xmax = xmax, xmin
                if ymin > ymax: ymin, ymax = ymax, ymin
                # 外扩一点，避免与点重合
                x = xmin - expand
                y = ymin - expand
                w = (xmax - xmin) + 2 * expand
                h = (ymax - ymin) + 2 * expand
                env_list[idx].set_rect(x, y, w, h)


    # ---------- UI 初始化 ----------
    def _init_ui_core(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)

        # 背景/前景散点
        self.bg_scatter = pg.ScatterPlotItem(size=26, brush='w', pen=None)
        self.bg_scatter.setZValue(5)
        self.plot_widget.addItem(self.bg_scatter)

        self.scatter = pg.ScatterPlotItem(size=17, pen=pg.mkPen(width=0.7, color='#222'))
        self.scatter.setZValue(10)
        self.plot_widget.addItem(self.scatter)

        # 预计算所有卫星坐标，背景点只做一次
        # sat_ids = np.arange(TOTAL_SATS)
        # self._all_cols = sat_ids // N
        # self._all_rows = sat_ids % N

        sat_ids = np.arange(self.total_sats)
        self._all_cols = sat_ids // self.N
        self._all_rows = sat_ids % self.N


        bg_spots = [{'pos': (self._all_cols[i], self._all_rows[i]), 'brush': 'w'}
                    for i in range(self.total_sats)]
        self.bg_scatter.setData(bg_spots)

        # 滑块 —— 空态禁用，待有数据再启用
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.on_slider)
        self.slider.setEnabled(bool(self.steps))
        self.layout.addWidget(self.slider)

        # 子区间控件
        range_layout = QtWidgets.QHBoxLayout()
        self.range_start_input = QtWidgets.QLineEdit()
        self.range_start_input.setPlaceholderText("区间起点")
        self.range_end_input = QtWidgets.QLineEdit()
        self.range_end_input.setPlaceholderText("区间终点")
        self.set_range_btn = QtWidgets.QPushButton("设定区间")
        self.set_range_btn.clicked.connect(self.set_subrange)
        self.exit_range_btn = QtWidgets.QPushButton("退出区间")
        self.exit_range_btn.clicked.connect(self.exit_subrange)
        range_layout.addWidget(self.range_start_input)
        range_layout.addWidget(self.range_end_input)
        range_layout.addWidget(self.set_range_btn)
        range_layout.addWidget(self.exit_range_btn)
        self.layout.addLayout(range_layout)

        # 跳转控件
        jump_layout = QtWidgets.QHBoxLayout()
        self.jump_input = QtWidgets.QLineEdit()
        self.jump_input.setPlaceholderText("跳转到 step")
        self.jump_button = QtWidgets.QPushButton("跳转")
        self.jump_button.clicked.connect(self.on_jump)
        jump_layout.addWidget(self.jump_input)
        jump_layout.addWidget(self.jump_button)
        self.layout.addLayout(jump_layout)

        # 单步按钮
        step_layout = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("<")
        self.next_btn = QtWidgets.QPushButton(">")
        self.prev_btn.clicked.connect(self.step_prev)
        self.next_btn.clicked.connect(self.step_next)
        step_layout.addWidget(self.prev_btn)
        step_layout.addWidget(self.next_btn)
        self.layout.addLayout(step_layout)

        self.label = QtWidgets.QLabel("Waiting for paper_dataresult..." if not self.steps else "")
        self.layout.addWidget(self.label)

        self.plot_widget.setRange(xRange=[-0.5, P-0.5], yRange=[-0.5, N-0.5])
        self.plot_widget.setLabel('bottom', "Orbit Plane Index (P)")
        self.plot_widget.setLabel('left', "Satellite Index in Plane (N)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)

        legend_str = "  ".join([
            f'<span style="color:{self.group_colors[g]};">&#9679;</span> {self.station_groups[g]["name"]}'
            for g in sorted(self.station_groups.keys()) if g < len(self.group_colors)
        ])
        self.legend_label = QtWidgets.QLabel(legend_str)
        self.layout.addWidget(self.legend_label)

    # ---------- 数据就绪后统一启用滑块 & 画首帧 ----------
    def _apply_steps_and_draw(self):
        if not self.steps:
            self.slider.setEnabled(False)
            self.label.setText("Waiting for paper_dataresult...")
            return
        self.slider.setEnabled(True)
        self.slider.setMinimum(self.steps[0])
        self.slider.setMaximum(self.steps[-1])
        self.slider.setValue(self.steps[0])
        self.plot_satellites(self.steps[0])

    # ---------- 基本交互 ----------
    def step_prev(self):
        if not self.steps: return
        val = self.slider.value()
        if val > self.steps[0]:
            self.slider.setValue(val - 1)

    def step_next(self):
        if not self.steps: return
        val = self.slider.value()
        if val < self.steps[-1]:
            self.slider.setValue(val + 1)

    def on_jump(self):
        if not self.steps: return
        val = self.jump_input.text().strip()
        if val.isdigit():
            step = int(val)
            if self.steps[0] <= step <= self.steps[-1]:
                self.slider.setValue(step)

    # ---------- 包络区域（保留你的原逻辑与接口） ----------
    def _ensure_list_store(self):
        """把旧的 {gid: RectangleEnvelope} 迁移为 {gid: [RectangleEnvelope, ...]}（就地兼容）"""
        for gid, val in list(self.envelopes.items()):
            if isinstance(val, RectangleEnvelope):
                self.envelopes[gid] = [val]

    def _hide_all_envelopes(self):
        """隐藏所有矩形（兼容单个/多个存储）"""
        # 先确保是列表存储
        self._ensure_list_store()
        for lst in self.envelopes.values():
            # 兼容万一某处又塞了单个
            if isinstance(lst, RectangleEnvelope):
                lst.hide()
            else:
                for env in lst:
                    env.hide()
    def register_envelope(self, group_id, color="deeppink", need_count=1):
        """
        确保某个分组有 need_count 个 RectangleEnvelope（支持多个）。
        兼容：如果之前是单个对象存储，会自动迁移为列表存储。
        """
        self._ensure_list_store()
        lst = self.envelopes.get(group_id, [])
        if isinstance(lst, RectangleEnvelope):
            # 极端容错：如果别处又放回了单个，转为列表
            lst = [lst]

        while len(lst) < need_count:
            env = RectangleEnvelope(color=color)
            lst.append(env)
            self.plot_widget.addItem(env.rect_item)

        self.envelopes[group_id] = lst
        return lst

    def update_envelopes(self, step, expand=0.3):
        # 若没有外部提供 envelope_regions，直接隐藏
        if not hasattr(self, "envelope_regions") or self.envelope_regions is None:
            for envelope in self.envelopes.values():
                envelope.hide()
            return

        # 查找 step 属于哪个区间
        interval = None
        for (start, end) in self.envelope_regions:
            if start <= step <= end:
                interval = (start, end)
                break

        for gid, envelope in self.envelopes.items():
            rect = None
            if interval is not None:
                region_map = self.envelope_regions[interval]
                rect = region_map.get(gid)
            if rect is not None:
                xmin, xmax, ymin, ymax = rect
                x = xmin - expand
                y = ymin - expand
                w = (xmax - xmin) + 2 * expand
                h = (ymax - ymin) + 2 * expand
                envelope.set_rect(x, y, w, h)
            else:
                envelope.hide()

    # ---------- 绘制 ----------
    def plot_satellites(self, step):
        if not self.group_data or not self.steps:
            return
        current_data = self.group_data.get(int(step), {"groups": {}, "all_mentioned": set()})

        # 彩色前景点
        colors = ['#FFF'] * self.total_sats
        for sat_id in range(self.total_sats):
            seeing_groups = []
            for gid in range(len(self.station_groups)):
                sats = current_data["groups"].get(gid, set())
                if sat_id in sats:
                    seeing_groups.append(gid)
            if seeing_groups:
                assigned_gid = min(seeing_groups)
                colors[sat_id] = self.group_colors[assigned_gid]
        fg_spots = [{'pos': (self._all_cols[i], self._all_rows[i]), 'brush': colors[i]}
                    for i in range(self.total_sats)]
        self.scatter.setData(fg_spots)

        # 连线
        if hasattr(self, "edges_by_step") and step in self.edges_by_step:
            self.draw_edges(step, self.edges_by_step[step])
        else:
            self.draw_edges(step, {})

        self.label.setText(f'Grouped Satellite Visibility (Step {step})')

        # if self.envelopesflag:
        #     self.update_envelopes(step)

        # 静态持久矩形：不让动态 update 覆盖/隐藏
        if getattr(self, "_static_envelopes", False):
            pass
        elif self.envelopesflag:
            self.update_envelopes(step)

        # route show
        self.update_src_dst_marker(step)
        self.draw_route_path(step)

    def on_slider(self, value):
        self.plot_satellites(value)

    def draw_edges(self, step, edges):
        # 清除旧线（保持你原有逻辑；后续可优化为复用）
        if hasattr(self, "_edges_line_items"):
            for item in self._edges_line_items:
                self.plot_widget.removeItem(item)
        self._edges_line_items = []

        # 画线
        for src, dsts in edges.items():
            for dst in dsts:

                opt = getoption(self._all_cols[src], self._all_rows[src],
                                self._all_cols[dst], self._all_rows[dst], N)
                if opt == 2:
                    item = self.draw_curved_edge(
                        self._all_cols[src], self._all_rows[src],
                        self._all_cols[dst], self._all_rows[dst],
                        curve=0.5, dash=False
                    )
                else:
                    item = self.draw_straight_edge(
                        self._all_cols[src], self._all_rows[src],
                        self._all_cols[dst], self._all_rows[dst],
                        dash=False
                    )

                # if abs(self._all_cols[src] - self._all_cols[dst]) > 1:
                #     item = self.draw_curved_edge(
                #         self._all_cols[src], self._all_rows[src],
                #         self._all_cols[dst], self._all_rows[dst],
                #         curve=0.5, dash=False
                #     )
                # else:
                #     item = self.draw_straight_edge(
                #         self._all_cols[src], self._all_rows[src],
                #         self._all_cols[dst], self._all_rows[dst],
                #         dash=False
                #     )
                self._edges_line_items.append(item)

        # 虚线（pending）
        pending_links = getattr(self, "pending_links_by_step", {}).get(step, {})
        for src, dsts in pending_links.items():
            for dst in dsts:
                # if abs(self._all_cols[src] - self._all_cols[dst]) > 1:
                #     item = self.draw_curved_edge(
                #         self._all_cols[src], self._all_rows[src],
                #         self._all_cols[dst], self._all_rows[dst],
                #         curve=0.5, dash=True
                #     )
                # else:
                #     item = self.draw_straight_edge(
                #         self._all_cols[src], self._all_rows[src],
                #         self._all_cols[dst], self._all_rows[dst],
                #         dash=True
                #     )
                opt = getoption(self._all_cols[src], self._all_rows[src],
                                self._all_cols[dst], self._all_rows[dst], N)
                if opt == 2:
                    item = self.draw_curved_edge(
                        self._all_cols[src], self._all_rows[src],
                        self._all_cols[dst], self._all_rows[dst],
                        curve=0.5, dash=True
                    )
                else:
                    item = self.draw_straight_edge(
                        self._all_cols[src], self._all_rows[src],
                        self._all_cols[dst], self._all_rows[dst],
                        dash=True
                    )

                self._edges_line_items.append(item)
        # 3) IG 链路 —— 蓝色虚线
        ig_links = getattr(self, "IG_link_by_step", {}).get(step, {})
        for src, dsts in ig_links.items():
            for dst in dsts:
                # if abs(self._all_cols[src] - self._all_cols[dst]) > 1:
                #     item = self.draw_curved_edge(
                #         self._all_cols[src], self._all_rows[src],
                #         self._all_cols[dst], self._all_rows[dst],
                #         curve=0.5, dash=True, color='#1E88E5'
                #     )
                # else:
                #     item = self.draw_straight_edge(
                #         self._all_cols[src], self._all_rows[src],
                #         self._all_cols[dst], self._all_rows[dst],
                #         dash=True, color='#1E88E5'
                #     )
                opt = getoption(self._all_cols[src], self._all_rows[src],
                                self._all_cols[dst], self._all_rows[dst], N)
                if opt == 2:
                    item = self.draw_curved_edge(
                        self._all_cols[src], self._all_rows[src],
                        self._all_cols[dst], self._all_rows[dst],
                        curve=0.5, dash=True, color='#1E88E5'
                    )
                else:
                    item = self.draw_straight_edge(
                        self._all_cols[src], self._all_rows[src],
                        self._all_cols[dst], self._all_rows[dst],
                        dash=True, color='#1E88E5'
                    )
                self._edges_line_items.append(item)

    def draw_straight_edge(self, x0, y0, x1, y1, dash=False,color=None, width=None):
        path = QPainterPath()
        path.moveTo(x0, y0)
        path.lineTo(x1, y1)
        item = QGraphicsPathItem(path)

        if color is None:
            # 默认行为：虚线=红色，实线=灰色
            color = 'red' if dash else '#888'
        if width is None:
            width = 1.2

        pen = pg.mkPen(
            color=color,              # ✅ 用传进来的 color
            width=width,              # ✅ 用传进来的 width
            style=QtCore.Qt.DashLine if dash else QtCore.Qt.SolidLine
        )
        item.setPen(pen)
        item.setZValue(3)
        self.plot_widget.addItem(item)
        return item

    def draw_curved_edge(self, x0, y0, x1, y1, curve=0.5, dash=False,color=None, width=None):
        path = QPainterPath()
        path.moveTo(x0, y0)
        ctrl_x = (x0 + x1) / 2
        ctrl_y = (y0 + y1) / 2 + curve * abs(x1 - x0)
        path.quadTo(ctrl_x, ctrl_y, x1, y1)
        item = QGraphicsPathItem(path)

        if color is None:
            # 默认行为：虚线=红色，实线=灰色
            color = 'red' if dash else '#888'
        if width is None:
            width = 1.2

        pen = pg.mkPen(
            color=color,              # ✅ 用传进来的 color
            width=width,              # ✅ 用传进来的 width
            style=QtCore.Qt.DashLine if dash else QtCore.Qt.SolidLine
        )
        item.setPen(pen)
        item.setZValue(3)
        self.plot_widget.addItem(item)
        return item

    # ---------- 子区间 ----------
    def set_subrange(self):
        if not self.full_steps:
            QtWidgets.QMessageBox.warning(self, "范围无效", "尚无数据")
            return
        try:
            start = int(self.range_start_input.text())
            end = int(self.range_end_input.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "输入错误", "请输入有效的区间起止 step")
            return
        steps = [s for s in self.full_steps if start <= s <= end]
        if not steps:
            QtWidgets.QMessageBox.warning(self, "范围无效", "未找到该区间内的数据 step")
            return
        self.subrange_steps = steps
        self.steps = steps
        self.slider.setMinimum(self.steps[0])
        self.slider.setMaximum(self.steps[-1])
        self.slider.setValue(self.steps[0])
        self.plot_satellites(self.steps[0])

    def exit_subrange(self):
        if self.subrange_steps is not None:
            self.subrange_steps = None
            self.steps = self.full_steps
            self.slider.setMinimum(self.steps[0])
            self.slider.setMaximum(self.steps[-1])
            self.slider.setValue(self.steps[0])
            self.plot_satellites(self.steps[0])


# ====== 包络矩形（保留） ======
class RectangleEnvelope:
    def __init__(self, color="deeppink", width=2, style=QtCore.Qt.DashLine, z=100):
        self.rect_item = QtWidgets.QGraphicsRectItem()
        pen = pg.mkPen(color=color, width=width, style=style)
        self.rect_item.setPen(pen)
        self.rect_item.setZValue(z)
        self.rect_item.setVisible(False)

    def set_rect(self, x, y, w, h):
        self.rect_item.setRect(x, y, w, h)
        self.rect_item.setVisible(True)

    def hide(self):
        self.rect_item.setVisible(False)


# ====== 后台线程：逐 step 产生边；也可顺带解析 XML ======
class EdgeWorker(QtCore.QObject):
    edges_ready = QtCore.pyqtSignal(int, dict)  # (step, edges)
    group_ready = QtCore.pyqtSignal(dict)       # group_data（可选）
    finished = QtCore.pyqtSignal()

    def __init__(self, start_ts, end_ts, config, xml_file=None, parser_func=None):
        super().__init__()
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.config = config
        self.N = config.N
        self.P = config.P
        self.xml_file = xml_file
        self.parser_func = parser_func
        self._running = True

    @QtCore.pyqtSlot()
    def run(self):
        # 1) 解析 XML（如果传入）
        if self.xml_file and self.parser_func:
            gd = self.parser_func(self.xml_file, self.start_ts, self.end_ts)
            self.group_ready.emit(gd)

        # 2) 逐 step 生成 edges
        N, P = self.N, self.P
        for step in range(self.start_ts, self.end_ts):
            if not self._running:
                break
            edges = {}
            for i in range(P - 1):
                for j in range(14, 32):
                    nownode = i * N + j
                    if i + 1 <= 17 and j + 2 <= 32:
                        next_node1 = (i + 1) * N + j + 2
                        edges.setdefault(nownode, set()).add(next_node1)
                    upnodes = i * N + (j + 1) % N
                    edges.setdefault(nownode, set()).add(upnodes)
                    downnodes = i * N + (j - 1 + N) % N
                    edges.setdefault(nownode, set()).add(downnodes)
            for i in range(P - 1):
                for j in range(0, 14):
                    nownode = i * N + j
                    next_node1 = (i + 1) * N + j
                    edges.setdefault(nownode, set()).add(next_node1)

            self.edges_ready.emit(step, edges)

        self.finished.emit()

    def stop(self):
        self._running = False
