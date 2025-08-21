# 下面是非阻塞版本，简单来说，可以画出图后，继续在控制台操作，这样可以省去每次读取xml文件的时间


from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg

# ====== 常量（与原始一致） ======
N = 36   # 每轨道卫星数
P = 18   # 轨道平面数
TOTAL_SATS = N * P

STATION_GROUPS = {
    0: {"name": "Group 0", "stations": list(range(0, 4))},
    1: {"name": "Group 1", "stations": list(range(4, 9))},
    2: {"name": "Group 2", "stations": [9]},
    3: {"name": "Group 3", "stations": [10]},
    4: {"name": "Group 4", "stations": list(range(11, 15))},
    5: {"name": "Group 5", "stations": list(range(15, 17))},
    6: {"name": "Group 6", "stations": list(range(17, 20))},
}

GROUP_COLORS = [
    '#FF0000',  # 红 (Group 0)
    '#00FF00',  # 绿 (Group 1)
    '#0000FF',  # 蓝 (Group 2)
    '#FFA500',  # 橙 (Group 3)
    '#800080',  # 紫 (Group 4)
    '#00FFFF',  # 青 (Group 5)
    '#FFFF00',  # 黄 (Group 6)
]

# ========== UI 主类 ==========
class SatelliteViewer(QtWidgets.QWidget):
    def __init__(self, group_data):
        pg.setConfigOption('background', 'w')
        super().__init__()
        # 允许空态启动
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
        兼容以下输入：
          - {gid: [(xmin,xmax,ymin,ymax), ...]}
          - {gid: (box1, box2_or_none)}   # 例如 calc_envelope_for_group 的返回
          - {gid: (None, None)} 或 {gid: None} 也能接受（等同于不画）
        """
        # 标记持久：后续刷新不覆盖
        self._static_envelopes = bool(persist)

        # 工具函数：把各种形式归一为“列表[box,...]”，过滤 None
        def _as_box_list(v):
            """
            v 可能是：
              - None
              - (xmin,xmax,ymin,ymax)
              - [ (....), (....), ... ]
              - (box1, box2_or_none)   # 典型：t1, t2
            返回：list[ (xmin,xmax,ymin,ymax), ... ]
            """
            if v is None:
                return []
            # 单个 box 4 元组
            if isinstance(v, (tuple, list)) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
                return [tuple(v)]
            # 二元组 (box1, box2_or_none)
            if isinstance(v, (tuple, list)) and len(v) == 2:
                out = []
                for b in v:
                    if b and isinstance(b, (tuple, list)) and len(b) == 4:
                        out.append(tuple(b))
                return out
            # 列表/可迭代：过滤掉 None
            if isinstance(v, (tuple, list)):
                out = []
                for b in v:
                    if b and isinstance(b, (tuple, list)) and len(b) == 4:
                        out.append(tuple(b))
                return out
            return []

        # 工具：修正顺序并做一点裁剪（防手抖）
        def _fix_box(box):
            xmin, xmax, ymin, ymax = box
            if xmin > xmax: xmin, xmax = xmax, xmin
            if ymin > ymax: ymin, ymax = ymax, ymin
            # 可选裁剪到画布范围（不想裁剪可注释掉）
            xmin = max(0, min(xmin, P - 1))
            xmax = max(0, min(xmax, P - 1))
            ymin = max(0, min(ymin, N - 1))
            ymax = max(0, min(ymax, N - 1))
            return xmin, xmax, ymin, ymax

        # 先把所有旧 envelope 隐掉，但不销毁
        self._hide_all_envelopes()
        self._ensure_list_store()

        if not rects_by_group:
            return

        for gid, raw in rects_by_group.items():
            rect_list = [_fix_box(b) for b in _as_box_list(raw)]
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

            for idx, (xmin, xmax, ymin, ymax) in enumerate(rect_list):
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
        sat_ids = np.arange(TOTAL_SATS)
        self._all_cols = sat_ids // N
        self._all_rows = sat_ids % N
        bg_spots = [{'pos': (self._all_cols[i], self._all_rows[i]), 'brush': 'w'}
                    for i in range(TOTAL_SATS)]
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

        self.label = QtWidgets.QLabel("Waiting for data..." if not self.steps else "")
        self.layout.addWidget(self.label)

        self.plot_widget.setRange(xRange=[-0.5, P-0.5], yRange=[-0.5, N-0.5])
        self.plot_widget.setLabel('bottom', "Orbit Plane Index (P)")
        self.plot_widget.setLabel('left', "Satellite Index in Plane (N)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)

        legend_str = "  ".join([
            f'<span style="color:{GROUP_COLORS[g]};">&#9679;</span> {STATION_GROUPS[g]["name"]}'
            for g in sorted(STATION_GROUPS.keys()) if g < len(GROUP_COLORS)
        ])
        self.legend_label = QtWidgets.QLabel(legend_str)
        self.layout.addWidget(self.legend_label)

    # ---------- 数据就绪后统一启用滑块 & 画首帧 ----------
    def _apply_steps_and_draw(self):
        if not self.steps:
            self.slider.setEnabled(False)
            self.label.setText("Waiting for data...")
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
        colors = ['#FFF'] * TOTAL_SATS
        for sat_id in range(TOTAL_SATS):
            seeing_groups = []
            for gid in range(len(STATION_GROUPS)):
                sats = current_data["groups"].get(gid, set())
                if sat_id in sats:
                    seeing_groups.append(gid)
            if seeing_groups:
                assigned_gid = min(seeing_groups)
                colors[sat_id] = GROUP_COLORS[assigned_gid]
        fg_spots = [{'pos': (self._all_cols[i], self._all_rows[i]), 'brush': colors[i]}
                    for i in range(TOTAL_SATS)]
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
                if abs(self._all_cols[src] - self._all_cols[dst]) > 1:
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
                self._edges_line_items.append(item)

        # 虚线（pending）
        pending_links = getattr(self, "pending_links_by_step", {}).get(step, {})
        for src, dsts in pending_links.items():
            for dst in dsts:
                if abs(self._all_cols[src] - self._all_cols[dst]) > 1:
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

    def draw_straight_edge(self, x0, y0, x1, y1, dash=False):
        path = QPainterPath()
        path.moveTo(x0, y0)
        path.lineTo(x1, y1)
        item = QGraphicsPathItem(path)
        pen = pg.mkPen(
            color='red' if dash else '#888',
            width=1.2,
            style=QtCore.Qt.DashLine if dash else QtCore.Qt.SolidLine
        )
        item.setPen(pen)
        item.setZValue(3)
        self.plot_widget.addItem(item)
        return item

    def draw_curved_edge(self, x0, y0, x1, y1, curve=0.5, dash=False):
        path = QPainterPath()
        path.moveTo(x0, y0)
        ctrl_x = (x0 + x1) / 2
        ctrl_y = (y0 + y1) / 2 + curve * abs(x1 - x0)
        path.quadTo(ctrl_x, ctrl_y, x1, y1)
        item = QGraphicsPathItem(path)
        pen = pg.mkPen(
            color='red' if dash else '#888',
            width=1.2,
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

    def __init__(self, start_ts, end_ts, N, P, xml_file=None, parser_func=None):
        super().__init__()
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.N, self.P = N, P
        self.xml_file = xml_file
        self.parser_func = parser_func  # 传入 read_snap_xml.parse_xml_group_data 以避免循环导入
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
