# 下面是非阻塞版本，简单来说，可以画出图后，继续在控制台操作，这样可以省去每次读取xml文件的时间


from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg

# ====== 常量（与原始一致） ======
N = 36  # 每轨道卫星数
P = 18  # 轨道平面数
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
class Onetopology(QtWidgets.QWidget):
    def __init__(self, rec):
        pg.setConfigOption('background', 'w')
        super().__init__()
        # 允许空态启动
        self.rec = rec
      #  self.steps = sorted(self.group_data.keys())

        self.subrange_steps = None
        self.envelopes = {}
        self._edge_lines = []
        self._edges_line_items = []
        self.envelopesflag = 0  # 默认不显示包络
        self.envelope_regions = getattr(self, "envelope_regions", None)  # 若外部后来设置
        self._edges_by_step = {}
        self._active_step = None  # 当前使用的 step key（从 edges_by_step 里选一个）

        # 新增：无 step 的边存储
        self.edges = {}  # {src: {dst, ...}, ...}
        self.pending_links = {}  # 可选虚线边，同结构
        self._init_ui_core()



        self.plot_satellites()

    @property
    def edges_by_step(self):
        return self._edges_by_step

    @edges_by_step.setter
    def edges_by_step(self, value):
        d = value or {}

        # 情况1：传进来就是单层 {src: set(dst)}
        if d and all(isinstance(v, set) for v in d.values()):
            self._edges_by_step = {0: d}
            step = 0
        else:
            # 情况2：两层 {step: {src: set(dst)}}
            self._edges_by_step = d
            if not self._edges_by_step:
                # 清空
                if hasattr(self, 'draw_edges'):
                    self.draw_edges(0, {})
                elif hasattr(self, '_redraw_edges'):
                    self.edges = {}
                    self.pending_links = {}
                    self._redraw_edges()
                return
            # 选一个 step（兼容字符串数字键）
            keys = list(self._edges_by_step.keys())
            try:
                step = min(keys)
            except TypeError:
                step = sorted(keys, key=lambda k: int(k))[0]

        # 触发绘制：完全复用你已有的绘图代码路径
        if hasattr(self, 'draw_edges'):
            self.draw_edges(step, self._edges_by_step[step])
        elif hasattr(self, '_redraw_edges'):
            self.edges = self._edges_by_step[step]  # dict: src -> set(dst)
            self.pending_links = getattr(self, 'pending_links_by_step', {}).get(step, {})
            self._redraw_edges()

    def refresh_edges(self, step=None):
        """从 self.edges_by_step 选一个 step 画边；默认取最小的 step。"""
        if not hasattr(self, "edges_by_step") or not self.edges_by_step:
            # 没有边数据就清空
            self.draw_edges(0, {})
            return
        if step is None:
            step = min(self.edges_by_step.keys())
        self.draw_edges(step, self.edges_by_step[step])

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

    def set_edges(self, edges, pending_links=None):
        """
        设置当前要绘制的边（无 step 版本）。
        edges/pending_links: dict[int, set[int]]
        """
        self.edges = edges or {}
        self.pending_links = pending_links or {}
        self._redraw_edges()

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

        self.label = QtWidgets.QLabel("Grouped Satellite Visibility (Rects)")

        self.layout.addWidget(self.label)

        self.plot_widget.setRange(xRange=[-0.5, P - 0.5], yRange=[-0.5, N - 0.5])
        self.plot_widget.setLabel('bottom', "Orbit Plane Index (P)")
        self.plot_widget.setLabel('left', "Satellite Index in Plane (N)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)

        # legend_str = "  ".join([
        #     f'<span style="color:{GROUP_COLORS[g]};">&#9679;</span> {STATION_GROUPS[g]["name"]}'
        #     for g in sorted(STATION_GROUPS.keys()) if g < len(GROUP_COLORS)
        # ])
        # self.legend_label = QtWidgets.QLabel(legend_str)
        # self.layout.addWidget(self.legend_label)

    # ---------- 数据就绪后统一启用滑块 & 画首帧 ----------
    def _apply_steps_and_draw(self):
        # if not self.steps:
        #     self.slider.setEnabled(False)
        #     self.label.setText("Waiting for data...")
        #     return
        # self.slider.setEnabled(True)
        # self.slider.setMinimum(self.steps[0])
        # self.slider.setMaximum(self.steps[-1])
        # self.slider.setValue(self.steps[0])
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



    def plot_satellites(self):
        """
        仅用 self.rec 进行分组上色。
        self.rec 形如:
            {
                4: [(9, 15, 32, 35)],
                0: [(0, 5, 9, 13), (17, 17, 30, 32)],
            }
        规则：落入任一矩形 (xmin<=x<=xmax 且 ymin<=y<=ymax) 的点归属该 gid；
              多组重叠时取较小的 gid。
        """
        # 0) 读取矩形定义
        rects = getattr(self, "rec", None)

        # 1) 计算归属
        owner = np.full(TOTAL_SATS, -1, dtype=int)  # -1 表示未归属
        if rects:
            x_all = self._all_cols  # 整数网格中心
            y_all = self._all_rows

            # 用排序保证较小 gid 优先占位
            for gid in sorted(rects.keys()):
                rect_list = rects.get(gid) or []
                if not rect_list:
                    continue

                # 该 gid 的总体掩码（多个矩形并集）
                gid_mask = np.zeros(TOTAL_SATS, dtype=bool)
                for (xmin, xmax, ymin, ymax) in rect_list:
                    if xmin > xmax:
                        xmin, xmax = xmax, xmin
                    if ymin > ymax:
                        ymin, ymax = ymax, ymin
                    m = (x_all >= xmin) & (x_all <= xmax) & (y_all >= ymin) & (y_all <= ymax)
                    gid_mask |= m

                # 仅填充尚未归属的位置
                pick = (owner == -1) & gid_mask
                owner[pick] = gid

            title = "Grouped Satellite Visibility (Rects)"
        else:
            # 没有提供 rects：全部置白并返回
            spots = [{'pos': (self._all_cols[i], self._all_rows[i]), 'brush': '#FFF'}
                     for i in range(TOTAL_SATS)]
            self.scatter.setData(spots)
            self.label.setText("Grouped Satellite Visibility (No rects)")
            return

        # 2) 上色绘制
        colors = ['#FFF'] * TOTAL_SATS
        for sid in range(TOTAL_SATS):
            gid = owner[sid]
            if 0 <= gid < len(GROUP_COLORS):
                colors[sid] = GROUP_COLORS[gid]

        fg_spots = [{'pos': (self._all_cols[i], self._all_rows[i]), 'brush': colors[i]}
                    for i in range(TOTAL_SATS)]
        self.scatter.setData(fg_spots)

        # 3) 标签与可选包络（不做动态包络）
        self.label.setText(title)
        # 补：根据 self.edges / self.pending_links 立即画边
        self._redraw_edges()
        # 如需把矩形轮廓同时画出来，可在外部调用：
        # self.show_envelopes_static(self.rec, persist=True)

    # def on_slider(self, value):
    #     self.plot_satellites(value)

    def _redraw_edges(self):
        # 清空旧线
        if hasattr(self, "_edges_line_items"):
            for item in self._edges_line_items:
                self.plot_widget.removeItem(item)
        self._edges_line_items = []

        # 实线边
        for src, dsts in (self.edges or {}).items():
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

        # 虚线边（可选）
        for src, dsts in (self.pending_links or {}).items():
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
    group_ready = QtCore.pyqtSignal(dict)  # group_data（可选）
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
