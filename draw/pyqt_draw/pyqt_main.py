from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import draw.read_snap_xml as read_snap_xml
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
    '#FFFF00',  # 黄   (Group 6)
]



def preprocess_envelopes(group_data, groups_to_envelope, N=36, y_preset_map={}):
    envelope_regions = {gid: {} for gid in groups_to_envelope}
    steps = sorted(group_data.keys())
    old_x_min_map = {gid: -1 for gid in groups_to_envelope}
    for step in steps:
        for gid in groups_to_envelope:
            group_sats = group_data[step]["groups"].get(gid, set())
            if gid==4 or gid==0:
                x_coords = [sid // N for sid in group_sats]
                if len(x_coords) == 0:
                    envelope_regions[gid][step] = None
                    continue
                x_min, x_max = min(x_coords), max(x_coords)
                rect_x = x_min - 0.5
                rect_width = 7
                if old_x_min_map[gid] == -1:
                    old_x_min_map[gid] = x_min
                elif old_x_min_map[gid] + rect_width > x_max:
                    rect_x = old_x_min_map[gid] - 0.5
                else:
                    old_x_min_map[gid] = x_min
                rect_y, rect_height = y_preset_map.get(gid, (0, N))
                envelope_regions[gid][step] = (rect_x, rect_y, rect_width, rect_height)
            else:
                envelope_regions[gid][step] = None
    return envelope_regions

class SatelliteViewer(QtWidgets.QWidget):
    def __init__(self, group_data):
        pg.setConfigOption('background', 'w')
        super().__init__()
        self.group_data = group_data
        self.steps = sorted(group_data.keys())
        self.envelopes = {}
        self.init_ui()
        self._edge_lines = []
        self.register_envelope(4, color="deeppink")
        self.register_envelope(0, color="orange")
        self.envelope_regions = preprocess_envelopes(
            self.group_data, [0, 4], N=36, y_preset_map={4: (18-0.5, 4), 0: (31, 5)}
        )
        self.plot_satellites(self.steps[0])

    def init_ui(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)
        self.envelopesflag = 0
        # 白色底点
        self.bg_scatter = pg.ScatterPlotItem(size=26, brush='w', pen=None)
        self.bg_scatter.setZValue(5)
        self.plot_widget.addItem(self.bg_scatter)

        # 彩色前景点
        self.scatter = pg.ScatterPlotItem(size=17, pen=pg.mkPen(width=0.7, color='#222'))
        self.scatter.setZValue(10)
        self.plot_widget.addItem(self.scatter)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(self.steps[0])
        self.slider.setMaximum(self.steps[-1])
        self.slider.setValue(self.steps[0])
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.on_slider)
        self.layout.addWidget(self.slider)


        # ---- 时间跳转 ----
        jump_layout = QtWidgets.QHBoxLayout()
        self.jump_input = QtWidgets.QLineEdit()
        self.jump_input.setPlaceholderText("跳转到 step")
        self.jump_button = QtWidgets.QPushButton("跳转")
        self.jump_button.clicked.connect(self.on_jump)
        jump_layout.addWidget(self.jump_input)
        jump_layout.addWidget(self.jump_button)
        self.layout.addLayout(jump_layout)



        # ---- 单步按钮 ----
        step_layout = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("<")
        self.next_btn = QtWidgets.QPushButton(">")
        self.prev_btn.clicked.connect(self.step_prev)
        self.next_btn.clicked.connect(self.step_next)
        step_layout.addWidget(self.prev_btn)
        step_layout.addWidget(self.next_btn)
        self.layout.addLayout(step_layout)


        self.label = QtWidgets.QLabel()
        self.layout.addWidget(self.label)

        self.plot_widget.setRange(xRange=[-0.5, P-0.5], yRange=[-0.5, N-0.5])
        self.plot_widget.setLabel('bottom', "Orbit Plane Index (P)")
        self.plot_widget.setLabel('left', "Satellite Index in Plane (N)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)

        legend_str = "  ".join([
            f'<span style="color:{GROUP_COLORS[g]};">&#9679;</span> {STATION_GROUPS[g]["name"]}'
            for g in sorted(STATION_GROUPS.keys())
            if g < len(GROUP_COLORS)
        ])
        self.legend_label = QtWidgets.QLabel(legend_str)
        self.layout.addWidget(self.legend_label)
    def step_prev(self):
        val = self.slider.value()
        if val > self.steps[0]:
            self.slider.setValue(val - 1)
    def step_next(self):
        val = self.slider.value()
        if val < self.steps[-1]:
            self.slider.setValue(val + 1)

    def register_envelope(self, group_id, color="deeppink"):
        envelope = RectangleEnvelope(color=color)
        self.envelopes[group_id] = envelope
        self.plot_widget.addItem(envelope.rect_item)
    def on_jump(self):
        val = self.jump_input.text().strip()
        if val.isdigit():
            step = int(val)
            # 限制范围
            if self.steps[0] <= step <= self.steps[-1]:
                self.slider.setValue(step)

    def update_envelopes(self, step):
        for gid, envelope in self.envelopes.items():
            rect = self.envelope_regions[gid].get(step)
            if rect is not None:
                envelope.set_rect(*rect)
            else:
                envelope.hide()

    def plot_satellites(self, step):
        current_data = self.group_data.get(int(step), {"groups": {}, "all_mentioned": set()})
        sat_ids = np.arange(TOTAL_SATS)
        all_cols = sat_ids // N
        all_rows = sat_ids % N

        # ==== 白色底点 ====
        bg_spots = [
            {'pos': (all_cols[i], all_rows[i]), 'brush': 'w'}
            for i in range(TOTAL_SATS)
        ]
        self.bg_scatter.setData(bg_spots)

        # ==== 彩色分组点 ====
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
        fg_spots = [
            {'pos': (all_cols[i], all_rows[i]), 'brush': colors[i]}
            for i in range(TOTAL_SATS)
        ]
        self.scatter.setData(fg_spots)

        # ==== 连线（如果有）====
        if hasattr(self, "edges_by_step") and step in self.edges_by_step:
            self.draw_edges(step, self.edges_by_step[step])
        else:
            self.draw_edges(step, {})

        self.label.setText(f'Grouped Satellite Visibility (Step {step})')

        if self.envelopesflag:
            self.update_envelopes(step)

    def on_slider(self, value):
        self.plot_satellites(value)

    def draw_edges(self, step, edges):
        # 清除旧的线
        if hasattr(self, "_edges_line_items"):
            for item in self._edges_line_items:
                self.plot_widget.removeItem(item)
        self._edges_line_items = []

        sat_ids = np.arange(TOTAL_SATS)
        all_cols = sat_ids // N
        all_rows = sat_ids % N

        # -------- 1. 先画实线 --------
        for src, dsts in edges.items():
            for dst in dsts:
                if abs(all_cols[src] - all_cols[dst]) > 1:
                    item = self.draw_curved_edge(
                        all_cols[src], all_rows[src], all_cols[dst], all_rows[dst], curve=0.5, dash=False
                    )
                else:
                    item = self.draw_straight_edge(
                        all_cols[src], all_rows[src], all_cols[dst], all_rows[dst], dash=False
                    )
                self._edges_line_items.append(item)

        # -------- 2. 再画虚线 --------
        pending_links = getattr(self, "pending_links_by_step", {}).get(step, {})
        for src, dsts in pending_links.items():
            for dst in dsts:
                if abs(all_cols[src] - all_cols[dst]) > 1:
                    item = self.draw_curved_edge(
                        all_cols[src], all_rows[src], all_cols[dst], all_rows[dst], curve=0.5, dash=True
                    )
                else:
                    item = self.draw_straight_edge(
                        all_cols[src], all_rows[src], all_cols[dst], all_rows[dst], dash=True
                    )
                self._edges_line_items.append(item)

    def draw_straight_edge(self, x0, y0, x1, y1, dash=False):
        path = QPainterPath()
        path.moveTo(x0, y0)
        path.lineTo(x1, y1)
        item = QGraphicsPathItem(path)
        # 按你的需求，虚线红色，实线灰色
        pen = pg.mkPen(
            color='red' if dash else '#888',  # 颜色同你原来风格
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
        # 参考你的用法
        pen = pg.mkPen(
            color='red' if dash else '#888',  # 虚线红，实线灰
            width=1.2,
            style=QtCore.Qt.DashLine if dash else QtCore.Qt.SolidLine
        )
        item.setPen(pen)
        item.setZValue(3)
        self.plot_widget.addItem(item)
        return item

    # def draw_edges(self, step, edges):
    #     # 1. 先移除上一次所有 QGraphicsPathItem 曲线
    #     if hasattr(self, "_edges_line_items"):
    #         for item in self._edges_line_items:
    #             self.plot_widget.removeItem(item)
    #     self._edges_line_items = []
    #
    #     # 2. 直线照常准备 xs, ys
    #     xs, ys = [], []
    #     sat_ids = np.arange(TOTAL_SATS)
    #     all_cols = sat_ids // N
    #     all_rows = sat_ids % N
    #
    #     for src, dsts in edges.items():
    #         for dst in dsts:
    #             if abs(all_cols[src] - all_cols[dst]) > 1:
    #                 # “跳线” 画贝塞尔曲线
    #                 item = self.draw_curved_edge(
    #                     all_cols[src], all_rows[src], all_cols[dst], all_rows[dst], curve=0.5
    #                 )
    #                 self._edges_line_items.append(item)
    #             else:
    #                 xs.extend([all_cols[src], all_cols[dst], np.nan])
    #                 ys.extend([all_rows[src], all_rows[dst], np.nan])
    #
    #     # 3. 直线继续用 PlotDataItem
    #     if not hasattr(self, "_edges_line"):
    #         self._edges_line = pg.PlotDataItem(pen=pg.mkPen(color='#888', width=1.2))
    #         self._edges_line.setZValue(1)
    #         self.plot_widget.addItem(self._edges_line)
    #     self._edges_line.setData(xs, ys)

    # def draw_curved_edge(self, x0, y0, x1, y1, curve=0.5):
    #     path = QPainterPath()
    #     path.moveTo(x0, y0)
    #     # 控制点横向/纵向适当偏移即可，例如横向跳两格y不变，可以让y加减1
    #     ctrl_x = (x0 + x1) / 2
    #     ctrl_y = (y0 + y1) / 2 + curve * abs(x1 - x0)
    #     path.quadTo(ctrl_x, ctrl_y, x1, y1)
    #     item = QGraphicsPathItem(path)
    #     pen = pg.mkPen(color='red', width=1.2)  # 修改这里，将颜色改为红色
    #     item.setPen(pen)
    #     item.setZValue(2)
    #     self.plot_widget.addItem(item)
    #     return item


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