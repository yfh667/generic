# -*- coding: utf-8 -*-
from __future__ import annotations

import matplotlib

matplotlib.use("TkAgg")  # 统一后端

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from matplotlib.widgets import Slider
from typing import Dict, Any, List, Tuple

# -*- coding: utf-8 -*-
import matplotlib
from matplotlib.patches import Rectangle   # ← 在文件顶部再加这一行

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import xml.etree.ElementTree as ET
import matplotlib.colors as mcolors
import sys
import matplotlib.patches as patches  # <--- 新增：导入 patches 模块

import draw.read_snap_xml as read_snap_xml
class DynamicGraphVisualizer:
    """
    一个功能强大的动态图表可视化工具，整合了以下功能：
    1. 动态节点着色（基于分组数据）。
    2. 动态边连接（直线和曲线）。
    3. 动态包络矩形高亮特定区域。
    4. 通过时间滑块进行交互式控制。
    """

    def __init__(
            self,
            group_data: Dict[int, Dict[str, Any]],
            adjacency_data: Dict[int, List[Tuple[int, int]]],
            N: int,
            P: int,
            station_groups: Dict[int, Dict[str, Any]],
            group_colors: List[str],
    ):
        """
        初始化可视化工具。

        Args:
            group_data (Dict): 包含每个时间步的卫星分组信息。
                               格式: {step: {'groups': {gid: {sats}}, 'all_mentioned': {sats}}}
            adjacency_data (Dict): 包含每个时间步的边连接信息。
                                   格式: {step: [(u, v), ...]}
            N (int): 每轨道卫星数。
            P (int): 轨道平面数。
            station_groups (Dict): 地面站分组的静态定义。
            group_colors (List): 每个组对应的颜色列表。
        """
        # --- 数据和配置 ---
        self.group_data = group_data
        self.adjacency_data = adjacency_data
        self.N = N
        self.P = P
        self.STATION_GROUPS = station_groups
        self.GROUP_COLORS = group_colors
        self.total_sats = N * P
        self.steps = sorted(self.group_data.keys())

        # --- 布局和样式 ---
        self._pos = {i: (i // N, i % N) for i in range(self.total_sats)}
        self._curved_edge_rad = 0.3

        # --- Matplotlib 对象 ---
        self._fig, self._ax = plt.subplots(figsize=(14, 9))
        self._fig.subplots_adjust(bottom=0.25, right=0.75)
        self._slider = None  # 将在 setup_widgets 中创建
        self._artists = []  # 存储当前帧绘制的所有元素，便于清理

        # --- 动态元素 ---
        self._scatter = None  # 卫星散点图对象
        self._rectangles = {}  # 存储包络矩形对象 {group_id: Rectangle}
        self._old_x_mins = {}  # 存储矩形上一帧的x_min，用于平滑移动

        # --- 初始化流程 ---
        self._setup_plot()
        self._create_dynamic_elements()
        self._setup_widgets()

        # 初始绘制第一帧
        self._update_frame(self.steps[0])

    def _setup_plot(self):
        """配置坐标轴、网格和标题等静态元素。"""
        self._ax.set_xlim(-0.5, self.P - 0.5)
        self._ax.set_ylim(-0.5, self.N - 0.5)
        self._ax.set_xticks(range(self.P))
        self._ax.set_yticks(range(self.N))
        self._ax.grid(True, linestyle='--', alpha=0.5)
        self._ax.set_xlabel('Orbit Plane Index (P)')
        self._ax.set_ylabel('Satellite Index in Plane (N)')
        self._ax.set_title('Dynamic Satellite Visibility and Connectivity')
        self._create_legend()

    def _create_legend(self):
        """创建并配置图例。"""
        legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markersize=10,
                       markerfacecolor=self.GROUP_COLORS[gid], markeredgecolor='k',
                       label=self.STATION_GROUPS[gid]["name"])
            for gid in self.STATION_GROUPS
        ]
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w', markersize=10,
                       markerfacecolor='white', markeredgecolor='#CCCCCC',
                       label='Not Visible')
        )
        self._ax.legend(
            handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0, title="Visible to Group"
        )

    def _create_dynamic_elements(self):
        """创建所有将在每帧更新的 matplotlib 对象。"""
        # 1. 创建卫星散点图对象
        self._scatter = self._ax.scatter(
            [], [], s=120, facecolors='white', edgecolors='#CCCCCC',
            linewidths=0.8, alpha=0.9, zorder=20
        )
        all_sat_pos = np.array([self._pos[i] for i in range(self.total_sats)])
        self._scatter.set_offsets(all_sat_pos)  # 设置所有点的位置 (只需一次)

        # 2. 创建包络矩形 (group 0 和 group 4)
        rect_configs = {
            4: {'color': 'deeppink', 'zorder': 10},
            0: {'color': 'blue', 'zorder': 10}
        }
        for gid, config in rect_configs.items():
            rect = patches.Rectangle(
                (0, 0), 0, 0,
                linewidth=2, edgecolor=config['color'], facecolor='none',
                linestyle='--', visible=False, zorder=config['zorder']
            )
            self._ax.add_patch(rect)
            self._rectangles[gid] = rect
            self._old_x_mins[gid] = -1  # 初始化平滑移动逻辑的状态

    def _setup_widgets(self):
        """创建滑块等交互式控件。"""
        slider_ax = self._fig.add_axes([0.2, 0.1, 0.65, 0.03])
        self._slider = Slider(
            ax=slider_ax, label='Time Step',
            valmin=min(self.steps), valmax=max(self.steps),
            valinit=min(self.steps), valstep=1
        )
        self._slider.on_changed(self._update_frame)

    def _clear_frame(self):
        """移除上一帧绘制的非持久性元素（主要是边）。"""
        for artist in self._artists:
            artist.remove()
        self._artists.clear()

    def _update_frame(self, step_val):
        """主更新函数，由滑块调用，协调所有元素的绘制。"""
        step = int(step_val)
        current_groups = self.group_data.get(step, {})
        current_edges = self.adjacency_data.get(step, [])

        self._clear_frame()

        # 按顺序绘制/更新各个图层
        self._update_nodes(current_groups)
        self._draw_edges(current_edges)
        self._update_rectangles(current_groups)

        self._ax.set_title(f'Satellite Visibility and Connectivity (Step {step})')
        self._fig.canvas.draw_idle()

    def _update_nodes(self, current_groups_data: Dict):
        """更新卫星节点的颜色和边框。"""
        groups = current_groups_data.get("groups", {})
        all_mentioned = current_groups_data.get("all_mentioned", set())

        facecolors = np.array([(1.0, 1.0, 1.0, 1.0)] * self.total_sats)
        edgecolors = np.array([(0.8, 0.8, 0.8, 1.0)] * self.total_sats)

        for sat_id in all_mentioned:
            if 0 <= sat_id < self.total_sats:
                seeing_gids = [gid for gid, sids in groups.items() if sat_id in sids]
                if seeing_gids:
                    assigned_gid = min(seeing_gids)
                    facecolors[sat_id] = mcolors.to_rgba(self.GROUP_COLORS[assigned_gid])
                    edgecolors[sat_id] = (0.0, 0.0, 0.0, 1.0)

        self._scatter.set_facecolors(facecolors)
        self._scatter.set_edgecolors(edgecolors)

    def _draw_edges(self, edges: List[Tuple[int, int]]):
        """绘制直线和曲线边。"""
        if not edges:
            return

        g = nx.Graph()
        g.add_nodes_from(range(self.total_sats))
        g.add_edges_from(edges)

        straight, curved = [], []
        for u, v in g.edges():
            y1, y2 = self._pos[u][1], self._pos[v][1]
            x1, x2 = self._pos[u][0], self._pos[v][0]
            if abs(y1 - y2) < 1e-9 and abs(x1 - x2) > 1:
                curved.append((u, v))
            else:
                straight.append((u, v))

        # 绘制直线
        if straight:
            art = nx.draw_networkx_edges(
                g, self._pos, ax=self._ax, edgelist=straight,
                edge_color="black", alpha=0.6, arrows=False
            )
            # --- START: MODIFIED BLOCK ---
            # Correctly handle single artist or list of artists
            if isinstance(art, list):
                self._artists.extend(art)
            else:
                self._artists.append(art)
            # --- END: MODIFIED BLOCK ---

        # 绘制曲线
        if curved:
            art = nx.draw_networkx_edges(
                g, self._pos, ax=self._ax, edgelist=curved,
                edge_color="red", width=1.5,
                arrows=True,
                arrowstyle='-',
                connectionstyle=f"arc3,rad={self._curved_edge_rad}"
            )
            # --- START: MODIFIED BLOCK ---
            # Correctly handle single artist or list of artists
            if isinstance(art, list):
                self._artists.extend(art)
            else:
                self._artists.append(art)
            # --- END: MODIFIED BLOCK ---
    def _update_rectangles(self, current_groups_data: Dict):
        """更新所有包络矩形的位置和可见性。"""
        groups = current_groups_data.get("groups", {})

        # --- Group 4 矩形逻辑 ---
        rect4 = self._rectangles.get(4)
        g4_sats = groups.get(4, set())
        if rect4 and g4_sats:
            x_coords = [sid // self.N for sid in g4_sats]
            x_min, x_max = min(x_coords), max(x_coords)

            rect_x = x_min - 0.5
            # 使用平滑逻辑
            if self._old_x_mins[4] != -1 and (self._old_x_mins[4] + 7 > x_max):
                rect_x = self._old_x_mins[4] - 0.5
            else:
                self._old_x_mins[4] = x_min

            rect4.set_xy((rect_x, 31.5))  # y=35.5 - 4
            rect4.set_width(7)
            rect4.set_height(4)
            rect4.set_visible(True)
        elif rect4:
            rect4.set_visible(False)
            self._old_x_mins[4] = -1

        # --- Group 0 矩形逻辑 ---
        rect0 = self._rectangles.get(0)
        g0_sats = groups.get(0, set())
        if rect0 and g0_sats:
            x_coords = [sid // self.N for sid in g0_sats]
            x_min, x_max = min(x_coords), max(x_coords)

            rect_x = x_min - 0.5
            # 使用平滑逻辑
            if self._old_x_mins[0] != -1 and (self._old_x_mins[0] + 7 > x_max):
                rect_x = self._old_x_mins[0] - 0.5
            else:
                self._old_x_mins[0] = x_min

            rect0.set_xy((rect_x, 8.5))
            rect0.set_width(7)
            rect0.set_height(5)
            rect0.set_visible(True)
        elif rect0:
            rect0.set_visible(False)
            self._old_x_mins[0] = -1

    def show(self):
        """显示交互式窗口。"""
        plt.show()


# ===================================================================
# 主程序入口 (Main Execution)
# ===================================================================
if __name__ == "__main__":
    # --- 1. 定义你的静态参数 ---
    N = 36  # 每轨道卫星数
    P = 18  # 轨道平面数

    STATION_GROUPS = {
        0: {"name": "Group 0"}, 1: {"name": "Group 1"}, 2: {"name": "Group 2"},
        3: {"name": "Group 3"}, 4: {"name": "Group 4"}, 5: {"name": "Group 5"},
        6: {"name": "Group 6"},
    }

    GROUP_COLORS = [
        '#FF0000', '#00FF00', '#0000FF', '#FFA500',
        '#800080', '#00FFFF', '#FFFF00',
    ]
    xml_file = "E:\Data\station_visible_satellites_648_8_h.xml"  # <<<<<<< 请替换为你的XML文件路径 >>>>>>>

    # dummy_file_name =
    # dummy_file_name =
    start_ts = 1202
    #  end_ts = 21431
    end_ts = 3320


    # 解析XML数据
    group_data = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)
    group_data_demo = read_snap_xml.modify_group_data(group_data, N=36, groupid=4)




    # --- 2. 准备你的动态数据 (这里使用伪数据作为示例) ---
    # 在实际使用中，你需要从你的 XML 或其他数据源加载这些数据。
    # `group_data` 来自你的第二个脚本
    # `adjacency_data` 是绘制连线所需的数据

    # 创建一些示例数据
    NUM_STEPS = 50
   # group_data_demo = {}
    adjacency_data_demo = {}

    for step in range(NUM_STEPS):
        # 模拟 group_data
        g_data = {'groups': {}, 'all_mentioned': set()}
        # 模拟 group 4 的移动区域
        start_x4 = (step * 2) % P
        g4_sats = {(x % P) * N + y for x in range(start_x4, start_x4 + 5) for y in range(32, 36)}
        g_data['groups'][4] = g4_sats
        g_data['all_mentioned'].update(g4_sats)

        # 模拟 group 0 的移动区域
        start_x0 = (P - 1 - step) % P
        g0_sats = {(x % P) * N + y for x in range(start_x0, start_x0 + 4) for y in range(9, 13)}
        g_data['groups'][0] = g0_sats
        g_data['all_mentioned'].update(g0_sats)
        group_data_demo[step] = g_data

        # # 模拟 adjacency_data (g4内部全连接, g0内部全连接)
        # adj_data = []
        # g4_list = list(g4_sats)
        # g0_list = list(g0_sats)
        # for i in range(len(g4_list)):
        #     for j in range(i + 1, len(g4_list)):
        #         adj_data.append((g4_list[i], g4_list[j]))
        # adjacency_data_demo[step] = adj_data

    # --- 3. 实例化并运行可视化工具 ---
    print("正在初始化可视化工具...")
    visualizer = DynamicGraphVisualizer(
        group_data=group_data_demo,
        adjacency_data=adjacency_data_demo,
        N=N,
        P=P,
        station_groups=STATION_GROUPS,
        group_colors=GROUP_COLORS
    )
    print("可视化工具已准备就绪。")
    visualizer.show()