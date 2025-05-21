# -*- coding: utf-8 -*-
"""
Dynamic Graph Visualizer
=======================

This script combines three capabilities you asked for:

1. **时间轴动画** (matplotlib `Slider`)
2. **区域高亮着色**（可自定义多组循环颜色）
3. **同行-跨列边自动弯曲**，避免与节点重叠

----
**输入数据结构**
```
adjacency_list_array : List[List[Tuple[int,int]]]
    第 *t* 个元素是一对节点 ID 构成的边表，对应时间步 *t*。

region_groups_array  : List[List[List[int]]]
    第 *t* 个元素是若干“区域”，每个区域是要着色的一组节点 ID。

N, P : int
    网格维度：每轨卫星数 *N*，轨道平面数 *P*。
```
两条列表 **长度必须一致**。

----
本轮修订
--------
* 修复 `TypeError: 'LineCollection' object is not iterable`
  * `networkx.draw_networkx_edges` 返回的是 `LineCollection`；
    不能用 `list.extend`，改为 `list.append` 单个对象。
* 其它逻辑不变。
"""
from __future__ import annotations

import math
from typing import List, Tuple, Sequence

import matplotlib
matplotlib.use("TkAgg")  # 后端统一
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import networkx as nx


class DynamicGraphVisualizer:
    """Interactive time-series visualisation for satellite topology graphs."""

    _DISTINCT_COLOURS = [
        "lightcoral",
        "lightgreen",
        "plum",
        "sandybrown",
        "khaki",
        "teal",
        "orchid",
        "salmon",
    ]

    def __init__(
        self,
        adjacency_list_array: Sequence[Sequence[Tuple[int, int]]],
        region_groups_array: Sequence[Sequence[Sequence[int]]],
        N: int,
        P: int,
        default_rad: float = 0.3,
        tolerance: float = 1e-9,
    ) -> None:
        if len(adjacency_list_array) != len(region_groups_array):
            raise ValueError("adjacency_list_array and region_groups_array must be the same length")
        self._adj = adjacency_list_array
        self._regions = region_groups_array
        self.N = N
        self.P = P
        self._default_rad = default_rad
        self._tol = tolerance
        self._pos = {i: (i // N, i % N) for i in range(N * P)}

        # Figure & slider
        self._fig, self._ax = plt.subplots(figsize=(max(6, P * 1.2), max(4, N * 1.2)))
        plt.subplots_adjust(bottom=0.25)
        slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])
        self._slider = Slider(slider_ax, "Time-step", 0, len(self._adj) - 1, valinit=0, valstep=1)
        self._slider.on_changed(self._update_frame)

        self._artists: List = []  # 存储当前帧的 Artist 便于清理

        self._setup_axes()
        self._update_frame(0)

    # --------------------- 公共 API ---------------------
    def show(self):
        """Blocking call – opens the interactive window."""
        plt.show()

    # ------------------- 内部工具函数 -------------------
    def _setup_axes(self):
        self._ax.set_xlim(-0.5, self.P - 0.5)
        self._ax.set_ylim(-0.5, self.N - 0.5)
        self._ax.set_xticks(range(self.P))
        self._ax.set_yticks(range(self.N))
        self._ax.set_xlabel("Orbit Plane Index (x)")
        self._ax.set_ylabel("Satellite Index in Plane (y)")
        self._ax.grid(True, linestyle="--", alpha=0.4)
        self._ax.set_aspect("equal", adjustable="box")

    def _graph_for_step(self, step: int) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.N * self.P))
        g.add_edges_from(self._adj[step])
        return g

    def _edge_split(self, g: nx.Graph):
        straight, curved = [], []
        for u, v in g.edges():
            x1, y1 = self._pos[u]
            x2, y2 = self._pos[v]
            if abs(y1 - y2) <= self._tol and abs(x1 - x2) > 1 + self._tol:
                curved.append((u, v))
            else:
                straight.append((u, v))
        return straight, curved

    def _node_colours(self, step: int):
        colours = ["skyblue"] * (self.N * self.P)
        for ridx, region in enumerate(self._regions[step]):
            colour = self._DISTINCT_COLOURS[ridx % len(self._DISTINCT_COLOURS)]
            for n in region:
                if 0 <= n < self.N * self.P:
                    colours[n] = colour
        return colours

    def _clear_artists(self):
        for art in self._artists:
            art.remove()
        self._artists.clear()

    def _update_frame(self, val):
        step = int(val)
        self._clear_artists()

        g = self._graph_for_step(step)
        straight, curved = self._edge_split(g)
        node_colours = self._node_colours(step)

        # 直线边
        if straight:
            art = nx.draw_networkx_edges(g, self._pos, ax=self._ax,
                                          edgelist=straight, edge_color="black", alpha=0.6)
            self._artists.append(art)
        # 曲线边
        if curved:
            conn_style = f"arc3,rad={self._default_rad}"
            art = nx.draw_networkx_edges(g, self._pos, ax=self._ax, edgelist=curved,
                                          edge_color="red", connectionstyle=conn_style, width=1.5)
            self._artists.append(art)

        # 节点和标签
        node_art = nx.draw_networkx_nodes(g, self._pos, ax=self._ax,
                                          node_color=node_colours, node_size=500)
        self._artists.append(node_art)
        label_art = nx.draw_networkx_labels(g, self._pos, ax=self._ax,
                                            font_size=8, font_weight="bold")
        self._artists.extend(label_art.values())

        self._ax.set_title(f"Time-step {step}")
        self._fig.canvas.draw_idle()

