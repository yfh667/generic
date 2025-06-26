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
#
----
本轮修订
--------
* 修复 `TypeError: 'LineCollection' object is not iterable`
  * `networkx.draw_networkx_edges` 返回的是 `LineCollection`；
    不能用 `list.extend`，改为 `list.append` 单个对象。
* 其它逻辑不变。
"""
from __future__ import annotations
import networkx as nx

import math
from typing import List, Tuple, Sequence

import matplotlib
matplotlib.use("TkAgg")  # 后端统一
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import networkx as nx
import graph.time_2d3 as time_2d
#
import genaric2.tegnode as tegnode
#
import draw.snapshotf_romxml as snapshotf_romxml


def in_same_region(node1, node2, region_groups: List[List[int]]) -> int:
    for region in region_groups:
        if node1 in region and node2 in region:
            return 1
    return 0
# ----------------------------- DEMO -----------------------------
if __name__ == "__main__":

    N = 10
    P = 10
    start_ts = 1500
    T=24

    end_ts = start_ts+T-1

    dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"

    regions_to_color = {}
    # Iterate over the time steps
    region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, start_ts, end_ts)
    target_time_step = len(region_satellite_groups)
    T = target_time_step
    for i in range(len(region_satellite_groups)):
        region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
        u = region_satellite_group[0]
        v = region_satellite_group[3]
        o = [u, v]
        regions_to_color[i] = o  # Corrected append to dictionary assignment



    adjacency_list_array: List[List[Tuple[int, int]]] = [[] for _ in range(T)]
    weight_list_array=[]
    for t in range(0, T):
        nodes = {}
        for x in range(P):  # x ∈ [0, P]
            for y in range(N):  # y ∈ [0, N]


                    nodes[(x, y, t)] = tegnode.tegnode(
                        asc_nodes_flag=False,  # 或初始为None等你自己定义
                        rightneighbor=None,
                        leftneighbor=None,
                        state=-1  # 默认初始为free
                    )
        #first we need complete the distinct inner links
        for group_idx, region in enumerate(regions_to_color[t]):
            for node_id in region:
                x_node, y_node = divmod(node_id, N)
                if x_node < P - 1:
                    neighbor_id = (x_node + 1) * N + y_node
                    if neighbor_id in region:
                        nodes[(x_node, y_node, t)].rightneighbor=((x_node + 1), y_node,t)
                        nodes[(x_node + 1, y_node, t)].leftneighbor=(x_node, y_node,t)


                        adjacency_list_array[t].append((node_id, neighbor_id))




        for i in range(P - 1):
            for j in range(N):
                if nodes[(i, j, t)].rightneighbor:
                    continue


                # if i==0 and j==2:
                #     print(1)



                nownode = i * N + j
                next_node1 = (i + 1) * N + j


                if not nodes[(i+1, j, t)].leftneighbor:
                    adjacency_list_array[t].append((nownode, next_node1))



                next_node2 = (i + 1) * N + (j + 1) % N

                if not nodes[(i+1, (j + 1) % N, t)].leftneighbor:
                    adjacency_list_array[t].append((nownode, next_node2))

                next_node3 = (i + 1) * N + (j - 1 + N) % N
                if not nodes[(i+1, (j - 1 + N) % N, t)].leftneighbor:
                    adjacency_list_array[t].append((nownode, next_node3))








                if i != P - 2:
                    next_node4 = (i + 2) * N + j
                    if not nodes[(i + 2, (j ) % N, t)].leftneighbor:
                        adjacency_list_array[t].append((nownode, next_node4))

                    # adjacency_list_array[t].append((nownode, next_node4))











        G = nx.Graph()
        G.add_edges_from(adjacency_list_array[t])  # ← 粘贴你的 400+ 条边
        S = regions_to_color[t][0]  # 源热点
        T = regions_to_color[t][1]  # 目标热点（可多点）
        # S = [2, 12, 88, 98]  # 源热点
        # T = [58]  # 目标热点（可多点）

        bw = nx.betweenness_centrality_subset(
            G,
            sources=S,
            targets=T,
            normalized=False,  # 不归一化 → 直接得到“出现次数”累加值
            weight=None  # 若有边权可传入
        )
        # 将 S 和 T 中的点的权重置为 0
        for node in set(S + T):
            bw[node] = 0.0

        weight_list_array.append(bw)





    # vis = time_2d.DynamicGraphVisualizer(adjacency_list_array, regions_to_color, N, P)
    # vis.show(block=True)          # 让 Qt 事件循环真正阻塞主线程
    vis = time_2d.DynamicGraphVisualizer(
        adjacency_list_array,
        regions_to_color,
        N,
        P,
        node_weights_array=weight_list_array  # ← 这一行传入节点权重
    )
    vis.show(block=True)
