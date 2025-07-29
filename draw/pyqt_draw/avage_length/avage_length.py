
from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import QGraphicsPathItem
import genaric2.tegnode as tegnode
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import draw.read_snap_xml as read_snap_xml
import draw.pyqt_draw.adjacent2xml as adjacent2xml
import draw.pyqt_draw.pyqt_main as pyqt_main
import draw.pyqt_draw.adjacent2xml as adjacent2xml
if __name__ == "__main__":
    start_ts = 1202
    end_ts = 3320
    N = 36
    groupid = 4


  #  xml_file = "E:\\Data\\test.xml"
    xml_file = "E:\\Data\\grid.xml"

    modify_edges_by_step = adjacent2xml.read_steps_from_xml(xml_file)

    xml_file2 = "E:\\Data\\station_visible_satellites_648_8_h.xml"

    group_data = read_snap_xml.parse_xml_group_data(xml_file2, start_ts, end_ts)
    group_data, offset = read_snap_xml.modify_group_data(group_data, N=36, groupid=4)


    # region1:
    #x:0-5,y9 -12

    region1 = []

    for i in range(6):
        for j in range(9,13):
            node = i*N + j
            region1.append(node)

    # region2:x 9-15,y 32-35


    region2 = []

    for i in range(9,16):
        for j in range(32,36):
            node = i*N + j
            region2.append(node)


    pathlength  =[]

    #
    time = 1871
    # 构建无向图
    G = nx.Graph()
    for node, neighbors in modify_edges_by_step[time].items():
        for nbr in neighbors:
            G.add_edge(node, nbr)

    # 计算所有 region1->region2 节点对的最短路径
    path_lengths = []
    for src in region1:
        for dst in region2:
            try:
                l = nx.shortest_path_length(G, src, dst)
                path_lengths.append(l)
            except nx.NetworkXNoPath:
                pass  # 跳过无路可达的情况

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 让中文可以显示

    plt.rcParams['axes.unicode_minus'] = False

    if path_lengths:
        plt.figure(figsize=(8, 5))
        plt.hist(path_lengths, bins=range(min(path_lengths), max(path_lengths) + 2), edgecolor='black', align='left')
        plt.xlabel("跳数")
        plt.ylabel("节点对数")
        plt.title("区域间最短路径跳数分布直方图")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    else:
        print("没有连通的点对，无法绘制直方图")



    # if path_lengths:
    #     print("平均最短路径长度:", np.mean(path_lengths))
    #     pathlength.append(np.mean(path_lengths))
    # else:
    #     print("没有连通的点对")

    # time = 1948
    # # 构建无向图
    # G = nx.Graph()
    # for node, neighbors in modify_edges_by_step[time].items():
    #     for nbr in neighbors:
    #         G.add_edge(node, nbr)
    #
    # # 计算所有 region1->region2 节点对的最短路径
    # path_lengths = []
    # for src in region1:
    #     for dst in region2:
    #         try:
    #             l = nx.shortest_path_length(G, src, dst)
    #             path_lengths.append(l)
    #         except nx.NetworkXNoPath:
    #             pass  # 跳过无路可达的情况
    #
    # if path_lengths:
    #     print("平均最短路径长度:", np.mean(path_lengths))
    #     pathlength.append(np.mean(path_lengths))
    # else:
    #     print("没有连通的点对")
    #
    #
    # time = 1939
    # # 构建无向图
    # G = nx.Graph()
    # for node, neighbors in modify_edges_by_step[time].items():
    #     for nbr in neighbors:
    #         G.add_edge(node, nbr)
    #
    # src = 193
    # paths = {}
    #
    # for dst in region2:
    #     try:
    #         path = nx.shortest_path(G, src, dst)
    #         length = len(path) - 1
    #         paths[dst] = (length, path)
    #     except nx.NetworkXNoPath:
    #         continue
    #
    #
    # def node_to_ij(node, N):
    #     return (node // N, node % N)
    #
    #
    # # 输出
    # for dst, (length, path) in paths.items():
    #     path_coords = [node_to_ij(n, N) for n in path]
    #     print(f"192({node_to_ij(src, N)}) → {dst}({node_to_ij(dst, N)}): 跳数 {length}, 路径 {path_coords}")
    #
    # # 平均最短跳数
    # if paths:
    #     avg_length = sum(length for length, _ in paths.values()) / len(paths)
    #     print("平均最短跳数:", avg_length)
    # else:
    #     print("192不可达region2任何节点")
    #
    # for time in range(start_ts, end_ts):
    #
    #     # 构建无向图
    #     G = nx.Graph()
    #     for node, neighbors in modify_edges_by_step[time].items():
    #         for nbr in neighbors:
    #             G.add_edge(node, nbr)
    #
    #     # 计算所有 region1->region2 节点对的最短路径
    #     path_lengths = []
    #     for src in region1:
    #         for dst in region2:
    #             try:
    #                 l = nx.shortest_path_length(G, src, dst)
    #                 path_lengths.append(l)
    #             except nx.NetworkXNoPath:
    #                 pass  # 跳过无路可达的情况
    #
    #     if path_lengths:
    #      #   print("平均最短路径长度:", np.mean(path_lengths))
    #         pathlength.append(np.mean(path_lengths))
    #     else:
    #         print("没有连通的点对")
    #
    # # 普通 set 去重（无序）
    # unique_pathlength = list(set(pathlength))
    # print(unique_pathlength)
    #
    # # 保持顺序去重（推荐）
    # unique_pathlength_ordered = []
    # seen = set()
    # for v in pathlength:
    #     if v not in seen:
    #         unique_pathlength_ordered.append(v)
    #         seen.add(v)
    # print(unique_pathlength_ordered)
    # #
    #
    #
    # print(1)

   #  # 我们接下来需要导出modify_edges_by_step，因为这个是转化后的拓扑形态
   #  app = QtWidgets.QApplication([])
   #  viewer = pyqt_main.SatelliteViewer(group_data)
   #  viewer.edges_by_step = modify_edges_by_step
   # # viewer.pending_links_by_step = modify_pending_links_by_step  # <<<<<<<< 新增
   #  viewer.envelopesflag = 1
   #  viewer.setWindowTitle("Grouped Satellite Visibility - High Performance (PyQtGraph)")
   #  viewer.resize(1200, 700)
   #  viewer.show()
   #  sys.exit(app.exec_())
