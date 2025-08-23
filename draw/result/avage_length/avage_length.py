
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


