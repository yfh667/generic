import genaric2.tegnode as tegnode
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
from typing import List, Tuple, Sequence
def initial_weight(P,N,T,regions_to_color):
   # regions_to_color = {}

    # for i in range(T):
    #     region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
    #     u = region_satellite_group[0]
    #     v = region_satellite_group[3]
    #     o = [u, v]
    #     regions_to_color[i] = o  # Corrected append to dictionary assignment



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
                        state=-1,  # 默认初始为free
                        importance=0,
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
    return weight_list_array,adjacency_list_array

if __name__ == '__main__':
    N = 10
    P = 10
    start_ts = 1500
    T = 24

    end_ts = start_ts + T - 1

    dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"

    region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, start_ts, end_ts)
    regions_to_color = {}


    for i in range(len(region_satellite_groups)):
        region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
        u = region_satellite_group[0]
        v = region_satellite_group[3]
        o = [u, v]
        regions_to_color[i] = o  # Corrected append to dictionary assignment

        # useage in here
    weight_list_array, adjacency_list_array=initial_weight(P,N,T,regions_to_color)


    ###########

    ########
    vis = time_2d.DynamicGraphVisualizer(
        adjacency_list_array,
        regions_to_color,
        N,
        P,
        node_weights_array=weight_list_array  # ← 这一行传入节点权重
    )
    vis.show(block=True)
