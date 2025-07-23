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
import numpy as np
import draw.snapshotf_romxml as snapshotf_romxml
from typing import List, Tuple, Sequence

from collections import deque
from collections import deque
import heapq

def k_shortest_paths(adjacency_list, start, end, k=3):
    """
    计算前k条最短路径

    参数:
    adjacency_list: 邻接表字典
    start: 起始节点
    end: 目标节点
    k: 需要的最短路径数量

    返回:
    list: 包含k个元组的列表，每个元组格式为(路径长度, 路径列表)
    """
    # 1. 基础验证
    if start not in adjacency_list or end not in adjacency_list:
        return []

    # 2. 使用BFS找到第一条最短路径
    def single_shortest_path(s, e):
        """BFS实现单条最短路径查找"""
        visited = set([s])
        queue = deque([(s, [s])])  # (当前节点, 路径)

        while queue:
            node, path = queue.popleft()
            if node == e:
                return path

            for neighbor in adjacency_list.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []  # 不可达

    # 3. 使用Yen's算法找前k条路径
    A = []  # 存储找到的路径
    B = []  # 优先队列 (路径长度, 路径)

    # 获取第一条路径
    first_path = single_shortest_path(start, end)
    if not first_path:
        return []  # 没有路径

    A.append((len(first_path) - 1, first_path))  # 保存路径长度和路径

    # 4. 迭代查找后续路径
    for ki in range(1, k):
        prev_path = A[-1][1]  # 获取上次找到的路径

        # 尝试从每个偏离点创建新路径
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i + 1]  # 从起点到偏离点的路径

            # 移除已用边（但保留偏离点）
            removed_edges = []
            for path in A:
                if len(path[1]) > i and root_path == path[1][:i + 1]:
                    u = path[1][i]
                    v = path[1][i + 1] if i + 1 < len(path[1]) else None
                    if v:
                        # 临时移除边
                        if v in adjacency_list[u]:
                            adjacency_list[u].remove(v)
                            removed_edges.append((u, v))
                        if u in adjacency_list[v]:
                            adjacency_list[v].remove(u)
                            removed_edges.append((v, u))

            # 从偏离点找新路径
            spur_path = single_shortest_path(spur_node, end)

            # 恢复移除的边
            for u, v in removed_edges:
                if v not in adjacency_list[u]:
                    adjacency_list[u].append(v)
                if u not in adjacency_list[v]:
                    adjacency_list[v].append(u)

            if spur_path:
                total_path = root_path[:-1] + spur_path
                path_tuple = (len(total_path) - 1, total_path)

                # 如果是新路径则加入候选
                if path_tuple not in B and total_path not in [path for _, path in A]:
                    heapq.heappush(B, path_tuple)

        # 从候选中选择最短的
        if not B:
            break  # 没有更多路径

        _, new_path = heapq.heappop(B)
        A.append((len(new_path) - 1, new_path))  # 保存路径长度和路径

    # 5. 格式化结果
    return [(len(path) - 1, path) for _, path in A[:k]]
def shortest_path(adjacency_list, start, end):
    """
    计算两个节点之间的最短路径

    参数:
    adjacency_list: 邻接表字典 {node: [neighbors]}
    start: 起始节点
    end: 目标节点

    返回:
    tuple: (distance, path)
        distance: 最短距离 (如果不可达返回-1)
        path: 节点路径列表 (如果不可达返回空列表)
    """
    # 初始化数据结构
    visited = set()
    queue = deque()
    predecessor = {}  # 记录前驱节点

    # 起点入队
    visited.add(start)
    queue.append(start)
    predecessor[start] = None

    # BFS遍历
    found = False
    while queue:
        current = queue.popleft()

        # 找到目标节点
        if current == end:
            found = True
            break

        # 遍历邻居
        for neighbor in adjacency_list.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                predecessor[neighbor] = current

    # 回溯路径
    if not found:
        return -1, []

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = predecessor[node]

    path.reverse()
    return len(path) - 1, path


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



# here he adjacncy_list_arrary is our main arrary

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

    # 创建邻接表字典
    adjacency_list = {}

    # 遍历所有边
    for u, v in adjacency_list_array[0]:
        # 添加 u->v 的连接
        if u not in adjacency_list:
            adjacency_list[u] = []
        adjacency_list[u].append(v)

        # 添加 v->u 的连接（无向图需要双向）
        if v not in adjacency_list:
            adjacency_list[v] = []
        adjacency_list[v].append(u)

    # 对每个节点的邻居列表排序（可选）
    for node in adjacency_list:
        adjacency_list[node].sort()


    for node in adjacency_list:
        # we need add the uo and down nodes
        xi = node//N
        yi = node%N
        upnodes = xi*N+(yi+1)%N
        adjacency_list[node].append(upnodes)
        downnode = xi*N+(yi-1+N)%N
        adjacency_list[node].append(downnode)


    # 打印结果（按节点ID排序）
    for node in sorted(adjacency_list.keys()):
        print(f"{node}: {adjacency_list[node]}")




    distance, path = shortest_path(adjacency_list, 2, 48)
    print(f"最短距离: {distance}")
    print(f"路径: {' → '.join(map(str, path))}")
    k=30
    paths = k_shortest_paths(adjacency_list, 12, 58, k)
    print(f"第{k}路径: {' → '.join(map(str, paths))}")

    ########
    vis = time_2d.DynamicGraphVisualizer(
        adjacency_list_array,
        regions_to_color,
        N,
        P,
       node_weights_array=weight_list_array  # ← 这一行传入节点权重
    )
    vis.show(block=True)



