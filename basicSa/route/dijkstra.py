import heapq
from collections import defaultdict


# def make_undirected(adjacency_list):
#     undirected_graph = defaultdict(list)
#     for node, edges in adjacency_list.items():
#         for neighbor, weight in edges:
#             # 检查是否已经存在该边
#             if (node, weight) not in undirected_graph[neighbor]:
#                 undirected_graph[node].append((neighbor, weight))
#                 undirected_graph[neighbor].append((node, weight))
#     return undirected_graph
def make_undirected(adjacency_list):
    undirected_graph = defaultdict(list)
    added_edges = set()  # 跟踪已添加的边 (node, neighbor)

    for node, edges in adjacency_list.items():
        for neighbor, weight in edges:
            # 确保边是无向的
            edge = tuple(sorted((node, neighbor)))  # 用排序的元组表示边
            if edge not in added_edges:
                undirected_graph[node].append((neighbor, weight))
                undirected_graph[neighbor].append((node, weight))
                added_edges.add(edge)  # 标记边已添加

    # 对每个节点的邻接表进行排序
    for node in undirected_graph:
        undirected_graph[node].sort()

    return undirected_graph
#  non-ori
def compute_dis_path(e, s):
    """
    输入：
    e: 邻接表，格式为 {节点: [(邻接节点, 边权重), ...], ...}
    s: 起点
    返回：
    dis: 从s到每个顶点的最短路长度
    paths: 每个顶点的最短路径
    """
    # 使用默认值为无穷大的字典初始化距离
    dis = defaultdict(lambda: float("inf"))
    dis[s] = 0
    prev = {s: None}
    q = [(0, s)]
    vis = set()

    while q:
        _, u = heapq.heappop(q)
        if u in vis:
            continue
        vis.add(u)

        # 遍历邻接节点和权重
        for v, w in e.get(u, []):
            if dis[v] > dis[u] + w:
                dis[v] = dis[u] + w
                prev[v] = u
                heapq.heappush(q, (dis[v], v))

    # 构建从起点到每个节点的路径
    paths = {s: [s]}
    for dest in dis:
        if dest == s or dest not in prev:
            continue
        path = []
        current = dest
        while current is not None:
            path.append(current)
            current = prev[current]
        paths[dest] = path[::-1]  # 逆序得到正向路径

    return dis, paths


def convert_to_adjacency_list(visibility_results):
    adjacency_list = {}
    n = len(visibility_results)
    for i in range(n):
        adjacency_list[i] = []
    for i in range(n):
        for j in range(n):
            weight = visibility_results[i][j]
            if weight != 0:
                adjacency_list[i].append((j, weight))
                # 添加反向边
                if i != j and (i, weight) not in adjacency_list[j]:
                    adjacency_list[j].append((i, weight))
  #  print(f"Adjacency list is: {adjacency_list}")
    return adjacency_list


