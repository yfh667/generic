# core/router.py
from collections import deque, defaultdict

import basicSa.route.dijkstra as  dijkstra
class SatelliteRouter:
    def __init__(self, P, N):
        self.P = P  # 轨道面数
        self.N = N  # 每轨道卫星数
        self.adj = defaultdict(list)

    def build_topology(self, base_adj):
        """构建完整拓扑（包含轨道内连接）"""
        # 清空现有拓扑
        self.adj = defaultdict(list)

        # 添加原始连接
        for node, neighbors in base_adj.items():
            self.adj[node].extend(neighbors)

        # 添加轨道内连接（双向）
        for i in range(self.P):
            for j in range(self.N):
                next_node = (i, (j + 1) % self.N)
                prev_node = (i, (j - 1) % self.N)

                if next_node not in self.adj[(i, j)]:
                    self.adj[(i, j)].append(next_node)
                if prev_node not in self.adj[(i, j)]:
                    self.adj[(i, j)].append(prev_node)


# here is the core route algorithm
    def find_path(self, start):
        test = defaultdict(list)

        for k in range(self.N * self.P):
            i = k // self.N
            j = k % self.N
            neighbors = self.adj[(i, j)]
            for neighbor in neighbors:
                neighbor_id = neighbor[0] * self.N + neighbor[1]
                test[k].append((neighbor_id, 1))
      #  basicSa = 1
      #  end = 89
        graph = dijkstra.make_undirected(test)
        distances, paths = dijkstra.compute_dis_path(graph, start)

        return distances,paths

        # if end in paths:
        #  #   print(f"Path from {basicSa} to {end}: {paths[end]}")
        #     return paths[end]



# next it is just visulazition
    def generate_route_adj(self, path):
        """根据路径生成路由邻接表"""


        route_adj = defaultdict(list)

        for i in range(1, len(path)):
            src = path[i - 1]
            src_i = src // self.N
            src_j = src % self.N
            dest = path[i]
            dest_i = dest // self.N
            dest_j = dest % self.N

            if src_i == dest_i and src_j == 0 and dest_j == self.N - 1:

                route_adj[(src_i, self.N)].append((dest_i, dest_j))
            else:
                route_adj[(src_i, src_j)].append((dest_i, dest_j))

        return route_adj





