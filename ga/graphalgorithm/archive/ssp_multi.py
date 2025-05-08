import heapq
from collections import deque
from math import inf


class Edge:
    def __init__(self, to, rev, capacity, cost):
        self.to = to  # 目标节点
        self.rev = rev  # 反向边在目标节点邻接表中的索引
        self.capacity = capacity  # 边容量
        self.cost = cost  # 单位流量成本


class MinCostFlow:
    def __init__(self, n):
        """初始化网络，n为普通节点数量（不包括后续添加的超级节点）"""
        self.n = n
        self.graph = [[] for _ in range(n)]  # 邻接表存储图结构

    def add_edge(self, fr, to, cost,cap ):
        """添加边及其反向边"""
        forward = Edge(to, len(self.graph[to]), cap, cost)
        backward = Edge(fr, len(self.graph[fr]), 0, -cost)
        self.graph[fr].append(forward)
        self.graph[to].append(backward)

    def solve(self, sources, source_supplies, sinks, total_demand):
        """
        多源多汇最小费用流算法
        :param sources: 源点列表（节点编号）
        :param source_supplies: 各源点供应量
        :param sinks: 汇点列表（节点编号）
        :param total_demand: 总需求流量
        :return: (实际流量, 总成本)
        """
        # 添加超级源点（s）和超级汇点（t）
        s = self.n
        t = self.n + 1
        self.n += 2
        self.graph.extend([[], []])  # 扩展邻接表

        # 连接超级源点到各源点
        for src, supply in zip(sources, source_supplies):
            self.add_edge(s, src, supply, 0)

        # 连接各汇点到超级汇点（容量设为无穷，不限制单个汇点流量）
        for sink in sinks:
            self.add_edge(sink, t, inf, 0)

        # 执行最小费用流算法
        flow = 0
        cost = 0
        h = [0] * self.n  # 势能数组
        prev_v = [0] * self.n  # 前驱节点
        prev_e = [0] * self.n  # 前驱边

        while flow < total_demand:
            # 使用Bellman-Ford算法寻找最短增广路
            dist = [inf] * self.n
            dist[s] = 0
            in_queue = [False] * self.n
            in_queue[s] = True
            q = deque([s])

            while q:
                v = q.popleft()
                in_queue[v] = False
                for i, edge in enumerate(self.graph[v]):
                    if edge.capacity > 0 and dist[edge.to] > dist[v] + edge.cost + h[v] - h[edge.to]:
                        dist[edge.to] = dist[v] + edge.cost + h[v] - h[edge.to]
                        prev_v[edge.to] = v
                        prev_e[edge.to] = i
                        if not in_queue[edge.to]:
                            q.append(edge.to)
                            in_queue[edge.to] = True

            if dist[t] == inf:  # 无法继续增广
                break

            # 更新势能
            for i in range(self.n):
                if dist[i] < inf:
                    h[i] += dist[i]

            # 计算本次增广的流量
            delta_flow = total_demand - flow
            v = t
            while v != s:
                edge = self.graph[prev_v[v]][prev_e[v]]
                delta_flow = min(delta_flow, edge.capacity)
                v = prev_v[v]

            # 更新流量和成本
            flow += delta_flow
            cost += delta_flow * h[t]  # 注意这里使用势能差计算成本

            # 更新残留网络
            v = t
            while v != s:
                edge = self.graph[prev_v[v]][prev_e[v]]
                edge.capacity -= delta_flow
                self.graph[v][edge.rev].capacity += delta_flow
                v = prev_v[v]

        return flow, cost


# ==================== 测试用例 ====================
# def test_case_1():
#     """基础测试：两个源点，两个汇点"""
#     mcf = MinCostFlow(4)  # 创建4个普通节点（0-3）
#
#     # 构建网络（示例简单网络）
#     mcf.add_edge(0, 2, 100, 1)  # 源点0到中间节点
#     mcf.add_edge(1, 2, 100, 1)  # 源点1到中间节点
#     mcf.add_edge(2, 3, 100, 1)  # 中间节点到汇点
#
#     # 设置参数
#     sources = [0, 1]
#     source_supplies = [60, 60]
#     sinks = [3]
#     total_demand = 120
#
#     flow, cost = mcf.solve(sources, source_supplies, sinks, total_demand)
#     print(f"测试案例1: 流量={flow}, 成本={cost} (预期: 120, 120)")
#
#
# def test_case_2():
#     """复杂网络测试（使用您之前的11节点网络）"""
#     mcf = MinCostFlow(13)  # 节点0-10
#
#     # 添加原始边（同您之前的网络）
#     mcf.add_edge(0, 1, 18, 60)
#     mcf.add_edge(12, 1, 30, 80)
#     mcf.add_edge(12, 2, 19, 60)
#     mcf.add_edge(12, 3, 15, 60)
#
#     mcf.add_edge(0, 2, 19, 60)
#     mcf.add_edge(0, 3, 17, 60)
#     mcf.add_edge(1, 4, 16, 40)
#     mcf.add_edge(1, 5, 14, 30)
#     mcf.add_edge(2, 5, 16, 50)
#     mcf.add_edge(2, 6, 17, 30)
#     mcf.add_edge(3, 6, 19, 40)
#     mcf.add_edge(4, 7, 19, 60)
#     mcf.add_edge(5, 4, 15, 20)
#     mcf.add_edge(5, 7, 16, 30)
#     mcf.add_edge(5, 8, 15, 40)
#     mcf.add_edge(6, 5, 15, 20)
#     mcf.add_edge(6, 9, 13, 40)
#     mcf.add_edge(7, 10, 18, 60)
#     mcf.add_edge(7, 8, 17, 30)
#     mcf.add_edge(8, 10, 19, 50)
#     mcf.add_edge(8, 9, 14, 30)
#     mcf.add_edge(9, 10, 17, 50)
#
#     mcf.add_edge(8, 11, 21, 60)
#     mcf.add_edge(9, 11, 16, 40)
#
#
#     # 设置参数
#     sources = [0, 12]  # 两个源点
#     source_supplies = [60, 60]  # 每个源点供应60
#     sinks = [11, 10]  # 两个汇点
#     total_demand = 120  # 总需求
#
#     flow, cost = mcf.solve(sources, source_supplies, sinks, total_demand)
#     print(f"测试案例2: 流量={flow}, 成本={cost}")
#
#
# if __name__ == "__main__":
#     test_case_1()
#     test_case_2()
