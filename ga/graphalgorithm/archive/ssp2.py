import heapq
import sys

INF = sys.maxsize # 用一个很大的数表示无穷大

class Edge:
    """表示图中的一条边"""
    def __init__(self, to, capacity, cost, rev):
        self.to = to              # 边的终点
        self.capacity = capacity  # 边的容量
        self.cost = cost          # 边的费用
        self.flow = 0             # 当前边的流量
        self.rev = rev            # 指向反向边的索引

class MinCostFlow:
    """实现基于SSP算法的最小费用流，可计算给定流量下的费用"""
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj = [[] for _ in range(num_nodes)] # 邻接列表表示图
        self.total_flow = 0
        self.total_cost = 0

    def add_edge(self, u, v,cost, capacity):
        """
        添加一条从u到v的边
        Args:
            u (int): 起点
            v (int): 终点
            capacity (int): 容量
            cost (int): 费用
        """
        # 添加正向边
        self.adj[u].append(Edge(v, capacity, cost, len(self.adj[v])))
        # 添加反向边（初始容量为0，费用为负）
        self.adj[v].append(Edge(u, 0, -cost, len(self.adj[u]) - 1))

    def ssp(self, source, sink, required_flow):
        """
        使用Successive Shortest Path算法计算达到指定流量的最小费用
        Args:
            source (int): 源点
            sink (int): 汇点
            required_flow (int): 需要达到的总流量

        Returns:
            tuple: (实际达到的总流量, 实现该流量的最小费用)
                   如果 required_flow 无法达到，则返回最大流量及其费用。
        """
        self.total_flow = 0
        self.total_cost = 0
        h = [0] * self.num_nodes # 势能初始化为0

        # 循环直到达到所需流量或者汇点不可达（说明所需流量不可行）
        while self.total_flow < required_flow:
            # 使用带势能的Dijkstra寻找最短路径
            dist = [INF] * self.num_nodes       # 存储最短距离（使用缩减费用）
            parent_node = [-1] * self.num_nodes # 存储最短路径上的前驱节点
            parent_edge_index = [-1] * self.num_nodes # 存储最短路径上前驱节点到当前节点的边在邻接列表中的索引
            dist[source] = 0
            pq = [(0, source)] # 优先队列存储 (缩减费用距离, 节点)

            while pq:
                d, u = heapq.heappop(pq)

                if d > dist[u]:
                    continue

                for i, edge in enumerate(self.adj[u]):
                    # 计算缩减费用: cost + h[u] - h[v]
                    reduced_cost = edge.cost + h[u] - h[edge.to]
                    # 计算残余容量
                    residual_capacity = edge.capacity - edge.flow

                    if residual_capacity > 0 and dist[u] + reduced_cost < dist[edge.to]:
                        dist[edge.to] = dist[u] + reduced_cost
                        parent_node[edge.to] = u
                        parent_edge_index[edge.to] = i
                        heapq.heappush(pq, (dist[edge.to], edge.to))

            # 如果汇点不可达，说明无法再找到增广路径。
            # 这意味着需要的流量 required_flow 无法达到。
            if dist[sink] == INF:
                break

            # 找到最短路径上的瓶颈容量，但不能超过剩余所需流量
            push = INF # 能够推送的最大流量
            cur = sink
            while cur != source:
                prev = parent_node[cur]
                edge_idx = parent_edge_index[cur]
                push = min(push, self.adj[prev][edge_idx].capacity - self.adj[prev][edge_idx].flow)
                cur = prev

            # 将推送流量限制在不超过剩余所需流量
            push = min(push, required_flow - self.total_flow)

            # 如果本次不能推送任何流量（例如，剩余所需流量为0），则停止
            if push <= 0:
                break

            # 沿着最短路径推送流量
            cur = sink
            while cur != source:
                prev = parent_node[cur]
                edge_idx = parent_edge_index[cur]
                edge = self.adj[prev][edge_idx]

                # 更新正向边的流量
                edge.flow += push
                # 更新反向边的流量
                reverse_edge = self.adj[edge.to][edge.rev]
                reverse_edge.flow -= push

                cur = prev

            # 更新总流量和总费用
            self.total_flow += push
            # Cost added is push * (shortest path distance in original costs)
            # Shortest path distance in original costs = shortest path distance in reduced costs + h[sink] - h[source]
            self.total_cost += push * (dist[sink] + h[sink] - h[source])

            # 更新势能 (使用本次 Dijkstra 计算出的 dist)
            for i in range(self.num_nodes):
                 if dist[i] != INF:
                     h[i] += dist[i]

        # 循环结束，self.total_flow 是实际达到的流量
        return self.total_flow, self.total_cost

# --- 应用到您提供的数据 ---

# 定义节点数量 (0到10，共11个节点)
num_nodes = 11
mcf = MinCostFlow(num_nodes)

# 添加边 (起点, 终点, 容量, 费用)
mcf.add_edge(0, 1, 18, 60)
mcf.add_edge(0, 2, 19, 60)
mcf.add_edge(0, 3, 17, 60)
mcf.add_edge(1, 4, 16, 40)
mcf.add_edge(1, 5, 14, 30)
mcf.add_edge(2, 5, 16, 50)
mcf.add_edge(2, 6, 17, 30)
mcf.add_edge(3, 6, 19, 40)
mcf.add_edge(4, 7, 19, 60)
mcf.add_edge(5, 4, 15, 20) # 注意: 5->4 是一条边，费用为20
mcf.add_edge(5, 7, 16, 30)
mcf.add_edge(5, 8, 15, 40)
mcf.add_edge(6, 5, 15, 20) # 注意: 6->5 是一条边，费用为20
mcf.add_edge(6, 9, 13, 40)
mcf.add_edge(7, 10, 18, 60)
mcf.add_edge(7, 8, 17, 30)
mcf.add_edge(8, 10, 19, 50)
mcf.add_edge(8, 9, 14, 30)
mcf.add_edge(9, 10, 17, 50)

# 定义源点、汇点和所需的总流量
source_node = 0
sink_node = 10
required_flow_amount = 30

# 运行SSP算法计算给定流量下的最小费用
actual_flow, min_cost = mcf.ssp(source_node, sink_node, required_flow_amount)

# 输出结果
if actual_flow == required_flow_amount:
    print(f"为达到总流量 {required_flow_amount} 的最小费用:")
    print(f"实际达到的总流量: {actual_flow}")
    print(f"最小费用: {min_cost}")
else:
    print(f"无法达到所需的总流量 {required_flow_amount}.")
    print(f"网络的最大流量为 {actual_flow}.")
    print(f"达到最大流量的最小费用为 {min_cost}")

# 打印每条原始边上的最终流量分配，这展示了流量的“路径”
print("\n每条原始边上的最终流量分配:")
# 遍历所有节点和其邻接边
for u in range(mcf.num_nodes):
    for edge in mcf.adj[u]:
        # 原始边通常是 cost >= 0 的那条（假设原始费用非负）
        # 并且只关心有正向净流量的边
        if edge.cost >= 0 and edge.flow > 0:
             print(f"{u} -> {edge.to}: 流量 = {edge.flow}, 容量 = {edge.capacity}, 费用 = {edge.cost}")