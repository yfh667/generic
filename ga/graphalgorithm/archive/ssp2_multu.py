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
    """实现基于SSP算法的最小费用流，支持单源单汇和多源多汇"""
    def __init__(self, num_original_nodes):
        # 这个类实例将保存原始图的定义
        self.num_original_nodes = num_original_nodes
        self.adj = [[] for _ in range(num_original_nodes)]

    def add_edge(self, u, v, capacity, cost):
        """
        向原始图添加一条边。
        Args:
            u (int): 起点 (原始节点索引 0 到 num_original_nodes-1)
            v (int): 终点 (原始节点索引 0 到 num_original_nodes-1)
            capacity (int): 容量
            cost (int): 费用
        """
        if u < 0 or u >= self.num_original_nodes or v < 0 or v >= self.num_original_nodes:
             raise ValueError(f"节点索引 ({u}, {v}) 超出原始图范围 (0-{self.num_original_nodes-1})")
        # 添加正向边
        self.adj[u].append(Edge(v, capacity, cost, len(self.adj[v])))
        # 添加反向边（初始容量为0，费用为负）
        self.adj[v].append(Edge(u, 0, -cost, len(self.adj[u]) - 1))

    # 这是一个内部的单源单汇 SSP 求解器，它对调用它的实例（即当前的图）进行操作
    def _ssp_solver(self, source, sink, required_flow=INF):
        """
        在当前图上求解从 source 到 sink 的最小费用流。
        Args:
            source (int): 源点节点索引
            sink (int): 汇点节点索引
            required_flow (int): 需要达到的总流量。如果 INF 则计算最大流。
        """
        total_flow = 0
        total_cost = 0
        num_nodes = len(self.adj) # 使用当前图实例的节点总数
        h = [0] * num_nodes # 势能初始化为0

        while total_flow < required_flow:
            # 使用带势能的Dijkstra寻找最短路径
            dist = [INF] * num_nodes
            parent_node = [-1] * num_nodes
            parent_edge_index = [-1] * num_nodes
            dist[source] = 0
            pq = [(0, source)]

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]: continue

                for i, edge in enumerate(self.adj[u]):
                    # 缩减费用: cost + h[u] - h[v]
                    reduced_cost = edge.cost + h[u] - h[edge.to]
                    residual_capacity = edge.capacity - edge.flow

                    if residual_capacity > 0 and dist[u] + reduced_cost < dist[edge.to]:
                        dist[edge.to] = dist[u] + reduced_cost
                        parent_node[edge.to] = u
                        parent_edge_index[edge.to] = i
                        heapq.heappush(pq, (dist[edge.to], edge.to))

            # 如果汇点不可达，说明无法再找到增广路径
            if dist[sink] == INF:
                break

            # 找到最短路径上的瓶颈容量，并限制不超过剩余所需流量
            push = INF
            cur = sink
            while cur != source:
                prev = parent_node[cur]
                edge_idx = parent_edge_index[cur]
                push = min(push, self.adj[prev][edge_idx].capacity - self.adj[prev][edge_idx].flow)
                cur = prev

            push = min(push, required_flow - total_flow)

            # 如果本次不能推送任何流量，则停止
            if push <= 0:
                break

            # 沿着最短路径推送流量
            cur = sink
            while cur != source:
                prev = parent_node[cur]
                edge_idx = parent_edge_index[cur]
                edge = self.adj[prev][edge_idx]
                edge.flow += push
                reverse_edge = self.adj[edge.to][edge.rev]
                reverse_edge.flow -= push
                cur = prev

            total_flow += push
            # 增加的费用 = 推送流量 * 原始路径费用
            # 原始路径费用 = 缩减费用距离 + h[sink] - h[source]
            total_cost += push * (dist[sink] + h[sink] - h[source])

            # 更新势能
            for i in range(num_nodes):
                 if dist[i] != INF:
                     h[i] += dist[i]

        return total_flow, total_cost

    def solve_multi_source_sink(self, sources, sinks, required_total_flow):
        """
        求解多源多汇最小费用流问题。
        Args:
            sources (list): 实际源点列表，每个元素是 (源点节点索引, 供应量)。
                            源点索引是原始图节点 (0 到 num_original_nodes-1)。
            sinks (list): 实际汇点列表。
                          汇点索引是原始图节点 (0 到 num_original_nodes-1)。
            required_total_flow (int): 总共需要从源点推送到汇点的流量。

        Returns:
            tuple: (实际达到的总流量, 实现该流量的最小费用, 原始边上的流量分配列表)
                   如果 required_total_flow 不可达，则返回最大可能流量及其费用。
        """
        num_original_nodes = self.num_original_nodes
        super_source = num_original_nodes # 超级源点索引
        super_sink = num_original_nodes + 1 # 超级汇点索引
        num_super_nodes = num_original_nodes + 2 # 包含超级节点的总节点数

        # 创建一个内部的 MinCostFlow 实例来表示包含超级节点的图
        # 它会在内部使用 _ssp_solver 方法
        super_mcf = MinCostFlow(num_super_nodes)

        # 复制原始图的边到超级图
        # 注意：这里需要创建新的 Edge 对象，而不是复制引用
        for u_orig in range(num_original_nodes):
            for edge_orig in self.adj[u_orig]:
                 super_mcf.add_edge(u_orig, edge_orig.to, edge_orig.capacity, edge_orig.cost)

        # 添加从超级源点到实际源点的边
        total_supply_from_sources = 0
        for node, supply in sources:
            if node < 0 or node >= num_original_nodes:
                 raise ValueError(f"源点节点 {node} 超出原始图范围 (0-{num_original_nodes-1})")
            if supply > 0:
                super_mcf.add_edge(super_source, node, supply, 0) # 容量=供应量，费用=0
                total_supply_from_sources += supply

        # 添加从实际汇点到超级汇点的边
        # 容量设置为总需求量或总供应量中较大者，确保能汇聚所有流量，费用为 0
        large_capacity = max(required_total_flow, total_supply_from_sources)
        if large_capacity == 0: large_capacity = 1 # 避免容量为0
        for node in sinks:
             if node < 0 or node >= num_original_nodes:
                raise ValueError(f"汇点节点 {node} 超出原始图范围 (0-{num_original_nodes-1})")
             super_mcf.add_edge(node, super_sink, large_capacity, 0) # 容量足够大，费用=0

        # 在包含超级节点的图上，从超级源点到超级汇点运行 SSP 算法
        # 目标流量是 required_total_flow
        actual_flow, min_cost = super_mcf._ssp_solver(super_source, super_sink, required_total_flow)

        # 从超级图的计算结果中提取原始边上的流量分配
        original_edge_flows = []
        # 遍历原始图的节点范围
        for u_super in range(num_original_nodes):
            # 遍历该节点在超级图中的邻接边
            for edge_super in super_mcf.adj[u_super]:
                # 这条边对应原始图中的边需要满足：
                # 1. 终点也是原始图节点
                # 2. 是原始的正向边 (假设原始费用非负)
                if edge_super.to < num_original_nodes and edge_super.cost >= 0:
                    # 只记录有正向净流量通过的边
                    if edge_super.flow > 0:
                         # 记录 (起点, 终点, 最终流量, 原始容量, 原始费用)
                         original_edge_flows.append((u_super, edge_super.to, edge_super.flow, edge_super.capacity, edge_super.cost))

        # 返回结果
        return actual_flow, min_cost, original_edge_flows

# --- 使用您提供的数据进行测试 (多源多汇) ---

# 定义原始节点数量 (0 到 10，共 11 个节点)
num_original_nodes = 13
mcf = MinCostFlow(num_original_nodes)

# 添加原始边 (节点 0-10 之间的边)
mcf.add_edge(0, 1, 60, 18)
mcf.add_edge(0, 2, 60, 19)
mcf.add_edge(0, 3, 60, 17)
mcf.add_edge(1, 4, 40, 16)
mcf.add_edge(1, 5, 30, 14)
mcf.add_edge(2, 5, 50, 16)
mcf.add_edge(2, 6, 30, 17)
mcf.add_edge(3, 6, 40, 19)
mcf.add_edge(4, 7, 60, 19)
mcf.add_edge(5, 4, 20, 15)
mcf.add_edge(5, 7, 30, 16)
mcf.add_edge(5, 8, 40, 15)
mcf.add_edge(6, 5, 20, 15)
mcf.add_edge(6, 9, 40, 13)
mcf.add_edge(7, 10, 60, 18)
mcf.add_edge(7, 8, 30, 17)
mcf.add_edge(8, 10, 50, 19)
mcf.add_edge(8, 9, 30, 14)
mcf.add_edge(9, 10, 50, 17)

# 添加额外边
mcf.add_edge(8, 11, 60, 21)
mcf.add_edge(9, 11, 40, 16)

# 添加从节点12出发的边
mcf.add_edge(12, 2, 60, 19)
mcf.add_edge(12, 3, 60, 15)



# 定义实际源点列表和供应量
# 基于图结构，假设源点是节点 0 和 3，各自供应 60
sources_list = [(0, 30), (12, 30)]

# 定义实际汇点列表
# 基于图结构，假设汇点是节点 7 和 10
sinks_list = [10, 11]

# 定义所需的总流量 (总供应量)
required_flow_amount = 60 # 60 + 60 = 120

# 调用 solve_multi_source_sink 方法求解
actual_flow, min_cost, original_edge_flows = mcf.solve_multi_source_sink(
    sources_list, sinks_list, required_flow_amount
)

# 输出结果
print(f"尝试在原始图 (节点 0-{num_original_nodes-1}) 中，从源点 {sources_list} 到汇点 {sinks_list}，总共推送 {required_flow_amount} 流量.")

if actual_flow == required_flow_amount:
    print(f"\n成功达到所需的总流量 {required_flow_amount}.")
    print(f"实现该流量的最小总费用: {min_cost}")
else:
    print(f"\n警告: 无法达到所需的总流量 {required_flow_amount}.")
    print(f"网络从指定源点到指定汇点的最大可能流量为 {actual_flow}.")
    print(f"达到最大流量的最小总费用: {min_cost}")


# 打印原始边上的最终流量分配，展示流量“路径”
print("\n原始边上的最终流量分配:")
if original_edge_flows:
    for u, v, flow, cap, cost in original_edge_flows:
         print(f"{u} -> {v}: 流量 = {flow}, 容量 = {cap}, 费用 = {cost}")
else:
    print("没有流量通过任何原始边。")