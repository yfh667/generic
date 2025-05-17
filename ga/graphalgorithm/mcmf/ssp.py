import heapq
from collections import deque

# def solve_mcmf(edges_data, source_node, sink_node, flow_demand):
#     """
#     最小费用最大流（SSP-SPFA版本）
#
#     参数
#     -------
#     edges_data : list[(u, v, cost, capacity)]
#         **顺序必须是 (from, to, cost, capacity)！**
#     source_node : int
#     sink_node   : int
#     flow_demand : int
#
#     返回
#     -------
#     dict  同前
#     """
#     # 1. 建图
#     n = max(max(u, v) for u, v, _, _ in edges_data) + 1
#     graph = [[] for _ in range(n)]
#
#     def _add_edge(fr, to, cap, cost):
#         graph[fr].append([to, len(graph[to]), cap,  cost, 0])   # 正向
#         graph[to].append([fr, len(graph[fr])-1, 0,  -cost, 0])  # 反向
#
#     # *** 这里按 (cost, capacity) 读入，再按 (capacity, cost) 存入 ***
#     for u, v, cst, cap in edges_data:          # ← cost 在前
#         _add_edge(u, v, cap, cst)              # ← capacity 先传入
#
#     # 2. 初始化
#     flow = 0
#     cost = 0
#     h = [0]*n
#     prev_v = [0]*n
#     prev_e = [0]*n
#
#     # 3. 主循环
#     while flow < flow_demand:
#         dist = [float('inf')]*n
#         in_q = [False]*n
#         dist[source_node] = 0
#         q = deque([source_node])
#         in_q[source_node] = True
#
#         # SPFA + 势能
#         while q:
#             u = q.popleft()
#             in_q[u] = False
#             for ei, e in enumerate(graph[u]):
#                 v, _, cap, cst, f = e
#                 if cap - f > 0 and dist[v] > dist[u] + cst + h[u] - h[v]:
#                     dist[v] = dist[u] + cst + h[u] - h[v]
#                     prev_v[v] = u
#                     prev_e[v] = ei
#                     if not in_q[v]:
#                         q.append(v)
#                         in_q[v] = True
#
#         if dist[sink_node] == float('inf'):
#             break
#
#         for i in range(n):
#             if dist[i] < float('inf'):
#                 h[i] += dist[i]
#
#         # 找瓶颈
#         d = flow_demand - flow
#         v = sink_node
#         while v != source_node:
#             u = prev_v[v]
#             ei = prev_e[v]
#             cap = graph[u][ei][2]
#             f   = graph[u][ei][4]
#             d = min(d, cap - f)
#             v = u
#         if d <= 0:
#             break
#
#         # 增广
#         flow += d
#         cost += d * h[sink_node]
#         v = sink_node
#         while v != source_node:
#             u = prev_v[v]
#             ei = prev_e[v]
#             graph[u][ei][4] += d
#             rev = graph[u][ei][1]
#             graph[v][rev][4] -= d
#             v = u
#
#     # 4. 输出
#     flow_details = []
#     for u in range(n):
#         for to, _, cap, cst, f in graph[u]:
#             if cst >= 0 and f > 0:
#                 flow_details.append(
#                     {"from": u, "to": to, "flow": f,
#                      "capacity": cap, "cost": cst})
#
#     status = "Optimal" if flow == flow_demand else "Infeasible"
#     return {
#         "status": status,
#         "total_flow": flow,
#         "total_cost": cost,
#         "flow_details": flow_details
#     }
from collections import deque
from collections import deque

def solve_mcmf(edges_data, source_node, sink_node, flow_demand):
    """
    最小费用最大流（SSP-SPFA版本）

    参数
    -------
    edges_data : list[(u, v, cost, capacity)]
        **顺序必须是 (from, to, cost, capacity)！**
    source_node : int
    sink_node   : int
    flow_demand : int

    返回
    -------
    dict  同前
    """
    # 1. 建图
    n = max(max(u, v) for u, v, _, _ in edges_data) + 1
    graph = [[] for _ in range(n)]

    def _add_edge(fr, to, cap, cost):
        graph[fr].append([to, len(graph[to]), cap,  cost, 0])   # 正向
        graph[to].append([fr, len(graph[fr])-1, 0,  -cost, 0])  # 反向

    # *** 这里按 (cost, capacity) 读入，再按 (capacity, cost) 存入 ***
    for u, v, cst, cap in edges_data:          # ← cost 在前
        _add_edge(u, v, cap, cst)              # ← capacity 先传入

    # 2. 初始化
    flow = 0
    cost = 0
    h = [0]*n
    prev_v = [0]*n
    prev_e = [0]*n

    # 3. 主循环
    while flow < flow_demand:
        dist = [float('inf')]*n
        in_q = [False]*n
        dist[source_node] = 0
        q = deque([source_node])
        in_q[source_node] = True

        # SPFA + 势能
        while q:
            u = q.popleft()
            in_q[u] = False
            for ei, e in enumerate(graph[u]):
                v, _, cap, cst, f = e
                if cap - f > 0 and dist[v] > dist[u] + cst + h[u] - h[v]:
                    dist[v] = dist[u] + cst + h[u] - h[v]
                    prev_v[v] = u
                    prev_e[v] = ei
                    if not in_q[v]:
                        q.append(v)
                        in_q[v] = True

        if dist[sink_node] == float('inf'):
            break

        for i in range(n):
            if dist[i] < float('inf'):
                h[i] += dist[i]

        # 找瓶颈
        d = flow_demand - flow
        v = sink_node
        while v != source_node:
            u = prev_v[v]
            ei = prev_e[v]
            cap = graph[u][ei][2]
            f   = graph[u][ei][4]
            d = min(d, cap - f)
            v = u
        if d <= 0:
            break

        # 增广
        flow += d
        cost += d * h[sink_node]
        v = sink_node
        while v != source_node:
            u = prev_v[v]
            ei = prev_e[v]
            graph[u][ei][4] += d
            rev = graph[u][ei][1]
            graph[v][rev][4] -= d
            v = u

    # 4. 输出
    flow_details = []
    for u in range(n):
        for to, _, cap, cst, f in graph[u]:
            if cst >= 0 and f > 0:
                flow_details.append(
                    {"from": u, "to": to, "flow": f,
                     "capacity": cap, "cost": cst})

    # 如果流量不等于需求，状态为 Infeasible
    if flow == flow_demand:
        status = "Optimal"
    else:
        status = "Infeasible"  # 状态修改为 Infeasible

    return {
        "status": status,
        "total_flow": flow,
        "total_cost": cost,
        "flow_details": flow_details
    }
