from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

import pulp
def solve_fixed_charge_network_flow(edges_data, source_node, sink_node, flow_demand):
    """
    解决固定费用网络流问题。

    参数:
    edges_data (list of tuples): 网络中所有边的列表。
                                 每条边是一个元组 (u, v, fixed_cost, capacity)，其中:
                                 u: 起点节点
                                 v: 终点节点
                                 fixed_cost: 使用该边的固定费用 (如果流量 > 0)
                                 capacity: 该边的最大容量
    source_node (int): 源节点的标识符。
    sink_node (int): 汇节点的标识符。
    flow_demand (float or int): 需要从源节点输送到汇节点的总流量。

    返回:
    dict: 包含求解结果的字典。
          如果找到最优解:
          {
              "status": "Optimal",
              "total_fixed_cost": total_fixed_cost_value,
              "flow_details": [
                  {"from": u, "to": v, "fixed_cost": fc, "flow": flow_on_edge, "capacity": cap},
                  ...
              ]
          }
          如果问题不可行或无界，或未找到最优解:
          {
              "status": "Infeasible" (或其他状态字符串),
              "total_fixed_cost": None,
              "flow_details": None
          }
    """

    # 1. 创建问题实例
    prob = LpProblem("FixedChargeOnlyMinCostFlow", LpMinimize)
   # prob.solve(pulp.PULP_CBC_CMD(msg=False))  # 添加 msg=False 参数

    # 2. 创建变量
    edge_flow_vars = {}  # 存储流量变量 x_uv
    edge_usage_vars = {}  # 存储边的使用指示变量 y_uv
    fixed_cost_terms = []  # 存储目标函数中的固定成本项

    # 为每条边创建变量，并构建目标函数相关的成本项
    for i, edge_info in enumerate(edges_data):
        u, v, fixed_cost_for_edge, capacity = edge_info

        # 流量变量 x_uv_i (索引i用于区分平行边，尽管当前数据格式不直接支持)
        x_uv = LpVariable(f"x_{u}_{v}_{i}", lowBound=0, upBound=capacity, cat='Continuous')
        edge_flow_vars[(u, v, i)] = x_uv

        # 使用指示变量 y_uv_i (二元)
        y_uv = LpVariable(f"y_{u}_{v}_{i}", cat='Binary')
        edge_usage_vars[(u, v, i)] = y_uv

        # 将固定成本项添加到列表，用于后续构建目标函数
        fixed_cost_terms.append(fixed_cost_for_edge * y_uv)

        # 约束：如果流量 x_uv > 0, 则 y_uv 必须为 1 (x_uv <= capacity * y_uv)
        prob += x_uv <= capacity * y_uv, f"Link_x_y_{u}_{v}_{i}"

    # 3. 设置目标函数: 最小化总固定费用
    prob += lpSum(fixed_cost_terms), "TotalFixedCost"

    # 4. 添加流量守恒约束
    # 首先，获取网络中所有参与边的节点集合
    nodes_in_network = set()
    for u, v, _, _ in edges_data:
        nodes_in_network.add(u)
        nodes_in_network.add(v)

    # 确保源节点和汇节点（如果它们可能没有出现在edges_data中，例如孤立节点）被考虑
    # 但实际上，如果源/汇没有边，对于正需求是不可行的
    # nodes_in_network.add(source_node)
    # nodes_in_network.add(sink_node)

    for node in nodes_in_network:
        # 计算流入该节点的总流量
        flow_in = lpSum(edge_flow_vars[(u_in, n_curr, idx)]
                        for idx, (u_in, n_curr, _, _) in enumerate(edges_data) if n_curr == node)

        # 计算从该节点流出的总流量
        flow_out = lpSum(edge_flow_vars[(n_curr, v_out, idx)]
                         for idx, (n_curr, v_out, _, _) in enumerate(edges_data) if n_curr == node)

        # 根据节点类型（源、汇、中间节点）设置流量守恒（或需求/供应）
        if node == source_node:
            prob += flow_out - flow_in == flow_demand, f"FlowConservation_Source_{node}"
        elif node == sink_node:
            prob += flow_in - flow_out == flow_demand, f"FlowConservation_Sink_{node}"
        else:
            prob += flow_in - flow_out == 0, f"FlowConservation_Intermediate_{node}"

    # 如果源点或汇点不在 `nodes_in_network` 中 (即没有连接的边),
    # 且 flow_demand > 0, PuLP 通常会因找不到变量而报错或判定不可行。
    # 一个更鲁棒的做法是确保源点和汇点的约束总是被尝试添加，
    # PuLP会处理那些没有相关变量的约束（通常导致不可行）。
    # 然而，对于一个有意义的问题，源和汇点必须是 `nodes_in_network` 的一部分。
    if source_node not in nodes_in_network and flow_demand > 0:
        # 添加一个无用的约束以强制处理源节点，但这通常表示图有问题
        # 或者更简单地，可以提前检查并报错
        pass  # 通常，如果源节点没有出边， flow_out 会是0，导致不可行
    if sink_node not in nodes_in_network and flow_demand > 0:
        pass  # 类似地，如果汇点没有入边，flow_in 会是0，导致不可行

    # 5. 求解问题
   # prob.solve(PULP_CBC_CMD(msg=False)) # 可以取消注释以隐藏求解器日志
    #prob.solve()
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 6. 处理并返回结果
    status_str = LpStatus[prob.status]

    if status_str == 'Optimal':
        optimal_cost = value(prob.objective)
        flow_details_list = []
        for idx, edge_spec in enumerate(edges_data):
            u, v, fixed_c, cap = edge_spec
            # 使用原始键 (u,v,i) 来获取变量
            flow_val = value(edge_flow_vars.get((u, v, idx)))
            usage_val = value(edge_usage_vars.get((u, v, idx)))

            if usage_val is not None and usage_val > 0.5:  # 检查边是否被使用
                flow_details_list.append({
                    "from": u,
                    "to": v,
                    "fixed_cost": fixed_c,
                    "flow": round(flow_val if flow_val is not None else 0, 5),  # 四舍五入流量值
                    "capacity": cap
                })

        return {
            "status": status_str,
            "total_fixed_cost": optimal_cost,
            "flow_details": flow_details_list
        }
    else:
        return {
            "status": status_str,
            "total_fixed_cost": None,
            "flow_details": None
        }
