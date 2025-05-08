
import ga.graphalgorithm.fcnfp as fcnfp
def add_super_nodes(edges_data, sources, sinks):
    """
    将多源多汇问题转换为单源单汇问题（通过添加超级源点和超级汇点）

    参数:
    edges_data (list): 原始边数据，格式与之前一致
    sources (dict): 源节点及其供应量，如 {0:50, 12:50}
    sinks (list): 汇节点列表，如 [10, 11]

    返回:
    tuple: (转换后的边数据, 超级源点ID, 超级汇点ID, 总需求)
    """
    # 生成唯一ID（假设原节点ID均为非负整数）
    super_source = max(max(u, v) for u, v, *_ in edges_data) + 1
    super_sink = super_source + 1

    # 添加超级源点连接
    new_edges = edges_data.copy()
    for node in sources:
        # 添加边：超级源点 -> 原源点，容量=该源点的供应量，固定费用0
        new_edges.append((super_source, node, 0, sources[node]))

    # 添加汇点连接
    for node in sinks:
        # 添加边：原汇点 -> 超级汇点，容量=极大值（模拟无限），固定费用0
        new_edges.append((node, super_sink, 0, 1e9))

    total_demand = sum(sources.values())
    return new_edges, super_source, super_sink, total_demand


def solve_multi_source_sink_with_super_nodes(edges_data, sources, sinks):
    """
    通过超级节点解决多源多汇问题的入口函数
    """
    # 步骤1：添加超级节点
    modified_edges, super_src, super_snk, total_demand = add_super_nodes(edges_data, sources, sinks)

    # 步骤2：调用原有单源单汇函数
    result = fcnfp.solve_fixed_charge_network_flow(
        edges_data=modified_edges,
        source_node=super_src,
        sink_node=super_snk,
        flow_demand=total_demand
    )

    # 步骤3：过滤结果（移除超级节点相关的边）

    if result["flow_details"] is not None:
            original_nodes = set(u for u, _, *_ in edges_data) | set(v for _, v, *_ in edges_data)
            result["flow_details"] = [
                detail for detail in result["flow_details"]
                if detail["from"] in original_nodes and detail["to"] in original_nodes
            ]
    else:
        return 0
    return result

