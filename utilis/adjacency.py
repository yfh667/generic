def complete_undirected_graph(adjacency_list):
    """
    补全无向图的邻接表，确保所有边的双向性

    参数:
        adjacency_list: 原始的邻接表（字典形式，值可以是集合或列表）

    返回:
        补全后的对称邻接表（所有值转为集合）
    """
    # 首先复制原始邻接表（避免修改原数据）
    fixed_adj = {node: set(neighbors) for node, neighbors in adjacency_list.items()}

    # 遍历每个节点及其邻居
    for node in list(fixed_adj.keys()):  # 使用list()避免字典大小变化问题
        for neighbor in fixed_adj[node]:
            # 如果邻居不在邻接表中，先添加它
            if neighbor not in fixed_adj:
                fixed_adj[neighbor] = set()
            # 确保邻居的邻接列表包含当前节点
            if node not in fixed_adj[neighbor]:
                fixed_adj[neighbor].add(node)

    return fixed_adj
