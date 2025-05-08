import heapq


def dijkstra_shortest_path(adjacency_list, start, end):
    """
    使用Dijkstra算法计算最短路径
    :param adjacency_list: 邻接表（字典形式，如 {0: {1: 5, 2: 3}}）
    :param start: 起点节点
    :param end: 终点节点
    :return: (最短路径列表, 最短距离)
    """
    # 初始化距离字典（默认无穷大）
    distances = {node: float('inf') for node in adjacency_list}
    distances[start] = 0

    # 优先队列（存储 (距离, 当前节点)）
    priority_queue = [(0, start)]

    # 记录前驱节点（用于回溯路径）
    predecessors = {}

    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)

        # 如果到达终点，提前退出
        if current_node == end:
            break

        # 遍历邻居
        for neighbor in adjacency_list.get(current_node, []):
            # 假设所有边权重为1（未加权图）
            distance = current_dist + 1

            # 如果找到更短路径
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # 回溯构建路径
    path = []
    if end in predecessors or end == start:
        current = end
        while current != start:
            path.append(current)
            current = predecessors.get(current)
        path.append(start)
        path.reverse()

    return path, distances.get(end, float('inf'))


