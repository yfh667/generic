from collections import deque


def shortest_path(graph, start, end):
    """
    使用BFS算法寻找图中两点之间的最短路径

    参数:
        graph: 图的邻接表表示，字典形式
        start: 起点
        end: 终点

    返回:
        最短路径的节点列表，如果不存在路径则返回None
    """
    # 如果起点或终点不在图中，直接返回None
    if start not in graph or end not in graph:
        return None

    # 特殊情况：起点和终点相同
    if start == end:
        return [start]

    # 初始化队列和已访问字典
    queue = deque()
    queue.append(start)
    visited = {start: None}  # 键是节点，值是前驱节点

    while queue:
        current = queue.popleft()

        # 遍历当前节点的所有邻居
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)

                # 如果找到终点，回溯构建路径
                if neighbor == end:
                    path = []
                    node = end
                    while node is not None:
                        path.append(node)
                        node = visited[node]
                    return path[::-1]  # 反转路径

    # 如果没有找到路径
    return None

if __name__ == '__main__':
    # 示例使用
    graph = {
        # 这里放你提供的整个图结构
        2: {1, 3, 12},
        0: {1, 9, 10},
        # ... 其余节点 ...
        97: {96, 98}
    }

    # 查找从节点0到节点99的最短路径
    path = shortest_path(graph, 0, 99)
    print(path)
