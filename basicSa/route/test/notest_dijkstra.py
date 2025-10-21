

from basicSa.route  import dijkstra
if __name__ == '__main__':

    visibility_matrix = [
        [0, 2, 0, 0, 1],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 4, 0],
        [0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0]
    ]

    # Convert the visibility matrix to an adjacency list
    test_adjacency_list = dijkstra.convert_to_adjacency_list(visibility_matrix)
    print(test_adjacency_list)
    # next 我们得到graph
    graph = dijkstra.make_undirected(test_adjacency_list)
    print(dict(graph))  # 将 defaultdict 转换为普通 dict
    # 最后graph写入最短路径里面
    test_distances, test_paths = dijkstra.compute_dis_path(graph, 0)
    # Display the adjacency list to the user


    print(dict(test_distances))  # 显示最短路径的距离
    print(test_paths)  # 显示最短路径的节点顺序


