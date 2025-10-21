from basicSa.route  import dijkstra





def StationA2B(stationnum, start, end, visibility_matrix):
    test_adjacency_list = dijkstra.convert_to_adjacency_list(visibility_matrix)
    #  print(test_adjacency_list)
    # next 我们得到graph
    graph = dijkstra.make_undirected(test_adjacency_list)
    #print(dict(graph))  # 将 defaultdict 转换为普通 dict
    # 最后graph写入最短路径里面
    distances, paths = dijkstra.compute_dis_path(graph, start)

    if end in paths:
    #    print(f"Path from {basicSa} to {end}: {paths[end]}")
        return distances, paths[end]
    else:
     #   print(f"No path from {basicSa} to {end} found.")
        return distances, None

