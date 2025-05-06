# 示例用法
import graph_ga.adjact2chrom as a2c
import graph_ga.dijstra as dij
import graph_ga.chrom2adjact as c2a
import graph_ga.plotgraph as plotgraph
if __name__ == "__main__":
    # 邻接表示例（无权重图）
    N=4
    P=4
    base_adjacency_list = {
        0: {5},
        1: {9},
        2: {7},
        3: {6},
        4: {8},
        5: {},
        6: {10},
        7: {11},
        8: {12},
        9: {14},
        10: {13},
        11: {15}
    }

    adjacency_list = c2a.full_adjacency_list(base_adjacency_list, N, P)
    plotgraph.plot_graph_with_auto_curve(adjacency_list,N,P)

    chrom = a2c.adjacent2chrom(adjacency_list, N, P)
    print(f"chrom is {chrom}")

    start = 0
    end = 14

    indictor =0
    path, distance = dij.dijkstra_shortest_path(adjacency_list, start, end)
    print(f"最短路径: {path}")  # 输出: [0, 3, 6, 7, 8]
    print(f"最短距离: {distance}")  # 输出: 4
    indictor = indictor+distance

    start = 3
    end = 13
    path, distance = dij.dijkstra_shortest_path(adjacency_list, start, end)
    indictor = indictor + distance

    print(indictor)