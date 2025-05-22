
import genaric2.initiallink as initiallink
import copy



def initialize_individual(P, N, T, nodes, setuptime):
    # P = 4  # x轴
    # N = 3  # y轴
    # T = 2  # z轴

    nodes_copy = copy.deepcopy(nodes)  # 深复制nodes

    for i in range(P - 1):
        initiallink.initialize_snap_random_nodes(i, N, P, T, setuptime, nodes_copy)

    return nodes_copy
