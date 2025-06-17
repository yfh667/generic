
import genaric2.initiallink as initiallink
import copy

import genaric2.distinct_initial as distinct_initial

def initialize_individual(P, N, T, nodes, setuptime):
    # P = 4  # x轴
    # N = 3  # y轴
    # T = 2  # z轴

    nodes_copy = copy.deepcopy(nodes)  # 深复制nodes

    for i in range(P - 1):
        initiallink.initialize_snap_random_nodes(i, N, P, T, setuptime, nodes_copy)

    return nodes_copy


def initialize_individual_region(regions_to_color,P, N, T, setuptime):
    # P = 4  # x轴
    # N = 3  # y轴
    # T = 2  # z轴
    nodes = distinct_initial.distinct_initial(P, N, T, setuptime, regions_to_color)
    nodes=initialize_individual(P, N, T, nodes, setuptime)
    return nodes