
import genaric2.initiallink as initiallink
import copy

import genaric2.distinct_initial as distinct_initial
import genaric2.probability.mean_shortest_path as mean_shortest_path
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

    # here we need initial the weight of the nodes.
    weight_list_array, adjacency_list_array = mean_shortest_path.initial_weight(P, N, T, regions_to_color)

    for i in range(T):
        weight = weight_list_array[i]

        for k in range(P):
            for j in range(N):
                nodes[(k, j, i)].importance=weight[k * N + j]



    return nodes


def initialize_individual_grid(P, N, T, nodes, setuptime):
    # P = 4  # x轴
    # N = 3  # y轴
    # T = 2  # z轴

    nodes_copy = copy.deepcopy(nodes)  # 深复制nodes


    initiallink.initialize_snap_grid_nodes( N, P, T, setuptime, nodes_copy)


    return nodes_copy

def initialize_individual_grid(P, N, T, nodes, setuptime):
    # P = 4  # x轴
    # N = 3  # y轴
    # T = 2  # z轴

    nodes_copy = copy.deepcopy(nodes)  # 深复制nodes


    initiallink.initialize_snap_grid_nodes( N, P, T, setuptime, nodes_copy)


    return nodes_copy


def initialize_individual_grid_full_graph(P, N, T, nodes, setuptime):
    # P = 4  # x轴
    # N = 3  # y轴
    # T = 2  # z轴

    nodes_copy = copy.deepcopy(nodes)  # 深复制nodes


    initiallink.initialize_snap_full_nodes( N, P, T, setuptime, nodes_copy)


    return nodes_copy



def initialize_individual_region_grid(regions_to_color,P, N, T, setuptime):
    # P = 4  # x轴
    # N = 3  # y轴
    # T = 2  # z轴
    nodes = distinct_initial.distinct_initial(P, N, T, setuptime, regions_to_color)
    nodes=initialize_individual_grid(P, N, T, nodes, setuptime)
    return nodes

def initialize_individual_region_full_graph(regions_to_color,P, N, T, setuptime):
    # P = 4  # x轴
    # N = 3  # y轴
    # T = 2  # z轴
    nodes = distinct_initial.distinct_initial(P, N, T, setuptime, regions_to_color)
    nodes=initialize_individual_grid_full_graph(P, N, T, nodes, setuptime)
    return nodes

