import networkx as nx
import matplotlib.pyplot as plt


def base_chrom2adjacent(chrom, N, P):
    adjacency_list = {i: set() for i in range(P * N)}  # 初始化邻接表

    for i in range(len(chrom)):
        # 添加右邻居（如果存在）
        if chrom[i] != -1:
            adjacency_list[i].add(chrom[i])
            adjacency_list[chrom[i]].add(i)  # 无向图，双向连接


    return adjacency_list


def full_adjacency_list(adjacency_list, N, P):
    # First convert all neighbor collections to sets if they aren't already
    for node in adjacency_list:
        if not isinstance(adjacency_list[node], set):
            adjacency_list[node] = set(adjacency_list[node])

    # Create a copy to avoid modifying during iteration
    original_nodes = list(adjacency_list.keys())

    for node in original_nodes:
        # Add reverse connections for existing edges
        for neighbor in list(adjacency_list[node]):
            if neighbor not in adjacency_list:
                adjacency_list[neighbor] = set()
            adjacency_list[neighbor].add(node)

        # Calculate left and right neighbors (same row)
        x = node // N  # current row
        y = node % N  # current column

        # Right neighbor (same row, next column)
        up_neighbor = x * N + (y + 1) % N
        adjacency_list[node].add(up_neighbor)

        adjacency_list[up_neighbor].add(node)

        # Left neighbor (same row, previous column)
        down_neighbor = x * N + (y - 1 + N) % N
        adjacency_list[node].add(down_neighbor)

        adjacency_list[down_neighbor].add(node)

    return adjacency_list


