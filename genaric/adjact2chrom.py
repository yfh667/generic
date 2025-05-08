import networkx as nx
import matplotlib.pyplot as plt


def adjacent2chrom(adjacency_list, N, P):
    # 将邻接表转换为右邻居数组（仅记录右邻居）
    a = [-1] * (P - 1) * N  # 初始化数组，-1表示无右邻居)

    for node, neighbors in adjacency_list.items():
        orbit = node // N
        for i in neighbors:
            negighbor_orbit = i // N
            if orbit != negighbor_orbit and negighbor_orbit > orbit:
                a[node] = i

    return a
