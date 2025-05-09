import genaric.adjact2chrom as a2c

import genaric.chrom2adjact as c2a

import genaric.plotgraph as plotgraph
distinct = [[17,18,24,25],[31,32,38,39]]


# 1first we need arrage the simplest topology for the intra-distinct
#it is simple ,we just use the adjcnct link for every node in the distinct.

N = 7
P = 9
distinct_adjacency_list = {}  # 最终要生成的结构：{node: {neighbor1, neighbor2, ...}}

for dist in distinct:  # 假设 distinct 是一个包含多个节点集合的列表
    for node in dist:
        i_node = node // N  # 计算行号
        j_node = node % N  # 计算列号

        # 检查右邻居是否存在（i_node < N-1 表示不在最右列）
        if i_node < N - 1:
            right_neighbor = (i_node + 1) * N + j_node

            # 如果右邻居也在当前集合 dist 中，则建立连接
            if right_neighbor in dist:
                # 初始化 node 的邻接集合（如果不存在）
                if node not in distinct_adjacency_list:
                    distinct_adjacency_list[node] = set()
                # 添加右邻居
                distinct_adjacency_list[node].add(right_neighbor)

chrom =  a2c.adjacent2chrom(distinct_adjacency_list, N, P)
print(chrom)
base_adjacency_list = c2a.base_chrom2adjacent(chrom, N,P)

adjacency_list = c2a.full_adjacency_list(base_adjacency_list,N,P)

plotgraph.plot_graph_with_auto_curve(adjacency_list, N, P)

 