
import genaric.chrom2adjact as c2a

import genaric.plotgraph as plotgraph
import genaric.adjact2chrom as a2c
# 定义邻接表（完整连接关系）
# adjacency_list = {
#     0: {1, 6},
#     1: [0, 2, 3],
#     2: [1, 5],
#     3: [0, 4,7],
#     4: [ 3, 5],
#     5: [2, 4,8]
# }
adjacency_list = {
    0: {5},
    1: {4},
    2: {7},
    3: {6},
    4: {8},
    5: {9},
    6: {10},
    7: {11},
    8: {12},
    9: {14},
    10: {13},
    11: {15}
}

N=36
P=18
chrom =  a2c.adjacent2chrom(adjacency_list, N, P)

base_adjacency_list = c2a.base_chrom2adjacent(chrom, N,P)

adjacency_list = c2a.full_adjacency_list(base_adjacency_list,N,P)

plotgraph.plot_graph_with_auto_curve(adjacency_list, N, P)

