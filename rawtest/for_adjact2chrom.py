
#

import graph_ga.adjact2chrom as a2c
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

N=4
P=4
o =  a2c.adjacent2chrom(adjacency_list, N, P)
print(o)

