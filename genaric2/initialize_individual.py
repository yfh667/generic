
import genaric2.initiallink as initiallink
import copy
import genaric2.structure2nodes as structure2nodes

def initialize_individual(P,N,T,nodes,left_port,setuptime):
    # P = 4  # x轴
    # N = 3  # y轴
    # T = 2  # z轴
    individual = {
        (x, y, z): -1
        for x in range(P)
        for y in range(N)
        for z in range(T)
    }
    nodes_copy = copy.deepcopy(nodes)  # 深复制nodes

    for i in range(P - 1):
        nowlink = initiallink.initialize_snap_random( i,N, P,T, left_port,setuptime,nodes_copy)

        for j in range(N):
            for k in range(T):
                individual[(i,j,k)] = nowlink[(j, k)]
    structure2nodes.structure2nodes(P, N, T, setuptime, nodes_copy, individual)
    return nodes_copy

