
import main.snapshotf_romxml as snapshotf_romxml

import genaric2.initiallink as initiallink
import genaric.chrom2adjact as c2a

import genaric.plotgraph as plotgraph




def initialize_individual(P,N,T,distinct):
    # P = 4  # x轴
    # N = 3  # y轴
    # T = 2  # z轴

    #individual = [[[-1 for _ in range(T)] for _ in range(N)] for _ in range(P)]
    individual = [[[-1 for _ in range(N)] for _ in range(P)] for _ in range(T)]
    # individual = [-1] * (P - 1) * N
    # left_link =[0]*(P)*N
   # right_link =  [[[-1 for _ in range(T)] for _ in range(N)] for _ in range(P-1)]
    right_link = [[[-1 for _ in range(N)] for _ in range(P)] for _ in range(T)]
    own_link = [[[-1 for _ in range(N)] for _ in range(P)] for _ in range(T)]

   # own_link =[[[-1 for _ in range(T)] for _ in range(N)] for _ in range(P)]

    for i in range(len(distinct)):
        regions = distinct[i]
        for region in regions:

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
                    right_link[right_neighbor] = 1
                    own_link[node] = right_neighbor
                    individual[node] = right_neighbor

    for i in range(P - 1):
        # if individual[i] == -1:

        nowlink = initiallink.initialize_snap(N, i, P, right_link,own_link)

        individual[i * N:(i + 1) * N] = nowlink

    return individual

start_ts = 1500
end_ts = 1505

dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"

regions_to_color = {}
# Iterate over the time steps
region_satellite_groups= snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, start_ts, end_ts)
target_time_step = len(region_satellite_groups)
T = target_time_step
for i in range(len(region_satellite_groups)):
        region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
        u =region_satellite_group[0]
        v = region_satellite_group[3]
        o = [u,v]
        regions_to_color[i] = o  # Corrected append to dictionary assignment


N = 10
P=10




chrom = initialize_individual(P,N,T,regions_to_color)

base_adjacency_list = c2a.base_chrom2adjacent(chrom, N,P)

adjacency_list = c2a.full_adjacency_list(base_adjacency_list,N,P)

plotgraph.plot_graph_with_auto_curve(adjacency_list, N, P)