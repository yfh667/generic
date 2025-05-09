

import ga.initiallink as initiallink
import genaric.chrom2adjact as c2a

import genaric.plotgraph as plotgraph




def initialize_individual(N,P,distinct):
    individual = [-1] * (P - 1) * N
    # left_link =[0]*(P)*N
    right_link = [-1] * (P) * N
    own_link = [-1] * (P) * N
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

        nowlink = initiallink.initialize_links3(N, i, P, right_link,own_link)

        individual[i * N:(i + 1) * N] = nowlink

    return individual

import os
import main.snapshotf_romxml as snapshotf_romxml

import random
import math
import matplotlib.pyplot as plt
import random
import numpy as np
from networkx.classes import neighbors
import genaric.chrom2adjact as c2a
import ga.graphalgorithm.fcnfp_multi as fcnfp_multi
import genaric.chrom2adjact as c2a
import genaric.plotgraph as plotgraph
import satnode.relative_position as relpos
import ga.graphalgorithm.adjact2weight as a2w
import ga.initiallink as initiallink
# 示例用法
import genaric.adjact2chrom as a2c
import genaric.dijstra as dij
import genaric.chrom2adjact as c2a
import genaric.plotgraph as plotgraph

target_time_step = 0
dummy_file_name = "E:\code\data\station_visible_satellites.xml"
# Extract and print the satellite lists for each region from the file
print(f"\nExtracting data for time step {target_time_step} from '{dummy_file_name}'...")
region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, target_time_step)

print(f"Satellite groups for time step {target_time_step}:")
for i, satellite_list in enumerate(region_satellite_groups):
    print(f"Region {i}: {satellite_list}")
distinct =  []


integer_satellite_groups_0 = [int(satellite_id) for satellite_id in region_satellite_groups[0]]

integer_satellite_groups_1 = [int(satellite_id) for satellite_id in region_satellite_groups[3]]


distinct.append(integer_satellite_groups_0)
distinct.append(integer_satellite_groups_1)


N = 36
P=18
# distinct = [[17,18,24,25],[31,32,38,39]]
chrom = initialize_individual(N,P,distinct)

base_adjacency_list = c2a.base_chrom2adjacent(chrom, N,P)

adjacency_list = c2a.full_adjacency_list(base_adjacency_list,N,P)

plotgraph.plot_graph_with_auto_curve_distinct(adjacency_list, N, P,distinct)