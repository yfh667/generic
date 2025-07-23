import draw.snapshotf_romxml as snapshotf_romxml
import graph.drawall as drawall
import genaric2.tegnode as tegnode

import graph.time_2d3 as time_2d
# --- Example Usage ---
import genaric2.distinct_initial as distinct_initial
import genaric2.action_table as action_table
import copy

import genaric2.distinct_initial as distinct_initial

import genaric2.writetoxml as writetoxml
import genaric2.initialize_individual as initialize_individual






##1.我们先获得原始的卫星分布数据，主要是热点区域的拓扑序列数据
N = 10
P=10
start_ts = 1500
end_ts = 1523

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

##2. 完成对热点区域预建链安排

T = target_time_step
setuptime=2


# nodes =distinct_initial.distinct_initial(P,N,T,setuptime,regions_to_color)
# nodes_copy = initialize_individual.initialize_individual(P,N,T,nodes,setuptime)

nodes_copy = distinct_initial.distinct_initial(P, N, T, setuptime, regions_to_color)
# or we can just use  below method
#nodes_copy = initialize_individual.initialize_individual_region(regions_to_color, P, N, T, setuptime)



writetoxml.nodes_to_xml(nodes_copy, "E:\\code\\data\\1\\hot.xml")


adjacency_list_array=action_table.action_map2_shanpshots(nodes_copy, P, N, T)


########here ,we could do more in the adjacncy_list_array





weight_list_array = []

for t in range(T):
    weight = {}
    for i in range(P):
        for j in range(N):
            key = i * N + j
            value = nodes_copy[(i, j, t)].importance
            weight[key] = value
    weight_list_array.append(weight)



vis = time_2d.DynamicGraphVisualizer(
        adjacency_list_array,
        regions_to_color,
        N,
        P,
        node_weights_array=weight_list_array  # ← 这一行传入节点权重
)
vis.show(block=True)
