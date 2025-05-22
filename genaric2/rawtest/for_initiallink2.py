from networkx.classes import neighbors

import main.snapshotf_romxml as snapshotf_romxml

import genaric2.initiallink as initiallink
import genaric.chrom2adjact as c2a

import genaric.plotgraph as plotgraph

import ga.graphalgorithm.adjact2weight as a2w
import os
import main.snapshotf_romxml as snapshotf_romxml
import graph.drawall as drawall
import genaric2.tegnode as tegnode
import genaric2.dyplot as dyplot
import graph.time_2d2 as time_2d
# --- Example Usage ---
import genaric2.distinct_initial as distinct_initial
import genaric2.action_table as action_table
import copy
import genaric2.structure2nodes as structure2nodes

import genaric2.writetoxml as writetoxml
import genaric2.initialize_individual as initialize_individual
# def structure2nodes(P,N,T,setuptime,nodes,chrom):
#     for i in range(P):
#         for j in range(N):
#             for k in range(T):
#                 if chrom[(i,j,k)] !=-1 and chrom[(i,j,k)] !=1:
#                     neighbor = chrom[(i,j,k)]
#                   #  print(neighbor)
#                     nodes[(i,j,k)].state = 0
#
#                     nodes[(i,j,k)].rightneighbor = chrom[(i,j,k)]
#
#                     nowtime = k+1
#                     while 1:
#                         if nowtime >= T:
#                             break
#                         if  chrom[(i,j,nowtime)]==1 :
#                             if nowtime<k+setuptime:
#                                 nodes[(i,j,nowtime)].state = 1
#                             else:
#                                 nodes[(i,j,nowtime)].state = 2
#                             nodes[(i, j, nowtime)].rightneighbor = chrom[(i, j, k)]
#                             nowtime = nowtime+1
#                         else:
#                             break

def nodes2structure(P,N,T,nodes):
    structure = {
        (x, y, z): -1
        for x in range(P)
        for y in range(N)
        for z in range(T)
    }
    for i in range(P):
        for j in range(N):
            for k in range(T):
                if nodes[(i,j,k)].state==0:
                    neighbor = nodes[(i,j,k)].rightneighbor
                    structure[(i,j,k)] = neighbor
                elif nodes[(i,j,k)].state!=-1:
                    structure[(i,j,k)] = 1
    return structure


def initialize_individual(P,N,T,nodes):
    # P = 4  # x轴
    # N = 3  # y轴
    # T = 2  # z轴

    nodes_copy = copy.deepcopy(nodes)  # 深复制nodes
    t1 = nodes_copy[(4, 8, 3)]
    t2 = nodes_copy[(4, 8, 4)]
    for i in range(P - 1):
       initiallink.initialize_snap_random_nodes( i,N, P,T,setuptime,nodes_copy)


    return nodes_copy



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

## nodes是所个体的基础骨架
#left_port 是所有个体随机生成的限制，可以认为，每个个体必须要满足同样的核心结构，就类似，每个个体可以长相不一样，基本上都要相同数量的骨头
nodes =distinct_initial.distinct_initial(P,N,T,setuptime,regions_to_color)






                    # 标记设链路期间所占用的时间
##3.生成初始个体，注意，这个还只是其中一个。

# nodes_copy=initialize_individual.initialize_individual(P,N,T,nodes,left_port,setuptime)

nodes_copy = initialize_individual(P,N,T,nodes)

#nodes_copy = nodes
#
# writetoxml.nodes_to_xml(nodes, "E:\\code\\data\\1\\nodes.xml")
#
#
# # 4.解码，
# structure2nodes.structure2nodes(P, N, T, setuptime, nodes, structure)


# writetoxml.nodes_to_xml(nodes_copy, "E:\\code\\data\\1\\nodes_copy.xml")

connection_list=action_table.action_map2_shanpshots(nodes_copy, P, N, T)

import genaric2.adj2adjacylist as adj2adjaclist

# adjacency_list = adj2adjaclist.adj2adjaclist(connection_list, N, P,T)
#
#
#
# import ga.graphalgorithm.mcmf.ssp_multi as ssp_multi
#
#
# full_adjacency_list = adjacency_list
#
# inter_link_bandwidth = 50
# intra_link_bandwidth = 100
#
# cost =1


# for i in range(2,3):
#     edge =a2w.adjacent2edge(full_adjacency_list[i],N,inter_link_bandwidth,intra_link_bandwidth,cost)
#
#     print(full_adjacency_list[i])
#     distinct =  regions_to_color[i]
#     print(distinct)
#
#     SOURCES = {}
#     for i in range(len(distinct[0])):
#         SOURCES[distinct[0][i]] = 150
#    # SOURCES = {17: 150, 18: 150, 24: 150, 25: 150}
#     SINKS = distinct[1]
#     # 使用新函数求解
#     multi_result = ssp_multi.solve_multi_source_sink_with_super_nodes(
#         edges_data=edge,
#         sources=SOURCES,
#         sinks=SINKS
#     )
#     cost = multi_result['total_cost']
#
#
#
#     print(cost)
#
#
#     if multi_result == 0:
#         print("No solution found")
#     else:
#         # 输出结果（与原有格式兼容）
#         cost = multi_result['total_cost']
#
#         print("\nMulti-source multi-sink solution via super nodes:")
#         print(f"Status: {multi_result['status']}")
#        # if multi_result['status'] == "Optimal":
#         print(multi_result['status'] )
#         print(f"Total Fixed Cost: {multi_result['total_cost']}")
#         if multi_result['flow_details']:
#             print("\nFlow Details (原始网络中的边):")
#             for detail in multi_result['flow_details']:
#                 print(f"  Edge ({detail['from']} -> {detail['to']}): "
#                       f"Fixed Cost={detail['cost']}, Flow={detail['flow']}, Capacity={detail['capacity']}")
#     #plotgraph.plot_graph_with_auto_curve_distinct(full_adjacency_list[i], N, P,distinct)
#
#     plotgraph.plot_graph_with_auto_curve_distinct(full_adjacency_list[i], N, P,distinct)





#
#


vis = time_2d.DynamicGraphVisualizer(connection_list, regions_to_color, N, P)
vis.show()


main_plotter, original_points_objs, all_coords = drawall.plot_multi_layer_topology(P, N, target_time_step)
main_plotter = drawall.apply_region_colors(main_plotter, P, N, target_time_step, regions_to_color, all_coords)
connections_list=action_table.action_map2connecttion_list(nodes_copy, P, N, T)
main_plotter = drawall.add_dashed_connections(main_plotter, connections_list)

main_plotter.show(viewup="z", title="Interactive 3D Topology")
#
print("1")
#
for coord, node in nodes_copy.items():
    print(coord, node)
#
