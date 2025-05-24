import draw.snapshotf_romxml as snapshotf_romxml
import graph.drawall as drawall


import graph.time_2d2 as time_2d
# --- Example Usage ---
import genaric2.distinct_initial as distinct_initial
import genaric2.action_table as action_table
import random
import math
import matplotlib.pyplot as plt
import random
import genaric2.adj2adjacylist as adj2adjaclist

import numpy as np
from networkx.classes import neighbors
import genaric.chrom2adjact as c2a
import ga.graphalgorithm.fcnfp_multi as fcnfp_multi
import genaric2.writetoxml as writetoxml
import genaric2.initialize_individual as initialize_individual
import draw.snapshotf_romxml as snapshotf_romxml
import copy

import genaric2.initiallink as initiallink
import genaric.chrom2adjact as c2a


import genaric.plotgraph as plotgraph

import ga.graphalgorithm.adjact2weight as a2w
import os
import draw.snapshotf_romxml as snapshotf_romxml
import graph.drawall as drawall
import genaric2.tegnode as tegnode

import graph.time_2d2 as time_2d
# --- Example Usage ---
import genaric2.distinct_initial as distinct_initial
import genaric2.action_table as action_table
# 定义目标函数
import genaric2.cross as cross


import genaric2.initialize_individual as initialize_individual

import ga.graphalgorithm.mcmf.ssp_multi as ssp_multi
inter_link_bandwidth = 50
intra_link_bandwidth = 100

cost =1

demand= 30


def test_plot(adjacency_list,distinct,cost):
    edge = a2w.adjacent2edge(adjacency_list , N, inter_link_bandwidth, intra_link_bandwidth, cost)



    SOURCES = {}
    for i in range(len(distinct[0])):
        SOURCES[distinct[0][i]] = 150
    # SOURCES = {17: 150, 18: 150, 24: 150, 25: 150}
    SINKS = distinct[1]
    # 使用新函数求解
    multi_result = ssp_multi.solve_multi_source_sink_with_super_nodes(
        edges_data=edge,
        sources=SOURCES,
        sinks=SINKS
    )


    if multi_result == 0:
        print("No solution found")
    else:
        # 输出结果（与原有格式兼容）
        cost = multi_result['total_cost']

        print("\nMulti-source multi-sink solution via super nodes:")
        print(f"Status: {multi_result['status']}")
        # if multi_result['status'] == "Optimal":
        print(multi_result['status'])
        print(f"Total Fixed Cost: {multi_result['total_cost']}")
        if multi_result['flow_details']:
            print("\nFlow Details (原始网络中的边):")
            for detail in multi_result['flow_details']:
                print(f"  Edge ({detail['from']} -> {detail['to']}): "
                      f"Fixed Cost={detail['cost']}, Flow={detail['flow']}, Capacity={detail['capacity']}")
    # plotgraph.plot_graph_with_auto_curve_distinct(full_adjacency_list[i], N, P,distinct)

    plotgraph.plot_graph_with_auto_curve_distinct(adjacency_list[i], N, P, distinct)


def decode_chromosome(P, N, T,chromosome):
    connection_list = action_table.action_map2_shanpshots(chromosome, P, N, T)
    adjacency_list = adj2adjaclist.adj2adjaclist(connection_list, N, P, T)


    return adjacency_list



import genaric2.cross as cross
def fitness_function(P,N,T,population):
    indictors = []
    decoded_values=[]
    for ind in population:
        adjacency_list = decode_chromosome(P,N,T,ind)
        decoded_values.append(adjacency_list)

   # decoded_values = [decode_chromosome(P,N,T,ind) for ind in population]
    for adjacency_list in decoded_values:
        indictor=0
        for i in range(3,T-2):
            edge = a2w.adjacent2edge(adjacency_list[i], N, inter_link_bandwidth, intra_link_bandwidth, cost)

            distinct = regions_to_color[i]

            SOURCES = {}
            for i in range(len(distinct[0])):
                SOURCES[distinct[0][i]] = demand

            # SOURCES = {17: 150, 18: 150, 24: 150, 25: 150}
            SINKS = distinct[1]
            # 使用新函数求解
            multi_result = ssp_multi.solve_multi_source_sink_with_super_nodes(
                edges_data=edge,
                sources=SOURCES,
                sinks=SINKS
            )


            if multi_result == 0:
                print("No solution found")
            else:
                # 输出结果（与原有格式兼容）
                onecost = multi_result['total_cost']
                indictor += onecost

            # if multi_result == 0:
            #     print("No solution found")
            # else:
            #     # 输出结果（与原有格式兼容）
            #     cost = multi_result['total_cost']
            #
            #     print("\nMulti-source multi-sink solution via super nodes:")
            #     print(f"Status: {multi_result['status']}")
            #     # if multi_result['status'] == "Optimal":
            #     print(multi_result['status'])
            #     print(f"Total Fixed Cost: {multi_result['total_cost']}")
            #     if multi_result['flow_details']:
            #         print("\nFlow Details (原始网络中的边):")
            #         for detail in multi_result['flow_details']:
            #             print(f"  Edge ({detail['from']} -> {detail['to']}): "
            #                   f"Fixed Cost={detail['cost']}, Flow={detail['flow']}, Capacity={detail['capacity']}")
            # # plotgraph.plot_graph_with_auto_curve_distinct(full_adjacency_list[i], N, P,distinct)
            #
            # plotgraph.plot_graph_with_auto_curve_distinct(adjacency_list[i], N, P, distinct)


        indictors.append(indictor)
    #  print(indictor)


    return indictors


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
base =distinct_initial.distinct_initial(P,N,T,setuptime,regions_to_color)

individual1 = writetoxml.xml_to_nodes("E:\\code\\data\\1\\individual1.xml")
individual2 = writetoxml.xml_to_nodes("E:\\code\\data\\1\\individual2.xml")
#




child1, child2 = cross.crossover(individual1, individual2, P, N, T, setuptime)

population = [individual1,individual2,child1,child2]
fitness = fitness_function(P, N, T, population)
print(fitness)

connection_list=action_table.action_map2_shanpshots(individual1, P, N, T)
vis = time_2d.DynamicGraphVisualizer(connection_list, regions_to_color, N, P)
vis.show()
main_plotter, original_points_objs, all_coords = drawall.plot_multi_layer_topology(P, N, target_time_step)
main_plotter = drawall.apply_region_colors(main_plotter, P, N, target_time_step, regions_to_color, all_coords)
connections_list=action_table.action_map2connecttion_list(individual1, P, N, T)
main_plotter = drawall.add_dashed_connections(main_plotter, connections_list)

main_plotter.show(viewup="z", title="Interactive 3D Topology")

for coord, node in individual1.items():
    print(coord, node)
