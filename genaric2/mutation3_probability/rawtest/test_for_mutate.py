
import genaric2.writetoxml as writetoxml

import draw.snapshotf_romxml as snapshotf_romxml



import draw.snapshotf_romxml as snapshotf_romxml
import graph.drawall as drawall
import genaric2.tegnode as tegnode

import graph.time_2d2 as time_2d
# --- Example Usage ---
import genaric2.distinct_initial as distinct_initial
import genaric2.action_table as action_table
import copy


import genaric2.writetoxml as writetoxml
import genaric2.initialize_individual as initialize_individual


import random
import genaric2.TopoSeqValidator as TopoSeqValidator
import genaric2.mutation2.mutate as mutate
inter_link_bandwidth = 50
intra_link_bandwidth = 100
cost = 1
demand = 30

maxhop = 30

if __name__ == '__main__':
    ##1.我们先获得原始的卫星分布数据，主要是热点区域的拓扑序列数据
    N = 10
    P = 10
    start_ts = 1500
    end_ts = 1523

    dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"

    regions_to_color = {}
    # Iterate over the time steps
    region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, start_ts, end_ts)
    target_time_step = len(region_satellite_groups)
    T = target_time_step
    for i in range(len(region_satellite_groups)):
        region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
        u = region_satellite_group[0]
        v = region_satellite_group[3]
        o = [u, v]
        regions_to_color[i] = o  # Corrected append to dictionary assignment

    ##2. 完成对热点区域预建链安排

    T = target_time_step
    setuptime = 2


    #nodes_copy = initialize_individual.initialize_individual_region(regions_to_color,P, N, T, setuptime)




    individual1 = writetoxml.xml_to_nodes("E:\\code\\data\\1\\hot.xml")
    # fitness, indictors_seq = fitness_function(P, N, T, [individual1], regions_to_color)


    child1,_ = mutate.mutate_f(individual1, regions_to_color, inter_link_bandwidth, intra_link_bandwidth, cost, P, N, T,
                             setuptime)


    # flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(individual1, P, N, T, setuptime)


    flag1, connection1_test, connection2_test = TopoSeqValidator.TologialSequenceValidator(child1, P, N, T, setuptime)
    writetoxml.nodes_to_xml(child1, "E:\\code\\data\\2\\mutate_child2.xml")

    vis = time_2d.DynamicGraphVisualizer(connection1_test, regions_to_color, N, P)
    vis.show()

    main_plotter, original_points_objs, all_coords = drawall.plot_multi_layer_topology(P, N, target_time_step)
    main_plotter = drawall.apply_region_colors(main_plotter, P, N, target_time_step, regions_to_color, all_coords)

    main_plotter = drawall.add_dashed_connections(main_plotter, connection1_test)

    main_plotter.show(viewup="z", title="Interactive 3D Topology")
    print("1")