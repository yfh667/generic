# 从XML读取节点数据
import draw.snapshotf_romxml as snapshotf_romxml
import graph.drawall as drawall

import graph.time_2d2 as time_2d
# --- Example Usage ---
import genaric2.distinct_initial as distinct_initial
import genaric2.action_table as action_table

import genaric2.initialize_individual as initialize_individual

import genaric2.cross as cross



import genaric2.writetoxml as writetoxml



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


print(1)


individual1 = writetoxml.xml_to_nodes("E:\\code\\data\\1\\individual2.xml")
connection_list=action_table.action_map2_shanpshots(individual1, P, N, T)
vis = time_2d.DynamicGraphVisualizer(connection_list, regions_to_color, N, P)
vis.show()
main_plotter, original_points_objs, all_coords = drawall.plot_multi_layer_topology(P, N, target_time_step)
main_plotter = drawall.apply_region_colors(main_plotter, P, N, target_time_step, regions_to_color, all_coords)
connections_list=action_table.action_map2connecttion_list(individual1, P, N, T)
main_plotter = drawall.add_dashed_connections(main_plotter, connections_list)

main_plotter.show(viewup="z", title="Interactive 3D Topology")



