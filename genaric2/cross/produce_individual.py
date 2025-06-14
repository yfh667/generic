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



import genaric2.cross as cross


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

individual1 = initialize_individual.initialize_individual(P,N,T,base,setuptime)

individual2 = initialize_individual.initialize_individual(P,N,T,base,setuptime)

writetoxml.nodes_to_xml(individual1, "E:\\code\\data\\1\\individual1.xml")
writetoxml.nodes_to_xml(individual2, "E:\\code\\data\\1\\individual2.xml")