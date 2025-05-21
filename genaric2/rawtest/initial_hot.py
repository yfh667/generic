import os
import main.snapshotf_romxml as snapshotf_romxml
import graph.drawall as drawall
import genaric2.tegnode as tegnode
import genaric2.dyplot as dyplot
import graph.time_2d2 as time_2d
# --- Example Usage ---
import genaric2.distinct_initial as distinct_initial
# Define a dummy file name for demonstration
P_val, N_val = 10, 10

# Define region color information
# Structure: {layer index: [[region 1 point indices], [region 2 point indices], ...]}
# Point indices are based on the indices within a single layer (0 to N*P-1)
regions_to_color = {}

# Specify the time step you want to extract (e.g., 0)

start_ts = 1500
end_ts = 1523

dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"

# Extract and print the satellite lists for each region from the file


# Assuming snapshotf_romxml.extract_region_satellites_from_file returns the relevant data
# I'm not sure if the `extract_region_satellites_from_file` function is implemented, but it should be
# Uncomment and adjust this line as needed based on your actual function:
# region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, target_time_step)

# Iterate over the time steps
region_satellite_groups= snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, start_ts, end_ts)
target_time_step = len(region_satellite_groups)
print(f"\nExtracting data for time step {target_time_step} from '{dummy_file_name}'...")
for i in range(len(region_satellite_groups)):
        region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
        u =region_satellite_group[0]
        v = region_satellite_group[3]
        o = [u,v]
        regions_to_color[i] = o  # Corrected append to dictionary assignment


main_plotter, original_points_objs, all_coords = drawall.plot_multi_layer_topology(P_val, N_val, target_time_step)


main_plotter = drawall.apply_region_colors(main_plotter, P_val, N_val, target_time_step, regions_to_color, all_coords)






P = P_val  # 举例
N = N_val
T = target_time_step
setuptime=2



nodes =distinct_initial.distinct_initial(P,N,T,setuptime,regions_to_color)

import genaric2.action_table as action_table

connections_list=action_table.action_map2connecttion_list(nodes, P, N, T)
adj_list_array=action_table.action_map2_shanpshots(nodes, P, N, T)



print(adj_list_array)



vis = time_2d.DynamicGraphVisualizer(adj_list_array, regions_to_color, N, P)
vis.show()


# 4. Add dashed connections
main_plotter = drawall.add_dashed_connections(main_plotter, connections_list)
print("Vedo window is about to display. Please use the mouse for interaction:")
print("- Left-click and drag: Rotate")
print("- Middle-click and drag (or Shift + Left-click drag): Pan")
print("- Scroll wheel (or Right-click drag): Zoom")
main_plotter.show(viewup="z", title="Interactive 3D Topology")

for coord, node in nodes.items():
    print(coord, node)

