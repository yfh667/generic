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



# 3. Define the points that need to be connected (using actual 3D coordinates)
# Ensure these coordinates are consistent with SPACING and LAYER_HEIGHT settings
# connections_list = [
#     [(0, 0, 0), (1, 2, 4)],  # Vertical connection
#     [(1, 1, 1), (1, 1, 2)],  # Cross-layer connection
#     [(2, 2, 0), (2, 2, 2)],  # Same-layer diagonal connection
# ]
#
# # 4. Add dashed connections
# main_plotter = drawall.add_dashed_connections(main_plotter, connections_list)

# # 假设 P, N, T 是已知的维度上限

# # 初始化三维数组 nodes，键为 (x, y, z)，值为 tegnode 实例
# nodes = {}
#
#
# for x in range(P ):      # x ∈ [0, P]
#     for y in range(N ):  # y ∈ [0, N]
#         for z in range(T ):  # z ∈ [0, T]
#             nodes[(x, y, z)] = tegnode.tegnode(
#                 asc_nodes_flag=False,  # 或初始为None等你自己定义
#                 rightneighbor=None,
#                 state=-1  # 默认初始为free
#             )
#
# def find_the_start(time_now, node_id, neighbor_id, group_idx, region_satellite_groups):
#     while time_now > 0:
#         region = region_satellite_groups[time_now][group_idx]
#         if node_id in region and neighbor_id in region:
#             time_now -= 1
#         else:
#             break
#     return time_now
#
# # 分配状态：设定链接、工作状态、设置右邻居指向等
# def assign_state(nodes, start_time, end_time, start_node_id, end_node_id, setuptime, N):
#     if end_time - start_time > setuptime:
#         x_start, y_start = divmod(start_node_id, N)
#         x_end, y_end = divmod(end_node_id, N)
#
#         # 设置链接建立时刻
#         nodes[(x_start, y_start, start_time)].state = 0
#         nodes[(x_start, y_start, start_time)].asc_nodes_flag = 1
#         nodes[(x_start, y_start, start_time)].rightneighbor = (x_end, y_end, start_time + setuptime)
#
#         # 设置设置阶段状态
#         for t in range(start_time+1, start_time + setuptime):
#             nodes[(x_start, y_start, t)].state = 1
#             nodes[(x_start, y_start, t)].asc_nodes_flag = 1
#           #  nodes[(x_start, y_start, t)].rightneighbor = (x_end, y_end, start_time + setuptime)
#
#         # 设置工作阶段状态
#         for t in range(start_time + setuptime, end_time + 1):
#             nodes[(x_start, y_start, t)].state = 2
#             nodes[(x_start, y_start, t)].asc_nodes_flag = 1
#             nodes[(x_start, y_start, t)].rightneighbor = (x_end, y_end, t)
#
#
# # 主循环：从后往前处理区域，建立右邻居连接
# for t in range(T - 1, -1, -1):
#     region_groups = regions_to_color[t]
#     for group_idx, region in enumerate(region_groups):
#         for node_id in region:
#             x_node, y_node = divmod(node_id, N)
#             if nodes[(x_node, y_node, t)].state == -1:
#                 # 检查是否有右邻居（非最右列）
#                 if x_node < P - 1:
#                     neighbor_id = (x_node + 1) * N + y_node
#                     if neighbor_id in region:
#
#                         start_t = find_the_start(t - 1, node_id, neighbor_id, group_idx, regions_to_color)
#                         assign_state(nodes,  start_t + 1,t, node_id, neighbor_id, setuptime, N)
#
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


