

import genaric2.writetoxml as writetoxml
import draw.snapshotf_romxml as snapshotf_romxml
import graph.time_2d2 as time_2d
import genaric2.action_table as action_table
import graph.drawall as drawall
import genaric2.mutation.mutate3 as mutate

if __name__ == '__main__':
    N = 10
    P=10
    T=23
    start_ts = 1500
    setuptime=2
    end_ts = 1523
    dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"

    # 提取区域信息
    regions_to_color = {}
    region_satellite_groups = snapshotf_romxml.extract_region_satellites_from_file(dummy_file_name, start_ts, end_ts)
    target_time_step = len(region_satellite_groups)
    T = target_time_step
    for i in range(len(region_satellite_groups)):
        region_satellite_group = [[int(point) for point in region] for region in region_satellite_groups[i]]
        u = region_satellite_group[0]
        v = region_satellite_group[3]
        o = [u, v]
        regions_to_color[i] = o
    individual1 = writetoxml.xml_to_nodes("E:\\code\\data\\1\\individual1.xml")

  # uptime, down_time=find_time_period_for_establishment(individual1, (0, 1, 0), P, N, T)
    ##------------------------------##
    #usage
    mutate.disconenct_mutate((0, 1, 3), individual1, P, N, T,setuptime)
    ##------------------------------##


    connection_list = action_table.action_map2_shanpshots(individual1, P, N, T)
    # print("uptime", uptime)
    # print("down_time", down_time)
    vis = time_2d.DynamicGraphVisualizer(connection_list, regions_to_color, N, P)
    vis.show()

    # 3D 拓扑图可视化
    main_plotter, original_points_objs, all_coords = drawall.plot_multi_layer_topology(P, N, target_time_step)
    main_plotter = drawall.apply_region_colors(main_plotter, P, N, target_time_step, regions_to_color, all_coords)

    main_plotter = drawall.add_dashed_connections(main_plotter, connection_list)
    main_plotter.show(viewup="z", title="Interactive 3D Topology")