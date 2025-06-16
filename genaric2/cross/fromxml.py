import argparse
import draw.snapshotf_romxml as snapshotf_romxml
import graph.drawall as drawall
import graph.time_2d2 as time_2d
import genaric2.action_table as action_table
import genaric2.writetoxml as writetoxml

def main(individual_file):
    # 参数设置
    N = 10
    P = 10
    start_ts = 1500
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

    # 读取个体 XML 数据
    individual1 = writetoxml.xml_to_nodes(individual_file)

    # 2D 动态图可视化
    connection_list = action_table.action_map2_shanpshots(individual1, P, N, T)
    vis = time_2d.DynamicGraphVisualizer(connection_list, regions_to_color, N, P)
    vis.show()

    # 3D 拓扑图可视化
    main_plotter, original_points_objs, all_coords = drawall.plot_multi_layer_topology(P, N, target_time_step)
    main_plotter = drawall.apply_region_colors(main_plotter, P, N, target_time_step, regions_to_color, all_coords)

    main_plotter = drawall.add_dashed_connections(main_plotter, connection_list)
    main_plotter.show(viewup="z", title="Interactive 3D Topology")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize satellite topology_prof and regions.")
    parser.add_argument("--input", type=str, required=True, help="Path to individual1.xml")
    args = parser.parse_args()
    main(args.input)
