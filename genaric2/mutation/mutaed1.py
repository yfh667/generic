
import genaric2.writetoxml as writetoxml
import draw.snapshotf_romxml as snapshotf_romxml
import graph.time_2d2 as time_2d
import genaric2.action_table as action_table
import graph.drawall as drawall
def establishment_mutate(coordinate,chromosome,P, N, T):
    #we will continued this estavlishment and cover the next establishment until the second setabliashment
    x = coordinate[0]
    y=coordinate[1]
    t=coordinate[2]
    right_neighbor=chromosome[coordinate].rightneighbor

    #first we will find the next establishment
    uptime, down_time=find_time_period_for_establishment(coordinate,chromosome,  P, N, T)
    nexttime_period =find_one_time_period_for_establishment(coordinate,chromosome , P, N, T)
    for i in range(nexttime_period,down_time+1):
        chromosome[(x, y, i)].state = 2
        chromosome[(x, y, i)].rightneighbor=(right_neighbor[0],right_neighbor[1],i)




def find_one_time_period_for_establishment(coordinate,chromosome ,P, N, T):
    x, y, t = coordinate

    while t + 1 < T :
        t += 1
        if chromosome[(x, y, t)].state == -1:  # Node is setting up the link
            break
    return t



def find_time_period_for_establishment(coordinate,chromosome, P, N, T):
    # Extract x, y, and t values from coordinate
    x, y, t = coordinate

    # If the node at the given coordinate is setting up the link (state == 0)
    if chromosome[coordinate].state == 0:
        # Flag for finding the next establishment
        flag = 2
        # Loop through time steps to find the next establishment
        while t + 1 < T and flag:
            t += 1
            if chromosome[(x, y, t)].state == 0:  # Node is setting up the link
                flag -= 1  # Decrease flag to exit loop

        # The time when the node stops setting up the link
        down_time = t - 1

        return coordinate[2], down_time

    # If the node at the given coordinate is working (state == 1)
    elif chromosome[coordinate].state == 1:
        # Start with the current time step
        uptime = t

        # Loop backward to find the last time the node was working
        while uptime - 1 >=0:
            uptime -= 1
            if chromosome[(x, y, uptime)].state == 0:  # Node is setting up the link
                break

        # Find the next time step when the node starts setting up the link
        t = coordinate[2]
        flag = 2

        # Loop through time steps to find the next establishment
        while t + 1 < T and flag:
            t += 1
            if chromosome[(x, y, t)].state == 0:  # Node is setting up the link

                flag -= 1  # Decrease flag to exit loop


        # The time when the node stops setting up the link
        down_time = t - 1

        return uptime, down_time



if __name__ == '__main__':
    N = 10
    P=10
    T=23
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
    individual1 = writetoxml.xml_to_nodes("E:\\code\\data\\1\\individual1.xml")

  # uptime, down_time=find_time_period_for_establishment(individual1, (0, 1, 0), P, N, T)
    establishment_mutate((0, 1, 0), individual1, P, N, T)
    connection_list = action_table.action_map2_shanpshots(individual1, P, N, T)


    print("1")
    # print("uptime", uptime)
    # print("down_time", down_time)
    vis = time_2d.DynamicGraphVisualizer(connection_list, regions_to_color, N, P)
    vis.show()

    # 3D 拓扑图可视化
    main_plotter, original_points_objs, all_coords = drawall.plot_multi_layer_topology(P, N, target_time_step)
    main_plotter = drawall.apply_region_colors(main_plotter, P, N, target_time_step, regions_to_color, all_coords)

    main_plotter = drawall.add_dashed_connections(main_plotter, connection_list)
    main_plotter.show(viewup="z", title="Interactive 3D Topology")