import collections

import numpy as np

import basicSa.simulator.neighbor as neighbor
from basicSa.satellite_stk_alpha.contrib.route import SatelliteRouter
def getpaths(firstsnap, N,P,stationsnum):
    graph = collections.defaultdict(list)

    first_satellitesnap = firstsnap.satellites

    def add_edge(u, v):
        graph[u].append(v)
        graph[v].append(u)


    for i in range(P * N):
        orbit = i // N
        planeid = i % N
        current_node = (orbit, planeid)
        currentsatellite = first_satellitesnap[i]
        six_links = currentsatellite[6:12]
        if orbit <= P - 3:
            # IT HAS 7
            # neighbor_orbit = orbit+1
            for j in range(6):
                # neighbor_orbitid, neighbor_planeid, _ = sv._get_right_neighbor2(i, j + 1)
                neighbor_orbitid, neighbor_planeid, _ = neighbor._get_right_neighbor2(N, P, i, j + 1)
                neighbornode = (neighbor_orbitid, neighbor_planeid)

                if six_links[j] == 1:
                    add_edge(current_node, neighbornode)

        elif orbit <= P - 2:
            for j in range(4):

                neighbor_orbitid, neighbor_planeid, _ = neighbor._get_right_neighbor2(N, P, i, j + 1)

                neighbornode = (neighbor_orbitid, neighbor_planeid)
                if six_links[j] == 1:
                    add_edge(current_node, neighbornode)

    first_stationsnap = firstsnap.stations
    gndlink = np.empty(stationsnum, dtype=object)

    for i in range(len(gndlink)):
        gndlink[i] = []  # 每个地面节点关联一个空列表（用于存储可见卫星）

    for i in range(stationsnum):
        stationsnap = first_stationsnap[i][4:]

        for j in range(7):
            if stationsnap[j] != -1:
                gndlink[i].append(stationsnap[j])

    router = SatelliteRouter(P, N)
    router.build_topology(graph)
    targets = [[0, 1]]
    allpaths = []
    for target in targets:

        gndnode0_id = target[0]

        gndnode1_id = target[1]

        # below it is the
        gdnlink_copy0_len = len(gndlink[target[0]])
        gdnlink_copy1_len = len(gndlink[target[1]])

        start_satellites = gndlink[target[0]]
        end_satellites = gndlink[target[1]]

        # test
        end_satellites = [end_satellites[0]]
        min_path = None
        min_distance = float('inf')

        # 遍历每个起始卫星
        for start_satellite in start_satellites:
            # 计算从当前卫星到所有节点的最短路径
            distances, paths = router.find_path(start_satellite)

            # 遍历所有可达的终点节点,start_satellite to all nodes in start_satellites
            # 遍历所有可达的终点节点（paths的键是目标节点）
            for end_node in paths.keys():  # 明确遍历字典的键（目标节点）
                # 检查当前目标节点是否在目标卫星列表中

                if end_node in end_satellites:
                    current_distance = distances[end_node]
                    current_path = paths[end_node]
                    # 更新最短路径
                    new_path = [x + stationsnum for x in current_path]
                    allpaths.append([target[0]] + new_path + [target[1]])

                    if current_distance < min_distance:
                        #      min_path,min_distance =get_the_min_orbit(path1, min_distance, path2, distance2, gndnodes, N)

                        min_distance = current_distance
                        min_path = current_path


                    elif current_distance == min_distance:

                        pass
    return  allpaths
