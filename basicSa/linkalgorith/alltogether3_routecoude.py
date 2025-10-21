
# core/link_engine.py
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
from collections import defaultdict
import basicSa.satellite_stk_alpha.config.settings as settings
# from config.settings import SatelliteConfig
import basicSa.calculate_angular.calculate_angular as calculate_angular
from basicSa.satellite_stk_alpha.contrib.route import SatelliteRouter
from basicSa.satellite_stk_alpha.core.angularvector2 import SatelliteVector
import basicSa.utilis.Node as Node
from basicSa.los import  Sat2Gnd
import collections
import basicSa.simulator.linktype as linktype
from math import inf
def addtoroute(node:Node.Node,destid,nexthop):

    for i in range(5,15):
        if node.get_value(i) ==[-1,-1]:
            node.set_value(i, [destid, nexthop])
            return

    #print("ERROR addtoroute")

def get_the_min_orbit(path1,distance1,path2,distance2,gndnodes,N):
    min_path = []
    min_distance = []
    stationnum = len(gndnodes)
    orbit_id1 = set()
    for i in range(1,len(path1)-1):
        satid = path1[i]-stationnum
        orbitid = satid //N
        orbit_id1.add(orbitid)
    orbit_id2 = set()
    for i in range(1,len(path2)-1):
        satid = path2[i]-stationnum
        orbitid = satid //N
        orbit_id2.add(orbitid)

    if len(orbit_id1)>len(orbit_id2):
        min_path = path2
        min_distance = distance2
    else:
        min_path = path1
        min_distance = distance1

    return min_path ,min_distance
def alltogether_new(time,P, N, positions, gndnodes,targets,oldpath):
    # 使用向量化计算
   # target:[[A,B],[C,D]]


    gndlenth = len(gndnodes)

    ## sat-sat visible
    # 计算角速度
    sv = SatelliteVector(P, N)
    sv.load_from_3dview(positions)

    # 批量计算
   # angular_vel = sv.calculate_angular_velocity(time)
  #  los_mask = angular_vel < settings.SatelliteConfig.ANGULAR_VELOCITY

    linkflags = sv.calculate_angular_velocity_7_all(time)

    min_paths = []
    # 生成邻接表

    graph = collections.defaultdict(list)
    def add_edge(u, v):
        graph[u].append(v)
        graph[v].append(u)
   # adj_list = defaultdict(list)
    for i in range(P*N):
        orbit = i // N
        planeid = i % N
        current_node = (orbit, planeid)



        if orbit<=P-3:
            # IT HAS 7
           # neighbor_orbit = orbit+1
            for j in range(6):
                neighbor_orbitid,neighbor_planeid,_ = sv._get_right_neighbor2(i,j+1)
                neighbornode  =(neighbor_orbitid,neighbor_planeid)
                if linkflags[j][i]==1:
                    add_edge(current_node, neighbornode)

        elif orbit<=P-2:
            for j in range(4):
                # if i==80:
                #     print(1)
                neighbor_orbitid, neighbor_planeid, _ = sv._get_right_neighbor2(i, j + 1)
                neighbornode = (neighbor_orbitid, neighbor_planeid)
                if linkflags[j][i] == 1:
                    add_edge(current_node, neighbornode)

    adj_list =graph



    #
    for i in range(len(positions)):
        node1 = positions[i][0]  # 获取当前卫星节点
        plainid = i // N  # 计算当前卫星所在的平面编号
        oribit_id = i % N  # 计算当前卫星在平面中的轨道编号
        up_oribit_id = (oribit_id - 1) % N  # 获取上一轨道的编号（循环连接）
        down_oribit_id = (oribit_id + 1) % N  # 获取下一轨道的编号（循环连接）
        upnodeid = plainid * N + up_oribit_id  # 计算上一轨道卫星的全局编号
        downnodeid = plainid * N + down_oribit_id  # 计算下一轨道卫星的全局编号
        node1.linked_array[2]=upnodeid + gndlenth
        node1.linked_array[4] = downnodeid + gndlenth
        node1.set_value(2, [upnodeid + gndlenth, 4])  # 设置当前卫星的上链路连接
        node1.set_value(4, [downnodeid + gndlenth, 2])  # 设置当前卫星的下链路连接




    ## sat-gnd visible
    ##-------------------------------------------------------------__#####
    # next we need calculate the sat and the ground
    positionslength = len(positions)
    # 遍历每个地面站（行索引 i）
   # gndlink = np.zeros(len(gndnodes), dtype='int32')
    # 假设 gndnodes 是地面节点的列表/数组
    gndlink = np.empty(len(gndnodes), dtype=object)  # 用 empty 替代 zeros

    # 为每个元素初始化为空列表
    for i in range(len(gndlink)):
        gndlink[i] = []  # 每个地面节点关联一个空列表（用于存储可见卫星）

    for i, node1 in enumerate(gndnodes):
        # node1:gnd
        # node2:sat
        # 遍历每个卫星位置（列索引 j）→ 直接取字典的值（即元组）
        for j in range(positionslength):
            node2 = positions[j][0]  # 从元组中提取 Node 对象
            visibility = Sat2Gnd.Ground_Sat(node1, node2)
            if visibility:
                gndlink[i].append((j,visibility))





    ## sat-gnd visible
    ##-------------------------------------------------------------#####
    router = SatelliteRouter(P, N)
    router.build_topology(adj_list)
    allpaths = []
    for target in targets:
        start_satellite = -1
        end_satellite = -1

        if str(target) in oldpath:
            lastpath = oldpath[str(target)]
        else:
            lastpath = None  # 或其他默认值

        gndnode0_id = target[0]
        gndnode0 = gndnodes[gndnode0_id]

        gndnode1_id = target[1]
        gndnode1 = gndnodes[gndnode1_id]


        # below it is the
        gdnlink_copy0_len = len(gndlink[target[0]])
        gdnlink_copy1_len = len(gndlink[target[1]])

        if gdnlink_copy0_len and gdnlink_copy1_len:
                pass
        else:
            gndnode0.linked_array[0] = -1
            gndnode1.linked_array[0] = -1
            break

        start_satellites = gndlink[target[0]]
        end_satellites = gndlink[target[1]]
        real_endsatellite = []
        for i in range(len(end_satellites)):
            real_endsatellite.append(end_satellites[i][0])


        min_path = None
        min_distance = float('inf')


        # 遍历每个起始卫星
        # for start_satellite in start_satellites:
            # 计算从当前卫星到所有节点的最短路径
        for start_satellite in start_satellites:
            distances, paths = router.find_path(start_satellite[0])


            # 遍历所有可达的终点节点,start_satellite to all nodes in start_satellites
            # 遍历所有可达的终点节点（paths的键是目标节点）
            for end_node in paths.keys():  # 明确遍历字典的键（目标节点）
                # 检查当前目标节点是否在目标卫星列表中


                if end_node in real_endsatellite:
                    current_distance = distances[end_node]
                    current_path = paths[end_node]

                    new_path = [x + gndlenth for x in current_path]

                    allpaths.append([target[0]] + new_path + [target[1]])


                    # 更新最短路径
                    if not lastpath:
                        if current_distance < min_distance:

                            min_distance = current_distance
                            min_path = current_path

                        elif current_distance == min_distance:
                            #attenetion we need gu
                            if   lastpath:
                                # if time==62:
                                #     print(f"lastpath is {lastpath}")
                                #     print(f"current_path is {current_path} and min_path is {min_path}")
                                if  lastpath==linktype.calculate_path_signature(current_path,N):
                                    min_distance = current_distance
                                    min_path = current_path

                    else:
                        if lastpath==linktype.calculate_path_signature(current_path,N):
                            min_distance = current_distance
                            min_path = current_path
                            break




    # gndnode0 link table

    if min_path:
        if  not lastpath:

            oldpath[str(target)] = linktype.calculate_path_signature(min_path,N)


        new_path = [x + gndlenth for x in min_path]






        start_satellite = min_path[0]
        gndnode0.set_value(0,( start_satellite+gndlenth,0))
        gndnode0.linked_array[0] = start_satellite+gndlenth

        #gndnode0 raw route table,
     #   gndnode0 .set_value(5, (1, start_satellite+gndlenth))

        addtoroute(gndnode0, gndnode1_id, start_satellite+gndlenth)



        # the we set the sat -sat
        for i in range(len(min_path)):
            if i!=len(min_path)-1:
                node = positions[min_path[i]][0]
                addtoroute(node, gndnode1_id, min_path[i+1]+gndlenth)
                nodeid = min_path[i]
                nexthopid = min_path[i+1]
                nodeid_orbitid = nodeid //N
                nexthopid_orbitid = nexthopid // N
                if nodeid_orbitid==nexthopid_orbitid: ## in one orbit
                    pass

                else:
                    node.linked_array[1] = nexthopid+gndlenth
                    node.set_value(1, (nexthopid + gndlenth, 3))
                    positions[nexthopid][0].linked_array[3] = nodeid+gndlenth
                    positions[nexthopid][0].set_value(3, (nodeid + gndlenth, 1))

         #       node.set_value(5,(gndnode1_id,path[i+1]+gndlenth))
            else:
                node = positions[min_path[i]][0]
                # attention the last basicSa need set the link table and the route table
                #link table
                node.linked_array[0] = gndnode1_id
              #  node.set_value(0,[gndnode1_id,0])
                # route table, casue it is the route to the gnd ,so there is  no need to set the link table for gnd link
                addtoroute(node, gndnode1_id, 0)



##here we need transform the  raw route table to the ns3-route table
    for i in range(len(gndnodes) ):
        node_y = gndnodes[i]
        for i in range(5, 15):
            if node_y.get_value(i) != [-1, -1] :
                nodeid_next = node_y.get_value(i)[1]
                for j in range(5):
                    if node_y.get_value(j)[0] == nodeid_next:
                        node_y.set_value(i, [node_y.get_value(i)[0], j])
            else:
                break

    for i in range(len(positions)):
        node_y = positions[i][0]
        for i in range(5, 15):
            if node_y.get_value(i) != [-1, -1]:
                nodeid_next = node_y.get_value(i)[1]
                if nodeid_next != 0:
                    for j in range(5):
                        if node_y.get_value(j)[0] == nodeid_next:
                            node_y.set_value(i, [node_y.get_value(i)[0], j])
            else:
                break

  #  print(allpaths)

    return  adj_list,allpaths

